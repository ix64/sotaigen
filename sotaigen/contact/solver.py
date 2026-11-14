"""Generation-time contact following for an already skinned shoulder patch.

IPC supplies vertex/face and edge/edge barriers and continuous collision detection.
This is a geometric fit to pose targets, not a calibrated tissue or muscle simulation.
The browser replays baked morphs; it does not run this numerical solver.
"""

import numpy as np


def vertex_filter(excluded, paired):
    """An ipctk collision filter: no pair touching an `excluded` vertex, none with both in `paired`.

    Built from ipctk's own filters so the broad phase never calls back into Python.
    Same-label pairs pass the negated filter, so it keeps the pairs whose two vertices are both excluded
    or both not. The second blocks same-label pairs: the excluded share one label, the paired another,
    and every other vertex has its own.
    """
    import ipctk

    excluded, paired = np.asarray(excluded, bool), np.asarray(paired, bool)
    labels = np.arange(2, len(excluded) + 2, dtype=np.int32)
    labels[paired] = 1
    labels[excluded] = 0
    return (~ipctk.make_vertex_patches_filter(excluded.astype(np.int32))
            & ipctk.make_vertex_patches_filter(labels))


class ContactPatch:
    """Continue a collision-free patch toward successive, closely spaced pose targets.

    Coordinates must be normalized by torso length. Mobility fades at the fixed collar.
    A differential-displacement penalty spreads contact over the surrounding skin;
    oriented-area protection prevents resolving penetration by flattening the axilla.
    """

    def __init__(self, rest, triangles, polygons, mobility, *, gap=0.002, intrinsic_area=False,
                 moving_boundary=False):
        import ipctk
        from scipy import sparse

        self.ipc = ipctk
        self.sparse = sparse
        self.intrinsic_area = intrinsic_area
        self.moving_boundary = moving_boundary
        self.x = np.asarray(rest, float).copy()
        self.polygons = np.asarray(polygons, int)
        first = (self.polygons[:, :, None] == self.polygons[:, None, :]).argmax(axis=2)
        self.unique_corners = first == np.arange(self.polygons.shape[1])
        self.corner_groups = []
        for pattern in np.unique(first, axis=0):
            rows = np.flatnonzero((first == pattern).all(axis=1))
            unique = np.flatnonzero(pattern == np.arange(len(pattern)))
            transform = np.zeros((len(unique), len(pattern)))
            transform[np.searchsorted(unique, pattern), np.arange(len(pattern))] = 1
            dofs = (3 * unique[:, None] + np.arange(3)).ravel()
            self.corner_groups.append((rows, dofs, np.kron(transform, np.eye(3))))
        self.corner_kind = np.empty(len(self.polygons), np.intp)
        for kind, (rows, _, _) in enumerate(self.corner_groups):
            self.corner_kind[rows] = kind
        self.gap = gap
        # Numerical clearance for interpolation; not an anatomical skin thickness.
        self.clearance = .02 * gap
        edges = ipctk.edges(triangles)
        self.mesh = ipctk.CollisionMesh(self.x, edges, triangles)
        if ipctk.has_intersections(self.mesh, self.x):
            raise ValueError("Contact continuation requires an intersection-free rest patch")
        self.fixed = np.asarray(mobility) == 0
        self.dofs = np.flatnonzero(np.repeat(~self.fixed, 3))
        row = np.arange(len(edges))
        difference = sparse.coo_matrix(
            (np.tile([1., -1.], len(edges)), (np.repeat(row, 2), edges.ravel())),
            shape=(len(edges), len(rest)),
        ).tocsr()
        self.hessian = sparse.kron(
            sparse.diags(np.where(self.fixed, 1., 1 / np.maximum(mobility, 1e-3)**2))
            + 5 * (difference.T @ difference), sparse.eye(3), format="csc",
        )
        original = self.x[self.polygons]
        normal = np.cross(original, np.roll(original, -1, axis=1)).sum(1)
        self.area = np.linalg.norm(normal, axis=1) * .5
        if np.any(self.area <= 1e-12):
            raise ValueError("Degenerate rest polygon")
        self.normal = normal / (2 * self.area[:, None])
        self.corner_dofs = (
            3 * self.polygons[:, :, None] + np.arange(3)
        ).reshape(len(polygons), -1)
        self._area_layout()
        # Dropping the fixed degrees of freedom keeps the others in order, so a sorted column stays sorted.
        self.free_index = np.full(self.x.size, -1)
        self.free_index[self.dofs] = np.arange(len(self.dofs))
        self.barrier = ipctk.BarrierPotential(gap, 1e8)
        self.collisions = ipctk.NormalCollisions()
        self.collisions.build(self.mesh, self.x, self.gap)
        if self.collisions.compute_minimum_distance(self.mesh, self.x) <= self.clearance**2:
            raise ValueError("Rest patch has insufficient numerical contact clearance")
        self.boundary_update_step = max(1e-6, .01 * float(np.median(
            np.linalg.norm(self.x[edges[:, 0]] - self.x[edges[:, 1]], axis=1))))

    def _area_layout(self):
        """Fix where each polygon's 12x12 area Hessian block lands in the summed sparse matrix.

        The blocks overlap, so their sum depends on the order duplicates are added in. This replays,
        once, how scipy turns these coordinates into a CSC matrix: entries bucketed by column in input
        order, each column sorted by row, then each run of equal rows summed from left to right. Every
        iteration then adds its values in that same order without sorting again, and the matrix it
        gives the solver is the same to the bit.
        """
        sparse, n = self.sparse, self.x.size
        width = self.corner_dofs.shape[1]
        rows = np.repeat(self.corner_dofs, width, axis=1).ravel()
        cols = np.tile(self.corner_dofs, (1, width)).ravel()
        bucketed = np.argsort(cols, kind="stable")
        probe = sparse.csc_matrix(
            (bucketed.astype(float), rows[bucketed], np.searchsorted(cols[bucketed], np.arange(n + 1))),
            shape=(n, n),
        )
        probe.has_sorted_indices = False
        probe.sort_indices()
        order = probe.data.astype(np.intp)
        key = cols[order] * n + rows[order]
        first = np.r_[True, key[1:] != key[:-1]]
        group = np.cumsum(first) - 1
        self.area_group = np.empty(len(order), np.intp)
        self.area_group[order] = group
        self.area_rank = np.empty(len(order), np.intp)
        self.area_rank[order] = np.arange(len(order)) - np.flatnonzero(first)[group]
        self.area_keys = key[first]
        self.area_indices = rows[order][first]
        self.area_indptr = np.searchsorted(cols[order][first], np.arange(n + 1))
        # The constant quadratic term on the union of both layouts, ready for the area values.
        quadratic = self.hessian.tocsc()
        quadratic.sum_duplicates()
        quadratic_keys = np.repeat(np.arange(n), np.diff(quadratic.indptr)) * n + quadratic.indices
        union = np.union1d(quadratic_keys, self.area_keys)
        self.union_indices, self.union_indptr = union % n, np.searchsorted(union // n, np.arange(n + 1))
        self.union_quadratic = np.zeros(len(union))
        self.union_quadratic[np.searchsorted(union, quadratic_keys)] = quadratic.data
        self.union_area = np.searchsorted(union, self.area_keys)

    def _assemble_area(self, active, blocks):
        """Sum the 12x12 blocks of the `active` polygons, the others being zero, into the area layout.

        Adding a zero changes no sum, so skipping the other polygons keeps every value; only a sum
        that is zero may change sign, and the solver drops zeros.
        """
        slots = (active[:, None] * blocks[0].size + np.arange(blocks[0].size)).ravel()
        group, rank, values = self.area_group[slots], self.area_rank[slots], blocks.ravel()
        total = np.zeros(len(self.area_keys))
        for k in range(rank.max() + 1 if len(rank) else 0):
            chosen = rank == k
            total[group[chosen]] += values[chosen]
        return total

    def _area_terms(self, points, normals, derivatives=False, *, project=True):
        if not derivatives:
            return self._area_parts(points, normals, False, project)
        _, gradient, values = self._area_parts(points, normals, True, project)
        n = points.size
        return gradient, self.sparse.csc_matrix((values, self.area_indices, self.area_indptr), shape=(n, n))

    def _area_parts(self, points, normals, derivatives, project=True):
        """The area penalty; with derivatives also its gradient and its Hessian in the area layout.

        Only compressed polygons have derivatives, so only they are differentiated.
        """
        p = points[self.polygons]
        area_vector = .5 * np.cross(p, np.roll(p, -1, axis=1)).sum(1)
        actual_area = np.maximum(np.linalg.norm(area_vector, axis=1), 1e-14)
        if self.intrinsic_area:
            normals = area_vector / actual_area[:, None]
            signed = actual_area
        else:
            signed = np.sum(area_vector * normals, axis=1)
        missing = np.maximum(.3 * self.area - signed, 0) / np.sqrt(self.area)
        value = 500 * np.sum(missing**2)
        if not derivatives:
            return value
        active = np.flatnonzero(missing > 0)
        if not len(active):
            return value, np.zeros(points.size), np.zeros(len(self.area_keys))
        p, normals, missing = p[active], normals[active], missing[active]
        area, actual_area = self.area[active], actual_area[active]
        gradient = -.5 * np.cross(
            np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1), normals[:, None, :]
        ) / np.sqrt(area[:, None, None])
        # Accumulated in the same order as np.add.at over every polygon, the others adding zeros.
        total = np.bincount(
            self.corner_dofs[active].ravel(),
            (1000 * missing[:, None, None] * gradient).ravel(),
            minlength=points.size,
        )
        gradient = gradient.reshape(len(active), -1)
        hessian = 1000 * gradient[:, :, None] * gradient[:, None, :]
        # Oriented polygon area is bilinear in neighbouring corners. Its exact
        # second derivative matters near a strongly bent, compressed fold; the
        # Gauss-Newton outer product alone can converge very slowly there.
        cross = np.zeros((len(normals), 3, 3))
        cross[:, 0, 1], cross[:, 0, 2] = -normals[:, 2], normals[:, 1]
        cross[:, 1, 0], cross[:, 1, 2] = normals[:, 2], -normals[:, 0]
        cross[:, 2, 0], cross[:, 2, 1] = -normals[:, 1], normals[:, 0]
        curvature = 500 * (missing / np.sqrt(area))[:, None, None] * cross
        corners = self.polygons.shape[1]
        for i in range(corners):
            a = slice(3 * i, 3 * i + 3)
            for j, sign in [((i + 1) % corners, 1), ((i - 1) % corners, -1)]:
                hessian[:, a, 3 * j:3 * j + 3] += sign * curvature
        if self.intrinsic_area:
            # d|C|/dx = J^T n, d²|C|/dx² also includes the change in n=C/|C|.
            span = np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)
            jacobian = np.zeros((len(p), corners, 3, 3))
            jacobian[:, :, 0, 1], jacobian[:, :, 0, 2] = span[:, :, 2], -span[:, :, 1]
            jacobian[:, :, 1, 0], jacobian[:, :, 1, 2] = -span[:, :, 2], span[:, :, 0]
            jacobian[:, :, 2, 0], jacobian[:, :, 2, 1] = span[:, :, 1], -span[:, :, 0]
            jacobian = .5 * jacobian.transpose(0, 2, 1, 3).reshape(len(p), 3, -1)
            tangent = np.eye(3) - normals[:, :, None] * normals[:, None, :]
            hessian -= (1000 * missing / (actual_area * np.sqrt(area)))[:, None, None] * (
                jacobian.transpose(0, 2, 1) @ tangent @ jacobian)
        if project:
            # Project in MATERIAL degrees of freedom. Projecting padded triangle
            # slots first would reintroduce a dependence on the duplicated corner.
            projected = np.zeros_like(hessian)
            for kind, (_, dofs, transform) in enumerate(self.corner_groups):
                rows = np.flatnonzero(self.corner_kind[active] == kind)
                if not len(rows):
                    continue
                packed = transform @ hessian[rows] @ transform.T
                values, vectors = np.linalg.eigh(packed)
                packed = (vectors * np.maximum(values, 0)[:, None, :]) @ vectors.transpose(0, 2, 1)
                projected[rows[:, None, None], dofs[None, :, None], dofs[None, None, :]] = packed
            hessian = projected
        return value, total, self._assemble_area(active, hessian)

    def _area_stepsize(self, points, step):
        """Keep material area positive throughout a linear Newton step.

        Projection onto the CURRENT face normal is a sufficient lower bound on
        area, not a permanent normal attachment to a bone. Its quadratic in the
        step parameter is checked analytically, so a step cannot jump across a
        flattened face even if its final unsigned area is large again.
        """
        p, d = points[self.polygons], step[self.polygons]
        c0 = .5 * np.cross(p, np.roll(p, -1, axis=1)).sum(1)
        c1 = .5 * (np.cross(p, np.roll(d, -1, axis=1))
                    + np.cross(d, np.roll(p, -1, axis=1))).sum(1)
        c2 = .5 * np.cross(d, np.roll(d, -1, axis=1)).sum(1)
        area = np.linalg.norm(c0, axis=1)
        normal = c0 / np.maximum(area[:, None], 1e-14)
        a, b, c = (c2 * normal).sum(1), (c1 * normal).sum(1), area - 1e-6 * self.area
        if (c <= 0).any():
            raise RuntimeError("Contact iterate has degenerate material area")
        roots = np.full((len(a), 2), np.inf)
        quadratic = np.abs(a) > 1e-14 * self.area
        discriminant = b * b - 4 * a * c
        live = quadratic & (discriminant >= 0)
        q = -.5 * (b[live] + np.where(b[live] >= 0, 1., -1.) * np.sqrt(discriminant[live]))
        roots[live, 0] = q / a[live]
        roots[live, 1] = np.divide(c[live], q, out=np.full_like(q, np.inf), where=q != 0)
        linear = ~quadratic & (b < 0)
        roots[linear, 0] = -c[linear] / b[linear]
        roots[roots <= 0] = np.inf
        first = float(roots.min())
        return .9 * first if first <= 1 else 1.

    def _free_block(self, hessian):
        """hessian[self.dofs][:, self.dofs] for a canonical CSC matrix, without sorting it again."""
        column = np.repeat(self.free_index, np.diff(hessian.indptr))
        row = self.free_index[hessian.indices]
        kept = (column >= 0) & (row >= 0)
        size = len(self.dofs)
        indptr = np.r_[0, np.cumsum(np.bincount(column[kept], minlength=size))]
        return self.sparse.csc_matrix((hessian.data[kept], row[kept], indptr), shape=(size, size))

    def advance(self, goal, rotations, *, initial=None):
        """Solve one pose step; raise on unsafe boundary motion or failed convergence.

        Every accepted Newton step is CCD-limited and energy-decreasing. This protects
        this solver's linear iterates, not arbitrary interpolation of baked morphs.
        """
        from scipy.sparse.linalg import spsolve

        ipc = self.ipc
        goal = np.asarray(goal, float)
        # Padding a triangle to four slots must not double-weight its last
        # material vertex when transporting the area constraint's normal.
        material_rotation = (
            (rotations[self.polygons] * self.unique_corners[:, :, None, None]).sum(1)
            / self.unique_corners.sum(1)[:, None, None]
        )
        normals = (material_rotation @ self.normal[:, :, None])[:, :, 0]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        x = self.x.copy() if initial is None else np.asarray(initial, float).copy()
        if x.shape != self.x.shape or not np.isfinite(x).all():
            raise ValueError("Invalid contact predictor")
        x[self.fixed] = goal[self.fixed]
        def admissible(start, candidate):
            return (ipc.compute_collision_free_stepsize(
                self.mesh, start, candidate, min_distance=self.clearance) == 1
                and (not self.intrinsic_area
                     or self._area_stepsize(start, candidate - start) == 1))

        hard_boundary = True
        if not admissible(self.x, x):
            # If the full kinematic predictor closes a contact, the previous free
            # vertices can sometimes still be used with the new fixed collar.
            x = self.x.copy()
            x[self.fixed] = goal[self.fixed]
            if not admissible(self.x, x):
                if not self.moving_boundary:
                    raise RuntimeError("Fixed collar motion crosses the local skin; subdivide or expand the patch")
                # IPC technical supplement (2020), §5: equality constraints for
                # moving boundary nodes. Start feasible and approach the prescribed
                # collar together with the surrounding skin, using an augmented
                # Lagrangian. Switch to exact DOF elimination only after a safe snap.
                x = self.x.copy()
                hard_boundary = False
        fixed_dofs = np.flatnonzero(np.repeat(self.fixed, 3))
        all_dofs = np.arange(x.size)
        dual = np.zeros(len(fixed_dofs))
        penalty = 1e6
        initial_residual = max(np.linalg.norm((self.x - goal).ravel()[fixed_dofs]), 1e-15)
        boundary_iterations = 0

        # The line search has already built the collisions at the step it accepts, and costed it.
        built = {}

        def energy(y):
            if "at" not in built or not np.array_equal(built["at"], y):
                self.collisions.build(self.mesh, y, self.gap, dmin=self.clearance)
                offset = (y - goal).ravel()
                built["at"], built["value"] = y.copy(), (
                    .5 * offset @ (self.hessian @ offset)
                    + self.barrier(self.collisions, self.mesh, y)
                    + self._area_terms(y, normals))
            value = built["value"]
            if not hard_boundary:
                offset = (y - goal).ravel()
                residual = offset[fixed_dofs]
                value += dual @ residual + .5 * penalty * (residual @ residual)
            return value

        alpha = 0.
        for iteration in range(1000):
            if not hard_boundary:
                snapped = x.copy()
                snapped[self.fixed] = goal[self.fixed]
                if admissible(x, snapped):
                    x = snapped
                    hard_boundary = True
                else:
                    boundary_iterations += 1
            value = energy(x)
            gradient = self.hessian @ (x - goal).ravel()
            gradient += self.barrier.gradient(self.collisions, self.mesh, x)
            _, area_gradient, area_values = self._area_parts(x, normals, True)
            gradient += area_gradient
            # The quadratic and area terms share one fixed layout; scipy's sum with the barrier then
            # drops the zeros, as summing the three matrices one by one did.
            fixed_layout = self.union_quadratic.copy()
            fixed_layout[self.union_area] += area_values
            hessian = self.sparse.csc_matrix(
                (fixed_layout, self.union_indices, self.union_indptr), shape=(x.size, x.size)
            ) + self.barrier.hessian(self.collisions, self.mesh, x, ipc.PSDProjectionMethod.CLAMP)
            dofs = self.dofs if hard_boundary else all_dofs
            if not hard_boundary:
                residual = (x - goal).ravel()[fixed_dofs]
                gradient[fixed_dofs] += dual + penalty * residual
                diagonal = np.zeros(x.size)
                diagonal[fixed_dofs] = penalty
                hessian += self.sparse.diags(diagonal, format="csc")
            step = np.zeros(x.size)
            step[dofs] = -spsolve(self._free_block(hessian) if hard_boundary else hessian, gradient[dofs])
            if not np.isfinite(step).all():
                raise RuntimeError("Non-finite contact solve")
            step = step.reshape(-1, 3)
            step_size = np.linalg.norm(step, axis=1).max()
            if not hard_boundary and step_size < self.boundary_update_step:
                progress = 1 - np.linalg.norm(residual) / initial_residual
                if progress < .99 and penalty < 1e8:
                    penalty = min(2 * penalty, 1e8)
                else:
                    dual += penalty * residual
                continue
            if hard_boundary and step_size < 1e-6:
                break
            alpha = ipc.compute_collision_free_stepsize(
                self.mesh, x, x + step, min_distance=self.clearance
            )
            if self.intrinsic_area:
                alpha = min(alpha, self._area_stepsize(x, step))
            slope = gradient @ step.ravel()
            for _ in range(30):
                candidate = x + alpha * step
                if alpha > 0 and energy(candidate) <= value + 1e-4 * alpha * slope:
                    x = candidate
                    break
                alpha *= .5
            else:
                raise RuntimeError("Contact line search stalled")
        else:
            raise RuntimeError(f"Contact solve did not converge: step={np.linalg.norm(step, axis=1).max():.3g}, alpha={alpha:.3g}")
        if ipc.has_intersections(self.mesh, x):
            raise RuntimeError("Contact result intersects")
        if not hard_boundary or not np.array_equal(x[self.fixed], goal[self.fixed]):
            raise RuntimeError("Contact boundary did not reach its prescribed position")
        self.x = x
        return x.copy(), {"iterations": iteration, "boundaryIterations": boundary_iterations,
                         "maxShift": float(
            np.linalg.norm(x - goal, axis=1).max())}
