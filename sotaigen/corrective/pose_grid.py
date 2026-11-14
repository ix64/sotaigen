"""Piecewise-linear pose tables whose refinement leaves unrelated cells unchanged.

The table stores ordinary bind-space morphs. It does not establish collision safety:
the resulting skinned surface, including between samples, must be checked separately.
"""

import itertools

import numpy as np


class Correctives(dict):
    """Named morph deltas and the body-specific table that drives them."""

    def __init__(self, deltas, driver):
        super().__init__(deltas)
        self.driver = driver


def from_samples(samples, vertex_count, twist_limit):
    """Expand two local bind-space bakes and retain their exact pose tables.

    A two-axis table fades out as the arm twists, over `twist_limit` degrees, as the sculpted shoulder
    driver does (sotaigen.corrective.axilla.TWIST_LIMIT).
    """
    deltas, sides = {}, {}
    domain, axes = None, None
    for side in ("left", "right"):
        data = samples[side]
        grid = PoseGrid(data["points"], data["cells"])
        dimension = grid.points.shape[1]
        if dimension not in (2, 3):
            raise ValueError("Runtime shoulder samples need two or three pose coordinates")
        local_axes = [0, 2] if dimension == 2 else [0, 1, 2]
        if axes is not None and local_axes != axes:
            raise ValueError("Both shoulder tables must use the same pose coordinates")
        axes = local_axes
        if not (grid.points == 0).all(axis=1).any():
            raise ValueError("Shoulder table must contain the unchanged rest pose")
        bounds = np.column_stack((grid.points.min(axis=0), grid.points.max(axis=0)))
        if domain is not None and not np.array_equal(bounds, domain):
            raise ValueError("Both shoulder tables must cover the same domain")
        domain = bounds
        ids, local = np.asarray(data["ids"], int), np.asarray(data["deltas"], float)
        if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or not len(ids)
                or ids.min() < 0 or ids.max() >= vertex_count
                or local.shape != (len(grid.points), len(ids), 3) or not np.isfinite(local).all()):
            raise ValueError("Invalid local shoulder morph samples")
        names = []
        for i, (point, delta) in enumerate(zip(grid.points, local, strict=True)):
            if (point == 0).all():
                if np.max(np.abs(delta)) > 1e-8:
                    raise ValueError("Shoulder bake moves the rest pose")
                names.append(None)
                continue
            name = f"axilla{side.title()}Sample{i}"
            names.append(name)
            expanded = np.zeros((vertex_count, 3))
            expanded[ids] = delta
            deltas[name] = expanded
        sides[side] = grid.metadata() | {"targets": names}
    driver = {"kind": "simplices", "axes": axes, "domain": domain.tolist(), "sides": sides,
              "targets": {side: [name for name in local["targets"] if name]
                          for side, local in sides.items()}}
    if axes == [0, 2]:
        # Match the production driver's twist fade. Contact is audited at zero twist;
        # twisted poses need a separate contact audit.
        driver["twistLimit"] = twist_limit
    return Correctives(deltas, driver)


def side_weights(turns, group):
    """{target: weight} of one side's simplex table (a correctives group of "kind": "simplices"), from
    sotaigen.corrective.drive's readings: the arm as swing and twist, the right side read as the left."""
    from sotaigen.corrective.drive import reading

    arm = np.array([reading(turns, group["bone"], read, scale) for read, scale in
                    zip(("swingX", "twist", "swingZ"), group["scales"], strict=True)])
    domain = np.asarray(group["domain"], float)
    axes = group.get("sampled", [0, 1, 2])
    names = group["targets"]
    out = dict.fromkeys((name for name in names if name), 0.)
    point = np.clip(arm[axes], domain[:, 0], domain[:, 1])
    gain = 1.
    if axes == [0, 2]:
        t = np.clip(1 - abs(arm[1]) / group["twistLimit"], 0., 1.)
        gain = t * t * (3 - 2 * t)
    exact = np.flatnonzero((np.asarray(group["points"]) == point).all(axis=1))
    if len(exact):
        name = names[int(exact[0])]
        if name:
            out[name] = float(gain)
        return out
    first = np.einsum("cij,cj->ci", np.asarray(group["inverse"]),
                      point - np.asarray(group["origins"]))
    weights = np.column_stack((first, 1 - first.sum(axis=1)))
    inside = np.flatnonzero(weights.min(axis=1) >= -1e-10)
    if not len(inside):
        raise ValueError("Baked shoulder pose grid has a hole")
    cell = int(inside[0])
    selected = np.maximum(weights[cell], 0.)
    selected /= selected.sum()
    for index, weight in zip(group["cells"][cell], selected, strict=True):
        if names[index]:
            out[names[index]] = float(weight * gain)
    return out


def groups(table, source):
    """A simplex table's sides as correctives groups (sotaigen.corrective.drive)."""
    out = []
    for side in ("left", "right"):
        group = {"name": f"axilla.{side}", "source": source, "kind": "simplices", "bone": f"{side}UpperArm",
                 "scales": [1, 1, 1] if side == "left" else [1, -1, -1], "sampled": table.get("axes", [0, 1, 2]),
                 "domain": table["domain"], **table["sides"][side]}
        if "twistLimit" in table:
            group["twistLimit"] = table["twistLimit"]
        out.append(group)
    return out


def pose_weights(pose, table):
    """Mirror intent before locating a sample; later pose entries take precedence.

    The sampled coordinates are the arm's swing and twist (skinning.swing_twist), as in axilla.
    """
    from sotaigen.corrective.drive import weights

    return weights(pose, groups(table, "ipc"))


class PoseGrid:
    @classmethod
    def rectilinear(cls, axes):
        """Triangulate each rectangular cell with the same ordered diagonal.

        A Cartesian 3-D point cloud has coplanar subsets. Building the simplices
        directly avoids zero-volume tetrahedra from an unconstrained Delaunay call.
        """
        axes = [np.asarray(axis, float) for axis in axes]
        if (not axes or any(axis.ndim != 1 or len(axis) < 2
                            or not np.isfinite(axis).all() or (np.diff(axis) <= 0).any()
                            for axis in axes)):
            raise ValueError("Pose axes must be finite, strictly increasing sequences")
        shape = tuple(len(axis) for axis in axes)
        points = list(itertools.product(*axes))
        cells = []
        for origin in itertools.product(*(range(n - 1) for n in shape)):
            for order in itertools.permutations(range(len(axes))):
                point = np.array(origin)
                cell = [np.ravel_multi_index(tuple(point), shape)]
                for axis in order:
                    point[axis] += 1
                    cell.append(np.ravel_multi_index(tuple(point), shape))
                cells.append(cell)
        return cls(points, cells)

    def __init__(self, points, cells):
        self.points = np.asarray(points, float)
        self.cells = np.asarray(cells, int)
        if self.points.ndim != 2 or not np.isfinite(self.points).all():
            raise ValueError("Pose coordinates must be a finite matrix")
        dimension = self.points.shape[1]
        if (self.cells.ndim != 2 or self.cells.shape[1] != dimension + 1
                or not len(self.cells) or self.cells.min() < 0
                or self.cells.max() >= len(self.points)):
            raise ValueError("Invalid pose simplex indices")
        corners = self.points[self.cells]
        self.origins = corners[:, -1]
        # Columns are the edges from the last corner to the other corners.
        self.inverse = np.linalg.inv((corners[:, :-1] - self.origins[:, None]).transpose(0, 2, 1))

    def coordinates(self, point):
        """Barycentric weights in every cell; negative weights mean outside."""
        first = np.einsum("cij,cj->ci", self.inverse, np.asarray(point) - self.origins)
        return np.column_stack((first, 1 - first.sum(axis=1)))

    def locate(self, point):
        weights = self.coordinates(point)
        inside = np.flatnonzero(weights.min(axis=1) >= -1e-10)
        if not len(inside):
            raise ValueError(f"Pose outside baked domain: {point}")
        index = int(inside[0])
        # Remove only floating-point boundary noise, never extrapolate a pose.
        selected = np.maximum(weights[index], 0.)
        return self.cells[index], selected / selected.sum()

    def insert(self, point):
        """Split all containing cells, including both sides of a shared edge/face.

        Re-running an unconstrained triangulation can flip neighbouring diagonals
        and undo previous validation. Here only cells containing the new sample
        change, and their outer faces keep exactly the same interpolated values.
        """
        point = np.asarray(point, float)
        if point.shape != (self.points.shape[1],) or not np.isfinite(point).all():
            raise ValueError("Invalid new pose coordinate")
        if np.linalg.norm(self.points - point, axis=1).min() < 1e-9:
            raise ValueError("Pose sample already exists")
        weights = self.coordinates(point)
        inside = weights.min(axis=1) >= -1e-10
        if not inside.any():
            raise ValueError("New pose lies outside the baked domain")
        index = len(self.points)
        new = []
        for cell, barycentric in zip(self.cells[inside], weights[inside], strict=True):
            for opposite, weight in enumerate(barycentric):
                # A point on this facet would make a zero-volume simplex.
                if weight > 1e-10:
                    new.append([*np.delete(cell, opposite), index])
        return PoseGrid(np.vstack((self.points, point)), np.vstack((self.cells[~inside], new)))

    def metadata(self):
        return {"points": self.points.tolist(), "cells": self.cells.tolist(),
                "origins": self.origins.tolist(), "inverse": self.inverse.tolist()}
