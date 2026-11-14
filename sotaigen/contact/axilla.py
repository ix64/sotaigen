"""Offline sampling of fold support and continuous local shoulder contact.

The samples feed pose-space baking; exported interpolation is audited separately.
The solver poses a single arm by the swing and twist the correctives read
(skinning.swing_twist), in degrees; its rest collar and the opposite arm are fixed.
A rectangular domain can include poses inside the trunk, where fitting can severely
stretch the chest skin. The sampler handles local shoulder contact; general
trunk/forearm contact is outside its sampled region.
"""

import numpy as np

from sotaigen.contact.axilla_ribbons import constrain, ribbons
from sotaigen.contact.solver import ContactPatch
from sotaigen.corrective import axilla
from sotaigen.form.axilla import patch_weights
from sotaigen.mesh import adjacency, triangulate, vertex_normals
from sotaigen.rig.skinning import pose_skins, smoothed, torso_length


class ShoulderSampler:
    def __init__(self, base, verts, bones, weights, side):
        if side not in ("left", "right"):
            raise ValueError("Shoulder side must be left or right")
        self.side = side
        self.sign = 1 if side == "left" else -1
        self.verts, self.bones, self.weights = verts, bones, weights
        lookup = {b["name"]: b for b in bones}
        self.torso = torso_length(bones)
        raw = np.asarray(base["verts"])
        support = patch_weights(verts, lookup[side + "UpperArm"]["head"], self.torso)
        allowed = ((raw[:, 0] * self.sign > 0)
                   & ~np.isin(base["segments"], ("head", "neck")))
        support *= allowed
        self.goal_ids, triangles, self.goal_polygons = axilla._patch_topology(
            base["faces"], raw, support)
        self.goal_free = support[self.goal_ids]
        self.links = adjacency(len(self.goal_ids), triangles.tolist())
        self.detail = (verts[self.goal_ids]
                       - smoothed(verts[self.goal_ids], *self.links, 18, free=self.goal_free))
        self.ribbons = [r for r in ribbons(base, verts, bones) if r.side == side]
        # Every vertex moved by either authored fold must be allowed to respond
        # to contact. The old shoulder-only collar fixed several posterior-fold
        # vertices even though the ribbon moved them, making some rolls infeasible.
        for ribbon in self.ribbons:
            support = np.maximum(support, ribbon.support) * allowed
        # The upper arm can press below the axillary shape patch. Reserve contact
        # mobility along the complete humerus and the adjacent lateral trunk down
        # to its adducted reach. A shoulder-centred collar otherwise fixes the very
        # arm/chest vertices that need to separate during roll plus adduction.
        # This expands only contact response, not the authored fold or breast goal.
        arm = lookup[side + "UpperArm"]
        axis = arm["tail"] - arm["head"]
        length = np.linalg.norm(axis)
        t = np.clip((verts - arm["head"]) @ axis / (length * length), 0., 1.)
        along_arm = arm["head"] + t[:, None] * axis
        along_trunk = np.broadcast_to(arm["head"], verts.shape).copy()
        along_trunk[:, 2] += np.clip(verts[:, 2] - arm["head"][2], -length, 0.)
        contact_zone = np.isin(base["segments"], ("trunk", "upperArm")) & allowed
        envelope = np.maximum(patch_weights(verts, along_arm, self.torso),
                              patch_weights(verts, along_trunk, self.torso))
        support = np.maximum(support, envelope * contact_zone)
        self.ids, triangles, polygons = axilla._patch_topology(base["faces"], raw, support)
        self.solver = ContactPatch(verts[self.ids] / self.torso, triangles, polygons,
                                   support[self.ids], intrinsic_area=True, moving_boundary=True)
        self.triangles = triangulate(base["faces"], raw)
        self.local_weights = weights[self.ids]

    def pose(self, angles):
        x, y, z = angles
        return axilla.sample_pose(self.side, x, y, z)

    def matrices(self, angles, *, full=False):
        skins = pose_skins(self.bones, self.pose(angles))
        weights = self.weights if full else self.local_weights
        return skins, np.einsum("vb,bij->vij", weights, skins[:, :3])

    def posed(self, angles, delta):
        _, blend = self.matrices(angles)
        return (np.einsum("vij,vj->vi", blend[:, :, :3], self.verts[self.ids] + delta)
                + blend[:, :, 3])

    def goal(self, angles):
        skins, blend = self.matrices(angles, full=True)
        native = np.einsum("vij,vj->vi", blend[:, :, :3], self.verts) + blend[:, :, 3]
        local = blend[self.ids]
        u, _, vt = np.linalg.svd(local[:, :, :3])
        u[:, :, -1] *= np.linalg.det(u @ vt)[:, None]
        rotation = u @ vt
        # Build the goal at the actual 3-D pose, preserving local support without
        # an extra interpolation or the exported morph table's twist fade.
        goal_rotation = rotation[np.searchsorted(self.ids, self.goal_ids)]
        native[self.goal_ids] = axilla.surface_goal(
            self.verts[self.goal_ids], native[self.goal_ids], self.goal_polygons,
            goal_rotation, self.goal_free, self.links, self.detail)
        supported = constrain(self.verts, native, skins, self.ribbons,
                              vertex_normals(native, self.triangles))
        return supported[self.ids] / self.torso, rotation, local

    def sample(self, angles, *, start=None):
        """Continue from rest or a checked (angles, bind delta) sample.

        Each trial advances at most five degrees on any axis. IPC checks its
        nonlinear optimization path, which is not the final morph interpolation.
        The straight chord between solved targets is recorded separately: a bake
        must audit its actual interpolation, not use these samples as safe frames.
        Failed convergence is rolled back and retried with a smaller pose step.
        """
        angles = np.asarray(angles, float)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError("Shoulder angles must be three finite values")
        if start is None:
            previous = np.zeros(3)
            self.solver.x = self.verts[self.ids] / self.torso
        else:
            previous, delta = start
            previous = np.asarray(previous, float)
            self.solver.x = self.posed(previous, delta) / self.torso
        ipc = self.solver.ipc
        if ipc.has_intersections(self.solver.mesh, self.solver.x):
            raise ValueError("Shoulder continuation starts with an intersected sample")
        stats = []

        def advance(a, b, depth=0):
            original = self.solver.x.copy()
            goal, rotation, blend = self.goal(b)
            _, old_blend = self.matrices(a)
            material = np.linalg.solve(old_blend[:, :, :3],
                (original * self.torso - old_blend[:, :, 3])[:, :, None])[:, :, 0]
            predictor = (np.einsum("vij,vj->vi", blend[:, :, :3], material)
                         + blend[:, :, 3]) / self.torso
            try:
                result, stat = self.solver.advance(goal, rotation, initial=predictor)
                safe = ipc.compute_collision_free_stepsize(self.solver.mesh, original, result)
                stat["straightChordSafeFraction"] = float(safe)
            except RuntimeError as error:
                self.solver.x = original
                if depth == 8:
                    raise RuntimeError(f"{self.side} shoulder at {b.tolist()}: {error}") from error
                middle = (a + b) * .5
                advance(a, middle, depth + 1)
                return advance(middle, b, depth + 1)
            stats.append(stat)
            return result, blend

        initial = previous.copy()
        count = max(1, int(np.ceil(np.max(np.abs(angles - initial)) / 5)))
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            point = initial + fraction * (angles - initial)
            result, blend = advance(previous, point)
            previous = point
        bound = np.linalg.solve(blend[:, :, :3],
                                (result * self.torso - blend[:, :, 3])[:, :, None])[:, :, 0]
        return bound - self.verts[self.ids], stats
