"""A finished rig's pose correctives, the shoulder's, the pelvis's and the hanging arm's, as one set of
morph targets.

What the exporter stores, what the pose metrics measure and what the runtime check compares all go
through here, so that they can't come to weight or apply the targets differently.
"""

from sotaigen.corrective import axilla, deltoid, drive, pectoral
from sotaigen.corrective import groin as groin_driver
from sotaigen.corrective import hang as hang_driver

# Where each group came from, by how production() made them: the sculpted shoulder and pelvis and the
# implicit-field hanging arm, or IPC contact tables.
SOURCES = {
    "sculpted": {"axilla": "sculpted", "groin": "sculpted", "hang": "field"},
    "ipc": {"axilla": "ipc", "groin": "ipc", "hang": "ipc"},
}


class Correctives:
    """Morph targets that a pose drives: deltas in the rig's frame, by target name.

    `shoulder` is what sotaigen.corrective.axilla.targets or sotaigen.contact.axilla_cache.targets
    gives, `groin` what sotaigen.contact.groin.targets or sotaigen.corrective.groin.sculpted_targets
    gives. Either may be left out. `hang` and `hang_table` are what sotaigen.contact.hang.targets or
    sotaigen.corrective.hang_field.targets gives; they continue below the shoulder's edge, on top of
    what it delivers there, so they need `shoulder`.
    """

    def __init__(
        self, shoulder=None, groin=None, hang=None, hang_table=None, contact=None, normals=None, sources=None
    ):
        self.shoulder, self.groin = shoulder, groin
        # Normal deltas already made elsewhere (the editor makes some in another process), by name.
        # A group that has all of its here isn't made again.
        self.normals = normals or {}
        # How production() made them, for the GLB to say: "ipc", or "sculpted" for the sculpted
        # shoulder and pelvis with the implicit-field hanging arm. None when made by hand.
        self.contact = contact
        # Each group's source, for the file to say: an IPC bake that failed fell back to the sculpture.
        self.sources = {**SOURCES.get(contact, {}), **(sources or {})}
        self.hang, self.hang_table = (hang, hang_table) if hang_table else (None, None)
        if self.hang_table is not None and shoulder is None:
            raise ValueError("the hanging-arm table sits on the shoulder's, which is missing")
        names = [set(group or ()) for group in (shoulder, groin, self.hang)]
        clash = (names[0] & names[1]) | (names[0] & names[2]) | (names[1] & names[2])
        if clash:
            raise ValueError(f"corrective groups share target names: {sorted(clash)}")

    @classmethod
    def production(cls, base, verts, bones, weights, *, field=None, contact=False, progress=None):
        """What a generated body ships with.

        By default the shoulder's and the pelvis's sculpted correctives and the hanging arm's
        implicit-field contact (sotaigen/corrective/hang_field.py), in seconds and without any cache,
        so one set of parameters gives one result everywhere. That contact reads the body's
        calibrated `field` (anatomy.calibrated, or pipeline.surface for a body), which is required
        here. With `contact` the IPC contact tables of the shoulder, the pelvis and the hanging arm
        instead (sotaigen[contact]): minutes per body, baked once per input and cached. Either way the
        pectoral plate's outer part follows a raised arm on the shoulder's targets
        (sotaigen/corrective/pectoral.py), which reads `field` too, and the raised shoulder's hollow
        and gathered deltoid go on them as well (sotaigen/corrective/deltoid.py). `progress`, if
        given, hears "shoulder", "pelvic" and "hang" as each starts.
        """
        if contact:
            # Imported here so that the default path doesn't load the contact solver.
            from sotaigen.contact.axilla_cache import targets as shoulder_targets
            from sotaigen.contact.groin import targets as groin_targets
            from sotaigen.contact.hang import targets as hang_targets
        else:
            if field is None:
                raise TypeError("the default correctives need the body's field for the hanging arm")
            # Imported here so that what only reads a GLB doesn't load the fields.
            from sotaigen.corrective.hang_field import targets as implicit_targets

            shoulder_targets, groin_targets = axilla.targets, groin_driver.sculpted_targets

        say = progress or (lambda stage: None)
        say("shoulder")
        shoulder = shoulder_targets(base, verts, bones, weights)
        sources = {}
        if contact:
            from sotaigen.contact.groin import failure as pelvic_failure

            # A failed shoulder bake gives the sculpted correctives back, without a simplex table.
            sources["axilla"] = "ipc" if hasattr(shoulder, "driver") else "sculpted"
            sources["groin"] = "sculpted" if pelvic_failure(base, verts, bones, weights) else "ipc"
        shoulder = pectoral.follow(shoulder, base, verts, bones, field)
        shoulder = deltoid.gather(shoulder, base, verts, bones)
        say("pelvic")
        groin = groin_targets(base, verts, bones, weights)
        say("hang")
        if contact:
            hang, table = hang_targets(base, verts, bones, weights)
        else:
            hang, table = implicit_targets(base, verts, bones, weights, field, shoulder)
        return cls(shoulder, groin, hang, table, contact="ipc" if contact else "sculpted", sources=sources)

    @property
    def targets(self):
        """Every target, shoulder first, then pelvic, then hanging arm."""
        return {**(self.shoulder or {}), **(self.groin or {}), **(self.hang or {})}

    def groups(self):
        """The driver table as files carry it, extras.sotaigen.correctives.groups: a group per side and
        part, in the order of `targets`, each with its source (sotaigen.corrective.drive)."""
        if getattr(self, "_groups", None) is not None:
            return self._groups
        source = lambda part: self.sources.get(part, "made")  # a group made by hand says so
        out = []
        if self.shoulder is not None:
            out += axilla.groups(axilla.driver_metadata(self.shoulder), source("axilla"))
        if self.groin is not None:
            out += groin_driver.groups(source("groin"), groin_driver.keys_of(self.groin))
        if self.hang_table is not None:
            out += hang_driver.groups(self.hang_table, source("hang"))
        # The targets don't change once made, so neither does their table.
        self._groups = out
        return out

    def metadata(self):
        """The driver table the runtime reads, under the key the GLB stores it by."""
        return {"correctives": {"groups": self.groups()}}

    def pose_weights(self, pose):
        """How much of each target a pose takes."""
        return drive.weights(pose, self.groups())

    def bind(self, verts, pose):
        """The rest skin with the targets the pose drives added, to skin into that pose."""
        if self.shoulder is not None:
            verts = axilla.bind_verts(verts, self.shoulder, pose)
        if self.groin is not None:
            verts = groin_driver.bind_verts(verts, self.groin, pose)
        if self.hang_table is not None:
            verts = hang_driver.bind_verts(verts, self.hang, pose, self.hang_table)
        return verts

    def normal_targets(self, base, verts, bones, weights, normals, *, triangles=None):
        """Normal deltas to go with the targets, for the targets that move the skin."""
        out = {}

        def made(group):
            return all(name in self.normals for name in group)

        if self.shoulder is not None:
            if made(self.shoulder):
                out |= {name: self.normals[name] for name in self.shoulder}
            else:
                out |= axilla.normal_targets(
                    base, verts, bones, weights, self.shoulder, normals, triangles=triangles
                )
        if self.groin is not None:
            if made(self.groin):
                out |= {name: self.normals[name] for name in self.groin}
            else:
                out |= groin_driver.normal_targets(
                    base, verts, bones, weights, self.groin, normals, triangles=triangles
                )
        if self.hang_table is not None:
            missing = [name for name in self.hang if name not in self.normals]
            out |= hang_driver.normal_targets(
                base, verts, bones, weights, self.shoulder, self.hang, self.hang_table, normals,
                triangles=triangles, names=set(missing),
            ) if missing else {}
            out |= {name: self.normals[name] for name in self.hang if name in self.normals}
        return out
