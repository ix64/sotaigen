"""The frozen contracts, against golden copies in tests/contracts/: what a downstream may rely on.

A change here breaks every downstream file made before it (docs/DECISIONS.md, 冻结的对外约定), so a
failure is a decision to make, not a test to update: change the golden copy only with a new major
version and a migration note.
"""

import hashlib
import json
from pathlib import Path

import pytest

from sotaigen.form.bodies import body_specs
from sotaigen.output.glb import head_attachments
from sotaigen.pipeline import rigged, template
from sotaigen_dev import contracts

GOLDEN = Path(__file__).resolve().parent / "contracts"
BONES = json.loads((GOLDEN / "bones.json").read_text())
TOPOLOGY = json.loads((GOLDEN / "topology.json").read_text())


def test_the_bones_come_parents_first_with_the_head_bones_last():
    seen = set()
    for bone in BONES:
        assert bone["parent"] is None or bone["parent"] in seen, bone
        seen.add(bone["name"])
    # Last only so that every other bone kept its column when they came (docs/systems/rig.md, 骨骼约定).
    assert [bone["name"] for bone in BONES[-3:]] == ["jaw", "leftEye", "rightEye"]


@pytest.mark.parametrize("name", sorted(body_specs()))
def test_every_body_has_the_contracts_bones_in_its_order(name):
    assert [{"name": b["name"], "parent": b["parent"]} for b in rigged(name).bones] == BONES


def test_the_template_has_the_contracts_topology():
    base = template()
    faces = [[int(i) for i in face] for face in base["faces"]]
    assert len(base["verts"]) == TOPOLOGY["vertices"]
    assert len(faces) == TOPOLOGY["faces"]
    assert hashlib.sha256(json.dumps(faces, separators=(",", ":")).encode()).hexdigest() == TOPOLOGY["facesSha256"]


@pytest.mark.parametrize("name", sorted(body_specs()))
def test_every_body_has_the_contracts_head_interface(name):
    rig = rigged(name)
    head = head_attachments(rig.base, rig.verts, rig.bones)
    assert head["region"] == TOPOLOGY["headRegion"]
    assert head["neckRing"] == TOPOLOGY["neckRing"]
    assert head["crownVertex"] == TOPOLOGY["crownVertex"]


# --- The parameter file, the typical bodies, the command line, the diagnostics, the GLB, the .blend ---
# Each against its golden copy, read from the code by sotaigen_dev.contracts.


def golden(name):
    return json.loads((GOLDEN / f"{name}.json").read_text())


def test_the_parameter_file_has_the_contracts_fields_names_and_units():
    assert contracts.parameters() == golden("parameters")


def test_the_typical_bodies_have_the_contracts_names():
    assert contracts.bodies() == golden("bodies")


def test_the_command_line_has_the_contracts_commands_arguments_and_exit_codes():
    assert contracts.command_line() == golden("cli")


def test_every_diagnostic_has_the_contracts_code_and_values():
    """A code may be added; one taken away, or its values changed, breaks a reader."""
    found, frozen = contracts.diagnostics(), golden("diagnostics")
    assert found["levels"] == frozen["levels"]
    assert found["codes"] == frozen["codes"]


def test_the_blend_scripts_name_things_as_the_contract_does():
    assert contracts.blend_names() == golden("blend")["names"]


@pytest.mark.slow
def test_a_glb_has_the_contracts_fields(tmp_path):
    """Made as a downstream makes it, from one body of each sex."""
    found = contracts.merged_glb([contracts.glb_of(name, tmp_path) for name in contracts.READ_FROM])
    assert found == golden("glb")


@pytest.mark.blender
def test_a_written_blend_holds_the_contracts_properties_bones_and_pose_assets(tmp_path):
    files = [contracts.blend_file(contracts.read_blend(name, tmp_path)) for name in contracts.READ_FROM]
    frozen = golden("blend")["file"]
    for key in ("info", "bones", "poseAssets"):
        assert sorted(set().union(*(f[key] for f in files))) == frozen[key], key
