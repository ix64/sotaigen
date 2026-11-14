"""The review tool's logic: selection, framing, layout and cache. Nothing here needs Blender."""

import itertools
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sotaigen.form.bodies import load_bodies
from sotaigen.form.head import extended
from sotaigen.output import blender
from sotaigen.pipeline import fingerprints, write_fingerprint
from sotaigen.rig import poses as real_poses
from sotaigen.template import load_template
from sotaigen_dev import review

SCRIPT = Path(review.__file__)


# Stand-ins for data/ and sotaigen/rig/poses.py: the logic under test must hold whatever bodies and poses
# the project defines, and these names and sizes are the tests' own, not the project's.
STATURES = {"petite": 1150.0, "regular": 1400.0, "tall": 1650.0}
STYLES = ("soft", "slender", "structural")
PRESETS = {f"{b}-{s}": {"stature": h} for b, h in STATURES.items() for s in STYLES}
POSES = {pose: {} for pose in ("rest", "armsDown", "armRoll", "reach", "crouch", "stride", "fist")}


@pytest.fixture(autouse=True)
def catalogue(monkeypatch):
    monkeypatch.setattr(review, "load_bodies", lambda: PRESETS)
    monkeypatch.setattr(review, "load_typical", lambda: [f"{b}-soft" for b in STATURES])
    monkeypatch.setattr(review, "POSES", POSES)
    review.stature.cache_clear()
    yield
    review.stature.cache_clear()


@pytest.fixture(scope="module")
def base():
    return extended(load_template())


def box(width, depth, height, floor=0.0, x=0.0, y=0.0):
    """Eight corners of a box standing at `floor`, a stand-in for a posed body."""
    corners = itertools.product((-width / 2, width / 2), (-depth / 2, depth / 2), (0.0, height))
    return (np.array(list(corners), float) + (x, y, floor)).round(4)


def write_obj(path, verts):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x} {y} {z}" for x, y, z in verts] + ["f 1 2 3"]
    path.write_text("\n".join(lines) + "\n")


def write_pose(build, preset, pose, verts):
    """A posed mesh, with the fingerprint a finished build of the current sources leaves."""
    path = review.pose_path(build, preset, pose)
    write_obj(path, verts)
    write_fingerprint(Path(build) / preset, {"pose": fingerprints()["pose"]})
    return path


def mesh_of(verts, name="m"):
    return review.Mesh(Path(f"{name}.obj"), name, np.asarray(verts, float))


def body_specs(meshes, view="front", pose="rest"):
    """One single-pane cell per named mesh, along one row."""
    axes = {"preset": list(meshes), "pose": [pose], "view": [view]}
    grid = review.arrange(axes, cols=["preset"])
    return grid, [
        review.CellSpec(cell, (review.Pane("", meshes[cell.preset]),)) for cell in grid.cells
    ]


@pytest.fixture
def build(tmp_path, monkeypatch):
    folder = tmp_path / "build"
    monkeypatch.setattr(review, "BUILD", folder)
    return folder


def plant_renders(layout, wire=False):
    """Stand in for Blender: fill the cache with frames whose body fills the projected box."""
    review.cache_dir().mkdir(parents=True, exist_ok=True)
    jobs = [(item.mesh, item.frame) for item in layout.placed]
    jobs += [(item.under, item.frame) for item in layout.placed if item.under is not None]
    for mesh, frame in jobs:
        image = Image.new("RGBA", (frame.width, frame.height), (0, 0, 0, 0))
        pixels = np.array(image)
        uv = review.project(mesh.verts, frame.view)
        scale = frame.mm_per_px
        columns = (uv[:, 0] - frame.u) / scale + frame.width / 2
        rows = frame.height / 2 - (uv[:, 1] - frame.v) / scale
        top, bottom = round(rows.min()), round(rows.max())
        left, right = round(columns.min()), round(columns.max())
        pixels[top:bottom, left:right] = (90, 95, 105, 255)
        Image.fromarray(pixels).save(review.cache_path(mesh, frame, wire))


# --- Selection ---


def test_selection_by_body_style_pose_view():
    axes = review.select_axes(sex="petite,tall", group="soft", pose="rest,crouch", view="front,side")
    assert axes == {
        "sex": ["petite", "tall"],
        "group": ["soft"],
        "pose": ["rest", "crouch"],
        "view": ["front", "side"],
    }


def test_selection_defaults_and_all():
    axes = review.select_axes(pose="all", view="all", group="all")
    assert axes["sex"] == ["petite", "regular", "tall"]
    assert axes["pose"] == list(POSES)
    assert axes["view"] == list(review.VIEWS)
    assert axes["group"] == list(STYLES)
    plain = review.select_axes()
    assert plain["group"] == ["soft"] and plain["pose"] == ["rest"]
    assert plain["view"] == ["front", "threequarter", "side", "back"]


def test_the_catalogue_is_what_the_project_defines(monkeypatch):
    monkeypatch.setattr(review, "load_bodies", load_bodies)
    names, bodies, styles = review.catalog()
    assert names == list(load_bodies())
    # Each template has its own groups, so every body is a body and a style, but not every body and
    # style make a body.
    parts = [name.rsplit("-", 1) for name in names]
    assert {body for body, _ in parts} == set(bodies)
    assert {style for _, style in parts} == set(styles)


def test_the_defaults_name_what_the_project_defines(monkeypatch):
    monkeypatch.setattr(review, "load_bodies", load_bodies)
    monkeypatch.setattr(review, "POSES", real_poses.POSES)
    axes = review.select_axes()
    assert set(axes["pose"]) <= set(real_poses.POSES)
    if "preset" in axes:
        defaults = set(axes["preset"])
    else:
        defaults = {f"{body}-{style}" for body in axes["sex"] for style in axes["group"]}
    assert defaults <= set(load_bodies())


def test_selection_by_preset_name():
    axes = review.select_axes(["regular-soft", "tall-slender"], pose="crouch")
    assert axes["preset"] == ["regular-soft", "tall-slender"]
    assert "sex" not in axes and "group" not in axes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"presets": ["regular-nope"]},
        {"sex": "h99"},
        {"group": "chunky"},
        {"pose": "rest,moonwalk"},
        {"view": "top"},
        {"focus": "tail"},
        {"pose": ","},
        {"presets": ["regular-soft"], "sex": "regular"},
    ],
)
def test_selection_rejects_unknown_names(kwargs):
    with pytest.raises(review.ReviewError) as caught:
        review.select_axes(**kwargs)
    assert caught.value.code == review.EXIT_USAGE


def test_unknown_pose_error_lists_the_known_ones():
    with pytest.raises(review.ReviewError, match="moonwalk.*armsDown"):
        review.select_axes(pose="moonwalk")


# --- Grid ---


def test_explicit_rows_and_cols():
    axes = review.select_axes(pose="rest,crouch", view="front,side")
    grid = review.arrange(axes, rows=["sex"], cols=["pose", "view"])
    assert grid.row_titles == ["petite", "regular", "tall"]
    assert grid.col_titles == ["rest / front", "rest / side", "crouch / front", "crouch / side"]
    assert len(grid.cells) == 12
    cell = next(c for c in grid.cells if c.row == 2 and c.col == 3)
    assert cell.preset == "tall-soft"
    assert cell.coord["pose"] == "crouch" and cell.coord["view"] == "side"


def test_bodies_default_to_columns_so_a_row_shares_a_ground_line():
    grid = review.arrange(review.select_axes(view="front,side"))
    assert grid.col_titles == ["petite", "regular", "tall"]
    assert grid.row_titles == ["front", "side"]


def test_unplaced_axes_are_not_lost():
    axes = review.select_axes(group="soft,slender", view="front,side", pose="rest")
    grid = review.arrange(axes, rows=["view"])
    assert len(grid.cells) == 3 * 2 * 2
    assert {c.preset for c in grid.cells} == {
        f"{b}-{s}" for b in ("petite", "regular", "tall") for s in ("soft", "slender")
    }


def test_one_long_axis_wraps_and_a_short_one_stays_in_a_row():
    many = review.arrange(review.select_axes(["regular-soft"], pose="all", view="side"))
    assert len(many.cells) == len(POSES)
    assert len(many.col_titles) < len(POSES) and len(many.row_titles) > 1
    assert len({(c.row, c.col) for c in many.cells}) == len(POSES)
    few = review.arrange(review.select_axes(["regular-soft"]))
    assert len(few.row_titles) == 1 and len(few.col_titles) == 4


@pytest.mark.parametrize(
    "rows,cols",
    [(["nope"], []), ([], ["nope"]), (["pose"], ["pose"])],
)
def test_bad_axes_are_refused(rows, cols):
    with pytest.raises(review.ReviewError) as caught:
        review.arrange(review.select_axes(), rows, cols)
    assert caught.value.code == review.EXIT_USAGE


def test_sheet_names_differ_with_the_selection():
    plain = review.select_axes(["regular-soft"], view="front")
    assert review.sheet_name(plain) == "regular-soft_rest_front.png"
    names = {
        review.sheet_name(plain),
        review.sheet_name(plain, wire=True),
        review.sheet_name(plain, against="/a"),
        review.sheet_name(plain, against="/b"),
        review.sheet_name(review.select_axes(["regular-soft"], view="side")),
        review.sheet_name(review.select_axes(["regular-soft"], view="front", focus="hip")),
    }
    assert len(names) == 6
    long = review.select_axes(["regular-soft"], pose="all", view="front")
    assert len(review.sheet_name(long)) < 60


# --- One scale, one ground line ---


def test_one_scale_across_a_sheet_of_different_sizes():
    _, specs = body_specs(
        {
            "petite-soft": mesh_of(box(400, 200, 1150)),
            "regular-soft": mesh_of(box(600, 250, 1400)),
            "tall-soft": mesh_of(box(900, 300, 1650)),
        }
    )
    scale, frames = review.plan_frames(specs)
    assert {frame.mm_per_px for row in frames for frame in row} == {scale}
    assert scale in review.SCALES
    # The largest body fits the default cell at this scale and would not at the next finer one.
    assert 1650 / scale <= review.BODY_CELL_PX
    assert 1650 / review.SCALES[review.SCALES.index(scale) - 1] > review.BODY_CELL_PX


def test_a_fixed_scale_is_used_as_given():
    _, specs = body_specs({"regular-soft": mesh_of(box(600, 250, 1400))})
    scale, frames = review.plan_frames(specs, mm_per_px=0.7)
    assert scale == 0.7 and frames[0][0].mm_per_px == 0.7


def test_pixel_heights_keep_the_mesh_ratio():
    heights = {"petite-soft": 1150.3, "regular-soft": 1399.7, "tall-soft": 1649.7}
    _, specs = body_specs({n: mesh_of(box(400, 200, h)) for n, h in heights.items()})
    _, frames = review.plan_frames(specs)
    pixels = {spec.grid.preset: row[0].ground - review.TOP_PAD for spec, row in zip(specs, frames)}
    for a, b in itertools.combinations(heights, 2):
        assert pixels[a] / pixels[b] == pytest.approx(heights[a] / heights[b], rel=0.01)


def test_lowest_vertex_lands_on_the_frame_ground_row():
    # A crouch lifts the feet off z=0: the body is placed by where it actually is.
    verts = box(400, 200, 1000, floor=442.0)
    for view in review.VIEWS:
        frame = review.body_frame(review.extent(verts, view), view, 2.0)
        lowest = review.project(verts, view)[:, 1].min()
        bottom = frame.v - frame.height / 2 * frame.mm_per_px
        row = frame.height - (lowest - bottom) / frame.mm_per_px
        assert row == pytest.approx(frame.ground)
        assert frame.ground == frame.height - review.GROUND_PAD


def test_a_row_shares_one_ground_line_whatever_the_heights():
    grid, specs = body_specs(
        {
            "petite-soft": mesh_of(box(400, 200, 1150)),
            "regular-soft": mesh_of(box(600, 250, 1000, floor=442.0)),
            "tall-soft": mesh_of(box(900, 300, 1650)),
        }
    )
    scale, frames = review.plan_frames(specs)
    layout = review.layout_sheet(grid, specs, frames, scale)
    grounds = {item.y + item.frame.ground for item in layout.placed}
    assert len(grounds) == 1
    assert {line[2] for line in layout.ground_lines} == grounds
    # A shorter body starts lower on the sheet by exactly its height difference.
    tops = {item.preset: item.y for item in layout.placed}
    assert tops["petite-soft"] - tops["tall-soft"] == pytest.approx((1650 - 1150) / scale, abs=1)


def test_panes_stay_inside_their_cells_and_never_overlap():
    meshes = {
        "petite-soft": mesh_of(box(1500, 200, 1150)),
        "tall-soft": mesh_of(box(100, 300, 1650)),
    }
    grid, specs = body_specs(meshes)
    scale, frames = review.plan_frames(specs)
    layout = review.layout_sheet(grid, specs, frames, scale)
    spans = sorted((p.x, p.x + p.frame.width) for p in layout.placed)
    assert spans[0][1] <= spans[1][0]
    assert all(p.x >= 0 and p.x + p.frame.width <= layout.width for p in layout.placed)
    assert all(p.y >= 0 and p.y + p.frame.height <= layout.height for p in layout.placed)


def test_every_frame_is_whole_pixels_with_the_body_inside():
    verts = box(777.7, 333.3, 1234.5, floor=12.3, x=40.0, y=-30.0)
    for view in review.VIEWS:
        frame = review.body_frame(review.extent(verts, view), view, 1.25)
        assert isinstance(frame.width, int) and isinstance(frame.height, int)
        uv = review.project(verts, view)
        half_w, half_h = frame.width / 2 * 1.25, frame.height / 2 * 1.25
        assert (np.abs(uv[:, 0] - frame.u) <= half_w).all()
        assert (np.abs(uv[:, 1] - frame.v) <= half_h).all()


def test_scale_bar_is_a_round_length_at_the_sheet_scale():
    for scale in review.SCALES:
        mm, px = review.scale_bar(scale)
        assert px >= review.BAR_MIN_PX
        assert px == round(mm / scale)
        leading = mm / 10 ** np.floor(np.log10(mm))
        assert leading == pytest.approx(1) or leading == pytest.approx(2) or leading == 5


def test_view_axes_match_the_coordinate_convention():
    right, up, toward = review.view_axes("front")
    # The camera sits at -Y looking at the body's front, wearer's left (+X) on the image's right.
    assert right == pytest.approx([1, 0, 0]) and up == pytest.approx([0, 0, 1])
    assert toward == pytest.approx([0, -1, 0])
    for view in review.VIEWS:
        right, up, toward = review.view_axes(view)
        matrix = np.array((right, up, toward))
        assert matrix @ matrix.T == pytest.approx(np.eye(3))
        assert np.linalg.det(matrix) == pytest.approx(1)
    # Every side-on view looks at the wearer's left.
    for view in ("side", "threequarter", "backquarter"):
        assert review.VIEWS[view][0] > 0


# --- Before and after ---


def pair_spec(view="front"):
    before = mesh_of(box(500, 250, 1200), "before")
    after = mesh_of(box(800, 300, 1600, x=30.0), "after")
    axes = {"preset": ["regular-soft"], "pose": ["rest"], "view": [view]}
    grid = review.arrange(axes)
    panes = (review.Pane("BEFORE", before), review.Pane("AFTER", after))
    return grid, [review.CellSpec(grid.cells[0], panes)]


def test_paired_panes_share_a_scale_a_ground_and_an_axis():
    grid, specs = pair_spec()
    scale, frames = review.plan_frames(specs)
    (before, after) = frames[0]
    assert before.mm_per_px == after.mm_per_px == scale
    assert before.u == after.u and before.width == after.width
    assert before.height < after.height
    layout = review.layout_sheet(grid, specs, frames, scale)
    assert layout.placed[0].tag == "BEFORE" and layout.placed[1].tag == "AFTER"
    assert layout.placed[0].x < layout.placed[1].x
    grounds = {p.y + p.frame.ground for p in layout.placed}
    assert len(grounds) == 1
    labels = [t.text for t in layout.texts]
    assert any(t.startswith("BEFORE") for t in labels) and any(
        t.startswith("AFTER") for t in labels
    )


def test_against_reads_the_other_build(build, tmp_path):
    other = tmp_path / "other"
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    write_pose(other, "regular-soft", "rest", box(500, 250, 1200))
    axes = review.select_axes(["regular-soft"], view="front")
    specs = review.cell_specs(review.arrange(axes), build, other)
    before, after = specs[0].panes
    assert before.tag == "BEFORE" and before.mesh.path.is_relative_to(other)
    assert after.tag == "AFTER" and after.mesh.path.is_relative_to(build)


def test_a_silhouette_against_another_build_adds_a_change_pane(build, tmp_path):
    other = tmp_path / "other"
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    write_pose(other, "regular-soft", "rest", box(500, 250, 1200))
    for look, tags in (
        ("clay", ["BEFORE", "AFTER"]),
        ("silhouette", ["BEFORE", "AFTER", "CHANGE"]),
    ):
        axes = review.select_axes(["regular-soft"], view="front", look=look)
        specs = review.cell_specs(review.arrange(axes), build, other)
        assert [pane.tag for pane in specs[0].panes] == tags, look
    before, after, change = specs[0].panes
    assert change.mesh is after.mesh and change.under is before.mesh


def test_the_change_frame_holds_both_bodies_on_the_shared_axis_and_ground():
    grid, (spec,) = pair_spec()
    before, after = spec.panes
    spec = review.CellSpec(
        spec.grid, (before, after, review.Pane("CHANGE", after.mesh, before.mesh))
    )
    scale, ((one, two, change),) = review.plan_frames([spec])
    assert change.u == one.u == two.u and change.width == one.width
    assert change.height >= max(one.height, two.height)
    for mesh in (before.mesh, after.mesh):
        uv = review.project(mesh.verts, change.view)
        rows = change.height / 2 - (uv[:, 1] - change.v) / scale
        assert rows.min() >= 0 and rows.max() <= change.ground + 1e-6
    layout = review.layout_sheet(grid, [spec], [[one, two, change]], scale)
    assert len({p.y + p.frame.ground for p in layout.placed}) == 1
    assert layout.placed[2].under is before.mesh
    assert any(t.text == "CHANGE" for t in layout.texts)


def test_a_change_tile_colours_what_is_in_both_only_after_and_only_before():
    shape = (4, 6, 4)
    before, after = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    before[:, :4, 3] = 255  # columns 0-3
    after[:, 2:, 3] = 255  # columns 2-5
    before[0, 0, 3] = after[0, 0, 3] = 0
    tile = np.asarray(review.change_tile(Image.fromarray(before), Image.fromarray(after)))
    assert tuple(tile[1, 0, :3]) == review.LOST_RGB
    assert tuple(tile[1, 3, :3]) == review.BOTH_RGB
    assert tuple(tile[1, 5, :3]) == review.GAINED_RGB
    assert tile[1, 3, 3] == 255 and tile[0, 0, 3] == 0
    half = np.zeros(shape, np.uint8)
    half[..., 3] = 128
    edge = np.asarray(review.change_tile(Image.fromarray(half), Image.fromarray(after)))
    # Half covered before, whole after: half grey, half red. 128 of 255 is a hair over a half.
    middle = (np.array(review.BOTH_RGB) + np.array(review.GAINED_RGB)) / 2
    assert np.abs(edge[1, 5, :3] - middle).max() <= 1


def test_a_silhouette_sheet_against_another_build_renders_both_into_the_change_frame(
    build, tmp_path, monkeypatch
):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    other = tmp_path / "other"
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    write_pose(other, "regular-soft", "rest", box(500, 250, 1200))
    axes = review.select_axes(["regular-soft"], view="front", look="silhouette")
    specs = review.cell_specs(review.arrange(axes), build, other)
    scale, frames = review.plan_frames(specs)
    plant_renders(review.layout_sheet(review.arrange(axes), specs, frames, scale))
    layout, out = review.make_sheet(axes, against=other, out=tmp_path / "sheet.png")
    assert [p.tag for p in layout.placed] == ["BEFORE", "AFTER", "CHANGE"]
    assert any(t.text == review.CHANGE_NOTE for t in layout.texts)
    change = layout.placed[2]
    pixels = np.asarray(Image.open(out).convert("RGB"))[
        change.y : change.y + change.frame.height, change.x : change.x + change.frame.width
    ]
    colours = {tuple(p) for p in pixels.reshape(-1, 3)}
    assert {review.BOTH_RGB, review.GAINED_RGB} <= colours


def test_wire_is_refused_on_silhouette(build):
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    axes = review.select_axes(["regular-soft"], look="silhouette")
    with pytest.raises(review.ReviewError, match="wire") as caught:
        review.make_sheet(axes, wire=True)
    assert caught.value.code == review.EXIT_USAGE


def test_against_needs_a_directory_and_its_inputs(build, tmp_path):
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    axes = review.select_axes(["regular-soft"], view="front")
    with pytest.raises(review.ReviewError, match="not a directory"):
        review.make_sheet(axes, against=tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(review.ReviewError, match="uv run sotaigen-dev build"):
        review.make_sheet(axes, against=empty)


# --- Focus ---


def test_every_focus_has_vertices_on_the_wearers_left(base):
    x = np.asarray(base["verts"])[:, 0]
    for name, focus in review.FOCUS.items():
        region = review.focus_region(name, base)
        assert region.size > 0, name
        if focus.left_only:
            assert (x[region] >= 0).all(), name


def test_focus_seam_vertices_touch_both_segments(base):
    segments = np.asarray(base["segments"])
    region = set(review.focus_region("elbow", base).tolist())
    assert set(segments[list(region)]) == {"upperArm", "lowerArm"}
    touching = 0
    for face in base["faces"]:
        found = set(segments[list(face)])
        if {"upperArm", "lowerArm"} <= found and (np.asarray(base["verts"])[face, 0] >= 0).all():
            touching += 1
            assert {i for i in face if segments[i] in found} <= region
    assert touching > 0


def posed_template(base):
    verts = np.asarray(base["verts"], float).copy()
    segments = np.asarray(base["segments"])
    # The head's vertices are zeros in the template; any distinct spot will do for framing.
    head = segments == "head"
    verts[head] = np.random.default_rng(1).normal((0, 0, 1300), 80, (head.sum(), 3))
    return verts


def test_focus_frame_follows_the_part(base):
    verts = posed_template(base)
    region = review.focus_region("knee", base)
    center, side = review.focus_sphere(verts, region, review.FOCUS["knee"].context)
    moved = verts.copy()
    moved[region] += (10.0, -20.0, 300.0)
    moved_center, moved_side = review.focus_sphere(moved, region, review.FOCUS["knee"].context)
    assert moved_center - center == pytest.approx((10, -20, 300))
    assert moved_side == pytest.approx(side)
    frame = review.focus_frame(center, side, "front", 0.5)
    moved_frame = review.focus_frame(moved_center, moved_side, "front", 0.5)
    assert moved_frame.v - frame.v == pytest.approx(300)
    assert moved_frame.u - frame.u == pytest.approx(10)
    assert frame.ground is None


def test_focus_frame_ignores_the_rest_of_the_body(base):
    verts = posed_template(base)
    region = review.focus_region("hand", base)
    others = np.setdiff1d(np.arange(len(verts)), region)
    moved = verts.copy()
    moved[others] += (500.0, 500.0, 500.0)
    original = review.focus_sphere(verts, region, 1.0)
    shifted = review.focus_sphere(moved, region, 1.0)
    assert shifted[0] == pytest.approx(original[0]) and shifted[1] == pytest.approx(original[1])


def test_focus_frame_holds_the_part_from_every_view(base):
    verts = posed_template(base)
    for name, focus in review.FOCUS.items():
        region = review.focus_region(name, base)
        center, side = review.focus_sphere(verts, region, focus.context)
        for view in review.VIEWS:
            frame = review.focus_frame(center, side, view, 0.5)
            uv = review.project(verts[region], view)
            half = frame.width / 2 * frame.mm_per_px
            assert (np.abs(uv[:, 0] - frame.u) <= half).all(), (name, view)
            assert (np.abs(uv[:, 1] - frame.v) <= half).all(), (name, view)


def test_focus_cells_share_a_scale_and_a_frame_size(base):
    small = posed_template(base)
    large = small * 1.5
    axes = {"preset": ["petite-soft", "tall-soft"], "pose": ["crouch"], "view": ["side"]}
    axes["focus"] = ["elbow", "hip"]
    grid = review.arrange(axes)
    meshes = {"petite-soft": mesh_of(small), "tall-soft": mesh_of(large)}
    specs = [review.CellSpec(c, (review.Pane("", meshes[c.preset]),)) for c in grid.cells]
    regions = {name: review.focus_region(name, base) for name in axes["focus"]}
    scale, frames = review.plan_frames(specs, regions)
    assert {f.mm_per_px for row in frames for f in row} == {scale}
    assert len({(f.width, f.height) for row in frames for f in row}) == 1
    # The hip of the larger body sets the frame.
    hip = review.focus_sphere(large, regions["hip"], review.FOCUS["hip"].context)[1]
    assert frames[0][0].width == pytest.approx(hip / scale, abs=1)


def test_focus_refuses_a_different_topology(build):
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    axes = review.select_axes(["regular-soft"], view="front", focus="hip")
    with pytest.raises(review.ReviewError, match="topology"):
        review.make_sheet(axes)


# --- Cache keys ---


FRAME = review.Frame("front", 1.0, 700.0, 300, 720, 2.0, 714)


def test_cache_key_is_stable():
    assert review.render_key("abc", FRAME, False) == review.render_key("abc", FRAME, False)


@pytest.mark.parametrize(
    "digest,frame,wire",
    [
        ("other", FRAME, False),
        ("abc", review.Frame("side", 1.0, 700.0, 300, 720, 2.0, 714), False),
        ("abc", review.Frame("front", 9.0, 700.0, 300, 720, 2.0, 714), False),
        ("abc", review.Frame("front", 1.0, 709.0, 300, 720, 2.0, 714), False),
        ("abc", review.Frame("front", 1.0, 700.0, 301, 720, 2.0, 714), False),
        ("abc", review.Frame("front", 1.0, 700.0, 300, 721, 2.0, 714), False),
        ("abc", review.Frame("front", 1.0, 700.0, 300, 720, 1.5, 714), False),
        ("abc", FRAME, True),
    ],
)
def test_cache_key_changes_with_each_input(digest, frame, wire):
    assert review.render_key(digest, frame, wire) != review.render_key("abc", FRAME, False)


def test_cache_key_changes_with_the_renderer():
    assert review.render_key("abc", FRAME, False, "a") != review.render_key(
        "abc", FRAME, False, "b"
    )


def test_cache_key_follows_content_not_path_or_time(tmp_path):
    verts = box(100, 100, 100)
    first, second = tmp_path / "a.obj", tmp_path / "deep" / "b.obj"
    write_obj(first, verts)
    write_obj(second, verts)
    os.utime(second, (1, 1))
    one, two = review.read_mesh(first), review.read_mesh(second)
    assert one.digest == two.digest
    assert review.cache_path(one, FRAME, False) == review.cache_path(two, FRAME, False)
    write_obj(second, verts + 1)
    assert review.read_mesh(second).digest != one.digest


def test_read_mesh_keeps_vertex_order_and_ignores_the_rest(tmp_path):
    path = tmp_path / "m.obj"
    path.write_text("# comment\nv 1 2 3\nvn 0 0 1\nv 4.5 -5 6\nf 1 2 1\n")
    assert review.read_mesh(path).verts.tolist() == [[1, 2, 3], [4.5, -5, 6]]


# --- Blender ---


def executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_blender_env_comes_first(tmp_path, monkeypatch):
    named = executable(tmp_path / "named")
    on_path = executable(tmp_path / "blender")
    monkeypatch.setenv("BLENDER", str(named))
    monkeypatch.setenv("PATH", str(tmp_path))
    assert blender.find_blender() == named
    monkeypatch.delenv("BLENDER")
    assert blender.find_blender() == on_path


def test_blender_app_is_the_last_resort(tmp_path, monkeypatch):
    app = executable(tmp_path / "Blender")
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    monkeypatch.setattr(blender, "BLENDER_APP", app)
    assert blender.find_blender() == app
    monkeypatch.setattr(blender, "BLENDER_APP", tmp_path / "absent")
    assert blender.find_blender() is None


def test_a_blender_that_names_nothing_does_not_fall_through(tmp_path, monkeypatch):
    executable(tmp_path / "blender")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("BLENDER", "/nonexistent")
    assert blender.find_blender() is None


def test_render_without_blender_exits_3(capsys, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    assert review.main(["render", "regular-soft"]) == 3
    assert "BLENDER" in capsys.readouterr().err


def test_render_without_blender_exits_3_as_a_process():
    env = {**os.environ, "BLENDER": "/nonexistent"}
    done = subprocess.run(
        [sys.executable, "-m", "sotaigen_dev.review", "render", "regular-soft"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 3
    assert "BLENDER" in done.stderr


def test_sheet_without_blender_exits_3_when_it_must_render(build, capsys, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    assert review.main(["sheet", "regular-soft", "--view", "front"]) == 3
    assert "BLENDER" in capsys.readouterr().err


def test_missing_input_says_to_build(build, capsys, tmp_path, monkeypatch):
    blender = executable(tmp_path / "blender")
    monkeypatch.setenv("BLENDER", str(blender))
    code = review.main(["render", "regular-soft", "--pose", "rest", "--view", "front"])
    assert code not in (0, 3)
    assert "uv run sotaigen-dev build" in capsys.readouterr().err


def test_unknown_names_are_not_mistaken_for_a_missing_blender(capsys, tmp_path, monkeypatch):
    blender = executable(tmp_path / "blender")
    monkeypatch.setenv("BLENDER", str(blender))
    assert review.main(["render", "regular-nope"]) == review.EXIT_USAGE
    assert "regular-nope" in capsys.readouterr().err


def test_a_failing_blender_is_reported_with_its_own_code(build, tmp_path, monkeypatch):
    broken = tmp_path / "blender"
    broken.write_text("#!/bin/sh\necho boom\nexit 7\n")
    broken.chmod(0o755)
    monkeypatch.setenv("BLENDER", str(broken))
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    assert review.main(["render", "regular-soft", "--pose", "rest", "--view", "front"]) == 1


def test_a_silent_blender_that_wrote_nothing_is_a_failure(build, tmp_path, monkeypatch):
    silent = tmp_path / "blender"
    silent.write_text("#!/bin/sh\nexit 0\n")
    silent.chmod(0o755)
    monkeypatch.setenv("BLENDER", str(silent))
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    assert review.main(["render", "regular-soft", "--pose", "rest", "--view", "front"]) == 1


def test_blender_is_called_in_background_with_factory_settings(build, tmp_path, monkeypatch):
    seen = tmp_path / "argv"
    fake = tmp_path / "blender"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {seen}\nexit 3\n')
    fake.chmod(0o755)
    monkeypatch.setenv("BLENDER", str(fake))
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    review.main(["render", "regular-soft", "--pose", "rest", "--view", "front"])
    argv = seen.read_text().split()
    assert "-b" in argv and "--factory-startup" in argv
    assert argv.index("--factory-startup") < argv.index("--python")
    assert argv[argv.index("--python") + 1] == str(review.RENDERER)


# --- Whole sheets, with the cache standing in for Blender ---


def prepare_sheet(build, heights):
    for preset, height in heights.items():
        write_pose(build, preset, "rest", box(height * 0.4, 200, height))
    axes = review.select_axes(list(heights), view="front")
    specs = review.cell_specs(review.arrange(axes, cols=["preset"]), build)
    grid = review.arrange(axes, cols=["preset"])
    scale, frames = review.plan_frames(specs)
    layout = review.layout_sheet(grid, specs, frames, scale)
    return axes, layout


HEIGHTS = {"petite-soft": 1150.3, "regular-soft": 1399.7, "tall-soft": 1649.7}


def test_sheet_measures_true_on_the_saved_image(build, tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    axes, layout = prepare_sheet(build, HEIGHTS)
    plant_renders(layout)
    placed, path = review.make_sheet(axes, cols=["preset"], out=tmp_path / "out" / "sheet.png")
    assert path.exists()
    results = review.measure_sheet(path, placed)
    assert {row["preset"] for row in results} == set(HEIGHTS)
    errors = review.ratio_errors(results)
    assert max(errors.values()) < review.MAX_RATIO_ERROR
    for row in results:
        assert row["gap_to_ground"] <= 1
        assert row["pixels"] * placed.mm_per_px == pytest.approx(row["z_span"], rel=0.01)


def test_ratio_errors_notice_a_stretched_body():
    rows = [
        {"pose": "rest", "view": "front", "z_span": 1000.0, "pixels": 500},
        {"pose": "rest", "view": "front", "z_span": 2000.0, "pixels": 1100},
    ]
    assert review.ratio_errors(rows)["rest", "front"] == pytest.approx(0.0909, abs=1e-3)


def test_sheet_pastes_renders_without_resampling(build, tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    axes, layout = prepare_sheet(build, {"regular-soft": 1399.7})
    item = layout.placed[0]
    review.cache_dir().mkdir(parents=True)
    # Noise changes under any resampling; only a pixel-for-pixel paste keeps it.
    noise = np.random.default_rng(2).integers(0, 200, (item.frame.height, item.frame.width, 3))
    tile = np.dstack((noise, np.full(noise.shape[:2], 255))).astype(np.uint8)
    Image.fromarray(tile).save(review.cache_path(item.mesh, item.frame, False))
    _, path = review.make_sheet(axes, cols=["preset"], out=tmp_path / "sheet.png")
    sheet = np.asarray(Image.open(path).convert("RGB"))
    pasted = sheet[item.y : item.y + item.frame.height, item.x : item.x + item.frame.width]
    assert (pasted == noise).all()


def test_a_render_of_the_wrong_size_is_refused(build, tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    axes, layout = prepare_sheet(build, {"regular-soft": 1399.7})
    item = layout.placed[0]
    review.cache_dir().mkdir(parents=True)
    image = Image.new("RGBA", (item.frame.width + 1, item.frame.height))
    image.save(review.cache_path(item.mesh, item.frame, False))
    with pytest.raises(review.ReviewError, match="size"):
        review.make_sheet(axes, cols=["preset"], out=tmp_path / "sheet.png")


def test_cached_sheet_needs_no_blender_and_a_changed_mesh_does(build, tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    axes, layout = prepare_sheet(build, {"regular-soft": 1399.7})
    plant_renders(layout)
    review.make_sheet(axes, cols=["preset"], out=tmp_path / "sheet.png")
    write_pose(build, "regular-soft", "rest", box(560, 200, 1400))
    with pytest.raises(review.ReviewError) as caught:
        review.make_sheet(axes, cols=["preset"], out=tmp_path / "sheet.png")
    assert caught.value.code == review.EXIT_NO_BLENDER


def test_default_output_goes_under_build_review(build, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    axes, layout = prepare_sheet(build, {"regular-soft": 1399.7})
    plant_renders(layout)
    _, path = review.make_sheet(axes, cols=["preset"])
    assert path == build / "reports" / "review" / review.sheet_name(axes)
    assert path.exists()


# --- The `render` contract ---


def test_render_canvas_is_shared_and_grounded():
    tall = box(800, 200, 1650)
    short = box(400, 200, 1150, floor=300.0)
    for view in review.VIEWS:
        a = review.render_canvas(tall, view)
        b = review.render_canvas(short, view)
        assert (a.width, a.height, a.ground) == (b.width, b.height, b.ground)
        assert (a.width, a.height) == review.RENDER_CANVAS
        assert a.u == b.u == 0
        assert a.mm_per_px == b.mm_per_px == review.RENDER_MM_PER_PX
        for verts, frame in ((tall, a), (short, b)):
            lowest = review.project(verts, view)[:, 1].min()
            row = frame.height / 2 - (lowest - frame.v) / frame.mm_per_px
            assert row == pytest.approx(frame.ground)


def test_render_canvas_grows_rather_than_crops():
    huge = box(3000, 200, 2500)
    frame = review.render_canvas(huge, "front")
    half = frame.width / 2 * frame.mm_per_px
    assert half >= 1500
    assert (frame.height - review.RENDER_GROUND_PX - review.TOP_PAD) * frame.mm_per_px >= 2500


def test_render_writes_the_agreed_paths(build, tmp_path, monkeypatch):
    fake = tmp_path / "blender"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("BLENDER", str(fake))
    verts = box(800, 200, 1400)
    write_pose(build, "regular-soft", "crouch", verts)
    mesh = review.read_mesh(review.pose_path(build, "regular-soft", "crouch"))
    review.cache_dir().mkdir(parents=True)
    for view in ("front", "side"):
        frame = review.render_canvas(mesh.verts, view)
        image = Image.new("RGBA", (frame.width, frame.height))
        image.save(review.cache_path(mesh, frame, False))
    argv = ["render", "regular-soft", "--pose", "crouch", "--view", "front,side"]
    assert review.main(argv) == 0
    for view in ("front", "side"):
        assert (build / "regular-soft" / "renders" / "crouch" / f"{view}.png").exists()


def test_review_does_not_import_the_pose_script():
    source = SCRIPT.read_text()
    assert "import pose" not in source and "from pose" not in source
    assert ".resize(" not in source


# --- Look and camera ---


def test_look_and_camera_are_axes_only_when_asked_for():
    plain = review.select_axes(["regular-soft"])
    assert "look" not in plain and "camera" not in plain
    axes = review.select_axes(["regular-soft"], look="clay,toon", camera="ortho,persp")
    assert axes["look"] == ["clay", "toon"] and axes["camera"] == ["ortho", "persp"]
    grid = review.arrange(axes, rows=["look", "camera"], cols=["view"])
    assert len(grid.row_titles) == 4 and grid.row_titles[3] == "toon / persp"
    assert {(c.look, c.camera) for c in grid.cells} == set(
        itertools.product(("clay", "toon"), ("ortho", "persp"))
    )


def test_one_command_gives_each_look_a_row():
    axes = review.select_axes(["regular-soft"], look="clay,toon,toon+persp", view="front,side")
    grid = review.arrange(axes)
    assert grid.row_titles == ["clay", "toon", "toon+persp"]
    assert grid.col_titles == ["front", "side"]
    rows = {c.row: (c.look, c.camera) for c in grid.cells}
    assert rows == {0: ("clay", "ortho"), 1: ("toon", "ortho"), 2: ("toon", "persp")}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"look": "chalk"},
        {"look": "toon+fisheye"},
        {"camera": "fisheye"},
        {"look": "toon+persp", "camera": "ortho"},
    ],
)
def test_bad_looks_and_cameras_are_refused(kwargs):
    with pytest.raises(review.ReviewError) as caught:
        review.select_axes(["regular-soft"], **kwargs)
    assert caught.value.code == review.EXIT_USAGE


def test_look_combinations_for_render():
    assert review.look_combinations(None, None) == [("clay", "ortho")]
    assert review.look_combinations("toon", "ortho,persp") == [
        ("toon", "ortho"),
        ("toon", "persp"),
    ]
    assert review.look_combinations("clay,toon+persp", None) == [
        ("clay", "ortho"),
        ("toon", "persp"),
    ]


def test_only_clay_ortho_takes_the_agreed_render_path(tmp_path):
    agreed = tmp_path / "regular-soft" / "renders" / "rest" / "front.png"
    assert review.render_path(tmp_path, "regular-soft", "rest", "front") == agreed
    other = review.render_path(tmp_path, "regular-soft", "rest", "front", "toon", "persp")
    assert other == agreed.with_name("front-toon-persp.png")


def test_wire_is_refused_on_toon(build):
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    axes = review.select_axes(["regular-soft"], look="toon")
    with pytest.raises(review.ReviewError, match="wire") as caught:
        review.make_sheet(axes, wire=True)
    assert caught.value.code == review.EXIT_USAGE


def test_cache_key_changes_with_look_and_camera():
    keys = {
        review.render_key("abc", FRAME, False),
        review.render_key("abc", replace(FRAME, look="toon"), False),
        review.render_key(
            "abc", replace(FRAME, camera="persp", distance=2100.0, aim=(0, 0, 700)), False
        ),
        review.render_key(
            "abc", replace(FRAME, camera="persp", distance=2200.0, aim=(0, 0, 700)), False
        ),
        review.render_key(
            "abc", replace(FRAME, camera="persp", distance=2100.0, aim=(0, 0, 900)), False
        ),
    }
    assert len(keys) == 5


def persp_spec(verts, view="front", preset="regular-soft", look="toon"):
    axes = {"preset": [preset], "pose": ["crouch"], "view": [view], "look": [f"{look}+persp"]}
    grid = review.arrange(axes)
    return grid, [review.CellSpec(grid.cells[0], (review.Pane("", mesh_of(verts)),))]


def in_frame(points, frame):
    half_w, half_h = frame.width / 2 * frame.mm_per_px, frame.height / 2 * frame.mm_per_px
    return (np.abs(points[:, 0] - frame.u) <= half_w).all() and (
        np.abs(points[:, 1] - frame.v) <= half_h
    ).all()


def test_the_lens_matches_project_on_the_aim_plane_and_enlarges_nearer():
    aim = np.array([0.0, 0.0, 700.0])
    on_plane = np.array([[300.0, 0.0, 900.0]])
    plane = review.perspective_plane(on_plane, "front", 2100.0, aim)
    assert plane == pytest.approx(review.project(on_plane, "front"))
    near = review.perspective_plane(np.array([[300.0, -300.0, 700.0]]), "front", 2100.0, aim)
    far = review.perspective_plane(np.array([[300.0, 300.0, 700.0]]), "front", 2100.0, aim)
    assert near[0, 0] == pytest.approx(300 * 2100 / 1800)
    assert far[0, 0] == pytest.approx(300 * 2100 / 2400)


@pytest.mark.parametrize("view", list(review.VIEWS))
def test_perspective_frame_crops_nothing(view):
    # Reaches far toward every camera, where a lens makes things largest.
    verts = np.vstack((box(1400, 900, 1400), box(100, 100, 300, floor=500.0, x=500.0, y=-600.0)))
    _, specs = persp_spec(verts, view)
    _, frames = review.plan_frames(specs)
    frame = frames[0][0]
    distance, aim = specs[0].lens(verts)
    assert frame.camera == "persp" and frame.look == "toon"
    assert in_frame(review.perspective_plane(verts, view, distance, aim), frame)


def test_perspective_camera_stays_put_whatever_the_pose():
    standing = box(400, 200, 1400)
    crouching = box(400, 200, 1000, floor=442.0)
    assert review.camera_distance("regular-soft") == pytest.approx(1400 * 1.08 * 50 / 36)
    assert review.camera_distance("petite-soft") < review.camera_distance("tall-soft")
    # Half a stature above whatever is lowest, so a crouch keeps the camera's height over the floor.
    assert review.camera_aim(standing, "regular-soft")[2] == pytest.approx(700)
    assert review.camera_aim(crouching, "regular-soft")[2] == pytest.approx(442 + 700)


def test_lens_and_shift_frame_the_window_from_the_fixed_camera():
    aim = (0.0, 0.0, 700.0)
    frame = review.Frame("front", 50.0, 400.0, 300, 600, 2.0, None, "toon", "persp", 2100.0, aim)
    lens, shift_x, shift_y = frame.lens()
    assert lens == pytest.approx(2100 * 36 / 1200)
    assert shift_x == pytest.approx(50 / 1200)
    assert shift_y == pytest.approx((400 - 700) / 1200)
    entry = review.shot_entry(frame, False, "out.png")
    assert entry["eye"] == pytest.approx([0, -2100, 700])
    assert entry["look"] == "toon" and entry["camera"] == "persp"


def test_a_50mm_frame_of_108_statures_has_a_50mm_lens():
    stature = review.stature("regular-soft")
    side_px = 1100
    mm_per_px = stature * 1.08 / side_px
    distance = review.camera_distance("regular-soft")
    frame = review.Frame(
        "front", 0.0, 700.0, 700, side_px, mm_per_px, None, "toon", "persp", distance, (0, 0, 700)
    )
    assert frame.lens() == pytest.approx((50.0, 0.0, 0.0))


def test_orthographic_shots_carry_no_lens():
    entry = review.shot_entry(FRAME, False, "out.png")
    assert entry["camera"] == "ortho" and "lens_mm" not in entry


def test_perspective_focus_holds_the_part(base):
    verts = posed_template(base)
    region = review.focus_region("hand", base)
    axes = {
        "preset": ["regular-soft"],
        "pose": ["reach"],
        "view": list(review.VIEWS),
        "focus": ["hand"],
        "look": ["toon+persp"],
    }
    grid = review.arrange(axes)
    specs = [review.CellSpec(c, (review.Pane("", mesh_of(verts)),)) for c in grid.cells]
    _, frames = review.plan_frames(specs, {"hand": region})
    assert len({(row[0].width, row[0].height) for row in frames}) == 1
    for spec, row in zip(specs, frames):
        distance, aim = spec.lens(verts)
        points = review.perspective_plane(verts[region], spec.view, distance, aim)
        assert in_frame(points, row[0]), spec.view


def test_perspective_cells_say_their_scale_does_not_hold():
    grid, specs = persp_spec(box(400, 200, 1400))
    scale, frames = review.plan_frames(specs)
    layout = review.layout_sheet(grid, specs, frames, scale)
    assert layout.bar is None
    warnings = [t.text for t in layout.texts if t.warn]
    assert review.PERSP_CELL_NOTE in warnings and review.PERSP_SHEET_NOTE in warnings
    assert "centre plane" in review.PERSP_CELL_NOTE
    assert not any("mm per pixel," in t.text for t in layout.texts)


def test_a_mixed_sheet_scopes_its_scale_bar_to_the_orthographic_cells():
    axes = {
        "preset": ["regular-soft"],
        "pose": ["rest"],
        "view": ["front"],
        "look": ["toon", "toon+persp"],
    }
    grid = review.arrange(axes)
    mesh = mesh_of(box(400, 200, 1400))
    specs = [review.CellSpec(c, (review.Pane("", mesh),)) for c in grid.cells]
    scale, frames = review.plan_frames(specs)
    assert {f.mm_per_px for row in frames for f in row} == {scale}
    layout = review.layout_sheet(grid, specs, frames, scale)
    assert layout.bar is not None
    assert any("orthographic cells only" in t.text for t in layout.texts)
    notes = [t.text for t in layout.texts if t.text == review.PERSP_CELL_NOTE]
    assert len(notes) == 1


def test_orthographic_sheets_stay_as_they_were():
    grid, specs = body_specs({"regular-soft": mesh_of(box(600, 250, 1400))})
    scale, frames = review.plan_frames(specs)
    layout = review.layout_sheet(grid, specs, frames, scale)
    assert not any(t.warn for t in layout.texts)
    assert layout.bar is not None
    frame = frames[0][0]
    assert (frame.look, frame.camera, frame.distance, frame.aim) == ("clay", "ortho", None, None)


def test_measuring_refuses_a_perspective_sheet(tmp_path):
    grid, specs = persp_spec(box(400, 200, 1400))
    scale, frames = review.plan_frames(specs)
    layout = review.layout_sheet(grid, specs, frames, scale)
    path = tmp_path / "sheet.png"
    Image.new("RGB", (layout.width, layout.height), "white").save(path)
    with pytest.raises(review.ReviewError, match="orthographic"):
        review.measure_sheet(path, layout)


def test_render_canvas_through_the_lens_crops_nothing():
    verts = box(1400, 900, 1400)
    for view in review.VIEWS:
        frame = review.render_canvas(verts, view, "toon", "persp", "regular-soft")
        assert frame.u == 0 and frame.camera == "persp"
        distance, aim = review.camera_distance("regular-soft"), review.camera_aim(verts, "regular-soft")
        assert in_frame(review.perspective_plane(verts, view, distance, aim), frame)


# --- Out-of-date inputs ---


def test_stale_poses_are_refused_with_the_command_to_fix_them(build, capsys):
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    write_fingerprint(build / "regular-soft", {"pose": "from an older source"})
    code = review.main(["sheet", "regular-soft", "--view", "front"])
    assert code == review.EXIT_FAILED
    err = capsys.readouterr().err
    assert "out of date" in err and "uv run sotaigen-dev build regular-soft --stages pose" in err


def test_poses_no_finished_build_vouches_for_are_stale(build):
    write_obj(review.pose_path(build, "regular-soft", "rest"), box(800, 300, 1600))
    assert review.stale_presets(build, ["regular-soft"]) == ["regular-soft"]


def test_fresh_poses_pass(build):
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    assert review.stale_presets(build, ["regular-soft"]) == []
    assert review.check_fresh(build, ["regular-soft"]) == []


def test_allow_stale_shows_them_and_marks_the_sheet(build, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    axes, layout = prepare_sheet(build, {"regular-soft": 1399.7})
    plant_renders(layout)
    write_fingerprint(build / "regular-soft", {"pose": "from an older source"})
    with pytest.raises(review.ReviewError, match="out of date"):
        review.make_sheet(axes, cols=["preset"], out=tmp_path / "sheet.png")
    placed, _ = review.make_sheet(
        axes, cols=["preset"], out=tmp_path / "sheet.png", allow_stale=True
    )
    assert any("OUT OF DATE" in t.text for t in placed.texts)
    assert "out-of-date" in capsys.readouterr().err


def test_render_refuses_stale_poses_too(build, tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", str(executable(tmp_path / "blender")))
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    write_fingerprint(build / "regular-soft", {})
    argv = ["render", "regular-soft", "--pose", "rest", "--view", "front"]
    assert review.main(argv) == review.EXIT_FAILED


def test_verify_refuses_stale_poses_and_accepts_the_flag_it_names(build, capsys, tmp_path):
    plant_renders(prepare_sheet(build, {"regular-soft": 1399.7})[1])
    write_fingerprint(build / "regular-soft", {"pose": "from an older source"})
    argv = ["verify", "regular-soft", "--view", "front", "-o", str(tmp_path / "sheet.png")]
    assert review.main(argv) == review.EXIT_FAILED
    assert "--allow-stale" in capsys.readouterr().err
    assert review.main([*argv, "--allow-stale"]) == 0
    printed = capsys.readouterr()
    assert "ok: worst ratio error" in printed.out
    assert "out-of-date" in printed.err


def test_verify_defaults_to_every_body(build, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("BLENDER", "/nonexistent")
    plant_renders(prepare_sheet(build, HEIGHTS)[1])
    assert review.main(["verify", "--view", "front", "-o", str(tmp_path / "sheet.png")]) == 0
    out = capsys.readouterr().out
    assert all(preset in out for preset in HEIGHTS)


def test_the_against_build_may_be_older(build, tmp_path):
    other = tmp_path / "other"
    write_pose(build, "regular-soft", "rest", box(800, 300, 1600))
    write_obj(review.pose_path(other, "regular-soft", "rest"), box(500, 250, 1200))
    axes = review.select_axes(["regular-soft"], view="front")
    grid = review.arrange(axes)
    specs = review.cell_specs(grid, build, other)
    assert len(specs[0].panes) == 2
    assert review.check_fresh(build, [c.preset for c in grid.cells]) == []


def test_cells_are_wide_enough_for_their_labels():
    axes = {
        "preset": ["regular-soft"],
        "pose": ["armsDown"],
        "view": ["backquarter"],
        "look": ["toon"],
    }
    grid = review.arrange(axes)
    specs = [review.CellSpec(grid.cells[0], (review.Pane("", mesh_of(box(50, 50, 1400))),))]
    scale, frames = review.plan_frames(specs)
    layout = review.layout_sheet(grid, specs, frames, scale)
    x0, _, x1, _ = layout.boxes[0]
    assert x1 - x0 >= len("armsDown / backquarter / toon ortho") * review.LABEL_CHAR_PX
