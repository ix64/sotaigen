"""sotaigen_dev.displacement on made-up builds: the split of a move, the groups, the picture and the files."""

import json

import numpy as np
from PIL import Image

from sotaigen.mesh import write_obj
from sotaigen.pipeline import template
from sotaigen_dev import displacement

# An octahedron: six points 141 mm or more apart, eight outward triangles.
POINTS = np.array(
    [(100.0, 0, 0), (-100.0, 0, 0), (0, 100.0, 0), (0, -100.0, 0), (0, 0, 100.0), (0, 0, -100.0)]
)
TRIANGLES = np.array(
    [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
)


def test_a_move_splits_into_outward_and_sliding():
    after = POINTS.copy()
    after[0, 0] += 2.0  # straight out
    after[4, 1] += 1.0  # along the surface
    move, along, slide = displacement.split_move(POINTS, after, TRIANGLES)
    assert along[0] > 1.99 and slide[0] < 1e-9
    assert abs(along[4]) < 1e-9 and abs(slide[4] - 1.0) < 1e-9
    assert np.all(move[1:4] == 0)


def test_limbs_are_told_apart_by_side_and_the_trunk_is_not():
    labels = np.array(["arm", "arm", "trunk", "trunk"])
    side = displacement.sides(np.array([30.0, -30.0, 30.0, 0.5]))
    assert list(side) == ["left", "right", "left", ""]
    keys = displacement.group_keys(labels, side, {"arm"})
    assert list(keys) == ["arm.left", "arm.right", "trunk", "trunk"]


def test_the_summary_names_the_largest_places_and_where_most_of_it_is():
    after = POINTS.copy()
    after[0, 0] += 3.0
    after[1, 0] -= 1.0
    regions = np.array(["arm", "arm", "trunk", "trunk", "trunk", "trunk"])
    segments = np.array(["upperArm", "upperArm", "trunk", "trunk", "trunk", "trunk"])
    summary, *_ = displacement.summarise(POINTS, after, TRIANGLES, regions, segments, POINTS[:, 0])
    assert not summary["identical"]
    assert [move["vertex"] for move in summary["largest"]] == [0, 1]
    assert summary["largest"][0]["segment"] == "upperArm.left"
    assert summary["largest"][0]["outward"] > 2.99
    assert list(summary["regions"]) == ["arm.left", "arm.right", "trunk"]
    assert summary["regions"]["arm.left"]["share"] == 0.75
    assert summary["whole"]["max"] == 3.0 and summary["whole"]["moved"] == 2
    still, *_ = displacement.summarise(POINTS, POINTS, TRIANGLES, regions, segments, POINTS[:, 0])
    assert still == {"identical": True, "whole": still["whole"]}


def test_moves_close_together_count_as_one_place():
    points = np.array([(0.0, 0, 0), (5.0, 0, 0), (100.0, 0, 0)])
    size = np.array([3.0, 2.9, 1.0])
    keys = np.array(["a", "a", "b"], dtype=object)
    found = displacement.largest(points, size, size, np.zeros(3), keys, apart=40.0)
    assert [move["vertex"] for move in found] == [0, 2]


def test_the_scale_rounds_up_to_one_two_or_five():
    assert [displacement.nice_scale(v) for v in (0.03, 0.13, 1.0, 1.2, 3.1, 9.51, 12)] == [
        0.1,
        0.2,
        1,
        2,
        5,
        10,
        20,
    ]


def test_the_nearest_triangle_owns_a_pixel():
    near = np.array([(0.0, 0, 0), (20.0, 0, 0), (0, 0, 20.0)])
    far = near + (0, 5.0, 0)  # further from a front camera, which looks along +y from -y
    points = np.vstack((far, near))
    owner, weights, *_ = displacement.rasterise(
        points, np.array([(0, 1, 2), (3, 4, 5)]), "front", 1.0
    )
    covered = owner >= 0
    assert set(np.unique(owner[covered])) == {1}
    # Half a 20 mm square, at 1 mm per pixel.
    assert abs(covered.sum() - 200) < 25
    assert np.allclose(weights[covered].sum(axis=1), 1.0)


def build_dir(root, name, points, faces, poses):
    folder = root / name / "poses"
    folder.mkdir(parents=True)
    for pose, offset in poses.items():
        write_obj(folder / f"{pose}.obj", points + offset, faces)
    return root


def test_the_command_writes_numbers_and_a_sheet(tmp_path):
    base = template()
    faces = base["faces"]
    points = np.array(base["verts"], float)
    upper_arm = (np.array(base["segments"]) == "upperArm") & (points[:, 0] > 0)
    grown = np.zeros_like(points)
    grown[upper_arm, 1] = -4.0  # forward, on the left upper arm only
    before = build_dir(tmp_path / "before", "toy", points, faces, {"rest": 0.0, "armsUp": 0.0})
    after = build_dir(tmp_path / "after", "toy", points, faces, {"rest": grown, "armsUp": 0.0})
    out = tmp_path / "out"
    code = displacement.main(
        ["--against", str(before), "--build", str(after), "--views", "front", "-o", str(out)]
    )
    assert code == 0
    report = json.loads((out / "displacement.json").read_text())["presets"]["toy"]
    assert report["armsUp"] == {"identical": True, "whole": report["armsUp"]["whole"]}
    rest = report["rest"]
    assert abs(rest["whole"]["max"] - 4.0) < 0.01
    assert rest["largest"][0]["segment"] == "upperArm.left"
    assert next(iter(rest["regions"])) == "arm.left"
    assert rest["regions"]["arm.left"]["share"] == 1.0
    picture = Image.open(out / "toy-rest.png")
    assert picture.width > 300 and picture.height > 600
    assert (out / "summary.txt").read_text().startswith("toy / rest: max 4.00 mm at upperArm.left")


def test_a_different_vertex_count_is_reported_not_compared(tmp_path):
    base = template()
    points = np.array(base["verts"], float)
    before = build_dir(tmp_path / "before", "toy", points, base["faces"], {"rest": 0.0})
    after = tmp_path / "after"
    (after / "toy" / "poses").mkdir(parents=True)
    write_obj(after / "toy" / "poses" / "rest.obj", points[:-1], [])
    code = displacement.main(
        [
            "--against",
            str(before),
            "--build",
            str(after),
            "--images",
            "none",
            "-o",
            str(tmp_path / "o"),
        ]
    )
    assert code == displacement.EXIT_INCOMPLETE
