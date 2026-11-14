"""Edge loops added to the template after extract.py, appended in batches so nothing numbered moves.

    uv run template/loops.py

Reads sotaigen/data/template/base-mesh.json and uv.npz as extract.py and extract_uv.py write them, adds
every batch in BATCHES in order and writes both back. A loop is one cut through a closed strip of
quads: each rung of the strip gets a vertex at its middle and each quad of the strip becomes two.
The new vertices go after all the ones there already, and their UVs are the middles of their
rungs' corners, so neither file needs Blender. See template/PROVENANCE.md.
"""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "sotaigen" / "data" / "template"
TEMPLATE = DATA / "base-mesh.json"
UV = DATA / "uv.npz"

# Each batch names one rung of every strip it cuts, by the template's vertices at its two ends.
BATCHES = {
    # The shoulder and axilla (docs/parts/shoulder-axilla.md): on each side the ring at the arm's
    # seam and the ring round the shoulder on the trunk's side, and the band of the trunk that runs
    # under both armpits.
    "axilla": [(1441, 2718), (5850, 7071), (1484, 1485), (5893, 5895), (1437, 1438)],
}


def edge(a, b):
    return (a, b) if a < b else (b, a)


def owners_of(faces):
    owners = {}
    for i, face in enumerate(faces):
        for k in range(len(face)):
            owners.setdefault(edge(face[k], face[(k + 1) % len(face)]), []).append(i)
    return owners


def far_rung(face, rung):
    """The rung across a quad from `rung`."""
    for k in range(4):
        if edge(face[k], face[(k + 1) % 4]) == rung:
            return edge(face[(k + 2) % 4], face[(k + 3) % 4])
    raise ValueError(f"{rung} is not an edge of {face}")


def strip(faces, owners, seed):
    """The quads of the closed strip through the rung `seed`, each with the rung it is entered by."""
    visits, rung = [], seed
    face = owners[seed][0]
    while True:
        if len(faces[face]) != 4:
            raise ValueError(f"the strip through {seed} reaches a face that isn't a quad")
        visits.append((face, rung))
        rung = far_rung(faces[face], rung)
        if rung == seed:
            break
        others = [f for f in owners[rung] if f != face]
        if not others:
            raise ValueError(f"the strip through {seed} ends at the neck")
        face = others[0]
    if len({f for f, _ in visits}) != len(visits):
        raise ValueError(f"the strip through {seed} crosses itself")
    return visits


def cut(base, corners, seed, splits):
    """One loop through the strip with the rung `seed`. `corners` holds each template face's UVs.

    Each quad split appends its second half; `splits` gets the quad's index and the corner its
    first half starts at, so the cut can be undone (sotaigen.template.before_loops)."""
    faces, verts = base["faces"], base["verts"]
    segments = base["segments"]
    parents = base["refined"]["parents"]
    middle = {}

    def halve(rung, face):
        if rung not in middle:
            a, b = rung
            middle[rung] = len(verts)
            verts.append([round((p + q) / 2, 2) for p, q in zip(verts[a], verts[b], strict=True)])
            # A rung across a seam goes with the part most of its quad belongs to.
            votes = [segments[v] for v in faces[face]]
            pick = a if votes.count(segments[a]) >= votes.count(segments[b]) else b
            segments.append(segments[pick])
            parents.append([a, b])
        return middle[rung]

    for index, rung in strip(faces, owners_of(faces), seed):
        face = faces[index]
        k = next(i for i in range(4) if edge(face[i], face[(i + 1) % 4]) == rung)
        a, b, c, d = (face[(k + j) % 4] for j in range(4))
        ua, ub, uc, ud = (corners[index][(k + j) % 4] for j in range(4))
        m, n = halve(edge(a, b), index), halve(edge(c, d), index)
        um, un = (ua + ub) / 2, (uc + ud) / 2
        faces[index] = [a, m, n, d]
        corners[index] = np.array([ua, um, un, ud])
        faces.append([m, b, c, n])
        corners.append(np.array([um, ub, uc, un]))
        splits.append([index, k])


def add_batches(base, corners, sizes):
    """The template and its UVs with BATCHES added. `corners` and `sizes` are uv.npz's arrays: the
    template's faces first, then the head's."""
    refined = base["refined"]
    if "batches" in refined:
        raise SystemExit("base-mesh.json has its loops already: rerun extract.py and extract_uv.py")
    offsets = np.concatenate(([0], np.cumsum(sizes.astype(int))))
    count = len(base["faces"])
    own = [corners[offsets[i] : offsets[i + 1]].astype(float) for i in range(count)]
    head, head_sizes = corners[offsets[count] :], sizes[count:]
    batches = [{"name": "joints", "count": len(refined["parents"])}]
    for name, seeds in BATCHES.items():
        before, splits = len(base["verts"]), []
        for seed in seeds:
            cut(base, own, edge(*seed), splits)
        rungs = [list(edge(*s)) for s in seeds]
        count = len(base["verts"]) - before
        batches.append({"name": name, "count": count, "rungs": rungs, "splits": splits})
    refined["batches"] = batches
    uv = np.vstack([*own, head])
    uv_sizes = np.concatenate([np.full(len(own), 4), head_sizes]).astype(sizes.dtype)
    return base, np.round(uv).astype(corners.dtype), uv_sizes


def main():
    base = json.loads(TEMPLATE.read_text())
    with np.load(UV) as data:
        corners, sizes = data["corners"], data["sizes"]
    base, corners, sizes = add_batches(base, corners, sizes)
    TEMPLATE.write_text(json.dumps(base, separators=(",", ":")))
    np.savez_compressed(UV, corners=corners, sizes=sizes)
    added = {b["name"]: b["count"] for b in base["refined"]["batches"]}
    print("BODY_LOOPS", len(base["verts"]), "verts", len(base["faces"]), "faces", added)


if __name__ == "__main__":
    main()
