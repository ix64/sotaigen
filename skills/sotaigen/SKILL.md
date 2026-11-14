---
name: sotaigen
description: Make a character's base body with sotaigen from the character's design, and fit the downstream project's own head, hair and clothes to it. Use when a downstream project has a character design (height, head count, measurements, turnarounds or illustrations, outfit) and needs a rigged, posable anime-style base body.
---

# A base body from a character design

sotaigen makes a rigged, posable anime-style base body: the body below the neck, a neutral head, bones
and weights. The character's face, hair, clothes and own measurements are the downstream project's. Go
through the steps below in order, and note every problem in the log (last section). What the tool can't
do yet is in the [known limitations](../../docs/en/reference/limitations.md); what downstream may rely
on is in the [contracts](../../docs/en/reference/README.md). A Chinese version of this guide is in
[SKILL.zh.md](SKILL.zh.md).

## Know the boundary

| The tool gives | The downstream project makes | Not there yet |
|---|---|---|
| A skin of fixed topology: every body has the same vertices in the same order | The face, eyes, brows, mouth and their expressions | Hand-to-head poses solved again after a head swap |
| A body solved from measurements in mm; targets it can't reach are reported as they are | The hair and its physics | Regions to hide under clothes, landmarks to fit clothes by |
| Bones, weights, correctives, the GLB, and a `.blend` whose rest pose can be changed (arm abduction, ankle spacing, palms down) | Clothes, shoes, accessories, and the extra bones and attachment points they need | Rendering and posing a parameter file from the command line |
| A neutral head and the neck's seam, with `jaw` and eye bones; features, ears and hair are placeholders | The character's own size and clothing constraints | VRM export |

Keep the character's parameters, meshes and pictures in the downstream repository only, never in this
tool's.

## 1. Set up

Installed as packages, without a checkout:

```sh
pip install sotaigen sotaigen-blender    # sotaigen-blender only for the .blend
```

The commands are `sotaigen check`, `glb`, `blend`, `swap-head` and `scale`. They exit 0 when done, 1 when
the body can't be made (a rule, a broken mesh, past a joint limit), 2 for a wrong command line or file,
and 3 when Blender (5.2 or later) or `sotaigen-blender` is missing. From Python:
`sotaigen.read_spec`, `generate`, `write_glb`, `write_blend` (`sotaigen/api.py`).

For a checkout, install the uv version in `pyproject.toml`; use fnm with `.node-version` for Node.
The full setup is in [CONTRIBUTING](../../CONTRIBUTING.md). Run `uv sync --all-extras` there and prefix the commands with
`uv run`; from the downstream repository add `--project <the checkout>`, paths staying
relative to where you are: `uv run --project ../sotaigen sotaigen check your-file.json`. Run
`pnpm install` too to look at bodies in the editor.

Blender is found through the `BLENDER` environment variable, then `blender` on `PATH`, then the macOS
app. Solves are cached in the user's cache folder (`~/Library/Caches/sotaigen` on macOS,
`$XDG_CACHE_HOME/sotaigen` on Linux), or at `SOTAIGEN_CACHE`.

## 2. Turn the design into measurements

Parameters are mostly millimetres, with a few plain ratios. Each sex has 34 required ones, and optional
ones on top (breasts are a female body's), such as `outerThighArc` and those of the belly and the breast
shape.

Where to look:

- What each one is and where it is measured: the [parameter tables](../../docs/en/reference/parameters.md),
  and each slider's `help` in `sotaigen/data/sliders.json`.
- Ranges: `ALL_SPEC` in `sotaigen/params/spec.py`, as shares of the torso length, head height, stature or
  crotch height. The JSON Schema `sotaigen/data/params.schema.json` states them per parameter.
- Rules across parameters: `sotaigen/params/rules.py`; `sotaigen.params.rules.spec_bounds` turns them into
  millimetre ranges before `check`.

While doing it:

- **Stature and head count are results, not inputs.** Head height, neck length, torso length and crotch
  height add up to the stature; take the shoe soles off a height in shoes.
- **The torso and the neck are measured from the jugular notch.** The torso from the notch to the crotch,
  the neck from the notch to the chin. Design sheets rarely give the notch: read it off a 3D master or a
  side view where the front midline turns from neck into chest, or estimate it at about 0.4 head heights
  below the chin. Such an estimate moves the neck's rules directly, so mark it.
- **Some parameters are not finished sizes:** `headDepth`, `definition`, `softness` and a few more are
  controls of the masses ("an internal control" in the tables); `chestDepth` and `waistDepth` leave out
  the breasts and the belly's contour. Where the design gives a finished size, generate, measure once, and
  come back to adjust.
- **There is no direct input for the difference between the bust and the underbust.** Adjust the breast's
  projection, base breadth and other shape parameters, then measure the generated result. The reachable
  difference depends on the body and the parameter combination.
- **Make a table first:** each parameter with its value, its source (written on the sheet, read off a view,
  inferred) and its uncertainty. Mark the values read from pictures: when something goes wrong later,
  suspect them first.

## 3. Pick a sex and a starting body

- The sex, `female` or `male`, decides which parameters and rules there are.
- The typical bodies are in `sotaigen/data/bodies/<sex>/typical/`, named by proportion and build. Their
  statures differ a lot:

| Group | Head count (female / male) | Stature mm (female / male) |
|---|---|---|
| `chibi` | 4.0 / 4.0 | 1049 / 1058 |
| `compact` | 4.95 / 4.96 | 1219 / 1173 |
| `slim` | 6.18 / 6.25 | 1453 / 1444 |
| `long` | 6.66 / 6.94 | 1605 / 1730 |
| `balanced` | 6.93 / 6.94 | 1609 / 1743 |
| `rounded` | 7.17 / 6.95 | 1611 / 1740 |

- Start from the one nearest in head count. **Don't copy its millimetres:** scale it to the character first,
  with `sotaigen scale start.json scaled.json --stature 1580` (or `--heads`, `--torso`), then change what
  you measured; otherwise everything you didn't measure stays at the starting body's size.
- The file's format is in [Parameter file](../../docs/en/reference/parameters.md). `base.name` names the
  starting body, as a record only:

```json
{
  "format": "sotaigen/params",
  "version": 1,
  "sex": "female",
  "base": { "name": "female-compact" },
  "generator": "your tool",
  "body": { "headHeight": 0, "...": "the required parameters and the optional ones used" }
}
```

Keep the file in the downstream repository.

## 4. Check

```sh
sotaigen check your-file.json
```

`check` solves and wraps once, usually in seconds, at most tens of seconds. It prints in two passes.

**When a parameter is out of range or breaks a rule,** it prints the problems only and makes no body:

- The command line prints English sentences; files store the same as [diagnostics](../../docs/en/reference/diagnostics.md)
  with codes. Ranges are reported first and rules only once every range passes, so expect a few rounds.
- A range problem gives the range in mm, such as `outside 176.4 to 504.7`.
- A rule's `short by` has no unit: it is a share of the torso or a ratio. The `mend one of:` line below it
  lists how much each single parameter would have to change, lengths in mm, the smallest change first,
  all within range. Changing a length such as the torso moves other shares, so prefer the parameter the
  rule is about.

**Once everything passes:**

- `valid` means no problem.
- `target:` lines are measurements the body can't reach.
- `surface:` and `wrap:` lines are problems of the surface and the wrap.
- `warning:` lines are seam warnings.
- A local contour's target (such as `upperThighGirth`) past what this build of body can reach gets a
  `target:` line with the reachable range, and no body.

**When there is a problem:**

- Don't move a value into range quietly. Check how the parameter is measured and where its value came
  from first.
- If the measurement is right, log it with the target, the range and the evidence, for the tool's
  maintainer to decide whether to widen it.
- To move on meanwhile, take the nearest value in range, and log how far it is from the design and where
  that showed later.

## 5. Generate

```sh
sotaigen glb your-file.json out.glb
```

- A body takes about fifteen seconds.
- When a measurement can't be reached, a `target:` line says so; the body is still made, but the command
  exits non-zero. Log the differences.
- Use the default correctives for routine generation. To check contact, select the experimental IPC
  contact tables (the editor's IPC export, or `--contact` in the developer tools). Each new body bakes
  for tens of minutes.

## 6. Look

Only the editor shows a body made from a parameter file and poses it:

```sh
pnpm run editor
```

Import the file on the page, and look from the front, the side and the back, and in each check pose.
`uv run sotaigen-dev review` only knows the bodies this tool comes with. Without the editor, drive the poses yourself
(step 7) and render them yourself.

Show renders alongside the character's turnarounds at the same scale for the user to judge the
proportions and appearance.

## 7. Read the GLB

The contract is in [GLB](../../docs/en/reference/glb.md) and [Coordinates and topology](../../docs/en/reference/coordinates-topology.md).
What a downstream needs most:

- **Vertex order:** the skin's first 12676 vertices are in a fixed order: 8781 from the lower body of
  Blender Studio's CC0 stylized base mesh, then 3313 of the head, then the added loops (400 at elbows and
  knees, then 182 at shoulders and armpits); loops added later go after them. UV seam copies come last,
  their originals in `extras.sotaigen.splitFrom`, so the rest vertices are POSITION's count less the length
  of `splitFrom`. Anything written by vertex index (weight transfer, clothes binding, masks) carries over
  between bodies. A downstream already on the same CC0 base mesh can map its first 8781 vertices one to one;
  the added loops interpolate by `refined.parents` in `sotaigen/data/template/base-mesh.json`.
- **Local axes:** Y along the bone, Z forward, X the axis it bends about.
- **The helper bones and the correctives need a runtime that drives them:** the helper bones' drivers (the
  shoulder girdle, the half and twist bones) are in `extras.sotaigen.drivers`. The skin has over a hundred
  corrective morph targets, weighted by the table in `extras.sotaigen.correctives`; their names are not
  contract, so find them by the names the table lists. `web/runtime/driver.mjs` is the reference
  implementation (`poseQuaternions`, `correctiveWeights`), callable from Node: bone poses in, every bone's
  rotation and every morph weight out. Blender and ordinary players don't read these tables; without them
  the twist, the shoulders, the armpits, the groin and the hanging arms deform wrongly.
- **Rest pose:** an A pose, the arms 35° off straight down (55° below horizontal), the legs hanging with
  only a gap against touching. The angle is the same for every body. For clothes bound in another pose
  (other arm abduction, the feet further apart), write the `.blend` in that pose and it becomes the file's
  rest pose, so the clothes' weights carry over by bone name with no posing:
  `sotaigen blend your-file.json out.blend --arm-abduction 60 --ankle-spacing 150`, or `--t-pose` (arms
  90° off vertical, straight, palms down). The GLB stays in the A pose. The file's control bones are at
  zero in the new rest pose, and the correctives are driven as before.
- **Check poses:** in `extras.sotaigen.checkPoses` (not contract, may change with versions), angles relative
  to the rest pose, not to standing: `armsDown` turns the upper arms 24° further down from rest, 35° − 24° =
  11° off vertical.
- **Features, ears and hair** are nodes of their own after the skin: the mouth bound to `jaw`, the irises,
  pupils and highlights to `leftEye` and `rightEye`, the rest to `head`, the expressions on their own
  meshes. The head's skin is not a node: it is in the `body` mesh, vertices 8781 to 12093, and the UV copies
  of them.

## 8. Attach the head and the hair

When the downstream head is ready, `sotaigen swap-head` puts it on the body's `.blend`; it needs Blender
and no hand cutting:

```sh
sotaigen swap-head body.blend your-head.blend out.blend
```

- **Replace the whole head above the neck ring:** the command above. It aligns and scales the head by its
  neck opening and crown, joins the opening to the neck ring and blends a band, transfers the weights,
  moves `jaw` and the eye bones to your head's, then poses a head turn, a nod and an open jaw to look for
  collapsed faces. It writes nothing when the seam doesn't close or faces flip. The report is in `out.json`
  and in the file's `rig["sotaigen"]["headSwap"]`.
- **Keep this tool's head shape and replace only the face and the hair:** add `--parts-only`, with the
  parts made on this body's `.blend` and left where they are. All placeholder parts are removed;
  `--keep-parts ears` keeps those named.

`your-head.blend` holds:

- A skin mesh open at the neck. The skin is the largest piece with an opening, or the one marked with the
  vertex group `headRegion`; when marked only that piece is checked, and when not, pieces whose boundary
  touches itself (generated hair often does) are skipped and stay parts. The neck opening may have any
  number of vertices, and its centre places the head; one of 36 vertices is welded vertex to vertex. A
  character whose eyes and mouth are carved openings in the skull goes this way too, the other openings
  staying.
- Bones named `jaw`, `leftEye`, `rightEye` in an armature (or empties of those names), at your head's jaw
  hinge and eyeball centres. A bone that isn't there stays where it was, and the file's `quality` says so.
- Teeth, tongue and lower face with a vertex group `jaw`, upper teeth with `head`, eyeballs with `leftEye`,
  `rightEye`: such pieces are bound by their groups. Pieces with no bone's vertex group (hair, say) take
  weights from the new skin, a strand far from the skin moving whole with its nearest place. Weights
  painted on the skin are kept away from the seam.

The coordinates are the body's: X to the wearer's left, -Y forward, Z up, any size and object transform.
Only the tool's own test heads have gone all the way through: log what a real head runs into.

Mind:

- The neck-to-head rule (neck girth at least 0.47 head heights) only protects this tool's head; the swap
  doesn't check it and poses the head instead. For a thinner neck, write the body with
  `sotaigen blend FILE OUT --for-head-swap`: only that rule is let through, the file's quality notes it,
  and the neck's own minimum still holds. After the swap, check the new head turning and nodding.
- The tool counts collapsed faces; the look is yours to judge: either way, look at the seam and under the
  chin when the head turns, nods and the jaw opens. The hand-to-head pose assets are still solved for the
  old head.

## 9. Dress

- Bind clothes and transfer their weights by the body's vertex indices, and they carry over to the other
  bodies.
- There are no extra bones for clothes, such as a sleeve's puff bones, a prop point on a hand or a bag's
  point: add them downstream.
- Clothes don't follow the corrective morph targets: where the armpits and the groin are corrected by
  them, the clothes need their own handling.
- The feet's spacing is not a parameter. `legClearance` is only a gap against the legs touching; for shoes
  made for a wider stance, write the `.blend` with `--ankle-spacing <mm>` (between the ankle centres): the
  thighs spread to it, the soles stay flat on the floor, and the pelvis sinks a few millimetres.
- Skin entirely under clothes may be hidden, but there are no ready regions to hide yet: decide them and
  log it.
- Look for interpenetration in the check poses, and at how the clothes look. Known trouble: the shoulders
  raised high to the side, the groin in deep hip flexion, and the feet in shoes.

## 10. Check the motion

The check poses are in step 7. At least: standing, arms down, arms up, crouch, head turn and nod, with and
without clothes.

## Log

Write one each time, keep it in the downstream repository, and hand it to the tool's maintainer at the
end. Each entry has these columns:

| Column | What goes in it |
|---|---|
| Step | which step above |
| Meant to | the result expected |
| What happened | the command, the output, the path of a screenshot |
| Worked around how | say so when it wasn't |
| Whose | the tool lacks something, this guide is wrong or unclear, or it is the downstream's own |
| Impact | blocked, needed hand work, or only awkward |

Facts and evidence only. Keep character measurements and assets in the downstream log. For a report
to the maintainer, reproduce the issue on a built-in or independently constructed test body, and include
only the parameters and steps needed to reproduce it without revealing the character.
