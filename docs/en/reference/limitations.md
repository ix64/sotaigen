# Known limitations

[中文](../../reference/limitations.md)

Known defects and pending validation in 0.1.0. Geometry changes from fixes follow the
[compatibility policy](README.md#how-far-compatibility-goes).

## Shapes and deformation

- **Groin:** when the hip flexes, the lower belly and the top of the thigh press together only in part,
  and the middle of the thigh's root stays stiff; past 85° of flexion a ring of creases shows, and deep
  flexion goes through itself.
- **Shoulder and armpit:** raised high to the side, the armpit shows a long slant and the inner upper arm
  a hollow; the back fold of the armpit has a small crease; raised forward to 95° and 130°, a few more
  folds appear where the armpit is pressed. A compound Euler angle that makes no sense turns the upper
  arm into the torso, and in a compound pose with a large twist the correctives fade to about half.
- **Clean ranges are narrower than people's:** the `.blend`'s joint limits follow human ranges of motion,
  and the mesh looks right over part of them only; past the clean range a pose creases or collapses.
- **Toon terminator:** a small step at the neck, and uneven boundaries at the ankles and wrists, all from
  the geometry rather than the export.
- **Clothes are unchecked:** no standard garment has been tried on in the check poses yet, and there are
  no regions to hide under clothes and no landmarks to fit clothes by.
- **Heavy arms hanging:** on `female-heavyarm`, an arm hanging at 24° (`armsDown`) puts the forearm into
  the side of the waist.
- **Contact is static:** the default correctives are sculpted or from the implicit field; the IPC contact
  tables solved by contact are experimental (`sotaigen[contact]`), in the GLB only, not in the `.blend`.
  Not every body's tables bake; a table that doesn't falls back to the default correctives, and the file
  says so (`contact.failed`): heavy-armed bodies where the arm lies against the body (`female-heavyarm`'s
  and `male-heavyarm`'s hanging-arm tables, `male-wideheavyarm`'s shoulder table), `female-smallbust`'s
  hanging-arm table (the solver fails numerically), and the groin tables of 8 small, slim or male bodies
  (with the hips bent unevenly to 50-95°, one face folds flat, or the midline between the legs crosses
  itself too much).

## Files and tools

- **The drivers read Euler angles:** the helper bones follow by `Rz · Ry · Rx` Euler angles. Going back
  from a quaternion, an absolute `y` past 90° gives another set of equivalent angles: the same rotation,
  but a different shoulder girdle.
- **Ordinary importers don't read the driver tables:** Blender's glTF importer and players in general
  don't read the two tables in `extras.sotaigen` and only play baked animation; to pose a body elsewhere,
  implement the [driver tables](glb.md#helper-bone-drivers), or use the `.blend`.
- **No VRM:** VRM 1.0 export is not done.
- **A parameter file can't be rendered directly:** the review tool knows only the bodies this tool comes
  with; make a GLB or a `.blend` from a downstream file to look at it.
- **The editor** makes one body at a time, a second page's request cancelling the first; it is for the
  local machine only.

## Platforms

- macOS and Linux are supported, Windows is not.
- So far it has only run on Apple Silicon macOS: Blender 5.2.3, Chrome, Apple's GPU. Other GPUs and
  Blender on Linux are only in the CI configuration, not yet run.
- The same parameters give the same bytes given the same locked dependencies (`uv.lock`).
