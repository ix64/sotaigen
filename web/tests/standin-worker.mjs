// Stands in for sotaigen_dev/editor/worker.py behind web/serve.mjs: the same protocol, answering every body
// with one real file after a short wait. So the page can be tested on its own, quickly. A few parameter
// values ask for what only the real worker could otherwise be made to do. A third argument, "no-blender",
// is a machine without Blender.

import { copyFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";

const [file, delay = "250", machine = ""] = process.argv.slice(2);
const say = (row) => process.stdout.write(`${JSON.stringify(row)}\n`);
const hasBlender = machine !== "no-blender";

say({ ready: true, version: "stand-in", blendVersion: "stand-in-blend", blender: hasBlender });

// A mesh too broken to export, by the value of hipDepth 187, as sotaigen.output.quality.broken finds it. Its GLB is
// still made for the page to show, unless that GLB could only be for saving.
const BROKEN = [
  { code: "wrap.flipped", level: "broken", values: { count: 3, at: { hip: 2, leg: 1 } } },
  { code: "skin.open", level: "broken", values: { edges: 2 } },
];

// What sotaigen.mend.search answers, by the value of hipDepth: 187 ends at once with a parameter of the leg
// group; 190, broken too, finds none; 191 takes as long as a test wants, to be stopped.
function mend({ id, body }) {
  if (body.hipDepth === 191) return say({ id, stage: "1/35" });
  say({ id, stage: "1/35" });
  const found = body.hipDepth === 190 ? { tried: 35, total: 35 } : { key: "kneeGirth", part: "leg", before: 200, after: 240.5, unit: "mm", tried: 3, total: 35 };
  return setTimeout(() => say({ id, ok: true, mend: found }), Number(delay));
}

// A .blend: the same file, or why not, by the value of hipDepth.
function blend({ id, out, body }) {
  if (!hasBlender) return say({ id, ok: false, noBlender: true });
  if ([187, 190, 191].includes(body.hipDepth)) return say({ id, ok: false, broken: BROKEN });
  if (body.hipDepth === 188) return say({ id, ok: false, problems: [{ code: "rule.hip_shallow_thigh", level: "error", values: { short: 0.01 } }] });
  if (body.hipDepth === 189) return say({ id, ok: false, error: "ValueError: hipBack asked past what this body can reach" });
  copyFileSync(file, out);
  return say({ id, ok: true, blendBytes: 1, seconds: 0.5 });
}

// A body's contact poses: two that are reached, and with hipDepth 190 one that isn't.
function touch({ id, out, body }) {
  const arm = { leftUpperArm: [60, 0, 0], leftLowerArm: [120, 0, 0] };
  const reports = { "touchCrown.left": { reached: body.hipDepth !== 190, arrivalMm: 31.5, place: 0 }, "touchCheek.left": { reached: true, arrivalMm: 0.2, place: 0 } };
  writeFileSync(out, JSON.stringify({ poses: { "touchCrown.left": arm, "touchCheek.left": arm }, reports }));
  say({ id, ok: true, seconds: 0.1 });
}

createInterface({ input: process.stdin }).on("line", (line) => {
  const request = JSON.parse(line);
  const { id, out, body, contact } = request;
  if (request.touch) {
    setTimeout(() => touch(request), Number(delay));
    return;
  }
  if (request.blend) {
    setTimeout(() => blend(request), Number(delay) * 3);
    return;
  }
  if (request.mend) {
    mend(request);
    return;
  }
  const broken = [187, 190, 191].includes(body.hipDepth) ? BROKEN : [];
  // Refused before any table is baked.
  if (contact && broken.length) {
    setTimeout(() => say({ id, ok: false, broken }), Number(delay));
    return;
  }
  // A body with its contact tables takes longer, and says which table it is on.
  if (contact) say({ id, stage: "shoulder" });
  setTimeout(() => {
    copyFileSync(file, out);
    say({
      id,
      ok: true,
      seconds: { wrap: 0.1, rig: 0.1, head: 0.01, parts: 0.01, surface: 0.01, export: 0.01, total: 0.25 },
      // A body the rules pass and the surface doesn't, as the real hipTissueMidline (sotaigen.form.checks.field_failures).
      failures:
        body.hipDepth === 186
          ? [{ code: "surface.hipTissueMidline", level: "warning", values: {} }]
          : broken.length
            ? [BROKEN[0]]
            : [],
      broken,
      contact: contact ? "ipc" : "sculpted",
    });
  }, Number(delay) * (contact ? 8 : 1));
});
