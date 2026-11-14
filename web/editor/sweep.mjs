// How much of each slider's range the rules leave on every preset, and on one preset the body at both
// usable ends, really made, with what the worker says about it and a picture. The ranges are provisional
// and the masses move, so run this again after a change to the form.
//
//   pnpm run editor-sweep [--name female-long --sliders hipDepth,jaw --out build/reports/editor-sweep]
//
// Writes rules.json, bodies.json and sheet.png. Its bodies go to the usual cache, so a second run is quick.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import { serve } from "../serve.mjs";
import { validate } from "./params-file.ts";
import { applySliders, sliderBounds, sliderStart } from "./sliders.ts";

// The system's Chrome by default; SOTAIGEN_CHROME_CHANNEL names another, or empty for Playwright's own Chromium.
const CHANNEL = (process.env.SOTAIGEN_CHROME_CHANNEL ?? "chrome") || undefined;

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");
const STEPS = 40;
const FRONT = [0, 0.02, 1];
const SIDE = [1, 0.02, 0.02];
const QUARTER = [0.6, 0.04, 1];
const BEHIND = [0.6, 0.04, -1];

// Where a slider's effect shows: bone, share of the height to fit, direction, lift. Not listed: its group's.
const GROUP_CAMERA = {
  proportion: ["spine", 1.12, QUARTER, 0],
  quick: ["spine", 1.12, QUARTER, 0],
  headHeight: ["head", 0.3, QUARTER, 0.05],
  neck: ["neck", 0.3, QUARTER, 0],
  shoulder: ["upperChest", 0.5, FRONT, 0.02],
  chest: ["upperChest", 0.42, QUARTER, 0],
  arm: ["leftLowerArm", 0.6, FRONT, 0],
  waist: ["spine", 0.62, SIDE, 0],
  hip: ["hips", 0.5, QUARTER, -0.02],
  glute: ["hips", 0.46, BEHIND, 0],
  leg: ["leftUpperLeg", 0.62, QUARTER, -0.08],
};
const CAMERA = {
  chestDepth: ["chest", 0.5, SIDE, 0],
  hipDepth: ["hips", 0.5, SIDE, 0],
  pelvisLean: ["hips", 0.62, SIDE, 0.04],
  headWidthRatio: ["head", 0.3, FRONT, 0.06],
  headDepth: ["head", 0.3, SIDE, 0.06],
  handLength: ["leftHand", 0.3, FRONT, 0],
  upperThighGirth: ["leftUpperLeg", 0.62, FRONT, -0.08],
  ankleGirth: ["leftFoot", 0.32, FRONT, 0.04],
  footLength: ["leftFoot", 0.32, SIDE, 0.02],
  legClearance: ["leftLowerLeg", 0.72, FRONT, -0.02],
  thighFrontShift: ["leftUpperLeg", 0.62, SIDE, -0.08],
  ankleHeight: ["leftFoot", 0.32, SIDE, 0.03],
};

const { values: args } = parseArgs({
  options: {
    name: { type: "string", default: "female-long" },
    sliders: { type: "string" },
    out: { type: "string", default: resolve(ROOT, "build", "reports", "editor-sweep") },
  },
});
mkdirSync(args.out, { recursive: true });

const definition = JSON.parse(readFileSync(resolve(HERE, "../../sotaigen/data/sliders.json"), "utf8"));
// The report is in Chinese, like the rest of what the diagnostic tools write.
const words = JSON.parse(readFileSync(resolve(HERE, "../messages/zh.json"), "utf8"));
const groupOf = Object.fromEntries(definition.groups.flatMap((group) => group.sliders.map((name) => [name, group])));
const names = args.sliders ? args.sliders.split(",") : Object.keys(definition.sliders);

const server = await serve(0);
const address = `http://127.0.0.1:${server.address().port}`;
const { SEXES } = await import(resolve(ROOT, "build", "editor", "params.generated.mjs"));
// Each preset is held to its own sex's ranges and rules.
const rangesFor = (base) => ({
  body: SEXES[base.sex].specBounds(base.body),
  derived: { heads: SEXES[base.sex].HEADS_RANGE, stature: SEXES[base.sex].STATURE_RANGE },
});
const { presets } = await (await fetch(`${address}/api/presets`)).json();

const moved = (base, name, value) => {
  const schema = SEXES[base.sex];
  const out = applySliders(base, { [name]: value }, definition, rangesFor(base), schema);
  return { out, ...validate(out.body, schema) };
};

const rules = {};
for (const name of names) {
  rules[name] = { label: words[`slider_${name}`], presets: {} };
  for (const [preset, base] of Object.entries(presets)) {
    // A parameter this sex doesn't have, such as a male body's breast: no range, nothing to sweep.
    const bounds = sliderBounds(definition.sliders[name], rangesFor(base));
    if (!bounds) continue;
    const [low, high] = bounds;
    let usable = 0;
    let first = null;
    let last = null;
    const reasons = new Set();
    for (let i = 0; i <= STEPS; i++) {
      const value = low + ((high - low) * i) / STEPS;
      const { errors } = moved(base, name, value);
      if (errors.length === 0) {
        usable += 1;
        first ??= value;
        last = value;
      }
      for (const line of errors) reasons.add(line.replace(/, short by.*/, ""));
    }
    rules[name].presets[preset] = { low, high, share: usable / (STEPS + 1), first, last, reasons: [...reasons] };
  }
}
writeFileSync(resolve(args.out, "rules.json"), JSON.stringify(rules, null, 1));

const browser = await chromium.launch({ channel: CHANNEL, headless: true });
const page = await browser.newPage({ viewport: { width: 1180, height: 760 } });
const settled = () =>
  page.waitForFunction(
    () => {
      const { rebuilder, state } = window.editor ?? {};
      return state?.body && rebuilder.phase !== "generating" && rebuilder.phase !== "adjusting" && !state.undrawn;
    },
    null,
    { timeout: 600000 },
  );
await page.goto(`${address}/?body=${args.name}`);
await settled();
await page.addStyleTag({ content: "main .pointer-events-none, main p { display: none !important }" });
const base = presets[args.name];

const load = async ({ body }) => {
  const text = JSON.stringify({ format: "sotaigen/params", version: 5, sex: base.sex, base: { name: args.name }, body });
  await page.evaluate((file) => window.editor.importFile(file), text);
  await settled();
  // The error's own words: a failure nobody wrote down can't be looked into.
  return page.evaluate(() => ({
    phase: window.editor.rebuilder.phase,
    error: window.editor.rebuilder.error?.message ?? null,
    failures: window.editor.state.report?.failures ?? [],
  }));
};
const aim = (camera) =>
  page.evaluate(([bone, span, direction, lift]) => {
    const view = window.editor.viewport;
    const { body } = view;
    body.mesh.geometry.computeBoundingBox();
    const height = body.mesh.geometry.boundingBox.max.y;
    const target = body.mesh.skeleton.getBoneByName(bone).getWorldPosition(view.controls.target.clone());
    target.y += lift * height;
    const toward = view.camera.position.clone().set(...direction).normalize();
    const distance = (span * height) / Math.min(1, view.camera.aspect) / (2 * Math.tan((view.camera.fov * Math.PI) / 360));
    return { target: target.toArray(), position: target.clone().addScaledVector(toward, distance).toArray() };
  }, camera);
const shoot = async ({ target, position }) => {
  await page.evaluate(
    ([to, from]) => {
      const view = window.editor.viewport;
      view.flight = null;
      view.controls.target.set(...to);
      view.camera.position.set(...from);
      view.controls.update();
      view.invalidate();
    },
    [target, position],
  );
  await page.waitForTimeout(250);
  return (await page.locator("main canvas").screenshot({ type: "jpeg", quality: 82 })).toString("base64");
};

const rows = [];
for (const name of names) {
  const spec = definition.sliders[name];
  if (!rules[name].presets[args.name]) {
    console.log(name.padEnd(18), `not a slider of ${args.name}`);
    continue;
  }
  const { low, high, first, last } = rules[name].presets[args.name];
  await load(base);
  const pose = await aim(CAMERA[name] ?? GROUP_CAMERA[groupOf[name].id]);
  const start = await shoot(pose);
  const ends = [];
  for (const value of [first ?? low, last ?? high]) {
    const { out, seams } = moved(base, name, value);
    const report = await load(out);
    ends.push({ value, seams, ...report, image: await shoot(pose) });
  }
  rows.push({ name, label: words[`slider_${name}`], group: words[`group_${groupOf[name].id}`], start: sliderStart(spec, base, SEXES[base.sex]), startImage: start, ends });
  console.log(name.padEnd(18), ends.map((end) => `${end.value.toFixed(3)} ${end.phase} ${end.error ?? ""} ${end.failures.join("; ")}`).join(" | "));
}
writeFileSync(
  resolve(args.out, "bodies.json"),
  JSON.stringify(
    rows.map(({ startImage: _startImage, ends, ...row }) => ({ ...row, ends: ends.map(({ image: _image, ...end }) => end) })),
    null,
    1,
  ),
);

// A start measured on the made body (abdomenFrontArc) isn't known from the parameters alone.
const number = (value) => (value === undefined ? "—" : Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(3));
const cell = (label, image, bad) =>
  `<figure><img src="data:image/jpeg;base64,${image}"><figcaption class="${bad ? "bad" : ""}">${label}</figcaption></figure>`;
const endCell = (word, end) => cell(`${word} ${number(end.value)}${end.phase !== "idle" ? " 失败" : end.failures.length ? " ⚠" : ""}`, end.image, end.phase !== "idle" || end.failures.length);
const sheet = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
await sheet.setContent(`<!doctype html><meta charset=utf-8><style>
body{margin:0;background:#fff;font:13px system-ui,"PingFang SC";color:#222;display:grid;grid-sex-columns:1fr 1fr;gap:12px 22px;padding:16px;width:1560px}
h3{margin:0 0 3px;font-size:14px;font-weight:600}h3 span{color:#888;font-weight:400;margin-left:6px;font-size:12px}
.row{display:grid;grid-sex-columns:repeat(3,1fr);gap:3px}figure{margin:0}img{width:100%;display:block;aspect-ratio:1;object-fit:cover}
figcaption{font-size:11px;color:#555;text-align:center;padding:1px 0}.bad{color:#b3362c}</style>
${rows.map((row) => `<div><h3>${row.label}<span>${row.name} · ${row.group}</span></h3><div class=row>${endCell("低", row.ends[0])}${cell(`预设 ${number(row.start)}`, row.startImage)}${endCell("高", row.ends[1])}</div></div>`).join("")}`);
await sheet.waitForTimeout(400);
await sheet.screenshot({ path: resolve(args.out, "sheet.png"), fullPage: true });
await browser.close();
server.close();
server.closeAllConnections();
console.log(`\n${args.out}/sheet.png`);
process.exit(0);
