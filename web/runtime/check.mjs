// The browser half of scripts/runtime_check.py: glTF Validator, three.js skinning in real Chrome,
// load times and screenshots. Run that script rather than this file; it writes the reference data.
//
//   node web/runtime/check.mjs --names a,b --validate extra.glb --out report.json --shots dir

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import validator from "gltf-validator";
import { chromium } from "playwright-core";
import { BUILD, serve } from "../serve.mjs";

// The system's Chrome by default; SOTAIGEN_CHROME_CHANNEL names another, or empty for Playwright's own Chromium.
const CHANNEL = (process.env.SOTAIGEN_CHROME_CHANNEL ?? "chrome") || undefined;

const LOAD_RUNS = 7;
const SHOTS = [
  ["female-long", "rest", true],
  ["female-long", "armsUp", true],
  ["female-long", "armsUp", false],
  ["female-chibi", "reach", true],
  ["female-balanced", "crouch", true],
];
// The whole body in toon, for creases along the seams, and its head.
const TOON_BODY_PRESET = "female-long";
const TOON_BODY_VIEWS = ["threequarter", "backquarter", "side", "back"];
const HEAD_PRESETS = ["female-chibi", "female-long", "female-balanced", "female-rounded"];
const HEAD_VIEWS = ["front", "threequarter", "side", "back"];
const HEAD_SHOTS = [
  ["rest", "rest", ""],
  ["happy", "rest", "&morph=happy:1"],
  ["headTurn", "headTurn", ""],
];
const DRIVEN_PRESET = "female-long";

const { values } = parseArgs({
  options: {
    names: { type: "string" },
    validate: { type: "string", multiple: true, default: [] },
    out: { type: "string" },
    shots: { type: "string" },
  },
});
const presets = values.names.split(",");

async function validate(path) {
  const report = await validator.validateBytes(new Uint8Array(readFileSync(path)), {
    uri: path,
    format: "glb",
    maxIssues: 0,
  });
  const { numErrors, numWarnings, numInfos, numHints, messages } = report.issues;
  return {
    path,
    validatorVersion: report.validatorVersion,
    numErrors,
    numWarnings,
    numInfos,
    numHints,
    messages: messages.map(({ code, message, severity, pointer }) => ({
      code,
      message,
      severity,
      pointer,
    })),
  };
}

function median(numbers) {
  const sorted = [...numbers].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// A new browser with a new profile each time: nothing cached, no compiled shader left from before.
async function loadTimes(base, preset) {
  const rows = [];
  for (let i = 0; i < LOAD_RUNS; i++) {
    const browser = await chromium.launch({ channel: CHANNEL, headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    await page.goto(`${base}/runtime/shot.html?body=${preset}&pose=rest`);
    await page.waitForFunction(() => window.bodyReady === true, null, { timeout: 60000 });
    rows.push(await page.evaluate(() => window.bodyStats));
    await browser.close();
  }
  const row = { preset, runs: LOAD_RUNS };
  for (const key of ["fetchMs", "parseMs", "firstFrameMs", "totalMs"]) {
    const values = rows.map((r) => r[key]);
    row[key] = median(values);
    row[`${key}Max`] = Math.max(...values);
  }
  return row;
}

async function screenshots(browser, base, folder) {
  mkdirSync(folder, { recursive: true });
  const taken = [];
  for (const [preset, pose, driven] of SHOTS.filter(([p]) => presets.includes(p))) {
    const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
    const flag = driven ? "" : "&driven=0";
    await page.goto(`${base}/runtime/shot.html?body=${preset}&pose=${pose}${flag}`);
    await page.waitForFunction(() => window.bodyReady === true, null, { timeout: 60000 });
    const path = join(folder, `${preset}-${pose}${driven ? "" : "-undriven"}.png`);
    await page.screenshot({ path });
    taken.push(path);
    await page.close();
  }
  async function shoot(page, query, name) {
    await page.goto(`${base}/runtime/shot.html?${query}`);
    await page.waitForFunction(() => window.bodyReady === true, null, { timeout: 60000 });
    const path = join(folder, name);
    await page.screenshot({ path });
    taken.push(path);
  }
  if (presets.includes(TOON_BODY_PRESET)) {
    const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
    for (const view of TOON_BODY_VIEWS) {
      await shoot(
        page,
        `body=${TOON_BODY_PRESET}&view=${view}&look=toon`,
        `${TOON_BODY_PRESET}-toon-${view}.png`,
      );
    }
    await page.close();
  }
  const page = await browser.newPage({ viewport: { width: 700, height: 700 } });
  for (const preset of HEAD_PRESETS.filter((p) => presets.includes(p))) {
    for (const [tag, pose, extra] of HEAD_SHOTS) {
      for (const view of HEAD_VIEWS) {
        await shoot(
          page,
          `body=${preset}&framing=head&look=toon&view=${view}&pose=${pose}${extra}`,
          `${preset}-head-${tag}-${view}.png`,
        );
      }
    }
  }
  await page.close();
  return taken;
}

// What each look costs per frame and when first drawn, in a new browser each time so that no
// compiled shader is left from before.
async function lookCost(base, preset) {
  const rows = {};
  for (const look of ["standard", "toon"]) {
    const runs = [];
    for (let i = 0; i < LOAD_RUNS; i++) {
      const browser = await chromium.launch({ channel: CHANNEL, headless: true });
      const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
      await page.goto(`${base}/runtime/shot.html?body=${preset}&look=${look}`);
      await page.waitForFunction(() => window.bodyReady === true, null, { timeout: 60000 });
      const first = await page.evaluate(() => window.bodyStats.firstFrameMs);
      runs.push({ ...(await page.evaluate(() => window.frameMs(120))), first, state: await page.evaluate(() => window.bodyState()) });
      await browser.close();
    }
    rows[look] = {
      ms: median(runs.map((run) => run.ms)),
      calls: runs[0].calls,
      triangles: runs[0].triangles,
      firstFrameMs: median(runs.map((run) => run.first)),
      firstFrameMsMax: Math.max(...runs.map((run) => run.first)),
      look: runs[0].state.look,
      colors: runs[0].state.colors,
    };
  }
  return rows;
}

const report = { validation: [], consistency: [], parts: [], loads: [], shots: [] };
for (const preset of presets) report.validation.push(await validate(join(BUILD, "bodies", preset, "body.glb")));
for (const path of values.validate) report.validation.push(await validate(path));

const server = await serve(0, { bodies: false });
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ channel: CHANNEL, headless: true });
report.browser = browser.version();
try {
  const page = await browser.newPage();
  const problems = [];
  page.on("pageerror", (error) => problems.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(message.text());
  });
  await page.goto(`${base}/runtime/check.html`);
  await page.waitForFunction(() => window.checkReady === true, null, { timeout: 60000 });
  for (const preset of presets) {
    report.consistency.push(await page.evaluate((name) => window.checkPreset(name), preset));
    report.parts.push(await page.evaluate((name) => window.checkParts(name), preset));
  }
  report.pageProblems = problems;
  await page.close();
  for (const preset of ["female-chibi", "female-long", "female-balanced"].filter((p) => presets.includes(p))) {
    report.loads.push(await loadTimes(base, preset));
  }
  if (values.shots) {
    report.shots = await screenshots(browser, base, values.shots);
    const preset = presets.includes(DRIVEN_PRESET) ? DRIVEN_PRESET : presets[0];
    report.lookCost = await lookCost(base, preset);
  }
} finally {
  await browser.close();
  server.close();
}
writeFileSync(values.out, JSON.stringify(report, null, 1));
