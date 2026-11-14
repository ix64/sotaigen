// The real Python worker on real bodies: what a stand-in can't vouch for. Few, because each takes
// seconds on a warm machine, and in order, because they share the one worker. The waits go to build/web/smoke.json.

import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import validator from "gltf-validator";
import { BODY, expect, test } from "./fixtures";

const OUT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "build", "web");
const REAL = ["uv", "run", "python", "-m", "sotaigen_dev.editor.worker"];
// A cold body takes about ten seconds; this leaves room for a cold solver cache on a loaded machine.
const LONG = 120000;

test.use({ command: REAL });
test.describe.configure({ mode: "serial" });

async function validate(bytes: Buffer, uri: string) {
  const { issues } = await validator.validateBytes(new Uint8Array(bytes), { uri, format: "glb", maxIssues: 0 });
  expect(issues.numErrors, `${uri} has errors`).toBe(0);
  expect(issues.numWarnings, `${uri} has warnings`).toBe(0);
}

test("a preset is byte for byte what uv run sotaigen-dev export wrote", async ({ editor, page }) => {
  await page.goto(`${editor.address}/?body=female-long`);
  await page.waitForFunction(() => window.editor?.ready === true);
  await editor.settled(LONG);
  expect(await editor.phase()).toBe("idle");
  const [saved] = await Promise.all([page.waitForEvent("download"), page.locator("#exportGlb").click()]);
  const bytes = readFileSync(await saved.path());
  expect(bytes.equals(readFileSync(BODY)), "not the bytes uv run sotaigen-dev export wrote").toBe(true);
  await validate(bytes, "female-long.glb");
});

// Blender reads the file, by the script tests/test_blend.py poses them with: it fails without the object
// `rig`. Python runs it, because Python is what knows where Blender is.
const READ_BLEND = `
import sys
from pathlib import Path
from sotaigen.output.blender import run_blender, script
blend, folder = map(Path, sys.argv[1:])
(folder / "pose.json").write_text('{"poses": {"rest": {}}, "central": [], "limits": [false]}')
run_blender(script("pose"), folder / "pose.json", folder / "read", blend=blend)
`;

// With Blender, the file. Without it the same export can only say so, which is a real case of its own:
// this isn't a skip, and the run's output says the file wasn't opened.
test("a .blend from the page opens in Blender with its rig, or the page says Blender is missing", async ({ editor, page }, info) => {
  await page.goto(`${editor.address}/?body=female-long`);
  await page.waitForFunction(() => window.editor?.ready === true);
  await editor.settled(LONG);
  const { blender } = await (await page.request.get(`${editor.address}/api/blend`)).json();
  await page.locator("#exportMenu").click();
  if (!blender) {
    console.log("Blender was not found, so no .blend was written or opened (set BLENDER=/path/to/blender).");
    await expect(page.locator("#exportBlend")).toBeDisabled();
    await expect(page.locator("#blendWhy")).toContainText("BLENDER");
    return;
  }
  const [saved] = await Promise.all([page.waitForEvent("download", { timeout: LONG }), page.locator("#exportBlend").click()]);
  expect(saved.suggestedFilename()).toBe("female-long.blend");
  await expect(page.locator("#blendNote")).toHaveText(".blend 导出完成");
  const folder = info.outputPath("blend");
  mkdirSync(folder, { recursive: true });
  const blend = join(folder, saved.suggestedFilename());
  await saved.saveAs(blend);
  const read = spawnSync("uv", ["run", "python", "-c", READ_BLEND, blend, folder], { cwd: resolve(OUT, "..", ".."), encoding: "utf8" });
  expect(read.status, read.stderr).toBe(0);
  const file = JSON.parse(readFileSync(join(folder, "read.json"), "utf8"));
  expect(file.objects).toEqual(expect.arrayContaining(["rig", "body"]));
  expect([file.info.format, file.info.version, file.info.name]).toEqual(["sotaigen/blend", 1, "female-long"]);
  expect(file.info.correctives).toEqual({
    "axilla.left": "sculpted", "axilla.right": "sculpted", groin: "sculpted", "hang.left": "field", "hang.right": "field",
  });
  expect(file.texts).toBe(0);
  expect(file.notSimple).toEqual([]);
  expect(file.autoexecFail).toBe(false);
});

test("a contact pose is solved by the real worker for the body on show", async ({ editor, page }) => {
  const replies: { reports: Record<string, { reached: boolean }>; poses: Record<string, Record<string, number[]>> }[] = [];
  page.on("response", async (response) => {
    if (/\/api\/touch\/[0-9a-f]+\.touch$/.test(response.url())) replies.push(await response.json());
  });
  await page.goto(`${editor.address}/?body=female-long&pose=touchCheek.left`);
  await page.waitForFunction(() => window.editor?.ready === true);
  await editor.settled(LONG);
  const started = Date.now();
  await expect.poll(() => replies.length, { timeout: LONG }).toBe(1);
  await expect.poll(() => page.evaluate(() => window.editor.state.touchJob)).toBeNull();
  const [{ reports, poses }] = replies;
  for (const name of ["touchCrown.left", "touchForehead.left", "touchCheek.left", "touchCheek.both", "touchMouth.left", "touchOcciput.left", "touchOcciput.both"]) {
    expect(reports[name]?.reached, name).toBe(true);
    expect(Object.keys(poses[name]).length, name).toBeGreaterThan(3);
  }
  mkdirSync(OUT, { recursive: true });
  await page.locator("#viewport, canvas").first().screenshot({ path: resolve(OUT, "smoke-touch-cheek.png") });
  console.log(`contact poses solved in ${((Date.now() - started) / 1000).toFixed(1)} s after the body landed`);
});

test("a moved slider makes a valid body of those parameters, and comes back from the cache", async ({ editor, page }) => {
  await page.goto(`${editor.address}/?body=female-long&panel=quick`);
  await page.waitForFunction(() => window.editor?.ready === true);
  await editor.settled(LONG);
  await editor.drag("丰满度", 0.8);
  await editor.settled(LONG);
  expect(await editor.phase()).toBe("idle");
  const timing = await page.evaluate(() => window.editor.log.timing);
  const landed = timing.at(-1);
  expect(landed.cached).toBe(false);
  mkdirSync(OUT, { recursive: true });
  writeFileSync(resolve(OUT, "smoke.json"), JSON.stringify({ browser: page.context().browser()?.version(), timing }, null, 1));
  const [saved] = await Promise.all([page.waitForEvent("download"), page.locator("#exportGlb").click()]);
  await validate(readFileSync(await saved.path()), "moved.glb");
  await page.locator("#undo").click();
  await editor.settled(LONG);
  await page.locator("#redo").click();
  await editor.settled(LONG);
  expect(await page.evaluate(() => window.editor.log.timing.at(-1).cached)).toBe(true);
});

test("a real worker killed in the middle is reported, and the next body still comes", async ({ editor, page, service }) => {
  await page.goto(`${editor.address}/?body=female-long&panel=quick`);
  await page.waitForFunction(() => window.editor?.ready === true);
  await editor.settled(LONG);
  await editor.drag("丰满度", 0.7);
  await page.waitForFunction(() => window.editor.rebuilder.running !== null);
  await page.waitForTimeout(1000);
  process.kill(-service.server.worker.child.pid, "SIGKILL");
  await editor.settled(LONG);
  expect(await editor.phase()).toBe("failed");
  await expect(editor.status()).toContainText("生成失败");
  await editor.drag("丰满度", 0.6);
  await editor.settled(LONG);
  expect(await editor.phase()).toBe("idle");
  await expect(editor.status()).toContainText("已生成");
});
