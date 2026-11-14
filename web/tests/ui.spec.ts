// What the editor does, against a stand-in worker that answers with one real file a moment later.

import { chmodSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { join } from "node:path";
import { cacheKey } from "../serve.mjs";
import { BODY, expect, STANDIN, test } from "./fixtures";

/** A request as any client could send it, Host and Origin included, which fetch won't set. */
function raw(
  address: string,
  path: string,
  { method = "GET", headers = {}, body = "", setHost = true }: { method?: string; headers?: Record<string, string>; body?: string; setHost?: boolean } = {},
) {
  const { hostname, port } = new URL(address);
  return new Promise<{ status: number; text: string }>((done, fail) => {
    const asked = request({ host: hostname, port, path, method, headers, setHost }, (response) => {
      let text = "";
      response.setEncoding("utf8");
      response.on("data", (piece) => (text += piece));
      response.on("end", () => done({ status: response.statusCode ?? 0, text }));
    });
    asked.on("error", fail);
    asked.end(body);
  });
}

test.describe("asking for bodies", () => {
  test("the first body arrives and says so", async ({ editor, page }) => {
    await editor.open();
    await expect(editor.status()).toHaveText("已生成");
    await expect(page.locator("#start")).toContainText("修长");
    expect(await page.evaluate(() => window.editor.state.body.parts.length)).toBeGreaterThan(0);
  });

  test("two sliders in a row make one body, and the old one stays until it comes", async ({ editor, page }) => {
    await editor.open();
    const before = editor.posts.length;
    await page.evaluate(() => {
      window.__first = window.editor.state.body;
    });
    await editor.group("整体").click();
    await editor.grab("丰满度", 0.85);
    await expect(editor.status()).toContainText("调整中");
    expect(editor.posts.length, "a drag asked for a body").toBe(before);
    await page.mouse.up();
    await page.waitForTimeout(100);
    await editor.drag("曲线感", 0.25);
    expect(editor.posts.length, "the second drag began after a request").toBe(before);
    await expect(editor.status()).toContainText("生成中");
    expect(await page.evaluate(() => window.editor.state.body === window.__first)).toBe(true);
    await page.waitForFunction(() => window.editor.state.body !== window.__first);
    await editor.settled();
    expect(editor.posts.length - before).toBe(1);
    await expect(editor.status()).toHaveText("已生成");
  });

  test.describe("with a body that takes a while", () => {
    // Long enough for the next drag to come while it is being made, on a busy machine too: a body the
    // worker has finished has nothing left to cancel, while the page is still loading its file.
    test.use({ command: [process.execPath, STANDIN, BODY, "4000"] });

    test("a change while a body is being made cancels it, and only the newest lands", async ({ editor, page, service }) => {
      await editor.open();
      await editor.group("整体").click();
      await editor.drag("丰满度", 0.8);
      await page.waitForFunction(() => window.editor.rebuilder.running !== null);
      await expect.poll(() => service.server.worker.job, { message: "the worker was never asked" }).not.toBeNull();
      const worker = service.server.worker.child.pid;
      const landed = await page.evaluate(() => window.editor.log.timing.length);
      await editor.drag("曲线感", 0.6);
      await expect.poll(() => page.evaluate(() => window.editor.rebuilder.cancelled)).toBe(1);
      await expect.poll(() => service.server.worker.child?.pid, { message: "the older body kept running" }).not.toBe(worker);
      await editor.settled();
      expect((await page.evaluate(() => window.editor.log.timing.length)) - landed).toBe(1);
    });
  });

  test("parameters seen before come from the cache at once", async ({ editor, page }) => {
    await editor.open();
    const last = () => page.evaluate(() => window.editor.log.timing.at(-1));
    await editor.start("Q版");
    await editor.settled();
    await editor.start("修长");
    await editor.settled();
    expect((await last()).cached).toBe(true);
    await editor.group("整体").click();
    await editor.drag("丰满度", 0.85);
    await editor.settled();
    await editor.drag("丰满度", 0.4);
    await editor.settled();
    await editor.drag("丰满度", 0.85);
    await editor.settled();
    expect((await last()).cached, "a slider's way back missed the cache").toBe(true);
  });

  test("a contact pose is solved by the server for the body on show", async ({ editor, page }) => {
    const asked: Record<string, unknown>[] = [];
    page.on("request", (request) => {
      if (request.url().endsWith("/api/touch")) asked.push(request.postDataJSON());
    });
    await editor.open();
    await page.getByRole("combobox", { name: "姿势" }).click();
    await page.getByRole("option", { name: "摸头顶", exact: true }).click();
    await expect(page.getByRole("combobox", { name: "姿势" })).toHaveText(/摸头顶/);
    await expect.poll(() => asked.length).toBe(1);
    await expect.poll(() => page.evaluate(() => window.editor.state.touchJob)).toBeNull();
    // The next body asks for its own; the same one again doesn't.
    await page.getByRole("combobox", { name: "姿势" }).click();
    await page.getByRole("option", { name: "手贴脸侧", exact: true }).click();
    await expect.poll(() => page.evaluate(() => window.editor.state.pose)).toBe("touchCheek.left");
    expect(asked.length).toBe(1);
    await editor.group("整体").click();
    await editor.drag("丰满度", 0.3);
    await editor.settled();
    await expect.poll(() => asked.length).toBe(2);
    expect(page.url()).toContain("pose=touchCheek.left");
  });

  test("a body's hand that doesn't reach says so", async ({ editor, page }) => {
    await editor.open();
    await page.evaluate(() => {
      const fetched = window.fetch;
      window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).endsWith("/api/touch") && init?.body) {
          const asked = JSON.parse(String(init.body));
          return fetched(input, { ...init, body: JSON.stringify({ ...asked, body: { ...asked.body, hipDepth: 190 } }) });
        }
        return fetched(input, init);
      };
    });
    await page.getByRole("combobox", { name: "姿势" }).click();
    await page.getByRole("option", { name: "摸头顶", exact: true }).click();
    await expect(page.getByText("这个身体的手够不到这里")).toBeVisible();
    await page.getByRole("combobox", { name: "姿势" }).click();
    await page.getByRole("option", { name: "手臂上举" }).click();
    await expect(page.getByText("这个身体的手够不到这里")).toBeHidden();
  });

  test("pose and look stay with the next body", async ({ editor, page }) => {
    await editor.open();
    await page.getByRole("combobox", { name: "姿势" }).click();
    await page.getByRole("option", { name: "手臂上举" }).click();
    await editor.display();
    await page.getByRole("radio", { name: "灰模" }).click();
    await editor.group("整体").click();
    await editor.drag("丰满度", 0.3);
    await editor.settled();
    expect(await page.evaluate(() => window.editor.state.pose)).toBe("armsUp");
    await expect(page.getByRole("combobox", { name: "姿势" })).toHaveText(/手臂上举/);
    await expect(page.getByRole("radio", { name: "灰模" })).toBeChecked();
    expect(page.url()).toContain("pose=armsUp");
  });
});

test.describe("what can go wrong", () => {
  test("parameters the rules refuse ask for nothing and say why, in Chinese", async ({ editor, page }) => {
    await editor.open();
    const before = editor.posts.length;
    await page.evaluate(() => {
      window.__first = window.editor.state.body;
    });
    await editor.group("肩胸").click();
    await editor.type("biacromialBreadth", "160");
    await expect(editor.status()).toContainText("参数无效，没有生成");
    await page.waitForTimeout(800);
    expect(editor.posts.length).toBe(before);
    expect(await page.evaluate(() => window.editor.state.body === window.__first)).toBe(true);
    await expect(editor.problems().first()).toContainText(/[颈手肩躯]/);
    await expect(page.locator("#exportGlb")).toBeDisabled();
    await editor.problems().getByRole("button", { name: "去调整" }).first().click();
    expect(await page.evaluate(() => window.editor.state.panel)).toBe("shoulder");
    await page.locator("#reset").click();
    await editor.settled();
    await expect(editor.status()).toHaveText("已生成");
  });

  test("a body that breaks an invariant still shows, says which, and can be saved", async ({ editor, page }) => {
    await editor.open();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "186");
    await editor.settled();
    await expect(editor.status()).toHaveText("已生成，但有检查未通过");
    await expect(editor.problems().first()).toContainText("髋部软组织越过了中线");
    await expect(page.locator("#exportGlb")).toBeEnabled();
  });

  test("the worker dying in the middle is an error, and the next move recovers", async ({ editor, page, service }) => {
    await editor.open();
    await editor.group("整体").click();
    await editor.drag("肌肉感", 0.8);
    await page.waitForFunction(() => window.editor.rebuilder.running !== null);
    process.kill(-service.server.worker.child.pid, "SIGKILL");
    await editor.settled();
    expect(await editor.phase()).toBe("failed");
    await expect(editor.status()).toContainText("生成失败");
    await expect(editor.problems().first()).toContainText("worker is not running");
    await expect(page.locator("#hint")).toContainText("再试一次");
    await editor.drag("肌肉感", 0.6);
    await editor.settled();
    expect(await editor.phase()).toBe("idle");
  });

  test("a failure the worker reports is shown as written", async ({ editor, page }) => {
    await editor.open();
    await page.route("**/api/body", (route) => route.fulfill({ json: { ok: false, error: "ValueError: made up for this check" } }));
    await editor.group("整体").click();
    await editor.drag("肌肉感", 0.8);
    await editor.settled();
    expect(await editor.phase()).toBe("failed");
    await expect(editor.problems().first()).toContainText("made up for this check");
  });

  test.describe("without Python", () => {
    test.use({ command: ["/no/such/program"] });
    test("the page says what to run", async ({ editor, page }) => {
      await page.goto(`${editor.address}/?body=female-long`);
      await page.waitForFunction(() => window.editor?.ready === true);
      await editor.settled();
      expect(await editor.phase()).toBe("failed");
      await expect(editor.status()).toHaveText("连不上生成服务");
      await expect(page.locator("#hint")).toContainText("pnpm run editor");
    });
  });

  test("opened without the service, the page says what to run", async ({ editor, page }) => {
    await page.route("**/api/**", (route) => route.fulfill({ status: 404, body: "not found" }));
    await page.goto(`${editor.address}/`);
    await expect(page.locator("#fatal")).toContainText("pnpm run editor");
  });

  test("an address that isn't a page offers the way back", async ({ editor, page }) => {
    await page.goto(`${editor.address}/no/such/page`);
    await expect(page.getByRole("link", { name: "回到编辑器" })).toBeVisible();
  });
});

test.describe("the server", () => {
  test("a path that can't be decoded is refused, and the server stays up", async ({ service }) => {
    expect((await raw(service.address, "/%")).status).toBe(400);
    expect((await raw(service.address, "/build/%E0%A4%A")).status).toBe(400);
    expect((await raw(service.address, "/api/contact")).status).toBe(200);
  });

  // As a body the cache let go between the check and the read: there, and then it can't be opened.
  test("a file that can't be read is a 404, and the server stays up", async ({ service }) => {
    const glb = join(service.server.worker.folder, "0123abcd.glb");
    writeFileSync(glb, "glb");
    chmodSync(glb, 0);
    expect((await raw(service.address, "/api/body/0123abcd.glb")).status).toBe(404);
    chmodSync(glb, 0o644);
    expect(await raw(service.address, "/api/body/0123abcd.glb")).toEqual({ status: 200, text: "glb" });
  });

  // Any page can POST text/plain here without a preflight, and a site whose name was pointed at this
  // address (DNS rebinding) would read build/ as its own.
  test("another site can neither ask for a body nor read one", async ({ service }) => {
    const { port } = new URL(service.address);
    const asked = JSON.stringify({ sex: "female", body: {}, contact: true });
    const cross = { method: "POST", headers: { origin: "https://elsewhere.example", "content-type": "text/plain" }, body: asked };
    expect((await raw(service.address, "/api/body", cross)).status).toBe(403);
    const otherPort = { method: "POST", headers: { origin: "http://127.0.0.1:1" }, body: asked };
    expect((await raw(service.address, "/api/body", otherPort)).status).toBe(403);
    expect(service.server.contactWorker.child, "a refused POST started the contact worker").toBeNull();
    for (const host of ["elsewhere.example", `elsewhere.example:${port}`, "127.0.0.1.elsewhere.example"]) {
      expect((await raw(service.address, "/build/bodies/female-long/body.glb", { headers: { host } })).status, host).toBe(403);
      expect((await raw(service.address, "/api/presets", { headers: { host } })).status, host).toBe(403);
    }
    // Without a Host at all, Node's own parser refuses it before the server is asked.
    expect((await raw(service.address, "/api/presets", { setHost: false })).status, "no Host").toBe(400);
    // This machine's own names, with or without the port, and a script that sends no Origin.
    for (const host of [`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`, "localhost"]) {
      expect((await raw(service.address, "/build/bodies/female-long/body.glb", { headers: { host } })).status, host).toBe(200);
    }
    const plain = await raw(service.address, "/api/body", { method: "POST", body: JSON.stringify({ sex: "female", body: { a: 1 } }) });
    expect(JSON.parse(plain.text).ok, plain.text).toBe(true);
    const own = { origin: `http://localhost:${port}`, host: `localhost:${port}` };
    const fromPage = await raw(service.address, "/api/body", { method: "POST", headers: own, body: JSON.stringify({ sex: "female", body: { a: 2 } }) });
    expect(JSON.parse(fromPage.text).ok, fromPage.text).toBe(true);
  });
});

test.describe("files", () => {
  test("a JSON file goes out and comes back the same, unknown fields included", async ({ editor, page }) => {
    await editor.open();
    await editor.group("整体").click();
    await editor.drag("丰满度", 0.7);
    await editor.settled();
    const key = await page.evaluate(() => window.editor.state.report.key);
    const [saved] = await Promise.all([page.waitForEvent("download"), editor.menu("#fileMenu", "#exportJson")]);
    expect(saved.suggestedFilename()).toBe("female-long.json");
    const written = JSON.parse(readFileSync(await saved.path(), "utf8"));
    await editor.start("Q版");
    await editor.settled();
    await editor.importText(JSON.stringify({ ...written, notes: "from a later version" }));
    // The server reads the file, so the body it holds comes after its answer, which says what it didn't know.
    await expect(editor.problems().first()).toContainText("notes");
    await editor.settled();
    expect(await page.evaluate(() => window.editor.state.report.key), "the file made another body").toBe(key);
    expect(await page.evaluate(() => window.editor.log.timing.at(-1).cached)).toBe(true);
    const [again] = await Promise.all([page.waitForEvent("download"), editor.menu("#fileMenu", "#exportJson")]);
    const reread = JSON.parse(readFileSync(await again.path(), "utf8"));
    expect(reread.body).toEqual(written.body);
    expect(reread.notes).toBe("from a later version");
    await expect(page.locator("#start")).toContainText("导入的文件");
  });

  test("a file that isn't one says so, and one with a bad value opens and says what's wrong", async ({ editor, page }) => {
    await editor.open();
    await editor.importText(JSON.stringify({ format: "glTF" }));
    await expect(editor.status()).toContainText("导入失败：不是参数文件");
    const text = await page.evaluate(() => window.editor.currentFile());
    const { body, ...rest } = JSON.parse(text);
    const { crotchHeight: _crotch, ...withoutLeg } = body;
    const before = editor.posts.length;
    await editor.importText(JSON.stringify({ ...rest, body: { ...withoutLeg, stature: 99 } }));
    await expect(editor.status()).toContainText("参数无效");
    await expect(editor.problems().filter({ hasText: "缺少参数：裆高（crotchHeight）" })).toHaveCount(1);
    await expect(editor.problems().filter({ hasText: "有不认识的参数：stature" })).toHaveCount(1);
    expect(editor.posts.length).toBe(before);
    await expect(page.locator("#exportGlb")).toBeDisabled();
    await page.locator("#fileMenu").click();
    await expect(page.locator("#exportJson")).toBeEnabled();
    await page.keyboard.press("Escape");
  });

  test("a GLB saved while waiting is the body that was asked for", async ({ editor, page }) => {
    await editor.open();
    await editor.group("整体").click();
    await editor.drag("丰满度", 0.55);
    const [saved] = await Promise.all([page.waitForEvent("download"), page.locator("#exportGlb").click()]);
    expect(saved.suggestedFilename()).toBe("female-long.glb");
    expect(await page.evaluate(() => window.editor.rebuilder.phase)).toBe("idle");
    // The stand-in answers every body with the same file, so the file says nothing: its address does.
    // It is the cache key of the parameters on the page, as the server makes it, not one the page read.
    const asked = await page.evaluate(() => window.editor.state.current);
    const key = cacheKey("stand-in", asked.sex, asked.body);
    expect(editor.posts.at(-1)?.parameters).toEqual({ sex: asked.sex, body: asked.body });
    expect(saved.url()).toBe(`${editor.address}/api/body/${key}.glb`);
    expect(await page.evaluate(() => window.editor.state.report.key)).toBe(key);
  });

  test("a broken mesh stays on the screen, and only exporting it is refused", async ({ editor, page }) => {
    await editor.open();
    await editor.group("髋部").click();
    const before = await page.evaluate(() => window.editor.state.report.key);
    await editor.type("hipDepth", "187");
    await editor.settled();
    // It is shown: this body's own file is on the screen, and the page says what is wrong with it.
    await expect(editor.status()).toHaveText("已生成，但网格损坏，不能导出");
    await expect(editor.problems().first()).toContainText("翻转");
    const asked = await page.evaluate(() => window.editor.state.current);
    const key = cacheKey("stand-in", asked.sex, asked.body);
    expect(key).not.toBe(before);
    expect(await page.evaluate(() => window.editor.state.report.key)).toBe(key);
    expect(await page.evaluate(() => window.editor.state.body !== null)).toBe(true);
    let saved = false;
    page.on("download", () => {
      saved = true;
    });
    await page.locator("#exportGlb").click();
    const note = page.locator("#glbNote");
    await expect(note).toContainText("拒绝导出 GLB，网格损坏");
    await expect(note).toContainText("翻转");
    await expect(note).toContainText("皮肤不闭合：2 条边");
    expect(saved).toBe(false);
    expect(await page.evaluate(() => window.editor.exportGlb())).toBeNull();
    expect(saved).toBe(false);
    // Not a state the export gets stuck in: a click is a question, and the next body has its own answer.
    await expect(page.locator("#exportGlb")).toBeEnabled();
    await page.getByRole("button", { name: "关闭 GLB 导出的消息" }).click();
    await expect(note).toHaveCount(0);
    await page.locator("#exportGlb").click();
    await expect(note).toContainText("拒绝导出 GLB");
    await editor.type("hipDepth", "186");
    await editor.settled();
    const [file] = await Promise.all([page.waitForEvent("download"), page.locator("#exportGlb").click()]);
    expect(file.suggestedFilename()).toBe("female-long.glb");
    await expect(note).toHaveCount(0);
  });

  test("a refused export says where it broke and looks for a parameter that ends it", async ({ editor, page, service }) => {
    await editor.open();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "187");
    await editor.settled();
    // Where it broke costs nothing, so the page says it as soon as the body is there, and goes there.
    const shown = editor.problems().first();
    await expect(shown).toContainText("髋部 2");
    await expect(shown).toContainText("腿部 1");
    await editor.group("腿部").click();
    await shown.getByRole("button", { name: "去调整" }).click();
    expect(await page.evaluate(() => window.editor.state.panel)).toBe("hip");
    // Nothing is searched for until an export is refused.
    expect(service.server.mendWorker.job).toBeNull();
    let searches = 0;
    page.on("request", (sent) => {
      if (sent.url().endsWith("/api/mend") && sent.method() === "POST") searches += 1;
    });
    await page.locator("#exportGlb").click();
    await expect(page.locator("#glbNote")).toContainText("髋部 2");
    const note = page.locator("#mendNote");
    await expect(note).toContainText("膝围");
    await expect(note).toContainText("200 mm");
    await expect(note).toContainText("240.5 mm");
    await note.locator("..").getByRole("button", { name: "去调整" }).click();
    expect(await page.evaluate(() => window.editor.state.panel)).toBe("leg");
    // Once for the same parameters, however many exports they refuse.
    expect(searches).toBe(1);
    await page.locator("#exportGlb").click();
    await editor.menu("#exportMenu", "#exportBlend");
    await expect(page.locator("#blendNote")).toContainText("拒绝导出 .blend");
    expect(searches).toBe(1);
    await page.getByRole("button", { name: "关闭找参数的消息" }).click();
    await expect(note).toHaveCount(0);
  });

  test("the IPC and .blend refusals look for it too, and say so when no one parameter ends it", async ({ editor, page }) => {
    await editor.open();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "187");
    await editor.settled();
    await editor.menu("#exportMenu", "#exportContact");
    await expect(page.locator("#contactNote")).toContainText("拒绝 IPC 导出");
    await expect(page.locator("#mendNote")).toContainText("膝围");
    await editor.type("hipDepth", "190");
    await editor.settled();
    await editor.menu("#exportMenu", "#exportBlend");
    await expect(page.locator("#blendNote")).toContainText("拒绝导出 .blend");
    await expect(page.locator("#mendNote")).toContainText("试了 35 项参数");
    await expect(page.locator("#mendNote").locator("..").getByRole("button", { name: "去调整" })).toHaveCount(0);
  });

  test("the search can be stopped, and moving on from the body stops it", async ({ editor, page, service }) => {
    await editor.open();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "191");
    await editor.settled();
    await page.locator("#exportGlb").click();
    const note = page.locator("#mendNote");
    await expect(note).toContainText("第 1/35 项");
    expect(service.server.mendWorker.job).not.toBeNull();
    await page.getByRole("button", { name: "关闭找参数的消息" }).click();
    await expect(note).toHaveCount(0);
    await expect.poll(() => service.server.mendWorker.job).toBeNull();
    // Asked again, then the slider moves on.
    await page.locator("#exportGlb").click();
    await expect(note).toContainText("第 1/35 项");
    await editor.type("hipDepth", "186");
    await expect(note).toHaveCount(0);
    await expect.poll(() => service.server.mendWorker.job).toBeNull();
  });

  test("a broken mesh is shown from the cache too, and still refused", async ({ editor, page }) => {
    await editor.open();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "187");
    await editor.settled();
    await editor.type("hipDepth", "186");
    await editor.settled();
    await editor.type("hipDepth", "187");
    await editor.settled();
    expect(await page.evaluate(() => window.editor.log.timing.at(-1).cached)).toBe(true);
    await expect(editor.status()).toHaveText("已生成，但网格损坏，不能导出");
    await page.locator("#exportGlb").click();
    await expect(page.locator("#glbNote")).toContainText("拒绝导出 GLB，网格损坏");
  });
});

test.describe("exporting with contact tables", () => {
  test("an IPC export runs beside the sliders and saves the body asked for under its own name", async ({ editor, page }) => {
    await editor.open();
    const asked = await page.evaluate(() => window.editor.state.current.body);
    const download = page.waitForEvent("download");
    await editor.menu("#exportMenu", "#exportContact");
    await expect(page.locator("#contactNote")).toContainText("正在烘肩腋");
    await page.locator("#exportMenu").click();
    await expect(page.locator("#exportContact")).toBeDisabled();
    await page.keyboard.press("Escape");
    // The sliders stay free, and what they do neither cancels the export nor changes its body.
    await editor.group("整体").click();
    await editor.drag("丰满度", 0.55);
    const saved = await download;
    expect(saved.suggestedFilename()).toBe("female-long-ipc.glb");
    await expect(page.locator("#contactNote")).toHaveText("IPC 导出完成");
    await page.locator("#exportMenu").click();
    await expect(page.locator("#exportContact")).toBeEnabled();
    await page.keyboard.press("Escape");
    const contactPosts = editor.posts.filter((post) => post.parameters.contact);
    expect(contactPosts).toHaveLength(1);
    expect(contactPosts[0].parameters.body).toEqual(asked);
    await editor.settled();
    expect(await page.evaluate(() => window.editor.state.current.body.upperThighGirth)).not.toBe(asked.upperThighGirth);
  });

  test("a broken mesh is refused before any table is baked, and says why", async ({ editor, page, service }) => {
    await editor.open();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "187");
    await editor.settled();
    let saved = false;
    page.on("download", () => {
      saved = true;
    });
    await editor.menu("#exportMenu", "#exportContact");
    const note = page.locator("#contactNote");
    await expect(note).toContainText("拒绝 IPC 导出，网格损坏");
    await expect(note).toContainText("皮肤不闭合：2 条边");
    await expect(note).not.toContainText("正在烘");
    expect(saved).toBe(false);
    expect(service.server.contactWorker.job).toBeNull();
    // Nothing is left behind for the export's cache to serve.
    expect(readdirSync(service.server.contactWorker.folder).filter((name) => name.endsWith(".part.glb"))).toEqual([]);
    await page.locator("#exportMenu").click();
    await expect(page.locator("#exportContact")).toBeEnabled();
    await page.keyboard.press("Escape");
    // The body on the screen is left as it was.
    await expect(editor.status()).toHaveText("已生成，但网格损坏，不能导出");
  });
});

test.describe("exporting a .blend", () => {
  const NEEDS_BLENDER = "需要 Blender。请安装它，或设置 BLENDER=/path/to/blender，然后重启编辑器。";

  test("it runs beside the sliders, says so, can't be started twice, and saves under the GLB's name", async ({ editor, page, service }) => {
    await editor.open();
    const asked = await page.evaluate(() => window.editor.state.current);
    const download = page.waitForEvent("download");
    const posts: string[] = [];
    page.on("request", (sent) => {
      if (sent.url().endsWith("/api/blend") && sent.method() === "POST") posts.push(sent.postData() ?? "");
    });
    await editor.menu("#exportMenu", "#exportBlend");
    await expect(page.locator("#blendNote")).toContainText("正在导出 .blend");
    await page.locator("#exportMenu").click();
    await expect(page.locator("#exportBlend")).toBeDisabled();
    await expect(page.locator("#exportBlend")).toContainText(".blend 导出进行中");
    await page.keyboard.press("Escape");
    // However it is asked for again, nothing more starts.
    expect(await page.evaluate(() => window.editor.exportBlend())).toBeNull();
    await editor.group("整体").click();
    await editor.drag("丰满度", 0.55);
    const saved = await download;
    expect(saved.suggestedFilename()).toBe("female-long.blend");
    // The body asked for at the click, under the key the server makes of it.
    const key = cacheKey("stand-in-blend", asked.sex, asked.body);
    expect(saved.url()).toBe(`${editor.address}/api/blend/${key}.blend`);
    expect(readFileSync(await saved.path()).equals(readFileSync(BODY))).toBe(true);
    await expect(page.locator("#blendNote")).toHaveText(".blend 导出完成");
    expect(posts).toHaveLength(1);
    expect(JSON.parse(posts[0])).toEqual({ sex: asked.sex, body: asked.body });
    await editor.settled();
    expect(await page.evaluate(() => window.editor.state.current.body.upperThighGirth)).not.toBe(asked.body.upperThighGirth);
    await page.getByRole("button", { name: "关闭 .blend 导出的消息" }).click();
    await expect(page.locator("#blendNote")).toHaveCount(0);
    expect(service.server.blendWorker.job).toBeNull();
  });

  test("the same body again comes from its cache, and the GLBs' cache isn't touched", async ({ editor, page, service }) => {
    await editor.open();
    const glbs = () => readdirSync(service.server.worker.folder).filter((name) => name.endsWith(".glb"));
    const answer = async () => {
      const reply = page.waitForResponse((seen) => seen.url().endsWith("/api/blend") && seen.request().method() === "POST");
      const [saved] = await Promise.all([page.waitForEvent("download"), editor.menu("#exportMenu", "#exportBlend")]);
      return { reply: await (await reply).json(), saved };
    };
    const before = glbs();
    const first = await answer();
    expect(first.reply.cached).toBe(false);
    const second = await answer();
    expect(second.reply.cached).toBe(true);
    expect(second.saved.url()).toBe(first.saved.url());
    expect(glbs()).toEqual(before);
    expect(readdirSync(service.server.blendWorker.folder).filter((name) => name.endsWith(".blend"))).toHaveLength(1);
  });

  test("a broken mesh is refused, and the page says which way it is broken", async ({ editor, page }) => {
    await editor.open();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "187");
    await editor.settled();
    let saved = false;
    page.on("download", () => {
      saved = true;
    });
    await editor.menu("#exportMenu", "#exportBlend");
    const note = page.locator("#blendNote");
    await expect(note).toContainText("拒绝导出 .blend，网格损坏");
    await expect(note).toContainText("翻转");
    await expect(note).toContainText("皮肤不闭合：2 条边");
    expect(saved).toBe(false);
    // Not a state the export gets stuck in.
    await expect(page.locator("#exportGlb")).toBeEnabled();
    await page.locator("#exportMenu").click();
    await expect(page.locator("#exportBlend")).toBeEnabled();
    await page.keyboard.press("Escape");
  });

  test("parameters the rules refuse can't be exported, as a GLB can't, and a refusal only the worker sees is said", async ({ editor, page }) => {
    await editor.open();
    await editor.group("肩胸").click();
    await editor.type("biacromialBreadth", "160");
    await expect(editor.status()).toContainText("参数无效，没有生成");
    await page.locator("#exportMenu").click();
    await expect(page.locator("#exportBlend")).toBeDisabled();
    await page.keyboard.press("Escape");
    expect(await page.evaluate(() => window.editor.exportBlend())).toBeNull();
    await expect(page.locator("#blendNote")).toHaveCount(0);
    await page.locator("#reset").click();
    await editor.settled();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "188");
    await editor.settled();
    await editor.menu("#exportMenu", "#exportBlend");
    await expect(page.locator("#blendNote")).toContainText("导出 .blend 失败");
    await expect(page.locator("#blendNote")).toContainText("髋");
    await editor.type("hipDepth", "189");
    await editor.settled();
    await editor.menu("#exportMenu", "#exportBlend");
    // A target past what the body can reach, as a GLB's generating says it.
    await expect(page.locator("#blendNote")).toContainText("ValueError: hipBack asked past");
  });

  test("a body whose checks only warn is exported anyway, as a GLB is, with the problem on the page", async ({ editor, page }) => {
    await editor.open();
    await editor.group("髋部").click();
    await editor.type("hipDepth", "186");
    await editor.settled();
    await expect(editor.problems().first()).toContainText("髋部软组织越过了中线");
    const [saved] = await Promise.all([page.waitForEvent("download"), editor.menu("#exportMenu", "#exportBlend")]);
    expect(saved.suggestedFilename()).toBe("female-long.blend");
    await expect(page.locator("#blendNote")).toHaveText(".blend 导出完成");
  });

  test("another page's .blend of another body replaces this one, which says so", async ({ editor, page, service }) => {
    await editor.open();
    await editor.menu("#exportMenu", "#exportBlend");
    await expect.poll(() => service.server.blendWorker.job !== null).toBe(true);
    const { sex, body } = await page.evaluate(() => window.editor.state.current);
    const other = await raw(service.address, "/api/blend", { method: "POST", body: JSON.stringify({ sex, body: { ...body, head: body.head + 1 } }) });
    expect(JSON.parse(other.text).ok).toBe(true);
    await expect(page.locator("#blendNote")).toHaveText("被另一个 .blend 导出取代了，比如另一个打开的页面。");
  });

  test("a worker that dies in the middle is said, and the next click starts another", async ({ editor, page, service }) => {
    await editor.open();
    await editor.menu("#exportMenu", "#exportBlend");
    await expect(page.locator("#blendNote")).toContainText("正在导出 .blend");
    await expect.poll(() => service.server.blendWorker.job !== null).toBe(true);
    process.kill(-service.server.blendWorker.child.pid, "SIGKILL");
    await expect(page.locator("#blendNote")).toContainText("导出 .blend 失败");
    const [saved] = await Promise.all([page.waitForEvent("download"), editor.menu("#exportMenu", "#exportBlend")]);
    expect(saved.suggestedFilename()).toBe("female-long.blend");
    await expect(page.locator("#blendNote")).toHaveText(".blend 导出完成");
  });

  test.describe("on a machine without Blender", () => {
    test.use({ command: [process.execPath, STANDIN, BODY, "250", "no-blender"] });

    test("the menu says so before any click, and starts nothing", async ({ editor, page, service }) => {
      await editor.open();
      await page.locator("#exportMenu").click();
      await expect(page.locator("#exportBlend")).toBeDisabled();
      await expect(page.locator("#blendWhy")).toHaveText(NEEDS_BLENDER);
      await page.keyboard.press("Escape");
      expect(await page.evaluate(() => window.editor.exportBlend())).toBeNull();
      expect(service.server.blendWorker.child, "a worker was started for it").toBeNull();
      // The other exports don't need it.
      await expect(page.locator("#exportGlb")).toBeEnabled();
    });

    test("found out only at the click, it is said, and the menu says so from then on", async ({ editor, page }) => {
      await page.route("**/api/blend", (route) => (route.request().method() === "GET" ? route.fulfill({ json: { blender: null } }) : route.continue()));
      await editor.open();
      await editor.menu("#exportMenu", "#exportBlend");
      await expect(page.locator("#blendNote")).toHaveText(NEEDS_BLENDER);
      await page.locator("#exportMenu").click();
      await expect(page.locator("#exportBlend")).toBeDisabled();
      await expect(page.locator("#blendWhy")).toBeVisible();
      await page.keyboard.press("Escape");
    });
  });

  test("another site can't start one, and a key that isn't there is a 404", async ({ service }) => {
    const asked = JSON.stringify({ sex: "female", body: {} });
    const cross = { method: "POST", headers: { origin: "https://elsewhere.example", "content-type": "text/plain" }, body: asked };
    expect((await raw(service.address, "/api/blend", cross)).status).toBe(403);
    expect(service.server.blendWorker.child, "a refused POST started the worker").toBeNull();
    expect((await raw(service.address, "/api/blend/0123abcd.blend")).status).toBe(404);
    expect(JSON.parse((await raw(service.address, "/api/blend")).text)).toEqual({ blender: true });
  });
});

test.describe("adjusting", () => {
  test("undo and redo cover sliders and presets, from the cache", async ({ editor, page }) => {
    await editor.open();
    await editor.group("胸部").click();
    const before = await editor.value("breastProjection").inputValue();
    await editor.type("breastProjection", "50");
    await editor.settled();
    await page.locator("#undo").click();
    await editor.settled();
    await expect(editor.value("breastProjection")).toHaveValue(before);
    await page.locator("#redo").click();
    await editor.settled();
    await expect(editor.value("breastProjection")).toHaveValue("50.00");
    expect(await page.evaluate(() => window.editor.log.timing.at(-1).cached)).toBe(true);
    await editor.start("匀称");
    await editor.settled();
    await page.keyboard.press("ControlOrMeta+z");
    await editor.settled();
    await expect(page.locator("#start")).toContainText("修长");
    await expect(editor.value("breastProjection")).toHaveValue("50.00");
  });

  test("switching sex keeps the group it has, and otherwise starts from its first body", async ({ editor, page }) => {
    await editor.open();
    await editor.start("男性");
    await editor.settled();
    await expect(page.locator("#start")).toContainText("男性 · 修长");
    await editor.start("女性");
    await editor.settled();
    await editor.start("匀称");
    await editor.settled();
    const groups = await page.evaluate(async () =>
      Object.keys((await (await fetch("/api/presets")).json()).presets)
        .filter((key) => key.startsWith("male-"))
        .map((key) => key.split("-")[1]),
    );
    await editor.start("男性");
    await editor.settled();
    const expected = groups.includes("balanced") ? "balanced" : groups[0];
    expect(await page.evaluate(() => window.editor.state.name)).toBe(`male-${expected}`);
  });

  test("the picker offers sexes, then that sex's groups, and a file picks its own sex", async ({ editor, page }) => {
    await editor.open();
    await page.locator("#start").click();
    await expect(page.getByRole("radio", { name: "女性", exact: true })).toBeChecked();
    await expect(page.getByRole("radio", { name: "修长", exact: true })).toBeChecked();
    await expect(page.getByRole("radio", { name: "男性", exact: true })).not.toBeChecked();
    await page.keyboard.press("Escape");
    await editor.start("男性");
    await editor.settled();
    const male = await page.evaluate(() => window.editor.currentFile());
    await editor.start("女性");
    await editor.settled();
    // A file from no typical body: its sex is still its own, and no group is chosen.
    const { base: _base, ...file } = JSON.parse(male);
    await editor.importText(JSON.stringify(file));
    await editor.settled();
    await expect(page.locator("#start")).toContainText("男性 · 导入的文件");
    await page.locator("#start").click();
    await expect(page.getByRole("radio", { name: "男性", exact: true })).toBeChecked();
    await expect(page.getByRole("radio", { name: "修长", exact: true })).not.toBeChecked();
    await expect(page.getByRole("radio", { name: "女性", exact: true })).not.toBeChecked();
    await page.getByRole("radio", { name: "修长", exact: true }).click();
    await page.keyboard.press("Escape");
    await editor.settled();
    expect(await page.evaluate(() => window.editor.state.name)).toBe("male-long");
  });

  test("a typed value is kept inside its range, and Escape leaves it alone", async ({ editor }) => {
    await editor.open();
    await editor.group("腿部").click();
    await editor.type("kneeGirth", "900");
    const top = await editor.value("kneeGirth").inputValue();
    // The range is in torso lengths, so how many millimetres it is depends on the body; the slider knows.
    const high = Number(await editor.thumb("膝围").getAttribute("aria-valuemax"));
    expect(Math.abs(Number(top) - high), "900 mm stayed as typed").toBeLessThanOrEqual(0.5);
    expect(high).toBeLessThan(900);
    await editor.value("kneeGirth").fill("50");
    await editor.value("kneeGirth").press("Escape");
    await expect(editor.value("kneeGirth")).toHaveValue(top);
  });

  test("holding compare shows the body from before, and letting go comes back", async ({ editor, page }) => {
    await editor.open();
    const compare = page.locator("#compare");
    await expect(compare).toBeDisabled();
    await editor.group("整体").click();
    await editor.drag("丰满度", 0.8);
    await editor.settled();
    await expect(compare).toBeEnabled();
    const box = (await compare.boundingBox()) as { x: number; y: number; width: number; height: number };
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await expect(compare).toHaveAttribute("aria-pressed", "true");
    await expect(compare).toContainText("调整前");
    await page.mouse.up();
    await expect(compare).toHaveAttribute("aria-pressed", "false");
    await page.keyboard.down("c");
    await expect(compare).toHaveAttribute("aria-pressed", "true");
    await page.keyboard.up("c");
    await expect(compare).toHaveAttribute("aria-pressed", "false");
  });

  test("what changed is counted, listed, and can be restored one by one", async ({ editor, page }) => {
    await editor.open();
    await editor.group("胸部").click();
    await editor.type("breastProjection", "50");
    await editor.settled();
    await expect(editor.group("胸部")).toContainText("1");
    await editor.group("已改动").click();
    await expect(page.locator("[data-changed]")).toHaveCount(1);
    await page.getByRole("button", { name: "恢复突出度" }).click();
    await expect(page.getByText("还没有改动")).toBeVisible();
  });

  test("a slider says what it means in millimetres, and a longer leg moves the stature and nothing else", async ({ editor, page }) => {
    await editor.open();
    await editor.group("腰腹").click();
    await page.getByRole("button", { name: "腰宽的说明" }).click();
    await expect(page.getByText("生成时校准到 ±2%")).toBeVisible();
    const waist = await editor.value("waistBreadth").inputValue();
    const derived = page.locator("#derived");
    await expect(derived).toHaveText(/^身高 1605 mm · 头身比 6\.66$/);
    await editor.group("比例").click();
    // female-long's crotch is at 749 mm: 51 mm more is 51 mm more stature, and the head stays 241 mm.
    await editor.type("crotchHeight", "800");
    await expect(derived).toHaveText(/^身高 1656 mm · 头身比 6\.88$/);
    await editor.group("腰腹").click();
    await expect(editor.value("waistBreadth")).toHaveValue(waist);
  });

  test("a figure estimated from a value says so", async ({ editor, page }) => {
    await editor.open("?body=female-balanced&panel=head");
    await expect(page.locator('[data-reading="headDepth"]')).toHaveText(/^头深约 \d+ mm$/);
  });

  test("the head count moves others, which say so and can't be restored alone", async ({ editor, page }) => {
    await editor.open();
    await editor.type("heads", "6");
    await expect(page.locator('[data-changed]:has(#slider-crotchHeight)')).toContainText("联动");
    await expect(page.getByRole("button", { name: "恢复裆高" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "恢复头身比" })).toBeVisible();
  });

  test("every group of sliders.json is on the rail", async ({ editor }) => {
    await editor.open();
    const { groups } = JSON.parse(readFileSync(new URL("../../sotaigen/data/sliders.json", import.meta.url), "utf8"));
    const words = JSON.parse(readFileSync(new URL("../messages/zh.json", import.meta.url), "utf8"));
    for (const { id } of groups) await expect(editor.group(words[`group_${id}`]), `${id} can't be reached`).toHaveCount(1);
  });

  test("a sex without a group's sliders has no such group, on the rail, from a click, or in the counts", async ({ editor, page }) => {
    const changes = page.getByRole("navigation").getByRole("button", { name: /^已改动/u });
    const counted = async () => Number((await changes.textContent())?.replace("已改动", "") || 0);
    // A male body has no breast: the address asks for 胸部 and gets the pectorals of 肩胸.
    await editor.open("?body=male-long&panel=chest");
    await expect(editor.group("胸部")).toHaveCount(0);
    await expect(editor.group("肩胸")).toHaveAttribute("aria-current", "true");
    await expect(page.locator("#slider-pectoralThickness")).toBeVisible();
    // The front of the chest, where a female body's click means 胸部.
    const chest = async () =>
      page.evaluate(() => {
        const { viewport } = window.editor;
        const position = viewport.body.mesh.skeleton.getBoneByName("upperChest").getWorldPosition(viewport.controls.target.clone());
        position.project(viewport.camera);
        const rect = viewport.renderer.domElement.getBoundingClientRect();
        return { x: rect.left + ((position.x + 1) / 2) * rect.width, y: rect.top + ((1 - position.y) / 2) * rect.height };
      });
    await editor.group("比例").click();
    await page.waitForFunction(() => window.editor.viewport.flight === null);
    const point = await chest();
    await page.mouse.click(point.x, point.y);
    await expect(editor.group("肩胸")).toHaveAttribute("aria-current", "true");
    // The breast's defaults follow the torso, which the proportions move; a male body still has none.
    await editor.group("比例").click();
    const torso = Number(await editor.value("torsoLength").inputValue());
    await editor.type("torsoLength", String(torso + 20));
    await editor.type("heads", "6");
    await editor.settled();
    await editor.group("已改动").click();
    const rows = await page.locator("[data-changed]").count();
    expect(rows).toBeGreaterThan(0);
    expect(await counted(), "the count isn't what the list shows").toBe(rows);
    await expect(page.locator('[id^="slider-breast"]')).toHaveCount(0);
    // The same click on a female body is 胸部, and the group is back.
    await editor.start("女性");
    await editor.settled();
    await expect(editor.group("胸部")).toHaveCount(1);
    await editor.group("比例").click();
    await page.waitForFunction(() => window.editor.viewport.flight === null);
    await page.mouse.click(point.x, point.y);
    await expect(editor.group("胸部")).toHaveAttribute("aria-current", "true");
  });

  test("search finds a slider in any group", async ({ editor, page }) => {
    await editor.open();
    await page.getByRole("searchbox", { name: "搜索参数" }).fill("腕");
    await expect(page.locator('[id^="slider-"]')).toHaveCount(1);
    await expect(page.locator("#slider-wristGirth")).toBeVisible();
    await page.getByRole("searchbox", { name: "搜索参数" }).fill("没有这项");
    await expect(page.getByText("没有找到这个参数")).toBeVisible();
  });

  test("advanced sliders are folded away until changed, and the ankle stays where it is set", async ({ editor, page }) => {
    await editor.open("?body=female-long&panel=leg");
    await expect(page.locator("#slider-ankleHeight")).toHaveCount(0);
    await page.locator("#advanced").click();
    const start = await editor.value("ankleHeight").inputValue();
    await editor.type("footLength", "140");
    await editor.settled();
    await expect(editor.value("ankleHeight")).toHaveValue(start);
    await editor.type("ankleHeight", "60");
    await editor.settled();
    expect(await page.evaluate(() => window.editor.state.current.body.ankleHeight)).toBe(60);
    await page.reload();
    await page.waitForFunction(() => window.editor?.ready === true);
    await expect(page.locator("#slider-ankleHeight")).toHaveCount(0);
  });

  test("the track marks what the rules refuse, says why, and follows the other sliders", async ({ editor, page }) => {
    await editor.open("?body=female-balanced&panel=leg");
    await page.locator("#advanced").click();
    const zone = page.locator('#slider-ankleHeight [data-zone="blocked"]').first();
    await expect(zone).toHaveAttribute("title", /脚踝相对脚长太低/);
    const before = (await zone.boundingBox())?.width as number;
    // A longer foot raises the lowest ankle the rules allow. female-balanced's foot is 214 mm.
    await editor.type("footLength", "250");
    await editor.settled();
    await expect.poll(async () => (await zone.boundingBox())?.width).toBeGreaterThan(before);
    await editor.drag("脚踝高度", 0.05);
    await expect(editor.status()).toContainText("参数无效");
  });
});

test.describe("looking", () => {
  test("the address follows what's chosen, and a reload comes back to it", async ({ editor, page }) => {
    await editor.open();
    await editor.group("臀部").click();
    await editor.start("匀称");
    await editor.settled();
    await editor.display();
    await page.getByRole("radio", { name: "灰模" }).click();
    await expect.poll(() => page.url()).toContain("inspect=true");
    expect(page.url()).toContain("panel=glute");
    expect(page.url()).toContain("body=female-balanced");
    expect(page.url()).toContain("look=standard");
    await page.reload();
    await page.waitForFunction(() => window.editor?.ready === true);
    await editor.settled();
    await expect(editor.group("臀部")).toHaveAttribute("aria-current", "true");
    await expect(page.getByRole("radio", { name: "灰模" })).toBeChecked();
    await expect(page.locator("#inspector")).toBeVisible();
  });

  test("each look puts its own materials on, and the next one takes them off", async ({ editor, page }) => {
    await editor.open();
    await editor.display();
    const seen = () =>
      page.evaluate(() => {
        const viewport = window.editor.viewport as any;
        const skeleton = viewport.world.children.some((child: any) => child.children?.some((line: any) => line.isLineSegments2));
        return { material: viewport.body.mesh.material.type as string, skeleton };
      });
    for (const [label, look, material, skeleton] of [
      ["剪影", "silhouette", "MeshBasicMaterial", false],
      ["雕塑", "matcap", "MeshMatcapMaterial", false],
      ["X 光", "xray", "MeshMatcapMaterial", true],
      ["灰模", "standard", "MeshStandardMaterial", false],
      ["赛璐璐", "toon", "MeshToonMaterial", false],
    ] as const) {
      await page.getByRole("radio", { name: label }).click();
      expect(await seen(), look).toEqual({ material, skeleton });
      if (look === "toon") await expect.poll(() => page.url()).not.toContain("look=");
      else await expect.poll(() => page.url()).toContain(`look=${look}`);
    }
    await page.goto(`${editor.address}/?body=female-long&look=xray&inspect=true`);
    await page.waitForFunction(() => window.editor?.ready === true);
    await editor.settled();
    await expect(page.getByRole("radio", { name: "X 光" })).toBeChecked();
    expect(await seen()).toEqual({ material: "MeshMatcapMaterial", skeleton: true });
  });

  test("choosing a group takes the camera there, unless told not to", async ({ editor, page }) => {
    await editor.open();
    const start = await editor.camera();
    await editor.group("臀部").click();
    await page.waitForFunction(() => window.editor.viewport.flight === null);
    const moved = await editor.camera();
    expect(moved[2], "the seat is seen from behind").toBeLessThan(0);
    expect(moved).not.toEqual(start);
    await page.locator("#follow").click();
    await page.waitForFunction(() => window.editor.viewport.flight === null);
    const held = await editor.camera();
    await editor.group("手臂").click();
    await page.waitForTimeout(450);
    expect(await editor.camera()).toEqual(held);
  });

  test("a click on the body chooses its group, a drag only turns it", async ({ editor, page }) => {
    await editor.open();
    const point = await page.evaluate(() => {
      const { viewport } = window.editor;
      const position = viewport.body.mesh.skeleton.getBoneByName("head").getWorldPosition(viewport.controls.target.clone());
      position.y += 0.08;
      position.project(viewport.camera);
      const rect = viewport.renderer.domElement.getBoundingClientRect();
      return { x: rect.left + ((position.x + 1) / 2) * rect.width, y: rect.top + ((1 - position.y) / 2) * rect.height };
    });
    const before = await editor.camera();
    await page.mouse.move(point.x, point.y);
    await page.mouse.down();
    await page.mouse.move(point.x + 60, point.y, { steps: 4 });
    await page.mouse.up();
    await expect(editor.group("比例")).toHaveAttribute("aria-current", "true");
    expect(await editor.camera()).not.toEqual(before);
    await page.getByRole("button", { name: "正面" }).click();
    await page.waitForFunction(() => window.editor.viewport.flight === null);
    const held = await editor.camera();
    const again = await page.evaluate(() => {
      const { viewport } = window.editor;
      const position = viewport.body.mesh.skeleton.getBoneByName("head").getWorldPosition(viewport.controls.target.clone());
      position.y += 0.08;
      position.project(viewport.camera);
      const rect = viewport.renderer.domElement.getBoundingClientRect();
      return { x: rect.left + ((position.x + 1) / 2) * rect.width, y: rect.top + ((1 - position.y) / 2) * rect.height };
    });
    await page.mouse.click(again.x, again.y);
    await expect(editor.group("头部")).toHaveAttribute("aria-current", "true");
    expect(await editor.camera()).toEqual(held);
  });

  test("the dark theme keeps a middle grey behind the body, and is remembered", async ({ editor, page }) => {
    await editor.open();
    await page.getByRole("button", { name: "换成深色" }).click();
    await expect(page.locator("html")).toHaveClass(/dark/);
    expect(await page.evaluate(() => window.editor.viewport.world.background.getHex())).toBe(0x85868b);
    await page.reload();
    await page.waitForFunction(() => window.editor?.ready === true);
    await expect(page.locator("html")).toHaveClass(/dark/);
  });

  test("a phone-sized window has no sideways scroll, and reaches every group", async ({ editor, page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await editor.open();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await editor.group("腿部").click();
    await expect(page.locator("#slider-upperThighGirth")).toBeVisible();
    await expect(page.locator("#inspect")).toBeVisible();
  });
});

test.describe("inspecting", () => {
  test("the panel opens from the toolbar and by key, and asks for no body", async ({ editor, page }) => {
    await editor.open();
    await expect(page.locator("#inspector")).toHaveCount(0);
    const before = editor.posts.length;
    await page.locator("#inspect").click();
    await expect(page.locator("#stats")).toContainText("顶点");
    await page.keyboard.press("i");
    await expect(page.locator("#inspector")).toHaveCount(0);
    await page.keyboard.press("i");
    await expect(page.locator("#inspector")).toBeVisible();
    expect(editor.posts.length).toBe(before);
  });

  test("each part switches off alone, each expression reaches its meshes, and all of it goes back", async ({ editor, page }) => {
    await editor.open("?body=female-long&panel=head&inspect=true");
    const start = await editor.picture();
    const posts = editor.posts.length;
    const parts: string[] = await page.evaluate(() => window.editor.state.partNames);
    expect(parts.length).toBeGreaterThan(0);
    for (const name of parts) {
      await page.locator(`#part-${name}`).click();
      const state = await page.evaluate(() => Object.fromEntries(window.editor.state.body.parts.map((part: any) => [part.userData.name, part.visible])));
      expect(state[name], `${name} stayed on`).toBe(false);
      expect(Object.entries(state).filter(([other, on]) => other !== name && !on), `${name} took others with it`).toEqual([]);
      expect(await editor.picture(), `${name} changed nothing on screen`).not.toBe(start);
      await page.locator(`#part-${name}`).click();
    }
    const morphs: string[] = await page.evaluate(() => window.editor.state.morphNames);
    expect(morphs.length).toBeGreaterThan(0);
    for (const name of morphs) {
      await page.locator(`#morph-${name} [role="slider"]`).focus();
      await page.keyboard.press("End");
      const reached = await page.evaluate(
        (morph) =>
          window.editor.state.body.meshes
            .filter((mesh: any) => mesh.morphTargetDictionary && morph in mesh.morphTargetDictionary)
            .map((mesh: any) => mesh.morphTargetInfluences[mesh.morphTargetDictionary[morph]]),
        name,
      );
      expect(reached.length, `${name} is on no mesh`).toBeGreaterThan(0);
      expect(reached.every((weight: number) => weight > 0.95), `${name} didn't arrive`).toBe(true);
      expect(await editor.picture(), `${name} changed nothing on screen`).not.toBe(start);
      await page.keyboard.press("Home");
    }
    for (const name of ["jaw", "gazeAcross", "gazeUp"]) {
      await page.locator(`#face-${name} [role="slider"]`).focus();
      await page.keyboard.press("End");
      expect(await editor.picture(), `${name} changed nothing on screen`).not.toBe(start);
    }
    await page.locator("#morphsReset").click();
    expect(await editor.picture(), "putting every control back changed the body").toBe(start);
    expect(editor.posts.length, "looking asked for a body").toBe(posts);
  });

  test("what's being looked at stays on the next body, and stays out of the file", async ({ editor, page }) => {
    await editor.open("?body=female-long&panel=hip&inspect=true");
    await page.locator("#part-hair").click();
    await page.locator('#morph-happy [role="slider"]').focus();
    await page.keyboard.press("End");
    await editor.type("hipBreadth", "234");
    await editor.settled();
    const state = await page.evaluate(() => {
      const { body } = window.editor.state;
      const face = body.meshes.find((mesh: any) => mesh.morphTargetDictionary && "happy" in mesh.morphTargetDictionary);
      return {
        hair: body.parts.find((part: any) => part.userData.name === "hair").visible,
        happy: face.morphTargetInfluences[face.morphTargetDictionary.happy],
        file: window.editor.currentFile(),
      };
    });
    expect(state.hair).toBe(false);
    expect(state.happy).toBe(1);
    expect(state.file).not.toContain("happy");
    await expect(page.locator("#morphsReset")).toBeVisible();
    await page.locator("#morphsReset").click();
    expect(await page.evaluate(() => window.editor.state.morphs)).toEqual({});
  });

  test("the helper bones can be left undriven, in a pose where it shows", async ({ editor, page }) => {
    await editor.open("?body=female-long&pose=armsUp&inspect=true");
    const driven = await editor.picture();
    await page.locator("#driven").click();
    expect(await editor.picture(), "the switch changed nothing").not.toBe(driven);
    await page.locator("#driven").click();
    expect(await editor.picture()).toBe(driven);
  });
});

test.describe("languages", () => {
  test.describe("in an English browser", () => {
    test.use({ locale: "en-US" });

    test("the page is in English, down to the rules' reasons and the slider's help", async ({ editor, page }) => {
      await editor.open();
      await expect(page.locator("html")).toHaveAttribute("lang", "en");
      await expect(page).toHaveTitle("Body editor");
      await expect(page.getByRole("heading", { name: "Body editor" })).toBeVisible();
      await expect(page.getByRole("status")).toHaveText("Generated");
      await page.getByRole("navigation").getByRole("button", { name: /^Neck/ }).click();
      await page.getByRole("button", { name: "About Neck girth" }).click();
      await expect(page.getByText(/^Around the neck just below the chin/)).toBeVisible();
      await page.getByRole("navigation").getByRole("button", { name: /^Shoulders/ }).click();
      await editor.type("biacromialBreadth", "160");
      await expect(page.getByRole("status")).toHaveText("The parameters are invalid; nothing was generated. Showing the last valid model");
      await expect(page.getByRole("list", { name: "Problems" }).getByRole("listitem").first()).toContainText(/short by/);
      await expect(page.getByRole("button", { name: "Adjust" }).first()).toBeVisible();
      expect(await page.locator("body").innerText()).not.toMatch(/\p{Script=Han}/u);
    });

    test("picking Chinese reloads the page in Chinese, and it stays so", async ({ editor, page }) => {
      await editor.open();
      await page.getByRole("button", { name: "Language" }).click();
      await page.getByRole("menuitem", { name: "中文" }).click();
      await page.waitForFunction(() => window.editor?.ready === true);
      await editor.settled();
      await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN");
      await expect(page.getByRole("heading", { name: "体型编辑器" })).toBeVisible();
      await page.reload();
      await page.waitForFunction(() => window.editor?.ready === true);
      await expect(page.getByRole("heading", { name: "体型编辑器" })).toBeVisible();
      await page.getByRole("button", { name: "界面语言" }).click();
      await page.getByRole("menuitem", { name: "English" }).click();
      await page.waitForFunction(() => window.editor?.ready === true);
      await expect(page.getByRole("heading", { name: "Body editor" })).toBeVisible();
    });
  });

  test("a Chinese browser gets the Chinese page", async ({ editor, page }) => {
    await editor.open();
    await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN");
    await expect(page.getByRole("status")).toHaveText("已生成");
  });
});
