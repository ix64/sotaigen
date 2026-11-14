// What the tests stand on: one service of its own for each test, and the page helpers that read like
// what a person does. Any uncaught error or console error on the page fails the test.

import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { test as base, expect, type Locator, type Page } from "@playwright/test";
import { serve } from "../serve.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
export const BODY = resolve(HERE, "..", "..", "build", "bodies", "female-long", "body.glb");
export const STANDIN = join(HERE, "standin-worker.mjs");

export interface Options {
  /** Instead of the stand-in. */
  command?: string[];
}

export class Editor {
  readonly posts: { at: number; parameters: { template: string; body: Record<string, number>; contact?: boolean } }[] = [];

  constructor(
    readonly page: Page,
    readonly address: string,
  ) {
    page.on("request", (request) => {
      if (request.url().endsWith("/api/body") && request.method() === "POST") {
        this.posts.push({ at: Date.now(), parameters: request.postDataJSON() });
      }
    });
  }

  async open(query = "?body=female-long") {
    await this.page.goto(`${this.address}/${query}`);
    await this.page.waitForFunction(() => window.editor?.ready === true);
    await this.settled();
  }

  /** On a body, or on being told why not. */
  async settled(timeout = 30000) {
    await this.page.waitForFunction(
      () => {
        const { rebuilder, state } = window.editor;
        return rebuilder.phase !== "generating" && rebuilder.phase !== "adjusting" && !state.undrawn;
      },
      null,
      { timeout },
    );
  }

  phase = () => this.page.evaluate(() => window.editor.rebuilder.phase as string);
  status = () => this.page.getByRole("status");
  problems = () => this.page.getByRole("list", { name: "问题" }).getByRole("listitem");

  group(name: string) {
    return this.page.getByRole("navigation").getByRole("button", { name: new RegExp(`^${name}`, "u") });
  }

  /** The build or style the body starts from, picked in the bar's popover, which then closes. */
  async start(label: string) {
    await this.page.locator("#start").click();
    await this.page.getByRole("radio", { name: label, exact: true }).click();
    await this.page.keyboard.press("Escape");
  }

  /** One item of a menu in the bar, such as #exportJson under #fileMenu. */
  async menu(trigger: string, item: string) {
    await this.page.locator(trigger).click();
    await this.page.locator(item).click();
    // Closing hands the focus back to the trigger, which would close a popover opened meanwhile.
    await expect(this.page.getByRole("menu")).toHaveCount(0);
  }

  /** The panel of how the body is seen, opened if it isn't. */
  async display() {
    if ((await this.page.locator("#inspector").count()) === 0) await this.page.locator("#inspect").click();
  }

  thumb(label: string): Locator {
    return this.page.getByRole("slider", { name: label, exact: true });
  }

  value(name: string) {
    return this.page.locator(`#value-${name}`);
  }

  /** A real drag with the mouse, left held: down on the thumb, along the track. */
  async grab(label: string, share: number) {
    const thumb = this.thumb(label);
    await thumb.scrollIntoViewIfNeeded();
    const box = (await thumb.boundingBox()) as NonNullable<Awaited<ReturnType<Locator["boundingBox"]>>>;
    const track = (await thumb.locator("xpath=ancestor::*[@data-slot='slider']").boundingBox()) as typeof box;
    const y = box.y + box.height / 2;
    await this.page.mouse.move(box.x + box.width / 2, y);
    await this.page.mouse.down();
    await this.page.mouse.move(track.x + track.width * share, y, { steps: 8 });
  }

  async drag(label: string, share: number) {
    await this.grab(label, share);
    await this.page.mouse.up();
  }

  async type(name: string, value: string) {
    await this.value(name).fill(value);
    await this.value(name).press("Enter");
  }

  // Until the server has read it (POST /api/params/read): the page takes the file in only then.
  async importText(text: string) {
    const read = this.page.waitForResponse((response) => response.url().endsWith("/api/params/read"));
    await this.page.locator("input[type=file]").setInputFiles({ name: "body.json", mimeType: "application/json", buffer: Buffer.from(text) });
    await read;
  }

  /** What the body looks like now, for telling whether something changed it. */
  async picture() {
    await this.page.waitForFunction(() => window.editor.viewport.flight === null);
    await this.page.waitForTimeout(120);
    return (await this.page.locator("main canvas").screenshot()).toString("base64");
  }

  camera() {
    return this.page.evaluate(() => window.editor.viewport.camera.position.toArray() as number[]);
  }
}

export const test = base.extend<Options & { service: { address: string; server: any }; editor: Editor; pageProblems: string[] }>({
  command: [undefined, { option: true }],
  service: async ({ command }, use, info) => {
    if (!existsSync(BODY)) throw new Error(`${BODY} is missing. Run: uv run sotaigen-dev export female-long`);
    const options = { command: command ?? [process.execPath, STANDIN, BODY], schema: false, cacheFolder: info.outputPath("cache") };
    const server: any = await serve(0, options as Parameters<typeof serve>[1]);
    await use({ address: `http://127.0.0.1:${server.address().port}`, server });
    server.close();
    server.closeAllConnections();
  },
  pageProblems: [
    async ({ page }, use) => {
      const problems: string[] = [];
      page.on("pageerror", (error) => problems.push(String(error)));
      page.on("console", (message) => {
        if (message.type() === "error" && !message.text().includes("Failed to load resource")) problems.push(message.text());
      });
      await use(problems);
      expect(problems, "the page logged errors").toEqual([]);
    },
    { auto: true },
  ],
  editor: async ({ page, service }, use) => {
    await use(new Editor(page, service.address));
  },
});

export { expect };
