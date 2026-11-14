// The compiled messages, run by pnpm run web-check after the build has compiled them. A message with
// variants none of which matches compiles fine and then prints its own key, which no type check sees:
// every one is rendered here, in every language, with a singular and a plural input. What the
// catalogues hold is checked without the npm packages, in tests/test_editor.py.

import assert from "node:assert/strict";
import { test } from "node:test";
import * as messages from "../src/paraglide/messages.js";
import { locales } from "../src/paraglide/runtime.js";

/** Inputs that give n for every name a message asks for. */
const every = (n) => new Proxy({}, { get: () => n });

test("every message renders in every language", () => {
  const broken = [];
  for (const [key, say] of Object.entries(messages)) {
    if (typeof say !== "function") continue;
    for (const locale of locales) {
      for (const n of [1, 2]) {
        const out = say(every(n), { locale });
        if (out === key || out.includes("undefined")) broken.push(`${locale} ${key}(${n}): ${out}`);
      }
    }
  }
  assert.deepEqual(broken, []);
});

test("English is the source and the fallback", async () => {
  const { baseLocale } = await import("../src/paraglide/runtime.js");
  assert.equal(baseLocale, "en");
  assert.deepEqual([...locales].sort(), ["en", "zh"]);
});
