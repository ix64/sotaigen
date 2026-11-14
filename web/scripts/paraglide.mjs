// The one place the message compiler is configured: vite.config.mjs runs it as a plugin, and
// pnpm run typecheck runs this file on its own first, so tsc sees the generated src/paraglide.
//
// The language: the one picked on the page (kept in this browser only), else the browser's, else
// English, the source and the fallback. See docs/systems/editor.md, "界面语言".

import { fileURLToPath } from "node:url";
import { compile } from "@inlang/paraglide-js";

const web = (p) => fileURLToPath(new URL(`../${p}`, import.meta.url));

/** @type {import("@inlang/paraglide-js").CompilerOptions} */
export const paraglide = {
  project: web("project.inlang"),
  outdir: web("src/paraglide"),
  strategy: ["localStorage", "preferredLanguage", "baseLocale"],
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await compile(paraglide);
}
