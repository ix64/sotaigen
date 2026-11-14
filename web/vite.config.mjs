// The pages, built into build/web/app/ for web/serve.mjs, or served by it with hot reload when it's
// started with --dev. There's no separate Vite port: the service and the pages share one.

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { paraglideVitePlugin } from "@inlang/paraglide-js";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { paraglide } from "./scripts/paraglide.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: HERE,
  // The page's words: web/messages compiled into src/paraglide, which git ignores.
  plugins: [react(), tailwindcss(), paraglideVitePlugin(paraglide)],
  resolve: { alias: { "@": resolve(HERE, "src") } },
  // The sliders' definition is the package's data (sotaigen/data/sliders.json), outside this folder.
  server: { fs: { allow: [resolve(HERE, "..")] } },
  build: {
    outDir: resolve(HERE, "../build/web/app"),
    emptyOutDir: true,
    chunkSizeWarningLimit: 2000,
    // The app, and the two plain pages web/runtime/check.mjs uses, which must not carry the app.
    rollupOptions: {
      input: {
        app: resolve(HERE, "index.html"),
        check: resolve(HERE, "runtime/check.html"),
        shot: resolve(HERE, "runtime/shot.html"),
      },
    },
  },
});
