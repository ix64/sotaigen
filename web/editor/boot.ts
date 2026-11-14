// Starts the editor, once however often its route is entered: it goes on holding its body and its
// history while another page is open.

import type { ParameterSexes } from "./params-file.ts";
import { type BodyParams, createEditor, type Editor, type EditorStart } from "./store";

// web/serve.mjs writes this when it starts, so it isn't part of what Vite bundles.
const SCHEMA = "/build/editor/params.generated.mjs";

/** Null for anything but the JSON asked for, such as another server's own page. */
async function fetchJsonOrNull(url: string) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

let booted: Promise<Editor | null> | null = null;

async function boot(start: EditorStart): Promise<Editor | null> {
  const [served, schema] = (await Promise.all([
    fetchJsonOrNull("/api/presets"),
    import(/* @vite-ignore */ SCHEMA).catch(() => null),
  ])) as [{ presets: Record<string, BodyParams> } | null, ParameterSexes | null];
  if (!served || !schema) return null;
  const editor = createEditor({ presets: served.presets, sexes: schema, start });
  // Whether the .blend export can work, which the menu says before anyone clicks. Without an answer the
  // click finds out, so a failure here is nothing to report.
  fetchJsonOrNull("/api/blend").then((reply) => editor.setBlender(reply?.blender ?? null));
  // For web/tests/, and for poking at it from the console.
  window.editor = {
    state: editor.state,
    rebuilder: editor.rebuilder,
    log: editor.log,
    renderer: editor.viewport.renderer,
    viewport: editor.viewport,
    importFile: editor.importFile,
    currentFile: editor.currentFile,
    exportGlb: editor.exportGlb,
    exportBlend: editor.exportBlend,
    ready: true,
  };
  return editor;
}

/** The editor, or null when the page wasn't opened from the local service. */
export function bootEditor(start: EditorStart) {
  booted ??= boot(start);
  return booted;
}
