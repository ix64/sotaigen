import { Box, Check, ChevronDown, Download, File, FileJson, Languages, Layers, Moon, Redo2, RotateCcw, Sun, Undo2, Upload } from "lucide-react";
import { useRef } from "react";
import { Hint } from "@/components/Hint";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { getLocale, type Locale, locales, setLocale } from "@/paraglide/runtime.js";
import { type Editor, useEditor } from "../store";
import { say } from "../text";
import { StartPicker } from "./StartPicker";

/** The parameter file: what the body is, without its geometry. */
function FileMenu({ editor }: { editor: Editor }) {
  const file = useRef<HTMLInputElement>(null);
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button id="fileMenu" variant="ghost" size="sm">
            <File />
            <span className="hidden lg:inline">{say("top_file")}</span>
            <ChevronDown className="opacity-60" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem id="import" onSelect={() => file.current?.click()}>
            <Upload /> {say("top_import_json")}
          </DropdownMenuItem>
          <DropdownMenuItem id="exportJson" onSelect={editor.exportJson}>
            <FileJson /> {say("top_export_json")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <input
        ref={file}
        id="file"
        type="file"
        accept=".json,application/json"
        hidden
        onChange={async (event) => {
          const [chosen] = event.target.files ?? [];
          if (chosen) await editor.importBlob(chosen);
          event.target.value = "";
        }}
      />
    </>
  );
}

/** The model: the quick one by default, the rigged .blend or the one with contact tables on asking. */
function ExportButton({ editor }: { editor: Editor }) {
  const exporting = useEditor(editor, (state) => state.exporting);
  const baking = useEditor(editor, (state) => state.contactExport?.phase === "running");
  const blending = useEditor(editor, (state) => state.blendExport?.phase === "running");
  const noBlender = useEditor(editor, (state) => state.blender === false);
  const invalid = useEditor(editor, (state) => state.problems.errors.length > 0);
  return (
    <div className="ml-1 flex">
      <Button id="exportGlb" size="sm" className="rounded-r-none" disabled={invalid || exporting} onClick={editor.exportGlb}>
        <Download />
        <span className="hidden sm:inline">{exporting ? say("top_exporting") : say("top_export_glb")}</span>
      </Button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button id="exportMenu" size="sm" aria-label={say("top_export_more")} className="rounded-l-none border-l border-primary-foreground/25 px-1.5">
            <ChevronDown />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64">
          <DropdownMenuItem id="exportBlend" disabled={invalid || blending || noBlender} onSelect={editor.exportBlend} className="items-start">
            <Box className="mt-0.5" />
            <span>
              {blending ? say("top_blend_running") : say("top_blend_export")}
              {!noBlender && <span className="mt-0.5 block text-xs text-muted-foreground">{say("top_blend_note")}</span>}
            </span>
          </DropdownMenuItem>
          {/* Beside the item, not in it: a disabled item is faded, and this has to be read. */}
          {noBlender && (
            <p id="blendWhy" className="px-2 pb-1.5 pl-8 text-xs text-muted-foreground">
              {say("blend_no_blender")}
            </p>
          )}
          <DropdownMenuItem id="exportContact" disabled={invalid || baking} onSelect={editor.exportGlbContact} className="items-start">
            <Layers className="mt-0.5" />
            <span>
              {baking ? say("top_contact_running") : say("top_contact_export")}
              <span className="mt-0.5 block text-xs text-muted-foreground">
                {say("top_contact_note")}
              </span>
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

// Each language by its own name, so it can be found from a page one can't read.
const LANGUAGE_NAMES: Record<Locale, string> = { en: "English", zh: "中文" };

/** The page's language. Picking one reloads the page in it, and this browser remembers it. */
function LanguageMenu() {
  const current = getLocale();
  return (
    <DropdownMenu>
      <Hint text={say("app_language")}>
        <DropdownMenuTrigger asChild>
          <Button id="language" variant="ghost" size="icon-sm" aria-label={say("app_language")}>
            <Languages />
          </Button>
        </DropdownMenuTrigger>
      </Hint>
      <DropdownMenuContent align="end">
        {locales.map((locale) => (
          <DropdownMenuItem key={locale} id={`language-${locale}`} lang={locale} onSelect={() => setLocale(locale)}>
            <Check className={locale === current ? "" : "invisible"} /> {LANGUAGE_NAMES[locale]}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function TopBar({ editor }: { editor: Editor }) {
  const canUndo = useEditor(editor, (state) => state.canUndo);
  const canRedo = useEditor(editor, (state) => state.canRedo);
  const theme = useEditor(editor, (state) => state.theme);
  const anyMoved = useEditor(editor, (state) => Object.keys(state.sliders).length > 0);
  return (
    <header className="flex h-12 shrink-0 items-center gap-1 border-b bg-card px-3">
      <h1 className="mr-1 hidden font-semibold whitespace-nowrap sm:block">{say("app_title")}</h1>
      <StartPicker editor={editor} />
      <span className="flex-1" />
      <Hint text={say("top_undo")} keys="⌘Z">
        <Button id="undo" variant="ghost" size="icon-sm" aria-label={say("top_undo")} disabled={!canUndo} onClick={editor.undo}>
          <Undo2 />
        </Button>
      </Hint>
      <Hint text={say("top_redo")} keys="⇧⌘Z">
        <Button id="redo" variant="ghost" size="icon-sm" aria-label={say("top_redo")} disabled={!canRedo} onClick={editor.redo}>
          <Redo2 />
        </Button>
      </Hint>
      <Hint text={say("top_reset_all_hint")}>
        <Button id="reset" variant="ghost" size="sm" disabled={!anyMoved} onClick={editor.resetAll}>
          <RotateCcw />
          <span className="hidden lg:inline">{say("top_reset_all")}</span>
        </Button>
      </Hint>
      <span className="mx-1 h-5 w-px bg-border" />
      <FileMenu editor={editor} />
      <ExportButton editor={editor} />
      <LanguageMenu />
      <Hint text={theme === "dark" ? say("top_to_light") : say("top_to_dark")}>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={theme === "dark" ? say("top_to_light") : say("top_to_dark")}
          onClick={() => editor.setTheme(theme === "dark" ? "light" : "dark")}
        >
          {theme === "dark" ? <Sun /> : <Moon />}
        </Button>
      </Hint>
    </header>
  );
}
