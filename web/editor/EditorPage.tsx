import { useEffect } from "react";
import { ParameterPanel } from "./components/ParameterPanel";
import { RegionRail } from "./components/RegionRail";
import { TopBar } from "./components/TopBar";
import { ViewportStage } from "./components/ViewportStage";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { Editor } from "./store";
import type { ViewName } from "./viewport";

const VIEW_KEYS: Record<string, ViewName> = { "1": "front", "2": "side", "3": "back", "4": "threequarter" };

const typing = (target: EventTarget | null) => target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;

function useShortcuts(editor: Editor) {
  useEffect(() => {
    function down(event: KeyboardEvent) {
      if (typing(event.target)) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) editor.redo();
        else editor.undo();
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (VIEW_KEYS[event.key]) editor.view(VIEW_KEYS[event.key]);
      if (event.key.toLowerCase() === "c" && !event.repeat) editor.hold(true);
      if (event.key.toLowerCase() === "i" && !event.repeat) editor.setInspecting(!editor.state.inspecting);
    }
    function up(event: KeyboardEvent) {
      if (event.key.toLowerCase() === "c") editor.hold(false);
    }
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [editor]);
}

export function EditorPage({ editor }: { editor: Editor }) {
  useShortcuts(editor);
  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex h-dvh flex-col">
        <TopBar editor={editor} />
        <div className="flex min-h-0 flex-1 flex-col-reverse md:flex-row">
          <aside className="flex min-h-0 flex-1 flex-col border-t bg-card md:flex-none md:flex-row md:border-t-0 md:border-r">
            <RegionRail editor={editor} />
            <ParameterPanel editor={editor} />
          </aside>
          <ViewportStage editor={editor} />
        </div>
      </div>
    </TooltipProvider>
  );
}
