import { useEffect, useRef } from "react";
import { Inspector } from "./Inspector";
import { StatusCard } from "./StatusCard";
import { ViewToolbar } from "./ViewToolbar";
import { type Editor, useEditor } from "../store";
import { say } from "../text";

export function ViewportStage({ editor }: { editor: Editor }) {
  const host = useRef<HTMLDivElement>(null);
  const hasBody = useEditor(editor, (state) => state.hasBody);
  useEffect(() => {
    if (host.current) editor.viewport.attach(host.current);
  }, [editor]);
  return (
    <main className="relative flex min-h-[45dvh] flex-1 overflow-hidden">
      <div className="relative min-w-0 flex-1 bg-[#eceae6] dark:bg-[#85868b]">
      <div ref={host} className="absolute inset-0 [&_canvas]:outline-none" />
      <StatusCard editor={editor} />
      {hasBody && (
        <p className="pointer-events-none absolute top-3.5 left-4 hidden text-xs text-black/50 xl:block">
          {say("viewport_hint")}
        </p>
      )}
      <ViewToolbar editor={editor} />
      </div>
      <Inspector editor={editor} />
    </main>
  );
}
