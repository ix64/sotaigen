import { getRouteApi } from "@tanstack/react-router";
import { useEffect } from "react";
import { EditorPage } from "./EditorPage";
import { type Editor, useEditor } from "./store";
import { say } from "./text";

const NEEDS_SERVICE = "\n\npnpm install\npnpm run editor\n";

const route = getRouteApi("/");

/** The address follows what's chosen, so a reload or a shared link comes back to it. Not the sliders. */
function useAddress(editor: Editor) {
  const navigate = route.useNavigate();
  const name = useEditor(editor, (state) => state.name);
  const pose = useEditor(editor, (state) => state.pose);
  const look = useEditor(editor, (state) => state.look);
  const panel = useEditor(editor, (state) => state.panel);
  const inspecting = useEditor(editor, (state) => state.inspecting);
  useEffect(() => {
    navigate({
      search: {
        body: name ?? undefined,
        pose: pose === "rest" ? undefined : pose,
        look: look === "toon" ? undefined : look,
        panel: panel === "proportion" ? undefined : panel,
        inspect: inspecting || undefined,
      },
      replace: true,
    });
  }, [navigate, name, pose, look, panel, inspecting]);
}

function Running({ editor }: { editor: Editor }) {
  useAddress(editor);
  return <EditorPage editor={editor} />;
}

export function EditorRoute() {
  const editor = route.useLoaderData();
  if (!editor) {
    return (
      <div id="fatal" className="grid h-dvh place-items-center p-6 whitespace-pre-wrap">
        {say("app_needs_service") + NEEDS_SERVICE}
      </div>
    );
  }
  return <Running editor={editor} />;
}
