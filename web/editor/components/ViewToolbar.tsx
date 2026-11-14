import { Focus, PanelRight, SplitSquareHorizontal } from "lucide-react";
import type { ReactNode } from "react";
import { Hint } from "@/components/Hint";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Toggle } from "@/components/ui/toggle";
import { isHandPose, isTouchPose } from "../regions";
import { type Editor, useEditor } from "../store";
import type { ViewName } from "../viewport";
import { poseLabel, say } from "../text";

// Called view_<id>, and shown on the button as view_<id>_short; ¾ reads the same in every language.
const VIEWS: { id: ViewName; keys: string }[] = [
  { id: "front", keys: "1" },
  { id: "side", keys: "2" },
  { id: "back", keys: "3" },
  { id: "threequarter", keys: "4" },
];
const viewShort = (id: ViewName) => (id === "threequarter" ? "¾" : say(`view_${id}_short`));

function Divider() {
  return <span aria-hidden className="mx-0.5 hidden h-5 w-px shrink-0 bg-border xl:block" />;
}

function PoseSelect({ editor }: { editor: Editor }) {
  const pose = useEditor(editor, (state) => state.pose);
  const poses = useEditor(editor, (state) => state.poses);
  const groups: [string, string[]][] = [
    [say("pose_group_body"), poses.filter((name) => !isHandPose(name) && !isTouchPose(name))],
    [say("pose_group_touch"), poses.filter(isTouchPose)],
    [say("pose_group_hand"), poses.filter(isHandPose)],
  ];
  return (
    <Select value={pose} onValueChange={editor.setPose} disabled={!poses.length}>
      <SelectTrigger id="pose" size="sm" aria-label={say("pose_label")} className="h-8 w-auto min-w-[6.5rem] shrink-0 border-0 bg-transparent shadow-none dark:bg-transparent">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {groups.map(([label, names]) =>
          names.length ? (
            <SelectGroup key={label}>
              <SelectLabel>{label}</SelectLabel>
              {names.map((name) => (
                <SelectItem key={name} value={name} data-pose={name}>
                  {poseLabel(name)}
                </SelectItem>
              ))}
            </SelectGroup>
          ) : null,
        )}
      </SelectContent>
    </Select>
  );
}

function CompareButton({ editor }: { editor: Editor }) {
  const comparing = useEditor(editor, (state) => state.comparing);
  const canCompare = useEditor(editor, (state) => state.canCompare);
  return (
    <Hint text={say("compare_hint")} keys="C">
      <Button
        id="compare"
        size="sm"
        variant={comparing ? "default" : "ghost"}
        aria-pressed={comparing}
        disabled={!canCompare}
        className="shrink-0 touch-none select-none"
        onPointerDown={(event) => {
          if (event.button !== 0 || !event.isPrimary) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          editor.hold(true);
        }}
        onPointerUp={() => editor.hold(false)}
        onPointerCancel={() => editor.hold(false)}
        onLostPointerCapture={() => editor.hold(false)}
        onBlur={() => editor.hold(false)}
        onKeyDown={(event) => {
          if (event.key !== " " && event.key !== "Enter") return;
          event.preventDefault();
          if (!event.repeat) editor.hold(true);
        }}
        onKeyUp={(event) => {
          if (event.key === " " || event.key === "Enter") editor.hold(false);
        }}
      >
        <SplitSquareHorizontal />
        {comparing ? say("compare_before") : say("compare")}
      </Button>
    </Hint>
  );
}

function ViewButton({ id, label, title, keys, onClick, disabled }: { id: string; label: ReactNode; title: string; keys?: string; onClick: () => void; disabled: boolean }) {
  return (
    <Hint text={title} keys={keys}>
      <Button id={id} variant="ghost" size="sm" aria-label={title} disabled={disabled} className="min-w-8 shrink-0 px-2 font-normal" onClick={onClick}>
        {label}
      </Button>
    </Hint>
  );
}

/** How the body is seen. None of this asks for a new body. */
export function ViewToolbar({ editor }: { editor: Editor }) {
  const hasBody = useEditor(editor, (state) => state.hasBody);
  const follow = useEditor(editor, (state) => state.follow);
  const inspecting = useEditor(editor, (state) => state.inspecting);
  return (
    <div className="pointer-events-none absolute inset-x-3 bottom-3 flex justify-center">
      <div
        role="toolbar"
        aria-label={say("toolbar_label")}
        className="pointer-events-auto flex max-w-full flex-wrap items-center justify-center gap-0.5 rounded-lg border bg-card/95 p-1 shadow-sm backdrop-blur-sm"
      >
        {VIEWS.map((view) => (
          <ViewButton
            key={view.id}
            id={view.id}
            label={viewShort(view.id)}
            title={say(`view_${view.id}`)}
            keys={view.keys}
            disabled={!hasBody}
            onClick={() => editor.view(view.id)}
          />
        ))}
        <Divider />
        <ViewButton id="full" label={say("crop_full")} title={say("crop_full_hint")} onClick={() => editor.crop("full")} disabled={!hasBody} />
        <ViewButton id="upper" label={say("crop_upper")} title={say("crop_upper_hint")} onClick={() => editor.crop("upper")} disabled={!hasBody} />
        <Hint text={say("follow_hint")}>
          <Toggle
            id="follow"
            size="sm"
            aria-label={say("follow_label")}
            pressed={follow}
            onPressedChange={editor.setFollow}
            className="shrink-0"
          >
            <Focus />
          </Toggle>
        </Hint>
        <Divider />
        <PoseSelect editor={editor} />
        <Divider />
        <CompareButton editor={editor} />
        <Hint text={say("inspect_hint")} keys="I">
          <Toggle id="inspect" size="sm" aria-label={say("inspect")} pressed={inspecting} onPressedChange={editor.setInspecting} className="shrink-0 px-2">
            <PanelRight />
            {/* Open, the panel says what it is beside it, and the bar has room for longer words. */}
            {!inspecting && say("inspect")}
          </Toggle>
        </Hint>
      </div>
    </div>
  );
}
