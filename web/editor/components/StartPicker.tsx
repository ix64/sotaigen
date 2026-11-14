import { ChevronDown } from "lucide-react";
import { Segmented } from "@/components/Segmented";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { type Editor, useEditor } from "../store";
import { figureLabel, say, sexLabel } from "../text";

/**
 * Where the body starts from: first a sex, then one of that sex's typical bodies, picked once
 * and then left. It lives in the bar, named by what it is, so the panel below is all sliders.
 */
export function StartPicker({ editor }: { editor: Editor }) {
  const name = useEditor(editor, (state) => state.name);
  const sex = useEditor(editor, (state) => state.picks.sex);
  const group = useEditor(editor, (state) => state.picks.group);
  const sexes = Object.keys(editor.typical);
  const bodies = editor.typical[sex] ?? [];
  // Sexes need not have the same groups: keep the group when the other sex has it, else start
  // from that sex's first typical body.
  const switchTo = (next: string) => {
    if (next === sex) return;
    const others = editor.typical[next];
    editor.choosePreset((others.find((body) => body.group === group) ?? others[0]).name);
  };
  const what = name ? figureLabel(group) : say("start_imported");
  return (
    <Popover>
      <PopoverTrigger
        id="start"
        aria-label={say("start_preset")}
        className="flex h-8 min-w-0 items-center gap-1 rounded-md px-2 text-muted-foreground outline-none hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 data-[state=open]:bg-muted data-[state=open]:text-foreground"
      >
        <span className="hidden text-xs sm:inline">{say("start_preset")}</span>
        <span className="truncate font-medium text-foreground">{`${sexLabel(sex)} · ${what}`}</span>
        <ChevronDown className="size-3.5 shrink-0" />
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 space-y-2 p-3">
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">{say("start_sex")}</div>
          <Segmented
            label={say("start_sex")}
            idPrefix="start-sex"
            value={sex}
            options={sexes.map((key) => ({ value: key, label: sexLabel(key), title: key }))}
            onChange={switchTo}
          />
        </div>
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">{say("start_figure")}</div>
          <Segmented
            label={say("start_figure")}
            idPrefix="start-group"
            value={name ? group : ""}
            options={bodies.map((body) => ({ value: body.group, label: figureLabel(body.group), title: body.name }))}
            onChange={(next) => editor.choosePreset(bodies.find((body) => body.group === next)!.name)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          {name ? say("start_replaces") : say("start_imported_note")}
        </p>
      </PopoverContent>
    </Popover>
  );
}
