import { ChevronRight, RotateCcw, Search, X } from "lucide-react";
import { useState } from "react";
import { SliderRow } from "./SliderRow";
import { Button } from "@/components/ui/button";
import { type Editor, definition, useEditor } from "../store";
import { cn } from "@/lib/utils";
import { groupLabel, say, sectionLabel, sliderLabel } from "../text";

function SearchBox({ editor }: { editor: Editor }) {
  const query = useEditor(editor, (state) => state.query);
  return (
    <div className="relative border-b">
      <Search className="pointer-events-none absolute top-1/2 left-4 size-3.5 -translate-y-1/2 text-muted-foreground" />
      <input
        type="search"
        id="search"
        aria-label={say("panel_search")}
        placeholder={say("panel_search")}
        autoComplete="off"
        value={query}
        onChange={(event) => editor.setQuery(event.target.value)}
        className="h-9 w-full bg-transparent pr-9 pl-9 outline-none placeholder:text-muted-foreground [&::-webkit-search-cancel-button]:hidden"
      />
      {query && (
        <button
          type="button"
          aria-label={say("panel_clear_search")}
          onClick={() => editor.setQuery("")}
          className="absolute top-1/2 right-3 grid size-6 -translate-y-1/2 place-items-center rounded-sm text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <X className="size-3.5" />
        </button>
      )}
    </div>
  );
}

/** The sliders the body's sex has, as the sex changes. */
function useOffered(editor: Editor) {
  useEditor(editor, (state) => state.base.sex);
  return (names: string[]) => names.filter(editor.offered);
}

/** Rows from several groups, each under its group's name. */
function GroupedRows({ editor, names }: { editor: Editor; names: string[] }) {
  const offered = useOffered(editor);
  return definition.groups.map((group) => {
    const rows = offered(group.sliders.filter((name) => names.includes(name)));
    if (!rows.length) return null;
    return (
      <section key={group.id} className="pb-1">
        <h3 className="px-4 pt-3 pb-0.5 text-[11px] text-muted-foreground">{groupLabel(group.id)}</h3>
        {rows.map((name) => (
          <SliderRow key={name} editor={editor} name={name} />
        ))}
      </section>
    );
  });
}

function Empty({ children }: { children: string }) {
  return <p className="p-6 text-center text-muted-foreground">{children}</p>;
}

function SearchResults({ editor, query }: { editor: Editor; query: string }) {
  const wanted = query.trim().toLowerCase();
  const names = Object.keys(definition.sliders).filter(
    (name) => name.toLowerCase().includes(wanted) || sliderLabel(name).toLowerCase().includes(wanted),
  );
  return names.length ? <GroupedRows editor={editor} names={names} /> : <Empty>{say("panel_not_found")}</Empty>;
}

function Changes({ editor }: { editor: Editor }) {
  // Joined, so that this renders again only when the list itself changes, not on every value.
  const joined = useEditor(editor, () => Object.keys(definition.sliders).filter(editor.isChanged).join());
  return joined ? (
    <GroupedRows editor={editor} names={joined.split(",")} />
  ) : (
    <Empty>{say("panel_no_changes")}</Empty>
  );
}

function AbdominalMeasurements({ editor }: { editor: Editor }) {
  const report = useEditor(editor, () => editor.abdominalReport());
  if (!report) return <p className="px-4 py-2 text-xs text-muted-foreground">{say("abdomen_pending")}</p>;
  const mm = (key: string) => (report.measured[key] * report.torsoMm).toFixed(1);
  return (
    <div className="mx-4 my-2 rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
      <div>{say("abdomen_measured", { front: mm("frontArc"), back: mm("backHollow") })}</div>
      <div className="mt-1" id="abdominal-depth">{say("abdomen_depth", { depth: mm("depth") })}</div>
      <p className="mt-1">{say("abdomen_method")}</p>
    </div>
  );
}

function ThighMeasurements({ editor }: { editor: Editor }) {
  const report = useEditor(editor, () => editor.thighReport());
  if (!report) return <p className="px-4 py-2 text-xs text-muted-foreground">{say("thigh_pending")}</p>;
  const mm = (key: string) => (report.measured[key] * report.torsoMm).toFixed(1);
  return (
    <div id="thigh-measurements" className="mx-4 my-2 rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
      <div>{say("thigh_measured", { girth: mm("thighGirth"), arc: mm("outerArc") })}</div>
      <p className="mt-1">{say("thigh_method")}</p>
    </div>
  );
}

function Region({ editor, id }: { editor: Editor; id: string }) {
  const group = definition.groups.find((row) => row.id === id);
  const moved = useEditor(editor, (state) => group?.sliders.filter((name) => name in state.sliders && editor.isChanged(name)).length ?? 0);
  const offered = useOffered(editor);
  if (!group) return null;
  const names = offered(group.sliders);
  return (
    <>
      <header className="sticky top-0 z-10 flex h-9 items-center justify-between bg-card/95 px-4 backdrop-blur-sm">
        <h2 className="font-medium">{groupLabel(group.id)}</h2>
        {moved > 0 && (
          <Button variant="ghost" size="xs" className="text-muted-foreground" onClick={() => editor.clearSliders(group.sliders)}>
            <RotateCcw /> {say("panel_reset_group")}
          </Button>
        )}
      </header>
      <Rows editor={editor} names={names.filter((name) => !definition.sliders[name].advanced)} />
      <Advanced key={group.id} editor={editor} names={names.filter((name) => definition.sliders[name].advanced)} />
    </>
  );
}

/** Rows in order, with a caption wherever the section changes. */
function Rows({ editor, names }: { editor: Editor; names: string[] }) {
  return names.map((name, index) => {
    const { section, measure } = definition.sliders[name];
    const startsSection = section && section !== definition.sliders[names[index - 1]]?.section;
    return (
      <div key={name}>
        {startsSection && <h3 className="px-4 pt-2.5 text-[11px] text-muted-foreground">{sectionLabel(section)}</h3>}
        <SliderRow editor={editor} name={name} />
        {measure === "abdomen" && <AbdominalMeasurements editor={editor} />}
        {measure === "thigh" && <ThighMeasurements editor={editor} />}
      </div>
    );
  });
}

/** The ones few bodies need. They stay out of the way, unless one of them has been changed. */
function Advanced({ editor, names }: { editor: Editor; names: string[] }) {
  const changed = useEditor(editor, () => names.filter(editor.isChanged).length);
  const [asked, setAsked] = useState<boolean | null>(null);
  const open = asked ?? changed > 0;
  if (!names.length) return null;
  return (
    <section className="mt-1 border-t">
      <button
        type="button"
        id="advanced"
        aria-expanded={open}
        onClick={() => setAsked(!open)}
        className="flex h-9 w-full items-center gap-1 px-4 text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/50"
      >
        <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} />
        {say("panel_advanced")}
        {changed > 0 && <span className="ml-1 text-[11px] text-primary tabular-nums">{changed}</span>}
      </button>
      {open && <Rows editor={editor} names={names} />}
    </section>
  );
}

/** What the sliders add up to. Results, shown live, because nothing sets them directly. */
function Derived({ editor }: { editor: Editor }) {
  const line = useEditor(editor, () => {
    const derived = editor.derived();
    return derived && say("panel_derived", { stature: derived.stature.toFixed(0), heads: derived.heads.toFixed(2) });
  });
  if (!line) return null;
  return (
    <p id="derived" className="border-b px-4 py-1.5 text-[11px] tabular-nums text-muted-foreground">
      {line}
    </p>
  );
}

/** The one thing being adjusted, and how to get to any other slider. */
export function ParameterPanel({ editor }: { editor: Editor }) {
  const panel = useEditor(editor, (state) => state.panel);
  const query = useEditor(editor, (state) => state.query);
  return (
    <div className="flex min-h-0 flex-1 flex-col md:w-[19.5rem] md:flex-none">
      <SearchBox editor={editor} />
      <Derived editor={editor} />
      <div className="min-h-0 flex-1 overflow-y-auto pb-3">
        {query.trim() ? (
          <SearchResults editor={editor} query={query} />
        ) : panel === "changes" ? (
          <Changes editor={editor} />
        ) : (
          <Region editor={editor} id={panel} />
        )}
        <p className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 px-4 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span aria-hidden className="zone-blocked h-1.5 w-4 rounded-[1px]" />
            {say("legend_blocked")}
          </span>
          <span className="flex items-center gap-1">
            <span aria-hidden className="h-1.5 w-4 rounded-[1px] bg-warn/45" />
            {say("legend_seam")}
          </span>
        </p>
      </div>
    </div>
  );
}
