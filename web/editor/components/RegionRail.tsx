import { type Editor, definition, useEditor } from "../store";
import { REGION_TIERS, type RegionId } from "../regions";
import { cn } from "@/lib/utils";
import { groupLabel, say } from "../text";

const groupOf = (id: RegionId) => definition.groups.find((group) => group.id === id);

function RailItem({
  editor,
  id,
  label,
  count,
}: {
  editor: Editor;
  id: RegionId | "changes";
  label: string;
  count: number;
}) {
  const active = useEditor(editor, (state) => state.panel === id && !state.query);
  return (
    <button
      type="button"
      id={`region-${id}`}
      aria-current={active ? "true" : undefined}
      onClick={() => editor.setPanel(id)}
      className={cn(
        "flex h-8 shrink-0 items-center justify-between gap-1.5 rounded-md px-2.5 whitespace-nowrap outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring/50",
        active ? "bg-accent font-medium text-accent-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground",
      )}
    >
      {label}
      {count > 0 && <span className="text-[11px] text-primary tabular-nums">{count}</span>}
    </button>
  );
}

/** A group, when the body's sex has any of its sliders: a male body has no bust. */
function RegionItem({ editor, id }: { editor: Editor; id: RegionId }) {
  const group = groupOf(id);
  const shown = useEditor(editor, () => editor.regionOffered(id));
  const count = useEditor(editor, () => group?.sliders.filter(editor.isChanged).length ?? 0);
  return group && shown ? <RailItem editor={editor} id={id} label={groupLabel(id)} count={count} /> : null;
}

/** Where the sliders live: the whole figure, then its places. Across the top on a narrow screen. */
export function RegionRail({ editor }: { editor: Editor }) {
  const changes = useEditor(editor, () => Object.keys(definition.sliders).filter(editor.isChanged).length);
  return (
    <nav
      aria-label={say("rail_label")}
      className="flex shrink-0 gap-1 overflow-x-auto border-b p-2 md:w-max md:min-w-[5.75rem] md:flex-col md:gap-0.5 md:overflow-x-visible md:overflow-y-auto md:border-r md:border-b-0"
    >
      {REGION_TIERS.map((tier) => (
        <div key={tier.id} className="flex items-center gap-1 md:mb-3 md:flex-col md:items-stretch md:gap-0.5">
          <div className="hidden px-2.5 pt-1 pb-0.5 text-[11px] text-muted-foreground/80 md:block">{say(`tier_${tier.id}`)}</div>
          {tier.regions.map((id) => (
            <RegionItem key={id} editor={editor} id={id} />
          ))}
        </div>
      ))}
      <div className="flex shrink-0 md:mt-auto md:flex-col md:border-t md:pt-2">
        <RailItem editor={editor} id="changes" label={say("rail_changes")} count={changes} />
      </div>
    </nav>
  );
}
