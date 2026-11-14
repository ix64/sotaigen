import { CircleHelp, RotateCcw } from "lucide-react";
import { useDeferredValue, useId, useMemo, useState } from "react";
import { Hint } from "@/components/Hint";
import { Slider } from "@/components/ui/slider";
import { translateRule } from "../messages.ts";
import { type Editor, formatValue, useEditor } from "../store";
import { cn } from "@/lib/utils";
import { listOf, say, sliderEnds, sliderHelp, sliderLabel } from "../text";

// These shape a breast: with no projection there is none for them to act on.
const NEEDS_BREAST = new Set(["breastBaseBreadth", "breastFoldDrop", "breastUpperPoleRatio", "breastGap", "breastApexRatio"]);

/** The value as text, and a place to type an exact one. Enter or leaving commits, Escape doesn't. */
function ValueField({
  id,
  label,
  value,
  span,
  step,
  disabled,
  changed,
  onCommit,
}: {
  id: string;
  label: string;
  value: number | undefined;
  span: number;
  step: number;
  disabled: boolean;
  changed: boolean;
  onCommit: (value: number) => void;
}) {
  const shown = formatValue(value, span);
  const [draft, setDraft] = useState<string | null>(null);

  function commit() {
    if (draft === null) return;
    const typed = Number(draft);
    setDraft(null);
    if (draft.trim() !== "" && Number.isFinite(typed) && draft !== shown) onCommit(typed);
  }

  return (
    <input
      id={id}
      aria-label={say("row_value_of", { label })}
      inputMode="decimal"
      autoComplete="off"
      disabled={disabled}
      value={draft ?? shown}
      onChange={(event) => setDraft(event.target.value)}
      onFocus={(event) => {
        setDraft(shown);
        event.target.select();
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") event.currentTarget.blur();
        if (event.key === "Escape") {
          setDraft(null);
          // Blur reads the state from this render, so the draft goes first.
          requestAnimationFrame(() => (event.target as HTMLInputElement).blur());
        }
        if ((event.key === "ArrowUp" || event.key === "ArrowDown") && typeof value === "number") {
          event.preventDefault();
          const next = value + (event.key === "ArrowUp" ? step : -step) * (event.shiftKey ? 10 : 1);
          onCommit(next);
          setDraft(formatValue(next, span));
        }
      }}
      className={cn(
        "h-6 w-[4.25rem] rounded-sm border border-transparent bg-transparent px-1 text-right text-xs tabular-nums outline-none hover:border-input focus:border-ring focus:bg-card focus:ring-2 focus:ring-ring/30 disabled:opacity-50",
        changed ? "font-medium text-primary" : "text-muted-foreground",
      )}
    />
  );
}

export function SliderRow({ editor, name }: { editor: Editor; name: string }) {
  const spec = editor.spec(name);
  useEditor(editor, (state) => state.report);
  const [low, high] = editor.bounds(name);
  const span = high - low;
  const value = useEditor(editor, () => editor.sliderValue(name));
  const reading = useEditor(editor, () => editor.reading(name));
  const miss = useEditor(editor, () => editor.targetMiss(name)?.join());
  const origin = useEditor(editor, () => editor.baseValue(name));
  const changed = useEditor(editor, () => editor.isChanged(name));
  const moved = useEditor(editor, (state) => name in state.sliders);
  const noBreast = useEditor(editor, (state) => !(state.current.body.breastProjection > 0));
  const base = useEditor(editor, (state) => state.base);
  // Its own value doesn't move its limits, and the others' can wait until the frame is free.
  const others = useEditor(editor, (state) =>
    JSON.stringify(Object.entries(state.sliders).filter(([other]) => other !== name)),
  );
  const settled = useDeferredValue(others);
  const labelId = useId();
  const [helpOpen, setHelpOpen] = useState(false);

  const awaitingMeasurement = Boolean(spec.measured) && typeof value !== "number";
  const disabled = (NEEDS_BREAST.has(name) && noBreast) || awaitingMeasurement;
  const label = sliderLabel(name);
  const help = sliderHelp(name);
  const ends = sliderEnds(name);
  // Says why the four below it are off, so it doesn't wait to be asked.
  const standing = name === "breastProjection" && noBreast ? say("row_breast_needed") : null;
  const linked = changed && !moved;
  const known = typeof value === "number";
  // biome-ignore lint/correctness/useExhaustiveDependencies: base and settled are what zones reads.
  const zones = useMemo(() => {
    if (disabled) return [];
    return editor.zones(name).map((zone) => ({
      ...zone,
      title: say(zone.level === "blocked" ? "row_zone_blocked" : "row_zone_seam", {
        reasons: zone.reasons.map((reason) => translateRule(reason, say)).join(say("join_reasons")),
      }),
    }));
  }, [editor, name, disabled, base, settled]);
  const refused = listOf(
    zones
      .filter((zone) => zone.level === "blocked")
      .map(({ from, to }) => say("row_range", { from: formatValue(low + span * from, span), to: formatValue(low + span * to, span) })),
  );
  const described = [ends && say("row_ends", { low: ends[0], high: ends[1] }), refused && say("row_refused", { ranges: refused })]
    .filter(Boolean)
    .join(say("join_sentences"));

  return (
    <div className="relative px-4 py-2" data-changed={changed || undefined}>
      {/* The dot and not only the colour, so a change reads without telling teal from grey. */}
      {changed && <span aria-hidden className="absolute top-[1.0625rem] left-1.5 size-1.5 rounded-full bg-primary" />}
      <div className="flex h-6 items-center gap-1">
        <span id={labelId} className={cn("truncate", disabled && "text-muted-foreground")}>
          {label}
        </span>
        {help && (
          <button
            type="button"
            aria-label={say("row_help_of", { label })}
            aria-expanded={helpOpen}
            onClick={() => setHelpOpen(!helpOpen)}
            className="rounded-full text-muted-foreground/70 outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            <CircleHelp className="size-3.5" />
          </button>
        )}
        {linked && (
          <Hint text={say("row_linked_hint")}>
            <span className="rounded-sm bg-muted px-1 text-[11px] text-muted-foreground">{say("row_linked")}</span>
          </Hint>
        )}
        <span className="flex-1" />
        {moved && changed && (
          <Hint text={say("row_reset_hint")}>
            <button
              type="button"
              aria-label={say("row_reset", { label })}
              onClick={() => editor.clearSliders([name])}
              className="grid size-6 place-items-center rounded-sm text-muted-foreground outline-none hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <RotateCcw className="size-3" />
            </button>
          </Hint>
        )}
        <ValueField
          id={`value-${name}`}
          label={label}
          value={typeof value === "number" ? value : undefined}
          span={spec.measured ? Math.min(99, span) : span}
          step={spec.step ?? span / 100}
          disabled={disabled}
          changed={changed}
          onCommit={(next) => editor.moveSlider(name, Math.min(high, Math.max(low, next)), { now: true })}
        />
        {spec.unit && <span className="text-xs text-muted-foreground">{spec.unit}</span>}
      </div>
      <Slider
        id={`slider-${name}`}
        min={low}
        max={high}
        step={spec.step ?? span / 400}
        value={known ? value : low}
        origin={typeof origin === "number" ? origin : known ? value : low}
        disabled={disabled}
        zones={zones}
        onPointerDownCapture={() => {
          if (!disabled) editor.grab();
        }}
        onValueChange={([next]) => editor.moveSlider(name, next)}
        thumbProps={{
          "aria-labelledby": labelId,
          "aria-valuetext": reading ?? (spec.unit && known ? `${value.toFixed(1)} ${spec.unit}` : undefined),
          "aria-description": described || undefined,
        }}
      />
      {(ends || reading) && (
        <div className="-mt-0.5 flex justify-between gap-2 text-[11px] text-muted-foreground">
          <span className="shrink-0">{ends?.[0]}</span>
          {reading && <span data-reading={name} className="min-w-0 truncate tabular-nums">{reading}</span>}
          <span className="shrink-0">{ends?.[1]}</span>
        </div>
      )}
      {help && helpOpen && <p className="mt-1 text-xs text-muted-foreground">{help}</p>}
      {miss && (
        <p data-miss={name} className="mt-1 text-xs text-warn">
          {say("row_target_miss", { target: Number(miss.split(",")[0]).toFixed(1), actual: Number(miss.split(",")[1]).toFixed(1) })}
        </p>
      )}
      {awaitingMeasurement && <p className="mt-1 text-xs text-muted-foreground">{say("row_awaiting")}</p>}
      {standing && <p className="mt-1 text-xs text-muted-foreground">{standing}</p>}
    </div>
  );
}
