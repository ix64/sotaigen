// What a slider's value does to a body: sotaigen_dev/editor/sliders.py for the page.
// No browser API here, and only types Node can strip, so that tests/test_editor.py can run this file
// under Node against the Python.

/** One entry of sliders.json's `sliders`. */
export type Slider = {
  step?: number;
  /** What the number is in, shown after it. */
  unit?: string;
  /** An absolute surface measurement supplied by the generator when not set in the file. */
  measured?: "frontArc" | "backHollow" | "outerArc";
  /** Parameters that keep their share of this one when it moves. */
  carries?: string[];
  /** Every parameter measured in shares of the torso keeps its share when this one moves (the torso). */
  carriesShares?: boolean;
  /**
   * Parameters this one needs beside it, written where the body has none when it moves: millimetres as
   * a share of the torso, the rest as they are. A breast drawn from nothing needs its shape.
   */
  brings?: Record<string, number>;
  /**
   * Parameters raised after this one moves, just enough to meet the named rule of the body's sex, and
   * never lowered: each is in shares of the torso and grows the rule's room one for one.
   */
  lifts?: Record<string, string>;
  /** Another millimetre figure to show from the value. */
  reading?: Reading;
  /** A caption above this slider in its group: the message section_<id>. */
  section?: string;
  /** Folded away until asked for. */
  advanced?: boolean;
  /** A card of the finished body's measurements, shown under this slider. */
  measure?: "abdomen" | "thigh";
} & (
  | { kind: "direct"; param: string }
  /** Sets the head count at the same stature, as `y * ratio ** law`, each length following the torso. */
  | { kind: "covary"; param: string; laws: Record<string, number> }
  /** Scales every millimetre to reach a stature. */
  | { kind: "scale"; param: string }
  /** Added `value * gain`, then `value * gain * torso`, then multiplied by `1 + value * gain`. */
  | { kind: "macro"; gains?: Record<string, number>; torsoGains?: Record<string, number>; scales?: Record<string, number> }
);

/**
 * (value + plusTorso × torso) × factor × unit, in mm, where unit is 1 for `self`, or the head or the
 * torso. `factorBy` adds to the factor by other parameters. The page shows it after the message
 * slider_<name>_reading, which says "about" where the figure is an estimate of the finished surface.
 */
export interface Reading {
  of: "self" | "head" | "torso";
  factor?: number;
  factorBy?: Record<string, number>;
  plusTorso?: number;
}

/** sliders.json. */
export interface SliderDefinition {
  note?: string;
  groups: { id: string; sliders: string[] }[];
  sliders: Record<string, Slider>;
}

/** Limits: `body` by parameter, `derived` for what the tools aim at. */
export type GroupRanges = { body: Record<string, [number, number]>; derived: Record<string, [number, number]> };

/** A body's parameters, in millimetres and plain numbers. */
export type Groups = { body: Record<string, number> };

/** What the head count, the scale and the defaults need from the generated tables. */
export interface SliderTables {
  ALL_SPEC: Record<string, [string, string, number, number, number | null]>;
  CHAIN: string[];
  MILLIMETRES: string[];
  TORSO_SHARES: string[];
  RULES: [string, (p: any) => number][];
  proportions: (body: any) => Record<string, number>;
  specBounds: (body: any) => Record<string, [number, number]>;
}

/** sotaigen.params.spec.JAW_BASE: the jaw's breadth over the head's at its narrowest. */
const JAW_BASE = 0.46;

export const stature = (body: Record<string, number>) => body.headHeight + body.neckLength + body.torsoLength + body.crotchHeight;
export const headCount = (body: Record<string, number>) => stature(body) / body.headHeight;

export function clamp(ranges: GroupRanges, kind: "body" | "derived", key: string, value: number): number {
  const [low, high] = ranges[kind][key];
  return Math.min(Math.max(value, low), high);
}

export function sliderBounds(spec: Slider, ranges: GroupRanges): [number, number] {
  if (spec.kind === "macro") return [-1, 1];
  const [kind, key] = spec.param.split(".") as ["body" | "derived", string];
  return ranges[kind][key];
}

const UNIT_PARAMETER: Record<string, string> = { head: "head", torso: "torsoLength", crotch: "crotchHeight", hip: "hipBreadth" };

/** What a body without optional parameter `key` has, as sotaigen.params.spec.default_of; undefined for its own shape. */
export function defaultOf(tables: Pick<SliderTables, "ALL_SPEC">, body: Record<string, number>, key: string): number | undefined {
  const row = tables.ALL_SPEC[key];
  if (!row || row[4] === null) return undefined;
  const unit = row[1] === "one" ? 1 : row[1] === "stature" ? stature(body) : body[UNIT_PARAMETER[row[1]]];
  return row[4] * unit;
}

export function sliderStart(
  spec: Slider,
  base: Groups,
  tables: Pick<SliderTables, "ALL_SPEC">,
  measured?: Partial<Record<string, number>>,
): number | undefined {
  if (spec.kind === "macro") return 0;
  const [kind, key] = spec.param.split(".");
  if (kind === "derived") return key === "heads" ? headCount(base.body) : stature(base.body);
  if (!Object.hasOwn(base.body, key)) {
    if (spec.measured) return measured?.[spec.measured];
    return defaultOf(tables, base.body, key);
  }
  return base.body[key];
}

/** The same stature in `value` head heights, every length following the torso and the laws. */
function setHeadCount(
  body: Record<string, number>,
  value: number,
  laws: Record<string, number>,
  tables: SliderTables,
): Record<string, number> {
  const ratio = value / headCount(body);
  const height = stature(body);
  const head = height / value;
  const neckLength = (body.neckLength / body.headHeight) * ratio ** laws.neckLength * head;
  const crotchHeight = body.crotchHeight * ratio ** laws.crotchHeight;
  const torsoLength = height - head - neckLength - crotchHeight;
  const scale = torsoLength / body.torsoLength;
  const out: Record<string, number> = { ...body, headHeight: head, neckLength, torsoLength, crotchHeight };
  out.kneeHeight = (body.kneeHeight * crotchHeight) / body.crotchHeight;
  // The limits of the new lengths, as the check that follows will read them.
  const limits: GroupRanges = { body: tables.specBounds(out), derived: {} };
  for (const key of tables.MILLIMETRES) {
    if (key in body && !tables.CHAIN.includes(key) && key !== "kneeHeight") {
      out[key] = clamp(limits, "body", key, body[key] * scale * ratio ** (laws[key] ?? 0));
    }
  }
  out.headWidthRatio = clamp(limits, "body", "headWidthRatio", body.headWidthRatio * ratio ** laws.headWidthRatio);
  // The jaw's law is on the part of its ratio past the narrowest jaw the head draws, as sotaigen.params.tools.
  const jaw = JAW_BASE + (body.jawBreadthRatio - JAW_BASE) * ratio ** laws.jawBreadthRatio;
  out.jawBreadthRatio = clamp(limits, "body", "jawBreadthRatio", jaw);
  return out;
}

// The { body } that `values` { slider: value } make of `base`, in file order. Whatever a moved slider
// writes stays inside its range, but for the lengths a tool works out, which must keep adding up. Direct
// sliders act on the result of preceding controls.
export function applySliders(
  base: Groups,
  values: Record<string, number>,
  definition: SliderDefinition,
  ranges: GroupRanges,
  tables: SliderTables,
): Groups {
  const out: Groups = { body: { ...base.body } };
  for (const [name, spec] of Object.entries(definition.sliders)) {
    if (!(name in values) || values[name] === sliderStart(spec, out, tables)) continue;
    const value = values[name];
    if (spec.kind === "macro") {
      for (const [key, gain] of Object.entries(spec.gains ?? {})) {
        out.body[key] = clamp(ranges, "body", key, out.body[key] + value * gain);
      }
      for (const [key, gain] of Object.entries(spec.torsoGains ?? {})) {
        out.body[key] = clamp(ranges, "body", key, out.body[key] + value * gain * out.body.torsoLength);
      }
      for (const [key, gain] of Object.entries(spec.scales ?? {})) {
        out.body[key] = clamp(ranges, "body", key, out.body[key] * (1 + value * gain));
      }
    } else if (spec.kind === "covary") {
      out.body = setHeadCount(out.body, clamp(ranges, "derived", "heads", value), spec.laws, tables);
    } else if (spec.kind === "scale") {
      const scale = clamp(ranges, "derived", "stature", value) / stature(out.body);
      for (const key of tables.MILLIMETRES) {
        if (key in out.body) out.body[key] *= scale;
      }
    } else {
      const key = spec.param.split(".")[1];
      const aim = clamp(ranges, "body", key, value);
      if (spec.carriesShares) {
        const ratio = aim / out.body.torsoLength;
        for (const other of tables.TORSO_SHARES) if (Object.hasOwn(out.body, other)) out.body[other] *= ratio;
      }
      for (const other of spec.carries ?? []) out.body[other] = (out.body[other] * aim) / out.body[key];
      out.body[key] = aim;
      for (const [other, amount] of Object.entries(spec.brings ?? {})) {
        if (Object.hasOwn(out.body, other)) continue;
        out.body[other] = tables.MILLIMETRES.includes(other) ? amount * out.body.torsoLength : amount;
      }
      for (const [other, wanted] of Object.entries(spec.lifts ?? {})) {
        const rule = tables.RULES.find(([ruleName]) => ruleName === wanted);
        if (!rule) throw new Error(`${name} lifts ${other} by a rule this sex doesn't have: ${wanted}`);
        const short = -rule[1](tables.proportions(out.body));
        if (short > 0) out.body[other] = clamp(ranges, "body", other, out.body[other] + short * out.body.torsoLength);
      }
    }
  }
  return out;
}

/** A reading's millimetres on this body. `value` is the slider's own. */
export function readingMm(reading: Reading, body: Record<string, number>, value: number): number {
  const unit = reading.of === "head" ? body.headHeight : reading.of === "torso" ? body.torsoLength : 1;
  let factor = reading.factor ?? 1;
  for (const [key, gain] of Object.entries(reading.factorBy ?? {})) factor += gain * body[key];
  return (value + (reading.plusTorso ?? 0) * body.torsoLength) * factor * unit;
}

/** Whole millimetres from 100 up, where a tenth means nothing; a true minus sign below zero. */
export function formatMm(mm: number): string {
  const size = Math.abs(mm);
  return `${mm < 0 ? "−" : ""}${size >= 100 ? size.toFixed(0) : size.toFixed(1)} mm`;
}
