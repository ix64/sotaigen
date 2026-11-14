// Which parts of a slider's range the rules refuse, given where everything else stands. These are the
// checks the page runs anyway (validate), asked in advance. No browser API here, and only types Node
// can strip, so that tests/test_editor.py can run this against the Python.

import type { Diagnostic } from "./messages.ts";
import type { ParameterSchema } from "./params-file.ts";
import { validate } from "./params-file.ts";
import type { GroupRanges, Groups, SliderDefinition } from "./sliders.ts";
import { applySliders, sliderBounds } from "./sliders.ts";

/** `blocked` won't be made at all. `warned` will, and may miss at a seam. */
export type ZoneLevel = "blocked" | "warned";

/** A stretch of the track, as shares of it, 0 to 1, and what it breaks: rules by name (sotaigen.params.rules),
 * anything else by its code and parameter. */
export interface Zone {
  from: number;
  to: number;
  level: ZoneLevel;
  reasons: string[];
}

const SAMPLES = 32;
// Enough halvings to land within a slider's own step, which is 1/400 of the track.
const HALVINGS = 5;
// Narrower than a slider's step, it's nothing anyone could see or land on.
const NARROWEST = 1 / 400;

// What it breaks, without by how much, which changes along the track.
const reason = ({ code, values }: Diagnostic) => (code.startsWith("rule.") ? code.slice(5) : `${code} ${values.key ?? ""}`.trim());

export function sliderZones({
  name,
  base,
  sliders,
  definition,
  ranges,
  schema,
}: {
  name: string;
  base: Groups;
  sliders: Record<string, number>;
  definition: SliderDefinition;
  ranges: GroupRanges;
  schema: ParameterSchema;
}): Zone[] {
  const [low, high] = sliderBounds(definition.sliders[name], ranges);
  const read = (share: number) => {
    const moved = { ...sliders, [name]: low + (high - low) * share };
    const { body } = applySliders(base, moved, definition, ranges, schema);
    const { errors, seams } = validate(body, schema);
    return { errors: errors.map(reason), seams: seams.map(reason) };
  };
  const rows = Array.from({ length: SAMPLES + 1 }, (_, index) => read(index / SAMPLES));

  // What's broken wherever this slider stands isn't this slider's doing, and it can't mend it. Counting
  // it would paint the whole track, and say nothing about where to go.
  const everywhere = (kind: "errors" | "seams") =>
    new Set(rows[0][kind].filter((line) => rows.every((row) => row[kind].includes(line))));
  const alwaysErrors = everywhere("errors");
  const alwaysSeams = everywhere("seams");
  const judge = ({ errors, seams }: { errors: string[]; seams: string[] }) => {
    const refused = errors.filter((line) => !alwaysErrors.has(line));
    if (refused.length) return { level: "blocked" as ZoneLevel, reasons: refused };
    const warned = seams.filter((line) => !alwaysSeams.has(line));
    return warned.length ? { level: "warned" as ZoneLevel, reasons: warned } : { level: null, reasons: [] };
  };

  // Where the level changes between two samples next to each other.
  const edge = (from: number, to: number) => {
    const level = judge(read(from)).level;
    let [inside, outside] = [from, to];
    for (let i = 0; i < HALVINGS; i++) {
      const middle = (inside + outside) / 2;
      if (judge(read(middle)).level === level) inside = middle;
      else outside = middle;
    }
    return (inside + outside) / 2;
  };

  const judged = rows.map(judge);
  const zones: Zone[] = [];
  let start = 0;
  for (let index = 1; index <= judged.length; index++) {
    if (index < judged.length && judged[index].level === judged[start].level) continue;
    const { level } = judged[start];
    if (level) {
      const from = start === 0 ? 0 : edge((start - 1) / SAMPLES, start / SAMPLES);
      const to = index === judged.length ? 1 : edge((index - 1) / SAMPLES, index / SAMPLES);
      const reasons = [...new Set(judged.slice(start, index).flatMap((row) => row.reasons))];
      if (to - from >= NARROWEST) zones.push({ from, to, level, reasons });
    }
    start = index;
  }
  return zones;
}
