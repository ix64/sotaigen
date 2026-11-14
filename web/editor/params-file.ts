// The parameter file (format in docs/systems/editor.md) and the page's own checks of parameters.
// The page doesn't read files itself: web/serve.mjs hands a file's text to sotaigen/params/file.py, the one
// reader. What the checks find are diagnostics, as sotaigen.params.rules makes them; tests/test_editor.py
// runs the two on the same bodies. No browser API here, and only types Node can strip.

import type { BodyParameters } from "/build/editor/params.generated.mjs";
import type { Diagnostic } from "./messages.ts";

/** A body's lengths as shares of its torso, which is what the rules read. */
export type Controls = Record<string, number>;

/** A rule's name, and how much room the parameters leave it: below zero it is broken. */
export type Rule = [string, (p: Controls) => number];

/** The file on disk, at the current version. Fields this format doesn't name are kept and written back. */
export interface ParameterFileJson {
  format: string;
  version: number;
  /** Whose body this is, such as "female": which parameters, ranges and rules hold. */
  sex: string;
  /** Where these parameters started from; information only. */
  base?: { name?: string };
  /** What wrote the file; information only. */
  generator?: string;
  body: Record<string, number>;
}

/**
 * A parameter file once read. `unknown` holds top-level fields this reader doesn't know. The groups
 * are only known to be objects until validate has seen them, because an old file should still open.
 */
export interface ParameterFile {
  sex: string;
  body: Record<string, number>;
  /** Where these parameters started from; information only. */
  base: { name?: string } | null;
  unknown: Record<string, unknown>;
}

/** One sex's view of the generated module: its tables, rules and conversions. */
export interface ParameterSchema {
  /** control, unit, offset, gain, and the default of an optional parameter in shares of its unit. */
  ALL_SPEC: Record<string, [string, string, number, number, number | null]>;
  REQUIRED: string[];
  MILLIMETRES: string[];
  CHAIN: string[];
  TORSO_SHARES: string[];
  NUMERIC_SLACK: number;
  STATURE_RANGE: [number, number];
  HEADS_RANGE: [number, number];
  RULES: Rule[];
  SEAM_RULES: Rule[];
  units: (body: BodyParameters) => Record<string, number>;
  proportions: (body: BodyParameters) => Controls;
  specBounds: (body: BodyParameters) => Record<string, [number, number]>;
}

/** The generated module from sotaigen/params/rules.py and sotaigen/params/spec.py: a view for each sex,
 * and the parameter file's format and version, which the page writes. */
export interface ParameterSexes {
  SEXES: Record<string, ParameterSchema>;
  FILE_FORMAT: string;
  FILE_VERSION: number;
  /** What the page writes as a file's generator: the tool and its version. */
  GENERATOR: string;
}

/** What validate finds: `errors` stop a build and `seams` are a warning. */
export interface Problems {
  errors: Diagnostic[];
  seams: Diagnostic[];
}

const error = (code: string, values: Record<string, unknown>): Diagnostic => ({ code, level: "error", values });

// The order sotaigen.params.rules gives them in: by code, then by parameter.
const ordered = (found: Diagnostic[]) =>
  found.sort((a, b) => {
    const [x, y] = [[a.code, String(a.values.key ?? "")], [b.code, String(b.values.key ?? "")]];
    return x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : x[1] < y[1] ? -1 : x[1] > y[1] ? 1 : 0;
  });

// Python writes a value that isn't a number as its repr; these are JSON values, written as JSON.
const shown = (value: unknown) => (typeof value === "string" ? `'${value}'` : JSON.stringify(value) ?? String(value));

// What is missing, unknown, not a number, or out of its range on this body's own lengths.
export function specProblems(values: Record<string, unknown>, schema: ParameterSchema): Diagnostic[] {
  const found: Diagnostic[] = [];
  for (const key of schema.REQUIRED) {
    if (!(key in values)) found.push(error("param.missing", { key }));
  }
  for (const key of Object.keys(values)) {
    if (!Object.hasOwn(schema.ALL_SPEC, key)) {
      found.push(error("param.unknown", { key }));
    } else if (typeof values[key] !== "number" || !Number.isFinite(values[key])) {
      found.push(error("param.not_number", { key, value: shown(values[key]) }));
    }
  }
  if (found.length) return ordered(found);
  const body = values as unknown as BodyParameters;
  const bounds = schema.specBounds(body);
  for (const [key, value] of Object.entries(values) as [string, number][]) {
    const [low, high] = bounds[key];
    const slack = schema.NUMERIC_SLACK * Math.max(Math.abs(low), Math.abs(high), 1);
    if (value < low - slack || value > high + slack) found.push(error("param.outside", { key, value, low, high }));
  }
  const { stature } = schema.units(body);
  const [low, high] = schema.STATURE_RANGE;
  // The stature tool scales to an end and lands a rounding error past it.
  const statureSlack = schema.NUMERIC_SLACK * Math.max(Math.abs(low), Math.abs(high), 1);
  if (stature < low - statureSlack || stature > high + statureSlack) found.push(error("param.outside", { key: "stature", value: stature, low, high }));
  return ordered(found);
}

// The rules that the applied parameters break, by how much, to three places as Python rounds it.
export function broken(rules: Rule[], p: Controls, slack: number, level: "error" | "warning" = "error"): Diagnostic[] {
  const found: Diagnostic[] = [];
  for (const [name, room] of rules) {
    const left = room(p);
    if (left < -slack) found.push({ code: `rule.${name}`, level, values: { short: Math.round(-left * 1000) / 1000 } });
  }
  return found;
}

// Layers 1 to 3 of sotaigen/params/rules.py, as the page needs them: at every slider event. `seams` are a
// warning and the rest stop a build, as in Python.
export function validate(body: Record<string, number>, schema: ParameterSchema): Problems {
  const ranges = specProblems(body, schema);
  if (ranges.length) return { errors: ranges, seams: [] };
  const p = schema.proportions(body as unknown as BodyParameters);
  return { errors: broken(schema.RULES, p, schema.NUMERIC_SLACK), seams: broken(schema.SEAM_RULES, p, schema.NUMERIC_SLACK, "warning") };
}

// The file for a body, in the format and version the generated module says, fields this format doesn't
// name first and written back as they came.
export function writeParameterFile(
  { sex, body, base, unknown }: Omit<ParameterFile, "unknown"> & { unknown?: Record<string, unknown> },
  { FILE_FORMAT, FILE_VERSION }: Pick<ParameterSexes, "FILE_FORMAT" | "FILE_VERSION">,
  generator?: string,
): string {
  const file: Partial<ParameterFileJson> = { format: FILE_FORMAT, version: FILE_VERSION, sex };
  if (base) file.base = base;
  if (generator) file.generator = generator;
  file.body = body;
  return `${JSON.stringify({ ...(unknown ?? {}), ...file }, null, 2)}\n`;
}
