// What the page says about what Python and the page's own checks find. Every finding is a diagnostic,
// {code, level, values} (sotaigen/diagnostics.py), and is put into words only here, where it is shown, by
// the messages of web/messages; a code not listed shows as its code and values, so a new check still says
// what's wrong. No browser API here, no import of the compiled messages, and only types Node can strip:
// tests/test_editor.py runs this file under Node without the npm packages, with a `say` that reads
// web/messages/zh.json.

import type { RegionId } from "./regions.ts";

/** A message of web/messages in the page's language, with its inputs. */
export type Say = (key: string, inputs?: Record<string, string | number>) => string;

/** One finding, as sotaigen.diagnostics.Diagnostic.json writes it and params-file.ts makes it. */
export interface Diagnostic {
  code: string;
  level: "error" | "broken" | "warning";
  values: Record<string, any>;
}

/** What a reply says went wrong: a diagnostic, or the text of an error that isn't one. */
export type Found = Diagnostic | string;

export interface ProblemMessage {
  text: string;
  /** The group to go to, to do something about it. */
  panel?: RegionId;
}

// Each rule's message and its group, by the rule's name in sotaigen/params/rules.py.
const RULES: Record<string, [string, RegionId]> = {
  shoulders_clear_chest: ["rule_shoulders_clear_chest", "shoulder"],
  arm_clear_ribs: ["rule_arm_clear_ribs", "arm"],
  arm_thick_high_shoulder: ["rule_arm_thick_high_shoulder", "arm"],
  arm_thick_narrow_shoulders: ["rule_arm_thick_narrow_shoulders", "arm"],
  torso_short: ["rule_torso_short", "proportion"],
  neck_thick_head: ["rule_neck_thick_head", "neck"],
  neck_thin_head: ["rule_neck_thin_head", "neck"],
  torso_long_head: ["rule_torso_long_head", "proportion"],
  thigh_thin_hip: ["rule_thigh_thin_hip", "leg"],
  foot_long_knee: ["rule_foot_long_knee", "leg"],
  ankle_low_foot: ["rule_ankle_low_foot", "leg"],
  ankle_high_foot: ["rule_ankle_high_foot", "leg"],
  knee_wide_calf: ["rule_knee_wide_calf", "leg"],
  hip_shallow_thigh: ["rule_hip_shallow_thigh", "hip"],
  neck_thin_length: ["rule_neck_thin_length", "neck"],
  wrist_thin_arm: ["rule_wrist_thin_arm", "arm"],
  pectoral_under_breast_thin: ["rule_pectoral_under_breast_thin", "shoulder"],
  hand_big_forearm: ["rule_hand_big_forearm", "arm"],
  hand_big_forearm_arm: ["rule_hand_big_forearm_arm", "arm"],
  wrist_thick_hand: ["rule_wrist_thick_hand", "arm"],
  wrist_thick_hand_arm: ["rule_wrist_thick_hand_arm", "arm"],
  ankle_thick_foot: ["rule_ankle_thick_foot", "leg"],
  ankle_thick_foot_knee: ["rule_ankle_thick_foot_knee", "leg"],
  ankle_thin_foot: ["rule_ankle_thin_foot", "leg"],
  foot_long_body: ["rule_foot_long_body", "leg"],
};

// sotaigen.form.checks.field_failures, by name: the message is surface_<name>.
const SURFACE: Record<string, RegionId> = {
  waistBreadth: "waist",
  hipBreadth: "hip",
  hipTissueMidline: "hip",
  legsPart: "leg",
  wristCentre: "arm",
  kneeCentre: "leg",
  kneeMidline: "leg",
  calfLean: "leg",
};

// The seams' groups: the message is seam_<part>.
const PART: Record<string, RegionId> = { hand: "arm", foot: "leg" };

// The parts of a body a wrap's flips and crossings are named by (sotaigen.form.wrap.body_part): groups of the
// page, each worded by its group_<id>. A part not listed shows as written.
const BROKEN_PARTS = new Set<string>(["neck", "shoulder", "chest", "arm", "waist", "hip", "glute", "leg", "quick", "proportion", "head"]);

/** {neck: 2, shoulder: 1} in words, and the group of the first, which has the most. */
function brokenParts(at: Record<string, number> | undefined, say: Say): { where: string; panel?: RegionId } {
  const rows = Object.entries(at ?? {});
  if (!rows.length) return { where: "" };
  const worded = rows.map(([part, count]) => `${BROKEN_PARTS.has(part) ? say(`group_${part}`) : part} ${count}`);
  const first = rows[0][0];
  return { where: say("failure_where", { parts: worded.join(say("join_list")) }), panel: BROKEN_PARTS.has(first) ? (first as RegionId) : undefined };
}


// A failed contact bake's message and group, by the group of correctives it was for.
const CONTACT: Record<string, [string, RegionId]> = {
  axilla: ["failure_contact_shoulder", "arm"],
  groin: ["failure_contact_pelvic", "leg"],
  hang: ["failure_contact_hang", "arm"],
};

const named = (key: string, labels: Record<string, string>, say: Say) =>
  labels[key] ? say("param_named", { label: labels[key], key }) : key;

const short = (value: number) => String(Number(value.toPrecision(4)));
const tenths = (value: number) => value.toFixed(1);

/** One rule's message, by its name, without how far it's missed by. */
export const translateRule = (name: string, say: Say) => (RULES[name] ? say(RULES[name][0]) : name);

/** A diagnostic in the page's words, and the group to go to. A seam rule's warning reads as a note. */
export function translate(found: Found, labels: Record<string, string>, say: Say): ProblemMessage {
  if (typeof found === "string") return { text: found };
  const { code, level, values: v } = found;
  const out = (key: string, inputs: Record<string, string | number> = {}, panel?: RegionId): ProblemMessage => ({ text: say(key, inputs), panel });
  const [kind, name] = code.split(/\.(.*)/s);
  if (kind === "rule" && RULES[name]) {
    const [key, panel] = RULES[name];
    const text = say("problem_rule", { text: say(key), by: v.short.toFixed(3) });
    return { text: level === "warning" ? say("problem_note", { text }) : text, panel };
  }
  if (kind === "surface" && SURFACE[name]) return out(`surface_${name}`, {}, SURFACE[name]);
  if (code === "param.missing") return out("problem_missing", { name: named(v.key, labels, say) });
  if (code === "param.unknown") return out("problem_unknown", { name: v.key });
  if (code === "param.not_number") return out("problem_not_number", { name: named(v.key, labels, say), value: v.value });
  if (code === "param.outside") return out("problem_outside", { name: named(v.key, labels, say), value: short(v.value), low: short(v.low), high: short(v.high) });
  if (code === "param.sex") return out("file_no_sex", { sex: v.sex });
  if (code === "target.missed") return out("failure_target", { name: named(v.key, labels, say), target: tenths(v.asked), actual: tenths(v.reached) });
  if (code === "wrap.flipped" || code === "wrap.midline") {
    const { where, panel } = brokenParts(v.at, say);
    return { text: say(code === "wrap.flipped" ? "failure_flipped" : "failure_midline", { count: v.count }) + where, panel };
  }
  if (code === "skin.open") return out("failure_open_skin", { count: v.edges });
  if (code === "foot.off_ground") return out("failure_ground", { mm: v.mm }, "leg");
  if (code === "seam.crease") return out("failure_crease", { part: say(`seam_${v.part}`), angle: v.degrees, limit: v.limit }, PART[v.part]);
  if (code === "seam.miss") return out("failure_miss", { part: say(`seam_${v.part}`), share: v.share.toFixed(2), limit: v.limit }, PART[v.part]);
  if (code === "contact.failed" && CONTACT[v.group]) return out(CONTACT[v.group][0], { detail: v.detail }, CONTACT[v.group][1]);
  if (code === "file.not_json") return out("file_not_json", { detail: v.detail });
  if (code === "file.not_object") return out("file_not_object");
  if (code === "file.format") return out("file_format", { format: v.format });
  if (code === "file.version") return out("file_bad_version", { version: v.version });
  if (code === "file.newer") return out("file_newer", { version: v.version, latest: v.latest });
  if (code === "file.no_sex") return out("file_sex_missing");
  if (code === "file.not_parameters") return out("file_not_parameters", { group: "body" });
  return { text: Object.keys(v ?? {}).length ? `${code} ${JSON.stringify(v)}` : code };
}

/** Several findings in the page's words, one line each. */
export const translateAll = (found: Found[], labels: Record<string, string>, say: Say) =>
  found.map((item) => translate(item, labels, say).text);

/** What sotaigen.mend.search found for a refused body: a parameter's move that ends it, or none. */
export interface Mend {
  tried: number;
  total: number;
  key?: string;
  part?: string;
  before?: number;
  after?: number;
  unit?: string;
}

export function translateMend(mend: Mend, labels: Record<string, string>, say: Say): ProblemMessage {
  if (!mend.key) return { text: say("mend_none", { total: mend.total }) };
  const unit = mend.unit ? ` ${mend.unit}` : "";
  return {
    text: say("mend_found", { name: named(mend.key, labels, say), before: `${mend.before}${unit}`, after: `${mend.after}${unit}` }),
    panel: mend.part && BROKEN_PARTS.has(mend.part) ? (mend.part as RegionId) : undefined,
  };
}

/** Every message key this file can ask for, so tests can check the catalogues have them. */
export const MESSAGE_KEYS = [
  ...Object.values(RULES).map(([key]) => key),
  ...Object.keys(SURFACE).map((name) => `surface_${name}`),
  ...Object.keys(PART).map((part) => `seam_${part}`),
  ...Object.values(CONTACT).map(([key]) => key),
  ...[...BROKEN_PARTS].map((part) => `group_${part}`),
  "failure_where",
  "mend_found",
  "mend_none",
  "mend_preparing",
  "mend_running",
  "mend_failed",
  "mend_dismiss",
];
