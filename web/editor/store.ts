// The page's state and everything that changes it. Every body on screen is a GLB made
// by the local Python worker a few seconds after the last release; until it arrives the last one stays.
// What is asked for and when belongs to rebuild.ts, what parameters mean to sliders.ts and
// params-file.ts. React only reads this, through useEditor.

import { useSyncExternalStore } from "react";
import type { BodyParameters } from "/build/editor/params.generated.mjs";
import { type LoadedBody, loadBody } from "../runtime/body.mjs";
import { type ParameterFile, type ParameterSexes, type Problems, validate, writeParameterFile } from "./params-file.ts";
import { Rebuilder } from "./rebuild.ts";
import definitionJson from "../../sotaigen/data/sliders.json";
import { applySliders, formatMm, type GroupRanges, headCount, readingMm, type Slider, type SliderDefinition, sliderBounds, sliderStart, stature } from "./sliders.ts";
import { sliderZones } from "./limits.ts";
import { type Diagnostic, type Found, type Mend, type ProblemMessage, translate, translateAll, translateMend } from "./messages.ts";
import { listOf, say, sliderLabel, sliderReading } from "./text";
import { isTouchPose, REGION_FOCUS, REGION_STAND_INS, type RegionId, TOUCH_POSES } from "./regions";
import { isLook } from "./looks";
import { type CropName, type LookName, type ViewName, Viewport } from "./viewport";

export const definition = definitionJson as unknown as Omit<SliderDefinition, "groups"> & {
  groups: { id: RegionId; sliders: string[] }[];
};

/** A body and whose it is, such as "female": its sex says which parameters, ranges and rules hold. */
export type BodyParams = { sex: string; body: Record<string, number> };
export type StatusKind = "good" | "warn" | "bad" | "busy";
export type Theme = "light" | "dark";
/** What the panel shows: a region, or every changed slider. */
export type PanelView = RegionId | "changes";

/** What /api/body says about a finished body: serve.mjs and sotaigen/generation.py answer. */
export interface AbdominalReport {
  reference: Record<string, number>;
  measured: Record<string, number>;
  torsoMm: number;
}

export interface BodyReport {
  abdomen?: AbdominalReport;
  thigh?: AbdominalReport & { limits: Record<string, [number, number]> };
  key: string;
  glb: string;
  cached: boolean;
  seconds?: Record<string, number>;
  /** What the masses couldn't reach, by parameter, as [asked, reached] in mm. */
  targets?: Record<string, [number, number]>;
  /** Invariants the finished body breaks. It is still shown; only `broken` keeps it from being exported. */
  failures: Diagnostic[];
  /** Why this mesh may not be exported: sotaigen.output.quality.broken's diagnostics. It is still shown. */
  broken?: Diagnostic[];
  /** Which pose correctives the GLB has: the sculpted ones, or the IPC contact tables too. */
  contact?: "sculpted" | "ipc";
}

/** Where things stand with a GLB that has its IPC contact tables, which takes minutes to make. */
export interface ContactExport {
  phase: "running" | "done" | "failed";
  /** The table being baked, as the worker says: "shoulder", "pelvic" or "hang". */
  stage: string | null;
  /** When the server started it, in ms since the epoch. */
  startedAt: number;
  /** What went wrong, or which tables fell back; empty when all went well. */
  messages: string[];
}

interface Built {
  input: BodyParams;
  report: BodyReport;
  loaded: LoadedBody;
  requestMs: number;
  loadMs: number;
}

interface Timing {
  key: string;
  cached: boolean;
  waitMs: number;
  requestMs: number;
  loadMs: number;
  seconds?: Record<string, number>;
}

/** What the page says about where things stand. */
/** Where things stand with a body's rigged .blend, which takes seconds to make. */
export interface BlendExport {
  phase: "running" | "done" | "failed";
  /** When it was asked for, in ms since the epoch. */
  startedAt: number;
  /** What went wrong; null when all went well. */
  message: string | null;
}

/** Where things stand with the contact pose on show, which the server solves for each body. */
export interface TouchJob {
  phase: "running" | "failed" | "short";
  message: string | null;
}

/** What the server solved for one body: each contact pose's angles and how it came out. */
interface TouchAnswer {
  poses: Record<string, Record<string, [number, number, number]>>;
  reports: Record<string, { reached: boolean; arrivalMm: number; place: number }>;
}

export interface Status {
  line: string;
  kind: StatusKind;
  problems: (ProblemMessage & { seam: boolean })[];
  hint: string;
  numbers: string;
  /** How far along the wait is, from the last generation's length; null when there's nothing to go on. */
  progress: number | null;
}

/** What undo goes back to: where the parameters started and what the sliders did to them. */
interface Snapshot {
  name: string | null;
  base: BodyParams;
  provenance: { name?: string } | null;
  unknown: Record<string, unknown>;
  notes: string[];
  sliders: Record<string, number>;
}

/** The numbers of the body on screen. */
export interface BodyStats {
  vertices: number;
  triangles: number;
  bones: number;
  parts: number;
  partVertices: number;
  bytes: number;
  fetchMs: number;
  parseMs: number;
}

export const FACE_BONES = {
  jaw: { min: 0, max: 27 },
  gazeAcross: { min: -49, max: 49 },
  gazeUp: { min: -38, max: 38 },
} as const;

export type FaceBones = Record<keyof typeof FACE_BONES, number>;

const RESTING_FACE: FaceBones = { jaw: 0, gazeAcross: 0, gazeUp: 0 };

/** The sliders as a pose: both eyes look the same way, so neither mirrors the other. */
function facePose({ jaw, gazeAcross, gazeUp }: FaceBones) {
  const eye: [number, number, number] = [gazeUp, 0, gazeAcross];
  return {
    ...(jaw ? { jaw: [jaw, 0, 0] as [number, number, number] } : {}),
    ...(gazeAcross || gazeUp ? { leftEye: eye, rightEye: eye } : {}),
  };
}

/** Looking for a parameter that ends the breakage of the body a refused export was for. */
export interface MendSearch {
  phase: "running" | "done" | "failed";
  /** How far it is, as "3/35"; null until the worker has said. */
  stage: string | null;
  startedAt: number;
  /** Once done, what it found, in the page's words, and the group to go to. */
  message: ProblemMessage | null;
}

export interface EditorState extends Snapshot {
  /**
   * The sex and group the start picker has chosen. After an import the sex is the file's and
   * the group the typical body it says it came from, or "" when that is not one of its sex's.
   */
  picks: { sex: string; group: string };
  current: BodyParams;
  problems: Problems;
  body: LoadedBody | null;
  report: BodyReport | null;
  frameNext: boolean;
  waitStart: number;
  undrawn: Timing | null;
  pose: string;
  poses: string[];
  look: LookName;
  inspecting: boolean;
  driven: boolean;
  partNames: string[];
  hiddenParts: string[];
  morphNames: string[];
  morphs: Record<string, number>;
  /** The face's bones in degrees: jaw, gazeAcross (to the wearer's left), gazeUp. */
  faceBones: FaceBones;
  stats: BodyStats | null;
  panel: PanelView;
  query: string;
  follow: boolean;
  comparing: boolean;
  canCompare: boolean;
  exporting: boolean;
  /** Why the last click on Export GLB saved nothing; null when it did, or when put away. */
  glbRefusal: string | null;
  /** The search for what ends a refused body's breakage; null when none, or when put away. */
  mendSearch: MendSearch | null;
  contactExport: ContactExport | null;
  blendExport: BlendExport | null;
  touchJob: TouchJob | null;
  /** Whether this machine has Blender, as the server says. Null until it has. */
  blender: boolean | null;
  importError: string | null;
  status: Status;
  canUndo: boolean;
  canRedo: boolean;
  theme: Theme;
  hasBody: boolean;
}

const HISTORY = 100;
// From release to the body drawn: the speed target in docs/ARCHITECTURE.md.
const SPEED_TARGET_MS = 10000;
// Moves of one slider this close together are one step of undo, as a drag or held arrow key is.
const SAME_STEP_MS = 800;
const PREFERENCES = "sotaigen.editor";

function readPreferences(): { theme?: Theme; follow?: boolean } {
  try {
    return JSON.parse(localStorage.getItem(PREFERENCES) ?? "{}");
  } catch {
    return {};
  }
}

function writePreferences(preferences: { theme: Theme; follow: boolean }) {
  try {
    localStorage.setItem(PREFERENCES, JSON.stringify(preferences));
  } catch {
    // A convenience. Without storage the page starts on the defaults next time.
  }
}

const seconds = (ms: number) => (ms / 1000).toFixed(1);

export function formatValue(value: unknown, span: number) {
  // A file can leave a parameter out or say something else. validate has already said so.
  if (typeof value !== "number" || !Number.isFinite(value)) return "?";
  return span >= 100 ? value.toFixed(0) : span >= 2 ? value.toFixed(2) : value.toFixed(3);
}

/** Where the page starts, from the address. */
export interface EditorStart {
  body?: string;
  pose?: string;
  look?: string;
  panel?: string;
  inspect?: boolean;
}

export function createEditor({
  presets,
  sexes,
  start = {},
}: {
  presets: Record<string, BodyParams>;
  sexes: ParameterSexes;
  start?: EditorStart;
}) {
  const schemaOf = (sex: string) => {
    if (!Object.hasOwn(sexes.SEXES, sex)) throw new Error(`there is no sex ${JSON.stringify(sex)}`);
    return sexes.SEXES[sex];
  };
  const firstSex = Object.values(presets)[0]?.sex ?? Object.keys(sexes.SEXES)[0];
  // The tables of the body's sex, chosen again whenever the parameters change.
  let schema = schemaOf(firstSex);
  // A slider's limits come from where it started, so they stay put while the body is being edited. A file
  // can leave parameters out; the first preset of its sex stands in, for the limits only.
  const anyBodyOf = (sex: string) => Object.values(presets).find((preset) => preset.sex === sex)?.body ?? {};
  const rangesFor = (base: BodyParams): GroupRanges => ({
    body: schema.specBounds({ ...anyBodyOf(base.sex), ...base.body } as BodyParameters),
    derived: { heads: schema.HEADS_RANGE, stature: schema.STATURE_RANGE },
  });
  let ranges = rangesFor({ sex: firstSex, body: anyBodyOf(firstSex) });
  const viewport = new Viewport();
  const preferences = readPreferences();
  const listeners = new Set<() => void>();
  const log: { timing: Timing[] } = { timing: [] };
  const profiles = new Map<string, AbdominalReport>();
  const profileReferences = new Map<string, Record<string, number>>();
  const thighProfiles = new Map<string, NonNullable<BodyReport["thigh"]>>();
  const thighBounds = new Map<string, [number, number]>();
  const thighNames = ["outerThighArc"];
  function thighLimitKey(pair: { body: Record<string, number> }, name: string) {
    return name + profileKey({ body: Object.fromEntries(Object.entries(pair.body).filter(([key]) => key !== name)) });
  }
  function profileKey(pair: { body: Record<string, number> }, reference = false) {
    return JSON.stringify(
      Object.entries(pair.body)
        .filter(([key]) => !(reference && ["abdomenFrontArc", "lumbarHollow", ...thighNames].includes(key)))
        .sort(([a], [b]) => a.localeCompare(b)),
    );
  }
  const undoStack: Snapshot[] = [];
  const redoStack: Snapshot[] = [];
  let lastStep: { key: string; at: number } | null = null;

  const state = {
    name: null,
    base: { sex: firstSex, body: {} },
    provenance: null,
    unknown: {},
    notes: [],
    sliders: {},
    picks: { sex: firstSex, group: "" },
    current: { sex: firstSex, body: {} },
    problems: { errors: [], seams: [] },
    body: null,
    report: null,
    frameNext: true,
    waitStart: 0,
    undrawn: null,
    pose: "rest",
    poses: [],
    look: "toon",
    inspecting: false,
    driven: true,
    partNames: [],
    hiddenParts: [],
    morphNames: [],
    morphs: {},
    faceBones: RESTING_FACE,
    stats: null,
    panel: "proportion",
    query: "",
    follow: preferences.follow ?? true,
    comparing: false,
    canCompare: false,
    exporting: false,
    glbRefusal: null,
    mendSearch: null,
    contactExport: null,
    blendExport: null,
    touchJob: null,
    blender: null,
    importError: null,
    status: { line: say("status_loading"), kind: "busy", problems: [], hint: "", numbers: "", progress: null },
    canUndo: false,
    canRedo: false,
    theme: preferences.theme === "dark" ? "dark" : "light",
    hasBody: false,
  } as EditorState;

  // The search belongs to the parameters it was asked about, and each try is a whole wrap: moving on from
  // them, or putting it away, kills it on the server.
  let mendAsked: { key: string; abort: AbortController; poll: ReturnType<typeof setInterval> } | null = null;

  function emit() {
    for (const listener of listeners) listener();
  }

  function set(changes: Partial<EditorState>) {
    Object.assign(state, changes);
    emit();
  }

  const touchAnswers = new Map<string, TouchAnswer>();
  const touchKey = ({ sex, body }: BodyParams) => JSON.stringify([sex, body]);
  let touchAsked: AbortController | null = null;
  let touchAskedKey = "";
  // The parameters of the body on show, which a contact pose is solved for.
  let shownParams: BodyParams | null = null;

  async function build({ sex, body }: BodyParams, signal: AbortSignal): Promise<Built> {
    const started = performance.now();
    let reply: any;
    try {
      const response = await fetch("/api/body", { method: "POST", body: JSON.stringify({ sex, body }), signal });
      reply = await response.json();
    } catch (error) {
      if (signal.aborted) throw error;
      throw Object.assign(new Error(say("error_unreachable", { error: String(error) })), { unavailable: true });
    }
    if (reply.superseded) throw new Error(say("error_superseded"));
    if (!reply.ok) {
      const lines: Found[] = reply.problems ?? [reply.error ?? say("error_no_reason")];
      throw Object.assign(new Error(translateAll(lines, sliderLabels(), say).join("\n")), { lines, unavailable: Boolean(reply.unavailable) });
    }
    const answered = performance.now();
    const loaded = await loadBody(reply.glb);
    return { input: { sex, body }, report: reply, loaded, requestMs: answered - started, loadMs: performance.now() - answered };
  }

  // The new body takes the old one's place, in the pose and look that are on.
  function land({ input, report, loaded, requestMs, loadMs }: Built) {
    if (report.abdomen) {
      profiles.set(profileKey(input), report.abdomen);
      // The report's measures are shares of the torso; the sliders are in millimetres.
      const { reference, torsoMm } = report.abdomen;
      profileReferences.set(
        profileKey(input, true),
        { ...profileReferences.get(profileKey(input, true)), ...Object.fromEntries(Object.entries(reference).map(([key, share]) => [key, share * torsoMm])) },
      );
      if (profiles.size > 100) profiles.delete(profiles.keys().next().value!);
      if (profileReferences.size > 100) profileReferences.delete(profileReferences.keys().next().value!);
    }
    if (report.thigh) {
      thighProfiles.set(profileKey(input), report.thigh);
      profileReferences.set(profileKey(input, true), { ...profileReferences.get(profileKey(input, true)), ...Object.fromEntries(Object.entries(report.thigh.reference).map(([key, share]) => [key, share * report.thigh!.torsoMm])) });
      for (const name of thighNames) {
        const key = definition.sliders[name].measured!;
        const [low, high] = report.thigh.limits[key].map(share => share * report.thigh!.torsoMm);
        // Keep rounded millimetre inputs off the exact edge of the solver's domain.
        const margin = Math.min(0.1, (high - low) / 100);
        thighBounds.set(thighLimitKey(input, name), [low + margin, high - margin]);
      }
      if (thighProfiles.size > 100) thighProfiles.delete(thighProfiles.keys().next().value!);
      if (thighBounds.size > 200) {
        thighBounds.delete(thighBounds.keys().next().value!);
        thighBounds.delete(thighBounds.keys().next().value!);
      }
    }
    const keepPrevious = Boolean(state.body) && !state.frameNext && state.report?.key !== report.key;
    viewport.land(loaded, keepPrevious);
    state.body = loaded;
    state.report = report;
    state.hasBody = true;
    state.poses = [...Object.keys(loaded.info.checkPoses), ...TOUCH_POSES];
    shownParams = input;
    if (!state.poses.includes(state.pose)) state.pose = "rest";
    state.partNames = loaded.parts.map((part) => part.userData.name);
    state.morphNames = loaded.morphNames;
    const geometry = loaded.mesh.geometry;
    state.stats = {
      vertices: geometry.attributes.position.count,
      triangles: (geometry.index?.count ?? 0) / 3,
      bones: loaded.mesh.skeleton.bones.length,
      parts: loaded.parts.length,
      partVertices: loaded.parts.reduce((sum, part) => sum + part.geometry.attributes.position.count, 0),
      bytes: loaded.bytes,
      fetchMs: loaded.fetchMs,
      parseMs: loaded.parseMs,
    };
    // A new body is a new shape on the same face: what's being looked at stays as it is.
    showPose();
    viewport.applyInspection(state.hiddenParts, state.morphs);
    viewport.setLook(state.look);
    if (state.frameNext) {
      viewport.frameFirst();
      if (state.follow && state.panel !== "changes" && REGION_FOCUS[state.panel]) {
        viewport.focus(state.panel, { instant: true });
      }
      state.frameNext = false;
    }
    state.undrawn = { key: report.key, cached: report.cached, waitMs: 0, requestMs, loadMs, seconds: report.seconds };
  }

  const rebuilder = new Rebuilder<BodyParams, Built>({ build, land, notify: describe });

  function waitingText() {
    const elapsed = (performance.now() - state.waitStart) / 1000;
    const generated = log.timing.findLast((row) => !row.cached);
    const now = elapsed.toFixed(0);
    return generated ? say("status_generating_last", { elapsed: now, last: seconds(generated.waitMs) }) : say("status_generating", { elapsed: now });
  }

  function waitingProgress() {
    const generated = log.timing.findLast((row) => !row.cached);
    if (!generated) return null;
    return Math.min(0.95, (performance.now() - state.waitStart) / generated.waitMs);
  }

  function lastTiming() {
    const last = log.timing.at(-1);
    if (!last) return "";
    if (last.cached) return say("timing_cached", { seconds: (last.waitMs / 1000).toFixed(2) });
    const s = last.seconds;
    // docs/ARCHITECTURE.md: a release has its body within 10 s. A busy machine can miss it; say so.
    const over = last.waitMs > SPEED_TARGET_MS ? `\n${say("timing_over", { target: SPEED_TARGET_MS / 1000 })}` : "";
    if (!s) return say("timing_waited", { seconds: seconds(last.waitMs) }) + over;
    const parts = (s.head + s.parts).toFixed(1);
    const [wrap, rig, total] = [s.wrap, s.rig, s.total].map((x) => x.toFixed(1));
    return say("timing_detail", { waited: seconds(last.waitMs), total, wrap, rig, parts }) + over;
  }

  function sliderLabels() {
    const labels: Record<string, string> = {};
    for (const [name, slider] of Object.entries(definition.sliders)) {
      if (slider.kind !== "macro") labels[slider.param.split(".")[1]] = sliderLabel(name);
    }
    return labels;
  }

  function describe() {
    const { errors, seams } = state.problems;
    const { phase, error } = rebuilder;
    const canCompare = Boolean(viewport.previous) && phase === "idle";
    if (!canCompare && viewport.comparing) {
      viewport.showBefore(false);
      state.comparing = false;
    }
    const failures = phase === "idle" ? (state.report?.failures ?? []) : [];
    const buildError = error as (Error & { lines?: Found[]; unavailable?: boolean }) | null;
    const labels = sliderLabels();
    const problems = [
      ...errors.map((found) => ({ ...translate(found, labels, say), seam: false })),
      ...(phase === "failed" && buildError ? (buildError.lines ?? [buildError.message]) : []).map((found) => ({
        ...translate(found, labels, say),
        seam: false,
      })),
      ...failures.map((found) => ({ ...translate(found, labels, say), seam: false })),
      ...seams.map((found) => ({ ...translate(found, labels, say), seam: true })),
      ...state.notes.map((text) => ({ text, seam: true })),
    ];
    const previous = state.body ? "_previous" : "";
    let line: string;
    let kind: StatusKind;
    if (state.importError) [line, kind] = [say("status_import_failed", { error: state.importError }), "bad"];
    else if (phase === "invalid") [line, kind] = [say(`status_invalid${previous}`), "bad"];
    else if (phase === "failed" && buildError?.unavailable) [line, kind] = [say("status_unreachable"), "bad"];
    else if (phase === "failed") [line, kind] = [say(`status_failed${previous}`), "bad"];
    else if (phase === "adjusting") [line, kind] = [say("status_adjusting"), "warn"];
    else if (phase === "generating") [line, kind] = [waitingText(), "busy"];
    else if (state.report?.broken?.length) [line, kind] = [say("status_broken"), "bad"];
    else if (failures.length) [line, kind] = [say("status_failures"), "bad"];
    else if (seams.length) [line, kind] = [say("status_seams"), "warn"];
    else [line, kind] = [say("status_done"), "good"];
    const hint =
      phase === "failed" && buildError?.unavailable
        ? `${say("status_hint_unreachable")}\nuv sync --all-extras\npnpm run editor`
        : phase === "failed"
          ? say("status_hint_retry")
          : say("status_hint_idle");
    state.canCompare = canCompare;
    set({
      status: {
        line,
        kind,
        problems,
        hint,
        numbers: phase === "idle" ? lastTiming() : "",
        progress: phase === "generating" ? waitingProgress() : null,
      },
    });
  }

  function parametersChanged(options?: { now?: boolean }) {
    schema = schemaOf(state.base.sex);
    ranges = rangesFor(state.base);
    state.current = { sex: state.base.sex, ...applySliders(state.base, state.sliders, definition, ranges, schema) };
    state.problems = validate(state.current.body, schema);
    if (mendAsked && mendAsked.key !== mendKey(state.current)) stopMendSearch();
    // Another sex can lack the group on screen.
    if (shownPanel(state.panel) !== state.panel) set({ panel: shownPanel(state.panel) });
    state.waitStart = performance.now();
    state.importError = null;
    rebuilder.want(state.current, state.problems.errors.length === 0, options);
  }

  // History

  const snapshot = (): Snapshot => ({
    name: state.name,
    base: state.base,
    provenance: state.provenance,
    unknown: state.unknown,
    notes: state.notes,
    sliders: state.sliders,
  });

  function checkpoint(key?: string) {
    const now = performance.now();
    if (key && lastStep?.key === key && now - lastStep.at < SAME_STEP_MS) {
      lastStep.at = now;
      return;
    }
    undoStack.push(snapshot());
    if (undoStack.length > HISTORY) undoStack.shift();
    redoStack.length = 0;
    lastStep = key ? { key, at: now } : null;
    state.canUndo = true;
    state.canRedo = false;
  }

  function restore(to: Snapshot) {
    const sourceChanged = to.base !== state.base;
    Object.assign(state, to);
    lastStep = null;
    state.canUndo = undoStack.length > 0;
    state.canRedo = redoStack.length > 0;
    state.picks = picksFor(to);
    if (sourceChanged) {
      viewport.clearPrevious();
      state.frameNext = true;
    }
    parametersChanged({ now: true });
    if (sourceChanged && rebuilder.phase === "idle") state.frameNext = false;
  }

  function undo() {
    const to = undoStack.pop();
    if (!to) return;
    redoStack.push(snapshot());
    restore(to);
  }

  function redo() {
    const to = redoStack.pop();
    if (!to) return;
    undoStack.push(snapshot());
    restore(to);
  }

  // Where parameters come from

  function beginSource() {
    viewport.clearPrevious();
    state.frameNext = true;
    state.sliders = {};
    parametersChanged({ now: true });
    // Loading exactly the displayed parameters does not cause another landing.
    if (rebuilder.phase === "idle") state.frameNext = false;
  }

  /**
   * A typical body's group: its name is <sex>-<group>, read against the sex it says it has,
   * not split at a hyphen, so a group may have one.
   */
  function groupOf(name: string | undefined): string | null {
    const preset = name === undefined ? undefined : presets[name];
    return preset && name!.startsWith(`${preset.sex}-`) ? name!.slice(preset.sex.length + 1) : null;
  }

  /** What the picker shows for a body: its own sex, and the group it is or came from, if that sex has it. */
  function picksFor({ name, base, provenance }: Pick<Snapshot, "name" | "base" | "provenance">) {
    const from = name ?? provenance?.name;
    const group = from !== undefined && presets[from]?.sex === base.sex ? groupOf(from) : null;
    return { sex: base.sex, group: group ?? "" };
  }

  /** The typical bodies of each sex, in the order they are served, for the start picker. */
  const typical: Record<string, { name: string; group: string }[]> = {};
  for (const [name, preset] of Object.entries(presets)) {
    const group = groupOf(name);
    if (group !== null) (typical[preset.sex] ??= []).push({ name, group });
  }

  function loadPreset(name: string) {
    const preset = presets[name];
    state.name = name;
    state.base = { sex: preset.sex, body: { ...preset.body } };
    state.provenance = { name };
    state.picks = picksFor(state);
    state.unknown = {};
    state.notes = [];
    beginSource();
  }

  function choosePreset(name: string) {
    if (!presets[name] || name === state.name) return;
    checkpoint();
    loadPreset(name);
  }

  // The server reads the file (sotaigen/params/file.py), so the page has no reader of its own to keep in step.
  async function readFile(text: string): Promise<ParameterFile> {
    const reply = await (await fetch("/api/params/read", { method: "POST", body: text })).json();
    if (!reply.ok) throw Object.assign(new Error(translateAll(reply.problems ?? [reply.error], sliderLabels(), say).join("; ")), { problems: reply.problems });
    return reply.file;
  }

  async function importFile(text: string) {
    const file = await readFile(text);
    schemaOf(file.sex);
    checkpoint();
    state.base = { sex: file.sex, body: file.body };
    state.unknown = file.unknown;
    state.provenance = file.base;
    state.name = null;
    const notes = Object.keys(file.unknown);
    state.notes = notes.length ? [say("import_unknown_fields", { fields: listOf(notes) })] : [];
    state.picks = picksFor(state);
    beginSource();
    return { unknown: notes, problems: state.problems };
  }

  async function importBlob(blob: Blob) {
    try {
      await importFile(await blob.text());
    } catch (error) {
      state.importError = (error as Error).message;
      describe();
    }
  }

  function currentFile() {
    return writeParameterFile(
      { sex: state.current.sex, body: state.current.body, base: state.provenance, unknown: state.unknown },
      sexes,
      sexes.GENERATOR,
    );
  }

  function saveLink(name: string, href: string) {
    const link = document.createElement("a");
    link.href = href;
    link.download = name;
    link.click();
  }

  function exportJson() {
    const href = URL.createObjectURL(new Blob([currentFile()], { type: "application/json" }));
    saveLink(`${state.provenance?.name ?? "body"}.json`, href);
    URL.revokeObjectURL(href);
  }

  /** A broken mesh's reasons, as the worker gave them, in the page's words. */
  function brokenReasons(found: Found[]) {
    return translateAll(found, sliderLabels(), say).join("; ");
  }

  const mendKey = ({ sex, body }: BodyParams) => JSON.stringify({ sex, body });

  function stopMendSearch() {
    if (mendAsked) {
      clearInterval(mendAsked.poll);
      mendAsked.abort.abort();
      mendAsked = null;
    }
    if (state.mendSearch) set({ mendSearch: null });
  }

  // After a refusal, for the parameters refused, and only once for each. The refusal has already been said.
  async function seekMend({ sex, body }: BodyParams) {
    const key = mendKey({ sex, body });
    if (mendAsked?.key === key || key !== mendKey(state.current)) return;
    stopMendSearch();
    const abort = new AbortController();
    const asked = { phase: "running" as const, stage: null, startedAt: Date.now(), message: null };
    const still = () => mendAsked?.abort === abort;
    const poll = setInterval(async () => {
      try {
        const { job } = await (await fetch("/api/mend")).json();
        if (still() && state.mendSearch?.phase === "running") set({ mendSearch: { ...asked, stage: job?.stage ?? state.mendSearch.stage } });
      } catch {
        // The answer to the search itself says what went wrong.
      }
    }, 1000);
    mendAsked = { key, abort, poll };
    set({ mendSearch: asked });
    try {
      const reply = await (await fetch("/api/mend", { method: "POST", body: JSON.stringify({ sex, body }), signal: abort.signal })).json();
      if (!still()) return;
      if (reply.superseded) return;
      if (!reply.ok) throw new Error(translateAll(reply.problems ?? [reply.error ?? say("error_no_reason")], sliderLabels(), say).join("\n"));
      const mend: Mend | null = reply.mend;
      // Neither flipped nor crossed: an open skin, which is no parameter's.
      set({ mendSearch: mend ? { ...asked, phase: "done", message: translateMend(mend, sliderLabels(), say) } : null });
    } catch (error) {
      if (still()) set({ mendSearch: { ...asked, phase: "failed", message: { text: say("mend_failed", { error: (error as Error).message }) } } });
    } finally {
      clearInterval(poll);
    }
  }

  // The file is what the page is showing once it is up to date, so there is no second body to make. A
  // broken mesh stays on the screen, to see what went wrong, and is refused only here.
  async function exportGlb() {
    set({ exporting: true, glbRefusal: null });
    try {
      // A failed body gets another try: the click says it is still wanted.
      if (rebuilder.phase === "failed") parametersChanged({ now: true });
      do await rebuilder.flush();
      while (rebuilder.phase === "generating");
      if (rebuilder.phase !== "idle" || !state.report) return null;
      if (state.report.broken?.length) {
        state.glbRefusal = say("glb_broken", { reasons: brokenReasons(state.report.broken) });
        void seekMend(state.current);
        return null;
      }
      saveLink(`${state.provenance?.name ?? "body"}.glb`, state.report.glb);
      return state.report.glb;
    } finally {
      state.exporting = false;
      describe();
    }
  }

  // The body as it is now, with its IPC contact tables, from a worker of its own on the server: the
  // sliders stay free meanwhile, and what they do doesn't change or cancel this file.
  async function exportGlbContact() {
    if (state.contactExport?.phase === "running" || state.problems.errors.length) return null;
    const { sex, body } = state.current;
    const name = `${state.provenance?.name ?? "body"}-ipc.glb`;
    set({ contactExport: { phase: "running", stage: null, startedAt: Date.now(), messages: [] } });
    const poll = setInterval(async () => {
      try {
        const { job } = await (await fetch("/api/contact")).json();
        if (job && state.contactExport?.phase === "running") {
          set({ contactExport: { ...state.contactExport, stage: job.stage, startedAt: job.startedAt } });
        }
      } catch {
        // The answer to the export itself says what went wrong.
      }
    }, 1000);
    try {
      const reply = await (await fetch("/api/body", { method: "POST", body: JSON.stringify({ sex, body, contact: true }) })).json();
      if (reply.superseded) throw new Error(say("contact_superseded"));
      if (reply.broken?.length) {
        set({ contactExport: { ...state.contactExport!, phase: "failed", messages: [say("contact_broken", { reasons: brokenReasons(reply.broken) })] } });
        void seekMend({ sex, body });
        return null;
      }
      if (!reply.ok) throw new Error(translateAll(reply.problems ?? [reply.error ?? say("error_no_reason")], sliderLabels(), say).join("\n"));
      saveLink(name, reply.glb);
      const fellBack = translateAll(
        (reply.failures as Diagnostic[]).filter((found) => found.code === "contact.failed"),
        sliderLabels(),
        say,
      );
      set({ contactExport: { ...state.contactExport!, phase: "done", messages: fellBack } });
      return reply.glb as string;
    } catch (error) {
      set({ contactExport: { ...state.contactExport!, phase: "failed", messages: [say("contact_failed", { error: (error as Error).message })] } });
      return null;
    } finally {
      clearInterval(poll);
    }
  }

  // The body as it is now, as a rigged .blend, from a worker of its own on the server: like the IPC
  // export, the sliders stay free meanwhile and what they do doesn't change this file.
  async function exportBlend() {
    if (state.blendExport?.phase === "running" || state.problems.errors.length || state.blender === false) return null;
    const { sex, body } = state.current;
    const name = `${state.provenance?.name ?? "body"}.blend`;
    const started = { phase: "running" as const, startedAt: Date.now(), message: null };
    const failed = (message: string) => set({ blendExport: { ...started, phase: "failed", message } });
    set({ blendExport: started });
    // Only for the clock on the page.
    const tick = setInterval(() => set({ blendExport: { ...started } }), 1000);
    try {
      const reply = await (await fetch("/api/blend", { method: "POST", body: JSON.stringify({ sex, body }) })).json();
      const labels = sliderLabels();
      if (reply.ok) {
        saveLink(name, reply.blend);
        set({ blendExport: { ...started, phase: "done" } });
        return reply.blend as string;
      }
      if (reply.noBlender) {
        state.blender = false;
        failed(say("blend_no_blender"));
      } else if (reply.broken) {
        failed(say("blend_broken", { reasons: brokenReasons(reply.broken) }));
        void seekMend({ sex, body });
      } else if (reply.superseded) {
        failed(say("blend_superseded"));
      } else {
        const lines: Found[] = reply.problems ?? [reply.error ?? say("error_no_reason")];
        failed(say("blend_failed", { error: translateAll(lines, labels, say).join("\n") }));
      }
    } catch (error) {
      failed(say("blend_failed", { error: (error as Error).message }));
    } finally {
      clearInterval(tick);
    }
    return null;
  }

  // Sliders

  const spec = (name: string): Slider => definition.sliders[name];

  /** Whether the body's sex has what this slider moves: a male body has no breast to set. */
  function offered(name: string) {
    const slider = spec(name);
    if (slider.kind !== "direct") return true;
    const [kind, key] = slider.param.split(".");
    return kind !== "body" || Object.hasOwn(schema.ALL_SPEC, key);
  }

  function sliderValue(name: string): number | undefined {
    const slider = spec(name);
    return slider.kind === "macro" ? (state.sliders[name] ?? 0) : sliderStart(slider, state.current, schema, profileReferences.get(profileKey(state.current, true)));
  }

  const baseValue = (name: string): number | undefined => sliderStart(spec(name), state.base, schema, profileReferences.get(profileKey(state.base, true)));

  /** Whether the body's sex has any slider of this group. One it has none of isn't shown. */
  function regionOffered(id: string) {
    const group = definition.groups.find((row) => row.id === id);
    return !group || group.sliders.some(offered);
  }

  /** The panel shown for one asked for, by the rail, a click, a problem or the address. */
  function shownPanel(panel: PanelView): PanelView {
    if (panel === "changes" || regionOffered(panel)) return panel;
    const instead = REGION_STAND_INS[panel];
    return instead && regionOffered(instead) ? instead : "proportion";
  }

  /**
   * Whether the slider's value differs from where its source started, whoever moved it. A slider the
   * sex doesn't have never has: a male body's breast defaults still follow its torso.
   */
  function isChanged(name: string) {
    if (!offered(name)) return false;
    const slider = spec(name);
    if (slider.measured && slider.kind === "direct" && !(slider.param.split(".")[1] in state.base.body)) {
      return name in state.sliders;
    }
    const value = sliderValue(name);
    const base = baseValue(name);
    return Number.isFinite(value) && Number.isFinite(base) && Math.abs((value as number) - (base as number)) > 1e-9;
  }

  /** Where the rules would stop this slider, with everything else as it is now. */
  function zones(name: string) {
    if (thighNames.includes(name)) return [];
    return sliderZones({ name, base: state.base, sliders: state.sliders, definition, ranges, schema });
  }

  function moveSlider(name: string, value: number, options?: { now?: boolean }) {
    checkpoint(name);
    state.sliders = { ...state.sliders, [name]: value };
    parametersChanged(options);
  }

  function clearSliders(names: string[]) {
    const kept = Object.fromEntries(Object.entries(state.sliders).filter(([name]) => !names.includes(name)));
    if (Object.keys(kept).length === Object.keys(state.sliders).length) return;
    checkpoint();
    state.sliders = kept;
    parametersChanged({ now: true });
  }

  function resetAll() {
    checkpoint();
    state.sliders = {};
    parametersChanged({ now: true });
  }

  // View

  function setPanel(asked: PanelView, { fromPick = false } = {}) {
    const panel = shownPanel(asked);
    const moved = panel !== state.panel;
    set({ panel, query: "" });
    // A click on the body is already where the person is looking, so the camera stays.
    if (panel !== "changes" && state.follow && moved && !fromPick) viewport.focus(panel);
  }

  function setFollow(follow: boolean) {
    set({ follow });
    writePreferences({ theme: state.theme, follow });
    if (follow && state.panel !== "changes") viewport.focus(state.panel);
  }

  function setTheme(theme: Theme) {
    set({ theme });
    viewport.setTheme(theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
    writePreferences({ theme, follow: state.follow });
  }

  function setPose(pose: string) {
    set({ pose });
    showPose();
  }

  function setDriven(driven: boolean) {
    set({ driven });
    showPose();
  }

  // A contact pose isn't in the body's file: the server solves it for the body on show, once.
  function showPose() {
    if (!isTouchPose(state.pose)) {
      touchAsked?.abort();
      touchAsked = null;
      set({ touchJob: null });
      viewport.applyPose(state.pose, state.driven);
      return;
    }
    const answer = shownParams && touchAnswers.get(touchKey(shownParams));
    if (!answer) {
      // Not an arm through the next head while it's solved.
      viewport.applyPose("rest", state.driven);
      if (shownParams) void solveTouch(shownParams);
      return;
    }
    const rotations = answer.poses[state.pose];
    const report = answer.reports[state.pose];
    if (!rotations) {
      viewport.applyPose("rest", state.driven);
      set({ touchJob: { phase: "failed", message: say("touch_none") } });
      return;
    }
    viewport.applyPose(state.pose, state.driven, rotations);
    set({
      touchJob: report?.reached === false ? { phase: "short", message: say("touch_short", { mm: report.arrivalMm.toFixed(0) }) } : null,
    });
  }

  async function solveTouch(params: BodyParams) {
    const key = touchKey(params);
    if (touchAsked && touchAskedKey === key) return;
    touchAsked?.abort();
    const asked = new AbortController();
    touchAsked = asked;
    touchAskedKey = key;
    set({ touchJob: { phase: "running", message: null } });
    try {
      const reply = await (
        await fetch("/api/touch", { method: "POST", body: JSON.stringify(params), signal: asked.signal })
      ).json();
      if (reply.superseded) return;
      if (!reply.ok) {
        const lines: Found[] = reply.problems ?? [reply.error ?? say("error_no_reason")];
        throw new Error(translateAll(lines, sliderLabels(), say).join("\n"));
      }
      const answer = (await (await fetch(reply.touch, { signal: asked.signal })).json()) as TouchAnswer;
      touchAnswers.set(key, answer);
      if (touchAnswers.size > 32) touchAnswers.delete(touchAnswers.keys().next().value!);
      if (touchAsked === asked) touchAsked = null;
      showPose();
    } catch (error) {
      if (asked.signal.aborted) return;
      touchAsked = null;
      set({ touchJob: { phase: "failed", message: say("touch_failed", { error: (error as Error).message }) } });
    }
  }

  function setPartVisible(name: string, visible: boolean) {
    const others = state.hiddenParts.filter((part) => part !== name);
    set({ hiddenParts: visible ? others : [...others, name] });
    viewport.applyInspection(state.hiddenParts, state.morphs);
  }

  function setMorph(name: string, value: number) {
    set({ morphs: { ...state.morphs, [name]: value } });
    viewport.applyInspection(state.hiddenParts, state.morphs);
  }

  function setFaceBone(name: keyof FaceBones, value: number) {
    set({ faceBones: { ...state.faceBones, [name]: value } });
    viewport.setFace(facePose(state.faceBones));
    showPose();
  }

  function resetMorphs() {
    set({ morphs: {}, faceBones: RESTING_FACE });
    viewport.setFace({});
    showPose();
    viewport.applyInspection(state.hiddenParts, state.morphs);
  }

  function setLook(look: LookName) {
    set({ look });
    viewport.setLook(look);
  }

  function hold(held: boolean) {
    if (held && !state.canCompare) return;
    viewport.showBefore(held);
    set({ comparing: viewport.comparing });
  }

  viewport.onPick = (region) => setPanel(region, { fromPick: true });
  viewport.onDrawn = () => {
    if (!state.undrawn) return;
    log.timing.push({ ...state.undrawn, waitMs: performance.now() - state.waitStart });
    state.undrawn = null;
    describe();
  };

  // Wherever the mouse is let go, and a drag can end outside the slider it began on.
  window.addEventListener("pointerup", () => rebuilder.release());
  window.addEventListener("pointercancel", () => rebuilder.release());
  window.addEventListener("blur", () => hold(false));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) hold(false);
  });
  setInterval(() => {
    if (rebuilder.phase === "generating") describe();
  }, 250);

  document.documentElement.classList.toggle("dark", state.theme === "dark");
  viewport.setTheme(state.theme);

  const names = Object.keys(presets);
  const asked = start.body ?? "";
  if (start.pose) state.pose = start.pose;
  if (isLook(start.look)) state.look = start.look;
  state.inspecting = Boolean(start.inspect);
  if (definition.groups.some((group) => group.id === start.panel) || start.panel === "changes") {
    state.panel = start.panel as PanelView;
  }
  loadPreset(names.includes(asked) ? asked : names.includes("female-long") ? "female-long" : names[0]);

  return {
    state,
    rebuilder,
    log,
    viewport,
    get ranges() {
      return ranges;
    },
    presets,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    bounds: (name: string) => thighBounds.get(thighLimitKey(state.current, name)) ?? sliderBounds(spec(name), ranges),
    spec,
    sliderValue,
    baseValue,
    abdominalReport: () => profiles.get(profileKey(state.current)),
    thighReport: () => thighProfiles.get(profileKey(state.current)),
    /** What the finished body reached, where that is not what this slider asks for. */
    targetMiss: (name: string) => {
      const slider = spec(name);
      if (slider.kind !== "direct" || rebuilder.phase !== "idle") return undefined;
      return state.report?.targets?.[slider.param.split(".")[1]];
    },
    /** The results no slider sets directly, on the current body. */
    derived: () => {
      const body = state.current.body;
      if (!schema.REQUIRED.every((key) => typeof body[key] === "number")) return null;
      return { stature: stature(body), heads: headCount(body), torso: body.torsoLength };
    },
    /** Another millimetre figure from the slider's value; undefined where it has none. */
    reading: (name: string) => {
      const slider = spec(name);
      const value = sliderValue(name);
      if (!slider.reading || typeof value !== "number") return undefined;
      const body = state.current.body;
      if (!schema.REQUIRED.every((key) => typeof body[key] === "number")) return undefined;
      return `${sliderReading(name)} ${formatMm(readingMm(slider.reading, body, value))}`;
    },
    isChanged,
    offered,
    regionOffered,
    zones,
    moveSlider,
    clearSliders,
    resetAll,
    choosePreset,
    typical,
    importBlob,
    importFile,
    currentFile,
    exportJson,
    exportGlb,
    exportGlbContact,
    exportBlend,
    setBlender: (blender: boolean | null) => set({ blender }),
    /** Puts away why the last click on Export GLB saved nothing. */
    dismissGlbRefusal: () => set({ glbRefusal: null }),
    dismissTouchJob: () => {
      if (state.touchJob?.phase !== "running") set({ touchJob: null });
    },
    /** Stops the search for what ends the breakage, or puts away what it found. */
    dismissMendSearch: stopMendSearch,
    /** Puts away what a finished .blend export said. A running one stays. */
    dismissBlendExport: () => {
      if (state.blendExport?.phase !== "running") set({ blendExport: null });
    },
    /** Puts away what a finished IPC export said. A running one stays. */
    dismissContactExport: () => {
      if (state.contactExport?.phase !== "running") set({ contactExport: null });
    },
    undo,
    redo,
    setPanel,
    setQuery: (text: string) => set({ query: text }),
    setFollow,
    setTheme,
    setPose,
    setDriven,
    setPartVisible,
    setMorph,
    setFaceBone,
    resetMorphs,
    setInspecting: (inspecting: boolean) => set({ inspecting }),
    setLook,
    crop: (crop: CropName) => viewport.crop(crop),
    view: (view: ViewName) => viewport.view(view),
    hold,
    grab: () => rebuilder.hold(),
  };
}

export type Editor = ReturnType<typeof createEditor>;

/** Reads a piece of the state, and renders again only when that piece changes. */
export function useEditor<T>(editor: Editor, select: (state: EditorState) => T): T {
  return useSyncExternalStore(editor.subscribe, () => select(editor.state));
}
