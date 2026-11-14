// The page's words, from the compiled messages of web/messages (web/scripts/paraglide.mjs). What a
// slider, group or pose is called is looked up by its id, since sliders.json and the worker name them
// by id; the words belong to the catalogue. Page code only: messages.ts and sliders.ts must stay
// runnable under Node without the npm packages, so they take `say` from here instead of importing it.

import { m } from "@/paraglide/messages.js";
import type { Say } from "./messages.ts";

type Inputs = Record<string, string | number>;
const messages = m as unknown as Record<string, ((inputs: Inputs) => string) | undefined>;

/** A message by its key. One the catalogue doesn't have shows as its key, which a test rules out. */
export const say: Say = (key, inputs = {}) => messages[key]?.(inputs) ?? key;
/** Whether the catalogue has this key: an optional word, such as a slider's ends. */
const has = (key: string) => key in messages;

export const sliderLabel = (name: string) => say(`slider_${name}`);
export const sliderHelp = (name: string) => say(`slider_${name}_help`);
export const sliderReading = (name: string) => say(`slider_${name}_reading`);
/** What the left and right ends of a slider mean, where it says. */
export const sliderEnds = (name: string): [string, string] | undefined =>
  has(`slider_${name}_low`) ? [say(`slider_${name}_low`), say(`slider_${name}_high`)] : undefined;
export const groupLabel = (id: string) => say(`group_${id}`);
export const sectionLabel = (id: string) => say(`section_${id}`);

// A typical body is named <sex>-<group>; the page shows group names only, never ages. Ones the
// catalogue doesn't know, such as a new preset, show by their key.
const known = (key: string, fallback: string) => (has(key) ? say(key) : fallback);
export const sexLabel = (key: string) => known(`sex_${key}`, key);
export const figureLabel = (key: string) => known(`figure_${key}`, key);
// A message's key can't hold a dot, as a contact pose's name does (touchCheek.both).
export const poseLabel = (key: string) => known(`pose_${key.replaceAll(".", "_")}`, key);
export const partLabel = (key: string) => known(`part_${key}`, key);
export const morphLabel = (key: string) => known(`morph_${key}`, key);

/** Several items in one line, the way this language lists them. */
export const listOf = (items: string[]) => items.join(say("join_list"));
