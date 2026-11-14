// How the page lays out sliders.json's groups: which are about the whole figure and which about one
// place on it, where the camera goes for that place, and which place a bone belongs to. What they're
// called is in web/messages, through text.ts.

export type RegionId =
  | "proportion"
  | "quick"
  | "head"
  | "neck"
  | "shoulder"
  | "chest"
  | "arm"
  | "waist"
  | "hip"
  | "glute"
  | "leg";

/**
 * Where to look for a region: around some bones, raised by `lift` of the body's height, fitting `span`
 * of that height, and from which side.
 */
export interface RegionFocus {
  bones: string[];
  span: number;
  direction: [number, number, number];
  lift?: number;
}

/** The left column's two tiers; each is named by the message tier_<id>. */
export const REGION_TIERS: { id: "whole" | "parts"; regions: RegionId[] }[] = [
  { id: "whole", regions: ["proportion", "quick"] },
  { id: "parts", regions: ["head", "neck", "shoulder", "chest", "arm", "waist", "hip", "glute", "leg"] },
];

// glTF axes: +Y up, the body faces +Z, the wearer's left is +X. The whole-figure groups have no focus.
// The waist is seen from the side, because its leans and arcs show only there.
export const REGION_FOCUS: Partial<Record<RegionId, RegionFocus>> = {
  head: { bones: ["head"], span: 0.34, direction: [0.55, 0.04, 1], lift: 0.06 },
  neck: { bones: ["neck"], span: 0.3, direction: [0.45, 0.05, 1] },
  shoulder: { bones: ["upperChest"], span: 0.46, direction: [0.35, 0.06, 1], lift: 0.02 },
  chest: { bones: ["upperChest"], span: 0.44, direction: [0.6, 0.02, 1] },
  arm: { bones: ["leftUpperArm", "leftLowerArm"], span: 0.56, direction: [0.5, 0.05, 1] },
  waist: { bones: ["spine"], span: 0.6, direction: [1, 0.03, 0.04] },
  hip: { bones: ["hips"], span: 0.48, direction: [0.6, 0, 1], lift: -0.02 },
  glute: { bones: ["hips"], span: 0.46, direction: [0.75, 0, -1] },
  leg: { bones: ["leftUpperLeg", "leftLowerLeg"], span: 0.66, direction: [0.5, 0.03, 1] },
};

/**
 * Where to go instead of a group the body's sex has no slider of. A male body has no breast, and
 * the front of its chest is the pectorals of the shoulder group.
 */
export const REGION_STAND_INS: Partial<Record<RegionId, RegionId>> = { chest: "shoulder" };

/**
 * The region a click on the body means. `behind` is whether the point faces backwards, which is what
 * tells the seat from the belly and the upper back from the bust: one bone carries both.
 */
export function regionOfBone(bone: string, behind: boolean): RegionId {
  const name = bone.toLowerCase();
  if (/shoulder/.test(name)) return "shoulder";
  if (/arm|hand|palm|thumb|index|middle|ring|little/.test(name)) return "arm";
  if (/leg|knee|foot|toe/.test(name)) return "leg";
  if (/neck/.test(name)) return "neck";
  if (/head|jaw|eye/.test(name)) return "head";
  if (/chest/.test(name)) return behind ? "shoulder" : "chest";
  if (/hips|glute|pelvis/.test(name)) return behind ? "glute" : "hip";
  if (/spine|waist/.test(name)) return "waist";
  return "proportion";
}

export const isHandPose = (key: string) => /hand|grip|fist|pinch|opposition/i.test(key);

/** The hand on the head and at the face, as sotaigen/rig/touch.py names them: solved by the server for each body. */
// The hand-to-head contact poses, by the names sotaigen.rig.touch gives them, as the .blend's pose assets:
// the left hand's, and both hands' where a contact has them.
export const TOUCH_POSES = [
  "touchCrown.left",
  "touchForehead.left",
  "touchCheek.left",
  "touchCheek.both",
  "touchMouth.left",
  "touchOcciput.left",
  "touchOcciput.both",
];
export const isTouchPose = (key: string) => TOUCH_POSES.includes(key);
