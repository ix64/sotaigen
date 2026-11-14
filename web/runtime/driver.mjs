// The helper bones' drive at runtime: sotaigen.rig.skeleton.sided, from the table an exported GLB carries.
//
// A pose is the rig's own: rotations in degrees by bone name, or by limb for both sides, as in
// sotaigen/rig/poses.py. It is an intent, not the final rotation of each bone: the upper arm gives part of
// its swing to the shoulder, half bones take half of a limb's swing, twist bones spread a roll. So
// animation data goes through here every frame too, never straight onto the bones.
//
// No three.js in this file: tests/test_export.py runs it under plain Node against sotaigen.rig.skeleton.sided,
// and that test is what keeps the two together.

/**
 * A rotation as [x, y, z, w].
 * @typedef {[number, number, number, number]} Quaternion
 */

/**
 * Angles (x, y, z) in degrees, as sotaigen.rig.skeleton.euler takes them.
 * @typedef {[number, number, number]} Euler
 */

/**
 * A pose: rotations by bone name, or by limb for both sides, as in sotaigen/rig/poses.py.
 * @typedef {Record<string, Euler>} Pose
 */

/**
 * The driver table, extras.sotaigen.drivers: sotaigen/output/glb.py driver_table. Every helper bone is
 * named in full with the bone it follows, so nothing here builds a bone's name.
 * @typedef {object} DriverTable
 * @property {string[]} central Names that keep to themselves; any other unsided name means both sides.
 * @property {{share: number, forwardLift: number, arms: {arm: string, girdle: string, side: number}[]}} rhythm
 *   The share of the upper arm's swing that the girdle takes, and of a forward raise that also elevates it;
 *   `side` is the sign of a side raise's Euler Z.
 * @property {{bone: string, of: string, roll?: number}[]} half Each half bone takes half of its limb's
 *   swing, and `roll` of its true twist where given.
 * @property {{bone: string, of: string, share: number}[]} twist Each twist bone takes `share` of the true
 *   twist about local Y of the bone it follows.
 * @property {Record<string, [number, number, number]>} glide Metres a bone slides per degree of its
 *   opening, in its rest frame: only the jaw.
 */

/**
 * One group of extras.sotaigen.correctives.groups: sotaigen.corrective.drive, which this mirrors.
 * @typedef {object} CorrectiveGroup
 * @property {string} name
 * @property {string} source Where its targets came from: "sculpted", "field" or "ipc".
 * @property {string} [kind] "simplices" for the experimental IPC shoulder table; otherwise a grid.
 * @property {{bone: string, read: string, scale?: number, keys: number[]}[]} [axes]
 * @property {{bone: string, read: string, limit: number}[]} [fades]
 * @property {number[][]} [samples] A key of each axis for each target.
 * @property {(string | null)[]} targets
 * @property {string} [bone] Simplices: the bone read.
 * @property {number[]} [scales] Simplices: the scale of swing X, twist and swing Z.
 * @property {number[]} [sampled] Simplices: which of swing X, twist and swing Z its points sample.
 * @property {number[][]} [domain] Simplices: bounds in sampled coordinate order.
 * @property {number} [twistLimit] Simplices of two axes: the twist fade.
 * @property {number[][]} [points] @property {number[][]} [cells] @property {number[][]} [origins]
 * @property {number[][][]} [inverse]
 */

/** Degrees: a fade is 0 from this far below its limit on, as sotaigen/corrective/drive.py's LIMIT_SLACK. */
export const LIMIT_SLACK = 1e-4;

/** A fade's share at a reading: ease(1 - |value| / limit), and 0 within LIMIT_SLACK of the limit.
 * @param {number} value @param {number} limit */
export function fadeShare(value, limit) {
  return Math.abs(value) >= limit - LIMIT_SLACK ? 0 : ease(1 - Math.abs(value) / limit);
}

/** @param {number} x */
function ease(x) {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

/** @param {number[]} keys @param {number} value */
function interval(keys, value) {
  const clipped = Math.min(keys[keys.length - 1], Math.max(keys[0], value));
  let i = 1;
  while (i < keys.length - 1 && clipped > keys[i]) i += 1;
  const t = ease((clipped - keys[i - 1]) / (keys[i] - keys[i - 1]));
  return new Map([[keys[i - 1], 1 - t], [keys[i], t]]);
}

const READS = /** @type {Record<string, number>} */ ({ swingX: 0, twist: 1, swingZ: 2 });

/**
 * The rotation the pose asks of each bone, before any helper takes a share: a limb's name for both
 * sides, the right mirrored, later entries winning, as sotaigen.rig.skinning.sided.
 * @param {Pose} pose @param {string[]} central
 */
function intents(pose, central) {
  /** @type {Record<string, Euler>} */
  const out = {};
  for (const [name, [x, y, z]] of Object.entries(pose)) {
    if (name === "rootTranslation") continue;
    if (name.startsWith("left") || name.startsWith("right") || central.includes(name)) out[name] = [x, y, z];
    else {
      out[`left${capital(name)}`] = [x, y, z];
      out[`right${capital(name)}`] = [x, -y, -z];
    }
  }
  return out;
}

/**
 * Every target's weight for a pose, from the correctives' groups, as sotaigen.corrective.drive.weights.
 * The pose's limb names mean both sides unless `central` lists them (the driver table's central).
 * @param {Pose} pose @param {CorrectiveGroup[]} groups @param {string[]} central
 * @returns {Record<string, number>}
 */
export function correctiveWeights(pose, groups, central) {
  const asked = intents(pose, central);
  /** @type {Record<string, Euler>} */
  const read = {};
  /** @param {string} bone @param {string} channel @param {number} [scale] */
  const reading = (bone, channel, scale = 1) => {
    read[bone] ??= swingTwist(asked[bone] ?? [0, 0, 0]);
    const values = read[bone];
    const value = channel === "largest" ? Math.max(...values.map(Math.abs)) : values[READS[channel]];
    return scale === 1 ? value : value * scale;
  };
  /** @type {Record<string, number>} */
  const out = {};
  for (const group of groups) {
    if (group.kind === "simplices") {
      Object.assign(out, simplexWeights(group, reading));
      continue;
    }
    if (group.kind) throw new Error(`Unsupported corrective table ${group.kind}`);
    /** @type {Map<string, number>} */
    const byBone = new Map();
    for (const fade of group.fades ?? []) {
      const share = fadeShare(reading(fade.bone, fade.read), fade.limit);
      byBone.set(fade.bone, byBone.has(fade.bone) ? /** @type {number} */ (byBone.get(fade.bone)) * share : share);
    }
    let gain = 1;
    [...byBone.values()].forEach((share, i) => {
      gain = i === 0 ? share : gain * share;
    });
    const tents = (group.axes ?? []).map((axis) => interval(axis.keys, reading(axis.bone, axis.read, axis.scale ?? 1)));
    (group.samples ?? []).forEach((sample, i) => {
      let weight = tents[0].get(sample[0]) ?? 0;
      for (let a = 1; a < tents.length; a += 1) weight *= tents[a].get(sample[a]) ?? 0;
      out[/** @type {string} */ (group.targets[i])] = weight * gain;
    });
  }
  return out;
}

/**
 * One side's experimental IPC shoulder table: sotaigen.corrective.pose_grid.side_weights.
 * @param {CorrectiveGroup} group @param {(bone: string, channel: string, scale?: number) => number} reading
 */
function simplexWeights(group, reading) {
  const { domain, points, cells, origins, inverse, targets } = /** @type {Required<CorrectiveGroup>} */ (group);
  const scales = group.scales ?? [1, 1, 1];
  const arm = ["swingX", "twist", "swingZ"].map((channel, i) => reading(/** @type {string} */ (group.bone), channel, scales[i]));
  const axes = group.sampled ?? [0, 1, 2];
  /** @type {Record<string, number>} */
  const weights = {};
  for (const name of targets) if (name) weights[name] = 0;
  const point = axes.map((axis, i) => Math.max(domain[i][0], Math.min(domain[i][1], arm[axis])));
  let gain = 1;
  if (axes.length === 2) {
    if (!group.twistLimit) throw new Error("Missing twist fade for two-axis shoulder table");
    gain = ease(1 - Math.abs(arm[1]) / group.twistLimit);
  }
  const exact = points.findIndex((sample) => sample.every((value, axis) => value === point[axis]));
  if (exact >= 0) {
    const name = targets[exact];
    if (name) weights[name] = gain;
    return weights;
  }
  for (let cell = 0; cell < cells.length; cell += 1) {
    const offset = point.map((value, axis) => value - origins[cell][axis]);
    const blend = inverse[cell].map((row) => row.reduce((sum, v, i) => sum + v * offset[i], 0));
    blend.push(1 - blend.reduce((sum, value) => sum + value, 0));
    if (blend.some((value) => value < -1e-10)) continue;
    const positive = blend.map((value) => Math.max(value, 0));
    const sum = positive.reduce((total, value) => total + value, 0);
    cells[cell].forEach((index, corner) => {
      const name = targets[index];
      if (name) weights[name] = (gain * positive[corner]) / sum;
    });
    return weights;
  }
  throw new Error("Baked shoulder pose grid has a hole");
}

// What a GLB's extras.sotaigen must say for this runtime to read it: sotaigen/output/metadata.py.
export const FORMAT = "sotaigen/glb";
export const VERSION = 1;

/** Throws unless `info`, a GLB's extras.sotaigen, is a file this runtime reads. @param {any} info */
export function checkInfo(info) {
  if (info?.format !== FORMAT) throw new Error(`not a sotaigen GLB: extras.sotaigen.format is ${JSON.stringify(info?.format)}`);
  if (info.version !== VERSION) throw new Error(`extras.sotaigen version ${info.version}, this runtime reads ${VERSION}`);
}

/** @type {Quaternion} */
const IDENTITY = [0, 0, 0, 1];
const DEGREES = Math.PI / 180;

// Project-space millimetres to glTF metres, matching sotaigen.output.glb.to_gltf.
/** @param {Pose} pose @returns {[number, number, number]} */
export function rootTranslation(pose) {
  const value = pose.rootTranslation ?? [0, 0, 0];
  if (value.length !== 3 || !value.every(Number.isFinite)) {
    throw new Error("rootTranslation must be three finite millimetre coordinates");
  }
  return [value[0] / 1000, value[2] / 1000, -value[1] / 1000];
}

/** @param {string} name */
function capital(name) {
  return name[0].toUpperCase() + name.slice(1);
}

/**
 * @param {Quaternion} a
 * @param {Quaternion} b
 * @returns {Quaternion}
 */
function multiply([ax, ay, az, aw], [bx, by, bz, bw]) {
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

// sotaigen.rig.skeleton.euler: rz @ ry @ rx.
/**
 * @param {Euler} euler
 * @returns {Quaternion}
 */
export function fromEuler([x, y, z]) {
  const half = (/** @type {number} */ degrees) => (degrees * DEGREES) / 2;
  /** @type {Quaternion} */
  const rx = [Math.sin(half(x)), 0, 0, Math.cos(half(x))];
  /** @type {Quaternion} */
  const ry = [0, Math.sin(half(y)), 0, Math.cos(half(y))];
  /** @type {Quaternion} */
  const rz = [0, 0, Math.sin(half(z)), Math.cos(half(z))];
  return multiply(multiply(rz, ry), rx);
}

// sotaigen.rig.skinning.swing_twist: how the pose correctives read a rotation, as [swing x, twist, swing z]
// in degrees. The twist about local Y comes first, then the swing, given as its rotation vector
// (x, 0, z), so a turn about one axis reads as its own angle. Blender's Swing and Y Twist channels
// split it the same way (blender/rig.py correctives).
/**
 * @param {Euler} euler
 * @returns {Euler}
 */
export function swingTwist(euler) {
  if (euler.filter((v) => v !== 0).length <= 1 && euler.every((v) => Math.abs(v) <= 180)) {
    // A turn about one axis, exactly, without the rounding of the general case.
    return [euler[0], euler[1], euler[2]];
  }
  let [qx, qy, qz, qw] = fromEuler(euler);
  if (qw < 0) [qx, qy, qz, qw] = [-qx, -qy, -qz, -qw];
  const r = Math.hypot(qw, qy);
  let twist = 0;
  let sx = qx;
  let sz = qz;
  // At a half-turn swing Blender's twist is atan2(0, 0), zero.
  if (r >= 1e-12) {
    twist = 2 * Math.atan2(qy, qw);
    sx = (qx * qw + qz * qy) / r;
    sz = (qz * qw - qx * qy) / r;
  }
  const n = Math.hypot(sx, sz);
  const scale = n > 1e-12 ? (2 * Math.atan2(n, r)) / n : 2;
  return [(scale * sx) / DEGREES, twist / DEGREES, (scale * sz) / DEGREES];
}

// sotaigen.rig.skeleton.half_swing: half of a rotation's swing, with its roll about local Y removed.
/**
 * @param {Euler} euler
 * @returns {Quaternion}
 */
function halfSwing(euler) {
  const [qx, qy, qz, qw] = fromEuler(euler);
  // The second column of the rotation matrix: where local Y ends up.
  const mx = 2 * (qx * qy - qz * qw);
  const my = 1 - 2 * (qx * qx + qz * qz);
  const mz = 2 * (qy * qz + qx * qw);
  // Y cross moved.
  const ax = mz;
  const az = -mx;
  const size = Math.hypot(ax, az);
  if (size < 1e-9) return IDENTITY;
  const angle = Math.atan2(size, my) / 2;
  const s = Math.sin(angle / 2) / size;
  return [ax * s, 0, az * s, Math.cos(angle / 2)];
}

// The pose as each bone's final local rotation: { bone: [x, y, z, w] }, relative to its rest. A
// bone left out stays at rest.
/**
 * @param {Pose} pose
 * @param {DriverTable} table
 * @returns {Record<string, Quaternion>}
 */
export function poseQuaternions(pose, table) {
  /** @type {Pose} */
  const eulers = {};
  for (const [name, [x, y, z]] of Object.entries(pose)) {
    if (name === "rootTranslation") continue;
    if (name.startsWith("left") || name.startsWith("right") || table.central.includes(name)) {
      eulers[name] = [x, y, z];
    } else {
      eulers[`left${capital(name)}`] = [x, y, z];
      eulers[`right${capital(name)}`] = [x, -y, -z];
    }
  }
  const { share, forwardLift, arms } = table.rhythm;
  for (const { arm: armName, girdle, side } of arms) {
    const arm = eulers[armName];
    if (!arm) continue;
    const forward = Math.max(arm[0], 0) * share;
    const up = (Math.max(arm[2] * side, 0) * share + Math.max(arm[0], 0) * forwardLift) * side;
    const [x, y, z] = eulers[girdle] ?? [0, 0, 0];
    eulers[girdle] = [x + forward, y, z + up];
    eulers[armName] = [arm[0] - forward, arm[1], arm[2] - up];
  }
  /** @type {Record<string, Quaternion>} */
  const halves = {};
  for (const { bone, of, roll } of table.half) {
    const rotation = eulers[of];
    if (!rotation) continue;
    halves[bone] = roll === undefined
      ? halfSwing(rotation)
      : multiply(halfSwing(rotation), fromEuler([0, swingTwist(rotation)[1] * roll, 0]));
  }
  /** @type {Pose} */
  const twists = {};
  for (const { bone, of, share: taken } of table.twist) {
    const rotation = eulers[of];
    if (!rotation) continue;
    // The driver's true twist, the upper arm's after the shoulder rhythm, as sotaigen.rig.skinning.sided.
    twists[bone] = [0, swingTwist(rotation)[1] * taken, 0];
  }
  Object.assign(eulers, twists);
  /** @type {Record<string, Quaternion>} */
  const out = {};
  for (const [name, euler] of Object.entries(eulers)) out[name] = fromEuler(euler);
  return Object.assign(out, halves);
}

// Where the pose moves a bone off its rest place, as well as turning it: { bone: [x, y, z] } in metres
// in its rest frame. Only the jaw does, sliding as it opens (sotaigen.rig.skinning.bone_local).
/**
 * @param {Pose} pose
 * @param {DriverTable} table
 * @returns {Record<string, [number, number, number]>}
 */
export function poseTranslations(pose, table) {
  /** @type {Record<string, [number, number, number]>} */
  const out = {};
  for (const [bone, [x, y, z]] of Object.entries(table.glide ?? {})) {
    const rotation = pose[bone];
    if (!rotation) continue;
    const opening = Math.max(rotation[0], 0);
    out[bone] = [x * opening, y * opening, z * opening];
  }
  return out;
}

// The inverse of sotaigen.rig.skeleton.euler, so that quaternion animation data can enter the driver: the angles
// (x, y, z) in degrees with y within +-90. The driver reads these angles, so a rotation's Euler
// angles past that range come back as a different, equal rotation and drive the helpers differently.
/**
 * @param {Quaternion} quaternion
 * @returns {Euler}
 */
export function eulerFromQuaternion([x, y, z, w]) {
  const sinY = 2 * (w * y - x * z);
  // Past the pole only x and z together are defined, so all of it goes to x.
  if (Math.abs(sinY) > 1 - 1e-12) {
    return [(2 * Math.atan2(x, w)) / DEGREES, Math.sign(sinY) * 90, 0];
  }
  return [
    Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) / DEGREES,
    Math.asin(sinY) / DEGREES,
    Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) / DEGREES,
  ];
}
