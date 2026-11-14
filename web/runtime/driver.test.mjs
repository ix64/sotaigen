import assert from "node:assert/strict";
import test from "node:test";
import { correctiveWeights, eulerFromQuaternion, poseTranslations, swingTwist } from "./driver.mjs";

const CENTRAL = ["hips", "spine", "chest", "upperChest", "neck", "head", "jaw"];

/**
 * A shoulder table of keys and samples as correctives groups, as sotaigen.corrective.axilla.groups writes them.
 * @param {any} table
 */
function axillaGroups(table) {
  return ["left", "right"].map((side) => ({
    name: `axilla.${side}`,
    source: "sculpted",
    axes: [
      { bone: `${side}UpperArm`, read: "swingX", scale: 1, keys: table.keys },
      { bone: `${side}UpperArm`, read: "swingZ", scale: side === "left" ? 1 : -1, keys: table.spreadKeys },
    ],
    fades: [{ bone: `${side}UpperArm`, read: "twist", limit: table.twistLimit }],
    samples: table.samples,
    targets: table.targets[side],
  }));
}

/** @param {any} pose @param {any} table */
const axillaWeights = (pose, table) => correctiveWeights(pose, axillaGroups(table), CENTRAL);

/**
 * A hanging-arm table as correctives groups, as sotaigen.corrective.hang.groups writes them.
 * @param {any} table
 */
function hangGroups(table) {
  return ["left", "right"].map((side) => ({
    name: `hang.${side}`,
    source: "field",
    axes: [
      { bone: `${side}UpperArm`, read: "swingX", scale: 1, keys: table.forwardKeys },
      { bone: `${side}UpperArm`, read: "swingZ", scale: side === "left" ? -1 : 1, keys: table.adductionKeys },
    ],
    fades: [
      { bone: `${side}UpperArm`, read: "twist", limit: table.twistLimit },
      { bone: `${side}LowerArm`, read: "largest", limit: table.elbowLimit },
      { bone: `${side}Hand`, read: "largest", limit: table.handLimit },
      { bone: `${side}UpperLeg`, read: "largest", limit: table.hipLimit },
    ],
    samples: table.sides[side].samples,
    targets: table.sides[side].targets,
  }));
}

/** @param {any} pose @param {any} table */
const hangWeights = (pose, table) => correctiveWeights(pose, hangGroups(table), CENTRAL);

/**
 * The Euler angles that the correctives read back as this swing and twist (sotaigen.rig.skinning.swing_euler):
 * the twist about Y, then the swing whose rotation vector is (x, 0, z), in degrees.
 * @param {number} x @param {number} twist @param {number} z
 * @returns {[number, number, number]}
 */
function swing(x, twist, z) {
  const r = Math.PI / 180;
  const angle = Math.hypot(x, z) * r;
  const s = angle > 0 ? Math.sin(angle / 2) / angle : 0;
  const [ax, , az, aw] = [x * r * s, 0, z * r * s, Math.cos(angle / 2)];
  const [by, bw] = [Math.sin((twist * r) / 2), Math.cos((twist * r) / 2)];
  return eulerFromQuaternion([ax * bw - az * by, aw * by, az * bw + ax * by, aw * bw]);
}

/** @param {Record<string, number>} actual @param {Record<string, number>} expected */
function close(actual, expected) {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort());
  for (const [key, value] of Object.entries(expected)) {
    assert.ok(Math.abs(actual[key] - value) < 1e-9, `${key}: ${actual[key]} != ${value}`);
  }
}

const table = {
  keys: [0, 20, 35],
  spreadKeys: [0, 35],
  twistLimit: 45,
  samples: [[0, 35], [20, 0], [20, 35], [35, 0], [35, 35]],
  targets: {
    left: ["leftS", "left20", "left20S", "left35", "left35S"],
    right: ["rightS", "right20", "right20S", "right35", "right35S"],
  },
};

test("the correctives read a rotation by its swing and twist", () => {
  for (const [x, twist, z] of [[27.5, 22.5, 17.5], [60, 0, 72], [95, -40, 110], [-20, 10, -24]]) {
    const read = swingTwist(swing(x, twist, z));
    [x, twist, z].forEach((value, i) => assert.ok(Math.abs(read[i] - value) < 1e-9, `${read} != ${[x, twist, z]}`));
  }
  // A turn about one axis reads as its own angle; Euler (60, 0, 72) has a true twist of 45.5.
  assert.deepEqual(swingTwist([0, 0, 72]).map(Math.round), [0, 0, 72]);
  assert.deepEqual(swingTwist([0, 40, 0]).map(Math.round), [0, 40, 0]);
  assert.deepEqual(swingTwist([60, 0, 72]).map((v) => Math.round(v * 10) / 10), [71.1, 45.5, 39]);
});

test("arm intent drives the shape keys before shoulder rhythm", () => {
  const weights = axillaWeights({ upperArm: swing(27.5, 22.5, 17.5), rightUpperArm: [120, 0, 0] }, table);
  // Halfway along both axes, faded to half by the twist; the right arm's later entry wins and clamps.
  close(weights, {
    leftS: 0, left20: 0.125, left20S: 0.125, left35: 0.125, left35S: 0.125,
    rightS: 0, right20: 0, right20S: 0, right35: 1, right35S: 0,
  });
  const both = axillaWeights({ rightUpperArm: [120, 0, 0], upperArm: [20, 0, 0] }, table);
  assert.equal(both.right35, 0);
  assert.equal(both.right20, 1);
  const sided = axillaWeights({ leftUpperArm: [10, 0, 0], rightUpperArm: swing(35, -45, 0) }, table);
  assert.equal(sided.left20, 0.5);
  assert.ok(sided.right35 < 1e-12);
  for (const pose of [{}, { upperArm: [-5, 0, 0] }, { upperArm: swing(35, 45, 0) }]) {
    assert.ok(Object.values(axillaWeights(pose, table)).every((weight) => weight < 1e-12));
  }
});

test("the shoulder grid mirrors side elevation and interpolates across both axes", () => {
  const samples = [[0, 72], [60, 0], [60, 72]];
  const grid = {
    keys: [0, 60], spreadKeys: [0, 72], twistLimit: 90, samples,
    targets: { left: ["ls", "lf", "lm"], right: ["rs", "rf", "rm"] },
  };
  close(axillaWeights({ upperArm: swing(30, 0, 36) }, grid), {
    ls: 0.25, lf: 0.25, lm: 0.25, rs: 0.25, rf: 0.25, rm: 0.25,
  });
  const unilateral = axillaWeights({ rightUpperArm: [0, 0, -72] }, grid);
  assert.equal(unilateral.rs, 1);
  assert.equal(unilateral.ls, 0);
  const override = axillaWeights({ rightUpperArm: [0, 0, -72], upperArm: [0, 0, 0] }, grid);
  assert.ok(Object.values(override).every(x => x === 0));
  // A table of a kind this runtime doesn't know is refused, not read as a grid.
  assert.throws(() => correctiveWeights({}, [{ ...axillaGroups(grid)[0], kind: "made up" }], CENTRAL), /Unsupported/);
});

test("the hanging-arm table starts below the shoulder's edge and fades off a relaxed hang", () => {
  const hang = {
    forwardKeys: [0, 20], adductionKeys: [24, 27, 30],
    sides: {
      left: { samples: [[0, 27], [0, 30], [20, 27], [20, 30]], targets: ["l0a", "l0b", "l20a", "l20b"] },
      right: { samples: [[0, 27], [0, 30], [20, 27], [20, 30]], targets: ["r0a", "r0b", "r20a", "r20b"] },
    },
    twistLimit: 90, elbowLimit: 30, handLimit: 30, hipLimit: 30,
  };
  const zero = (/** @type {Record<string, number>} */ w) => Object.values(w).every((v) => v === 0);
  assert.ok(zero(hangWeights({}, hang)));
  assert.ok(zero(hangWeights({ upperArm: [0, 0, -24] }, hang)));
  // The left arm adducts with negative Z, the right with positive; both read the same key.
  const down = hangWeights({ leftUpperArm: [0, 0, -30], rightUpperArm: swing(20, 0, 27) }, hang);
  assert.equal(down.l0b, 1);
  assert.ok(Math.abs(down.r20a - 1) < 1e-9);
  // Past the last key the correction holds.
  assert.equal(hangWeights({ leftUpperArm: [0, 0, -40] }, hang).l0b, 1);
  // A bent elbow, a turned hand or a flexed hip on that side fades it out.
  assert.ok(zero(hangWeights({ upperArm: [0, 0, -30], lowerArm: [30, 0, 0] }, hang)));
  assert.equal(hangWeights({ upperArm: [0, 0, -30], leftHand: [0, 0, 15] }, hang).l0b, 0.5);
  const hip = hangWeights({ upperArm: [0, 0, -30], rightUpperLeg: [45, 0, 0] }, hang);
  assert.equal(hip.l0b, 1);
  assert.equal(hip.r0b, 0);
});

test("only the jaw moves off its place, by how far it opens", () => {
  const table = /** @type {any} */ ({ glide: { jaw: [0, 0.002, -0.001] } });
  assert.deepEqual(poseTranslations({}, table), {});
  assert.deepEqual(poseTranslations({ head: [20, 0, 0] }, table), {});
  assert.deepEqual(poseTranslations({ jaw: [10, 3, -2] }, table).jaw, [0, 0.02, -0.01]);
  // Closing past rest doesn't slide it back.
  assert.deepEqual(poseTranslations({ jaw: [-5, 0, 0] }, table).jaw.map(Math.abs), [0, 0, 0]);
  assert.deepEqual(poseTranslations({ jaw: [10, 0, 0] }, /** @type {any} */ ({ glide: {} })), {});
});
