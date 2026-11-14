// Skinned positions as three.js really computes them, against sotaigen.rig.skeleton.pose_verts.
//
// Three readings per pose:
//   gpu    three's own skinning shader chunks on this page's WebGL, written to a float target.
//   cpu    three's SkinnedMesh.getVertexPosition.
//   clip   gpu again, with each rotation first turned into a quaternion and back, as animation
//          data would arrive at the driver.
//   undriven gpu again with only the named bones turned: what the check must be able to see.
// The body goes through the same setPose as the page's body. Nothing here knows a skinning formula.
//
// checkParts does the same for each mesh of the exported file, under morph weights as well, reads
// back the normals three.js holds, and renders the skin with its normals and the skin-alone file's.

import * as THREE from "three";
import { loadBody } from "./body.mjs";
import { eulerFromQuaternion } from "./driver.mjs";

/** @typedef {import("./body.mjs").BodyMesh} BodyMesh */
/** @typedef {import("./body.mjs").LoadedBody} LoadedBody */
/** @typedef {import("./driver.mjs").Pose} Pose */

/** Reads one vertex's position into `out`. @typedef {(i: number, out: number[]) => void} Reading */

const WIDTH = 512;
const MM_PER_M = 1000;

const VERTEX = /* glsl */ `
  #include <common>
  #include <morphtarget_pars_vertex>
  #include <skinning_pars_vertex>
  uniform ivec2 size;
  varying vec3 reading;
  void main() {
    #include <skinbase_vertex>
    #include <begin_vertex>
    #include <morphtarget_vertex>
    #include <skinning_vertex>
    reading = (modelMatrix * vec4(transformed, 1.0)).xyz;
    // One pixel per vertex, by the vertex's number.
    vec2 pixel = vec2(gl_VertexID % size.x, gl_VertexID / size.x) + 0.5;
    gl_Position = vec4(pixel / vec2(size) * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = 1.0;
  }
`;
const FRAGMENT = /* glsl */ `
  varying vec3 reading;
  void main() {
    gl_FragColor = vec4(reading, 1.0);
  }
`;

// A point cloud of the mesh's vertices that three treats as a skinned mesh of the same skeleton,
// with the same morph targets and the same array of weights.
/**
 * @param {THREE.WebGLRenderer} renderer
 * @param {BodyMesh} mesh
 * @returns {() => Reading}
 */
function gpuReader(renderer, mesh) {
  const count = mesh.geometry.attributes.position.count;
  const height = Math.ceil(count / WIDTH);
  const target = new THREE.WebGLRenderTarget(WIDTH, height, {
    type: THREE.FloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
  });
  const geometry = new THREE.BufferGeometry();
  for (const name of ["position", "skinIndex", "skinWeight"]) {
    geometry.setAttribute(name, mesh.geometry.attributes[name]);
  }
  geometry.morphAttributes = mesh.geometry.morphAttributes;
  geometry.morphTargetsRelative = mesh.geometry.morphTargetsRelative;
  const material = new THREE.ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms: { size: { value: new THREE.Vector2(WIDTH, height) } },
    blending: THREE.NoBlending,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  /** @type {any} */
  const points = new THREE.Points(geometry, material);
  points.isSkinnedMesh = true;
  points.skeleton = mesh.skeleton;
  points.bindMatrix = mesh.bindMatrix;
  points.bindMatrixInverse = mesh.bindMatrixInverse;
  points.morphTargetInfluences = mesh.morphTargetInfluences;
  points.morphTargetDictionary = mesh.morphTargetDictionary;
  points.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(points);
  const camera = new THREE.Camera();
  const pixels = new Float32Array(WIDTH * height * 4);
  return () => {
    renderer.setRenderTarget(target);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(target, 0, 0, WIDTH, height, pixels);
    renderer.setRenderTarget(null);
    return (i, out) => {
      out[0] = pixels[i * 4];
      out[1] = pixels[i * 4 + 1];
      out[2] = pixels[i * 4 + 2];
    };
  };
}

/**
 * @param {BodyMesh} mesh
 * @returns {() => Reading}
 */
function cpuReader(mesh) {
  const vector = new THREE.Vector3();
  return () => (i, out) => {
    mesh.getVertexPosition(i, vector);
    vector.applyMatrix4(mesh.matrixWorld);
    out[0] = vector.x;
    out[1] = vector.y;
    out[2] = vector.z;
  };
}

/**
 * @param {Reading} read
 * @param {Float64Array} reference
 * @param {number} count
 * @param {number} [from]
 */
function difference(read, reference, count, from = 0) {
  const at = [0, 0, 0];
  let worst = 0;
  let worstVertex = -1;
  let sum = 0;
  for (let i = 0; i < count; i++) {
    read(i, at);
    const dx = at[0] * MM_PER_M - reference[from + i * 3];
    const dy = at[1] * MM_PER_M - reference[from + i * 3 + 1];
    const dz = at[2] * MM_PER_M - reference[from + i * 3 + 2];
    const size = Math.hypot(dx, dy, dz);
    sum += size;
    if (size > worst) {
      worst = size;
      worstVertex = i;
    }
  }
  return { maxMm: worst, meanMm: sum / count, worstVertex };
}

// Every rotation through a quaternion and back, as it would come out of a clip.
/**
 * @param {Pose} pose
 * @returns {Pose}
 */
function throughQuaternions(pose) {
  /** @type {Pose} */
  const out = {};
  for (const [name, euler] of Object.entries(pose)) {
    if (name === "rootTranslation") {
      out[name] = euler;
      continue;
    }
    const q = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(.../** @type {[number, number, number]} */ (euler.map(THREE.MathUtils.degToRad)), "ZYX"),
    );
    out[name] = eulerFromQuaternion(q.toArray());
  }
  return out;
}

window.checkPreset = async function checkPreset(preset) {
  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(16, 16);
  const body = await loadBody(`/build/bodies/${preset}/body.glb`);
  const count = body.mesh.geometry.attributes.position.count;
  const readGpu = gpuReader(renderer, body.mesh);
  const readCpu = cpuReader(body.mesh);
  /** @type {Record<string, any>} */
  const rows = {};
  for (const [pose, rotations] of Object.entries(body.info.checkPoses)) {
    const response = await fetch(`/build/reports/runtime-check/reference/${preset}/${pose}.f64`);
    if (!response.ok) throw new Error(`no reference for ${preset} ${pose}`);
    const reference = new Float64Array(await response.arrayBuffer());
    if (reference.length !== count * 3) throw new Error(`${preset} ${pose}: reference size`);
    body.setPose(rotations);
    const gpu = difference(readGpu(), reference, count);
    const cpu = difference(readCpu(), reference, count);
    body.setPose(throughQuaternions(rotations));
    const clip = difference(readGpu(), reference, count);
    body.setNamedOnly(rotations);
    const undriven = difference(readGpu(), reference, count);
    rows[pose] = { gpu, cpu, clip, undriven };
  }

  // What one frame's posing costs, with every bone recomputed.
  const moving = body.info.checkPoses.reach;
  const start = performance.now();
  const frames = 2000;
  for (let i = 0; i < frames; i++) body.setPose(moving);
  const setPoseMs = (performance.now() - start) / frames;

  const gl = renderer.getContext();
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const glRenderer = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : "unknown";
  renderer.dispose();
  renderer.forceContextLoss();
  return { preset, count, glRenderer, setPoseMs, poses: rows };
};
// Pixels of one view of a scene, with the page's standard lights.
/**
 * @param {THREE.WebGLRenderer} renderer
 * @param {LoadedBody} body
 * @param {number} size
 */
function renderPixels(renderer, body, size) {
  const target = new THREE.WebGLRenderTarget(size, size, { samples: 0 });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeceae6);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8078, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(1.5, 2.5, 3);
  scene.add(key);
  scene.add(body.scene);
  body.scene.updateMatrixWorld(true);
  const height = new THREE.Box3().setFromBufferAttribute(body.mesh.geometry.attributes.position).max.y;
  const camera = new THREE.PerspectiveCamera(10, 1, 0.01, 50);
  camera.position.set(0.3, 0.92 * height, 2.2);
  camera.lookAt(0, 0.92 * height, 0);
  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  const pixels = new Uint8Array(size * size * 4);
  renderer.readRenderTargetPixels(target, 0, 0, size, size, pixels);
  renderer.setRenderTarget(null);
  target.dispose();
  scene.remove(body.scene);
  return pixels;
}

/**
 * @param {Uint8Array} a
 * @param {Uint8Array} b
 * @param {number} threshold
 */
function differingPixels(a, b, threshold) {
  let count = 0;
  for (let i = 0; i < a.length; i += 4) {
    const gap = Math.max(
      Math.abs(a[i] - b[i]),
      Math.abs(a[i + 1] - b[i + 1]),
      Math.abs(a[i + 2] - b[i + 2]),
    );
    if (gap > threshold) count++;
  }
  return count;
}

/**
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @param {number} from
 * @param {number} count
 */
function angleDegrees(a, b, from, count) {
  let worst = 0;
  for (let i = 0; i < count; i++) {
    const j = from + i * 3;
    const k = i * 3;
    const dot = a[k] * b[j] + a[k + 1] * b[j + 1] + a[k + 2] * b[j + 2];
    // Not acos: near 1 that can't tell float32 rounding from a hundredth of a degree.
    const x = a[k + 1] * b[j + 2] - a[k + 2] * b[j + 1];
    const y = a[k + 2] * b[j] - a[k] * b[j + 2];
    const z = a[k] * b[j + 1] - a[k + 1] * b[j];
    worst = Math.max(worst, Math.atan2(Math.hypot(x, y, z), dot));
  }
  return THREE.MathUtils.radToDeg(worst);
}

/** The correctives' group of this name, if the file has it. @param {import("./body.mjs").LoadedBody} body @param {string} name */
const groupOf = (body, name) => body.info.correctives?.groups.find((group) => group.name === name);

/** @param {import("./body.mjs").LoadedBody} body */
function checkPelvisControls(body) {
  const table = groupOf(body, "groin");
  if (!table || !table.samples) return;
  const names = /** @type {string[]} */ (table.targets);
  if (names.some(name => body.morphNames.includes(name))) {
    throw new Error("an automatic pelvis target appears among expression controls");
  }
  const sample = table.samples.findIndex(([left, right]) => left === 40 && right === 0);
  if (sample < 0) throw new Error("missing unilateral pelvis sample");
  const active = names[sample];
  /** @param {number} expected */
  function check(expected) {
    for (const name of names) {
      const index = body.mesh.morphTargetDictionary?.[name];
      const actual = index === undefined ? undefined : body.mesh.morphTargetInfluences?.[index];
      if (actual !== (name === active ? expected : 0)) throw new Error(`wrong pelvis weight for ${name}`);
    }
  }
  /** @type {Pose} */
  const pose = { leftUpperLeg: [40, 0, 0] };
  body.setPose(pose); check(1);
  body.setMorphs({}); check(1);
  body.setNamedOnly(pose); check(0);
  body.setPose({}); check(0);
}

/** Verify pose targets survive expression changes and clear when the bone driver is bypassed.
 * @param {import("./body.mjs").LoadedBody} body
 */
function checkAxillaControls(body) {
  const left = groupOf(body, "axilla.left");
  const right = groupOf(body, "axilla.right");
  if (!left || !right) return;
  const names = [...left.targets, ...right.targets].filter((name) => name !== null);
  /** @type {string|null} */
  let active = null;
  /** @type {string|null} */
  let opposite = null;
  let angle = 0;
  if (left.kind === "simplices") {
    const points = /** @type {number[][]} */ (left.points);
    const sample = points.findIndex(point => point[0] > 0 && point.slice(1).every(v => v === 0));
    if (sample < 0) throw new Error("missing pure-forward shoulder sample");
    angle = points[sample][0];
    const other = /** @type {number[][]} */ (right.points).findIndex(point => point[0] === angle && point.slice(1).every(v => v === 0));
    active = left.targets[sample];
    opposite = other < 0 ? null : right.targets[other];
  } else {
    const keys = left.axes?.[0].keys;
    if (!keys) throw new Error("missing shoulder keys");
    const sample = left.samples?.findIndex(([x, z]) => x === keys[1] && z === 0);
    if (sample == null || sample < 0) throw new Error("missing pure-forward shoulder sample");
    angle = keys[1];
    active = left.targets[sample];
    opposite = right.targets[sample];
  }
  if (!active || !opposite) throw new Error("missing paired shoulder target");
  if (names.some((name) => body.morphNames.includes(name))) {
    throw new Error("an automatic axilla target appears among the expression controls");
  }
  /** @param {number} expected */
  function check(expected) {
    for (const name of names) {
      const index = body.mesh.morphTargetDictionary?.[name];
      const actual = index === undefined ? undefined : body.mesh.morphTargetInfluences?.[index];
      if (actual !== (name === active ? expected : 0)) throw new Error(`wrong pose weight for ${name}: ${actual}`);
    }
  }
  /** @type {Pose} */
  const pose = { leftUpperArm: [angle, 0, 0] };
  body.setPose(pose);
  body.setMorphs({ happy: 0.4, [active]: 0, [opposite]: 1 });
  check(1);
  body.setNamedOnly(pose);
  check(0);
  body.setPose(pose);
  body.setRotations({});
  check(0);
  body.setMorphs({});
}

window.checkParts = async function checkParts(preset) {
  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(16, 16);
  const folder = `/build/reports/runtime-check/reference/parts/${preset}`;
  /** @type {any} */
  const manifest = await (await fetch(`${folder}/manifest.json`)).json();
  const body = await loadBody(`/build/bodies/${preset}/body.glb`);
  checkAxillaControls(body);
  checkPelvisControls(body);
  const byName = new Map(body.meshes.map((mesh) => [mesh.userData.name, mesh]));
  const found = [...byName.keys()];
  const wanted = manifest.meshes.map((/** @type {any} */ row) => row.name);
  if (JSON.stringify(found.sort()) !== JSON.stringify([...wanted].sort())) {
    throw new Error(`${preset}: meshes ${found} but the reference has ${wanted}`);
  }
  const readers = manifest.meshes.map((/** @type {any} */ row) => {
    const mesh = /** @type {BodyMesh} */ (byName.get(row.name));
    if (mesh.geometry.attributes.position.count !== row.count) {
      throw new Error(`${preset} ${row.name}: vertex count`);
    }
    return { ...row, mesh, gpu: gpuReader(renderer, mesh), cpu: cpuReader(mesh) };
  });

  /** @type {Record<string, any>} */
  const out = {};
  for (const row of readers) {
    out[row.name] = {
      count: row.count,
      hasTargets: row.targets.length > 0,
      gpu: 0,
      cpu: 0,
      worstAt: null,
      unmorphed: 0,
      frozen: 0,
    };
  }
  const restReads = {};
  for (const [setting, weights] of Object.entries(/** @type {Record<string, Record<string, number>>} */ (manifest.settings))) {
    const response = await fetch(`${folder}/${setting}.f64`);
    const reference = new Float64Array(await response.arrayBuffer());
    let from = 0;
    for (const pose of manifest.poses) {
      const rotations = body.info.checkPoses[pose];
      body.setPose(rotations);
      body.setMorphs(weights);
      for (const row of readers) {
        const gpu = difference(row.gpu(), reference, row.count, from);
        const cpu = difference(row.cpu(), reference, row.count, from);
        const line = out[row.name];
        if (gpu.maxMm > line.gpu) {
          line.gpu = gpu.maxMm;
          line.worstAt = `${setting} ${pose}`;
        }
        line.cpu = Math.max(line.cpu, cpu.maxMm);
        if (setting !== "zero") {
          // The same reference with no weights: what the check must be able to see of the targets.
          const influences = row.mesh.morphTargetInfluences;
          const saved = influences?.slice();
          influences?.fill(0);
          const still = difference(row.gpu(), reference, row.count, from);
          if (influences && saved) Object.assign(influences, saved);
          line.unmorphed = Math.max(line.unmorphed, still.maxMm);
        }
        if (setting === "zero" && pose !== "rest") {
          // The same reference with the rest pose: what the check must be able to see of the bones.
          body.setRotations({});
          const frozen = difference(row.gpu(), reference, row.count, from);
          body.setPose(rotations);
          line.frozen = Math.max(line.frozen, frozen.maxMm);
        }
        from += row.count * 3;
      }
    }
    if (from !== reference.length) throw new Error(`${preset} ${setting}: reference size`);
  }
  body.setMorphs({});
  body.setRotations({});

  // The normals three.js really holds, against what the file was given, seam copies included.
  const stored = new Float32Array(await (await fetch(`${folder}/normals.f32`)).arrayBuffer());
  /** @type {Record<string, any>} */
  const normals = {};
  let from = 0;
  for (const row of manifest.meshes) {
    const held = /** @type {BodyMesh} */ (byName.get(row.name)).geometry.attributes.normal.array;
    normals[row.name] = { maxDegrees: angleDegrees(held, stored, from, row.count) };
    from += row.count * 3;
  }
  const plain = await loadBody(`/build/reports/runtime-check/plain/${preset}.glb`);
  const given = body.mesh.geometry.attributes.normal;
  const own = plain.mesh.geometry.attributes.normal;
  normals.body.fromPlainDegrees = angleDegrees(given.array, own.array, 0, own.count);

  // Does it reach the image? The same skin rendered with its given normals and its plain ones, then
  // with the plain normals copied onto the given file's mesh, which must leave nothing to tell.
  for (const part of body.parts) part.visible = false;
  const SIZE = 480;
  const imageGiven = renderPixels(renderer, body, SIZE);
  const imagePlain = renderPixels(renderer, plain, SIZE);
  const swapped = given.array.slice();
  given.array.set(own.array);
  given.needsUpdate = true;
  const imageSwapped = renderPixels(renderer, body, SIZE);
  given.array.set(swapped);
  given.needsUpdate = true;
  const pixels = {
    size: SIZE,
    givenAgainstPlain: differingPixels(imageGiven, imagePlain, 4),
    swappedAgainstPlain: differingPixels(imageSwapped, imagePlain, 4),
  };

  renderer.dispose();
  renderer.forceContextLoss();
  return { preset, meshes: out, normals, pixels };
};
window.checkReady = true;
