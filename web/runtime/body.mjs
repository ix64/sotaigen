// An exported body in three.js: the GLB, its skinned meshes and the driver that poses them.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { checkInfo, correctiveWeights, fromEuler, poseQuaternions, poseTranslations, rootTranslation } from "./driver.mjs";

/** @typedef {import("./driver.mjs").DriverTable} DriverTable */
/** @typedef {import("./driver.mjs").Pose} Pose */
/** @typedef {import("./driver.mjs").Quaternion} Quaternion */

/**
 * extras.sotaigen of an exported GLB: sotaigen/output/glb.py glb_bytes and export_preset.
 * @typedef {object} BodyInfo
 * @property {{groups: import("./driver.mjs").CorrectiveGroup[]}} [correctives] The pose correctives' table.
 * @property {DriverTable} drivers
 * @property {{bones: string[], region: [number, number], neckRing: number[], crownVertex: number, eyeLine: number, eyeX: number}} [head]
 *   What a head swap may rely on: sotaigen/output/glb.py head_attachments.
 * @property {Record<string, Pose>} checkPoses sotaigen.rig.poses.POSES.
 * @property {number[]} [splitFrom] For each vertex after the source's, the one it copies at a UV seam.
 * @property {string} format "sotaigen/glb".
 * @property {number} version
 * @property {{name: string, version: string, fingerprint: string}} generator
 * @property {string} [name] The typical or test body whose parameters these are.
 * @property {object} parameters The parameter file the body was made from.
 * @property {string[]} expressions
 */

/** @typedef {THREE.BufferGeometry<Record<string, THREE.BufferAttribute>>} BodyGeometry */

/**
 * The file's material, or a look's while it is on: toon.mjs's, or web/editor/looks.ts's.
 * @typedef {THREE.MeshStandardMaterial | THREE.MeshToonMaterial | THREE.MeshBasicMaterial | THREE.MeshMatcapMaterial} BodyMaterial
 */

/** One skinned mesh of the file. @typedef {THREE.SkinnedMesh<BodyGeometry, BodyMaterial>} BodyMesh */

/**
 * What loadBody gives back.
 * @typedef {object} LoadedBody
 * @property {THREE.Group} scene
 * @property {BodyMesh} mesh The skin: the mesh named "body".
 * @property {BodyMesh[]} meshes The skin and the parts, sharing the skin's skeleton.
 * @property {BodyMesh[]} parts Everything but the skin.
 * @property {string[]} morphNames Manual expression targets across all meshes, in the order they first appear.
 * @property {BodyInfo} info
 * @property {number} bytes The file's size.
 * @property {number} fetchMs
 * @property {number} parseMs
 * @property {(rotations: Record<string, Quaternion>, pose?: Pose, driven?: boolean, moves?: Record<string, [number, number, number]>) => void} setRotations
 *   Each bone's final local rotation, relative to its rest. A bone left out stays at rest. `moves`
 *   is where the driver puts a bone off its rest place, in metres in its rest frame.
 * @property {(weights: Record<string, number>) => void} setMorphs An omitted expression goes to zero; pose-driven targets are preserved.
 * @property {(pose: Pose) => void} setPose The pose with the helper bones driven, as the rig means it.
 * @property {(pose: Pose) => void} setNamedOnly The pose on the named bones only.
 */

// What a player without the driver does: each named bone only, for showing what the driver adds.
/**
 * @param {Pose} pose
 * @param {DriverTable} table
 * @returns {Record<string, Quaternion>}
 */
function namedOnly(pose, table) {
  /** @type {Record<string, Quaternion>} */
  const out = {};
  for (const [name, [x, y, z]] of Object.entries(pose)) {
    if (name === "rootTranslation") continue;
    if (name.startsWith("left") || name.startsWith("right") || table.central.includes(name)) {
      out[name] = fromEuler([x, y, z]);
    } else {
      const limb = name[0].toUpperCase() + name.slice(1);
      out[`left${limb}`] = fromEuler([x, y, z]);
      out[`right${limb}`] = fromEuler([x, -y, -z]);
    }
  }
  return out;
}

/**
 * @param {string} url
 * @returns {Promise<LoadedBody>}
 */
export async function loadBody(url) {
  const start = performance.now();
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const bytes = await response.arrayBuffer();
  const fetched = performance.now();
  const gltf = await new GLTFLoader().parseAsync(bytes, "");
  const parsed = performance.now();

  /** @type {BodyInfo} */
  const info = gltf.parser.json.extras?.sotaigen;
  checkInfo(info);
  /** @type {BodyMesh[]} */
  const meshes = [];
  gltf.scene.traverse((object) => {
    if (/** @type {BodyMesh} */ (object).isSkinnedMesh) meshes.push(/** @type {BodyMesh} */ (object));
  });
  const mesh = /** @type {BodyMesh} */ (meshes.find((object) => object.userData.name === "body"));
  const parts = meshes.filter((object) => object !== mesh);
  const skeleton = mesh.skeleton;
  for (const object of meshes) {
    object.frustumCulled = false;
    object.renderOrder = object.material.userData.sotaigen?.renderOrder ?? 0;
    // The loader gives each node its own Skeleton over the same bones. One is enough, and it means
    // one bone texture and one update however many parts there are.
    if (object !== mesh) object.bind(skeleton, object.bindMatrix);
  }
  const bones = skeleton.bones;
  const rest = new Map(bones.map((bone) => [bone, bone.quaternion.clone()]));
  const restPlace = new Map(bones.map((bone) => [bone, bone.position.clone()]));
  const local = new THREE.Quaternion();
  const moved = new THREE.Vector3();
  const root = bones[0];
  const rootRest = root.position.clone();

  const groups = info.correctives?.groups ?? [];
  const drivenNames = new Set(groups.flatMap((group) => group.targets.filter((name) => name !== null)));
  const drivenTargets = [...drivenNames].map((name) => {
    const index = mesh.morphTargetDictionary?.[name];
    if (index === undefined) throw new Error(`missing body morph target ${name}`);
    return { name, index };
  });

  /**
   * @param {Record<string, Quaternion>} rotations @param {Pose} pose @param {boolean} driven
   * @param {Record<string, [number, number, number]>} moves
   */
  function setRotations(rotations, pose = {}, driven = false, moves = {}) {
    if (groups.length) {
      const weights = driven ? correctiveWeights(pose, groups, info.drivers.central) : {};
      for (const { name, index } of drivenTargets) {
        /** @type {number[]} */ (mesh.morphTargetInfluences)[index] = weights[name] ?? 0;
      }
    }
    const [x, y, z] = rootTranslation(pose);
    root.position.set(rootRest.x + x, rootRest.y + y, rootRest.z + z);
    for (const bone of bones) {
      bone.quaternion.copy(/** @type {THREE.Quaternion} */ (rest.get(bone)));
      const rotation = rotations[bone.name];
      if (rotation) bone.quaternion.multiply(local.fromArray(rotation));
      if (bone === root) continue;
      bone.position.copy(/** @type {THREE.Vector3} */ (restPlace.get(bone)));
      const move = moves[bone.name];
      if (move) {
        bone.position.add(moved.fromArray(move).applyQuaternion(/** @type {THREE.Quaternion} */ (rest.get(bone))));
      }
    }
    gltf.scene.updateMatrixWorld(true);
    skeleton.update();
  }

  // Reserve targets on the body that this file assigns to the pose driver.
  const morphNames = [
    ...new Set(meshes.flatMap((object) => Object.keys(object.morphTargetDictionary ?? {})
      .filter((name) => object !== mesh || !drivenNames.has(name)))),
  ];

  // A target of the same name on several meshes is one expression. A name not given goes to zero.
  /** @param {Record<string, number>} weights */
  function setMorphs(weights) {
    for (const object of meshes) {
      for (const [name, index] of Object.entries(object.morphTargetDictionary ?? {})) {
        if (object === mesh && drivenNames.has(name)) continue;
        /** @type {number[]} */ (object.morphTargetInfluences)[index] = weights[name] ?? 0;
      }
    }
  }

  return {
    scene: gltf.scene,
    mesh,
    meshes,
    parts,
    morphNames,
    info,
    bytes: bytes.byteLength,
    fetchMs: fetched - start,
    parseMs: parsed - fetched,
    setRotations,
    setMorphs,
    // The pose with the helper bones driven, as the rig means it.
    setPose: (pose) => setRotations(poseQuaternions(pose, info.drivers), pose, true, poseTranslations(pose, info.drivers)),
    setNamedOnly: (pose) => setRotations(namedOnly(pose, info.drivers), pose),
  };
}
