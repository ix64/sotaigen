// One exported body, the whole window and nothing else, as the address says. For web/runtime/check.mjs
// and for looking at a file by hand; people adjusting a body use the editor.
//
//   /runtime/shot.html?body= &standin=1 &pose= &driven=0 &look=toon &view=threequarter
//                      &framing=head|review &hide=partA,partB &morph=blink:1,happy:0.5
//
// The stand-in parts come from scripts/standin_parts.py. It leaves bodyReady, bodyStats, bodyState and
// frameMs on window for the check. No React: its load times are the body's own.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { type LoadedBody, loadBody } from "./body.mjs";
import { makeToon } from "./toon.mjs";

// sotaigen_dev/review.py's views, as glTF directions from the body to the camera.
const VIEWS: Record<string, [number, number, number]> = {
  front: [0, 0, 1],
  threequarter: [0.64, 0.08, 0.77],
  side: [1, 0, 0],
  backquarter: [0.64, 0.08, -0.77],
  back: [0, 0, -1],
};
const DEFAULT_DIRECTION: [number, number, number] = [0.35, 0.08, 1];

function modelUrl(choice: string) {
  return choice.startsWith("standin:")
    ? `/build/reports/runtime-check/standin/${choice.slice("standin:".length)}.glb`
    : `/build/bodies/${choice}/body.glb`;
}

const query = new URLSearchParams(location.search);
const pose = query.get("pose") ?? "";
const driven = query.get("driven") !== "0";
const look = query.get("look") === "toon" ? "toon" : "standard";
const start = Object.fromEntries(query) as Record<string, string | undefined>;

async function main() {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.append(renderer.domElement);

  const world = new THREE.Scene();
  world.background = new THREE.Color(0xeceae6);
  const hemisphere = new THREE.HemisphereLight(0xffffff, 0x8a8078, 1.6);
  world.add(hemisphere);
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(1.5, 2.5, 3);
  world.add(key);
  const grid = new THREE.GridHelper(2, 20, 0xb5b0a8, 0xcfcbc4);
  world.add(grid);

  const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
  world.add(camera);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener("change", draw);
  const toon = makeToon(world, camera, renderer);

  // Set by show before anything reads it past the checks below.
  let body = null as unknown as LoadedBody;

  function frame(shown: LoadedBody) {
    const height = (shown.mesh.geometry.boundingBox as THREE.Box3).max.y;
    const direction = new THREE.Vector3(...(VIEWS[start.view ?? ""] ?? DEFAULT_DIRECTION)).normalize();
    let distance: number;
    if (start.framing === "review") {
      // What review.py's persp camera does: 1.5 heights away at half the height, 1.08 heights across.
      camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(1.08 / 2 / 1.5));
      distance = 1.5 * height;
      controls.target.set(0, 0.5 * height, 0);
    } else if (start.framing === "head") {
      camera.fov = 30;
      const bone = shown.mesh.skeleton.getBoneByName("head") as THREE.Bone;
      const from = bone.getWorldPosition(new THREE.Vector3()).y;
      const span = 1.5 * (height - from);
      distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
      controls.target.set(0, (from + height) / 2 - 0.05 * span, 0);
    } else {
      // Three quarters from the wearer's left, with room for raised arms, by the body's own height.
      camera.fov = 30;
      distance = (1.45 * height) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
      controls.target.set(0, 0.55 * height, 0);
    }
    camera.updateProjectionMatrix();
    camera.position.copy(controls.target).addScaledVector(direction, distance);
    controls.update();
  }

  function draw() {
    toon.update();
    renderer.render(world, camera);
  }

  function resize() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    draw();
  }
  window.addEventListener("resize", resize);

  function applyLook(shown: LoadedBody) {
    if (look === "toon") {
      toon.on(shown, [hemisphere, key]);
      grid.visible = false;
    } else {
      toon.off();
      grid.visible = true;
    }
  }

  function applyAddress(shown: LoadedBody, chosenPose: string) {
    const hidden = (start.hide ?? "").split(",");
    for (const part of shown.parts) part.visible = !hidden.includes(part.userData.name);
    const asked = (start.morph ?? "")
      .split(",")
      .filter(Boolean)
      .map((item): [string, number] => {
        const [name, weight] = item.split(":");
        return [name, Number(weight)];
      });
    shown.setMorphs(Object.fromEntries(asked));
    applyLook(shown);
    const posed = shown.info.checkPoses[chosenPose] ?? {};
    if (driven) shown.setPose(posed);
    else shown.setNamedOnly(posed);
    draw();
  }

  async function show(choice: string) {
    const started = performance.now();
    const next = await loadBody(modelUrl(choice));
    body = next;
    world.add(body.scene);
    for (const mesh of body.meshes) mesh.geometry.computeBoundingBox();
    body.scene.updateMatrixWorld(true);
    frame(body);
    const wanted = pose || "rest";
    const chosenPose = wanted in body.info.checkPoses ? wanted : "rest";
    applyAddress(body, chosenPose);
    // readPixels waits for the GPU, so the first frame's time includes compiling and drawing it.
    renderer.getContext().readPixels(0, 0, 1, 1, 6408, 5121, new Uint8Array(4));
    const totalMs = performance.now() - started;
    const firstFrameMs = totalMs - body.fetchMs - body.parseMs;
    const geometry = body.mesh.geometry;
    window.bodyStats = {
      preset: choice,
      pose: chosenPose,
      bytes: body.bytes,
      vertices: geometry.attributes.position.count,
      triangles: (geometry.index as THREE.BufferAttribute).count / 3,
      bones: body.mesh.skeleton.bones.length,
      parts: body.parts.map((part) => part.userData.name),
      morphs: body.morphNames,
      partVertices: body.parts.reduce((n, p) => n + p.geometry.attributes.position.count, 0),
      fetchMs: body.fetchMs,
      parseMs: body.parseMs,
      firstFrameMs,
      totalMs,
    };
  }

  // What the look and the address have reached, for the browser check to read back.
  window.bodyState = () => ({
    look: toon.isOn() ? "toon" : "standard",
    parts: Object.fromEntries(body.parts.map((p) => [p.userData.name, p.visible])),
    colors: Object.fromEntries(body.meshes.map((m) => [m.userData.name, m.material.color.toArray()])),
    influences: Object.fromEntries(
      body.meshes
        .filter((m) => m.morphTargetDictionary)
        .map((m) => [
          m.userData.name,
          Object.fromEntries(
            Object.entries(m.morphTargetDictionary as Record<string, number>).map(([n, i]) => [
              n,
              (m.morphTargetInfluences as number[])[i],
            ]),
          ),
        ]),
    ),
  });

  // Frames drawn and waited for, in ms each, for comparing the looks.
  window.frameMs = (frames = 120) => {
    const gl = renderer.getContext();
    const pixel = new Uint8Array(4);
    const started = performance.now();
    for (let i = 0; i < frames; i++) {
      draw();
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    }
    return {
      ms: (performance.now() - started) / frames,
      calls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
    };
  };

  const { presets, standIns }: { presets: string[]; standIns: string[] } = await (await fetch("/api/exported")).json();
  if (presets.length === 0) throw new Error("No exported body. Run: uv run sotaigen-dev export");
  resize();
  const asked = start.body ?? "";
  const first = presets.includes(asked) ? asked : presets.includes("female-long") ? "female-long" : presets[0];
  await show(start.standin === "1" && standIns.includes(first) ? `standin:${first}` : first);
  window.bodyReady = true;
}

main();
