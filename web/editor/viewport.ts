// The 3D view: scene, camera and look. It knows nothing about parameters; the store hands it bodies.

import type { Pose } from "../runtime/driver.mjs";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { LoadedBody } from "../runtime/body.mjs";
import { type LookName, makeLooks } from "./looks";
import { REGION_FOCUS, type RegionId, regionOfBone } from "./regions";

export type ViewName = "front" | "side" | "back" | "threequarter";
export type CropName = "full" | "upper";
export type { LookName } from "./looks";

const DIRECTIONS: Record<ViewName, [number, number, number]> = {
  front: [0, 0, 1],
  side: [1, 0, 0],
  back: [0, 0, -1],
  threequarter: [0.6, 0.05, 1],
};
const FLIGHT_MS = 320;
// A press that moved further than this was an orbit, not a click on the body.
const CLICK_PX = 5;

// The dark one is a middle grey and not dark: the toon outline is near black and has to stay visible,
// because the body is judged by its look.
const THEMES = {
  light: { background: 0xeceae6, gridCenter: 0xb5b0a8, grid: 0xcfcbc4 },
  dark: { background: 0x85868b, gridCenter: 0x6c6d72, grid: 0x77787d },
};

const bodyHeight = (body: LoadedBody) => {
  const geometry = body.mesh.geometry;
  geometry.computeBoundingBox();
  return (geometry.boundingBox as THREE.Box3).max.y;
};

export function disposeBody(body: LoadedBody) {
  for (const mesh of body.meshes) {
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  body.mesh.skeleton.dispose();
}

export class Viewport {
  readonly renderer = new THREE.WebGLRenderer({ antialias: true });
  private readonly world = new THREE.Scene();
  private readonly hemisphere = new THREE.HemisphereLight(0xffffff, 0x8a8078, 1.6);
  private readonly key = new THREE.DirectionalLight(0xffffff, 2.2);
  // The near plane well out, so that W2's thin shells don't flicker at full-body distance.
  private readonly camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
  private readonly controls: OrbitControls;
  private readonly looks: ReturnType<typeof makeLooks>;
  private grid: THREE.GridHelper | null = null;
  private flight: { start: number; fromTarget: THREE.Vector3; toTarget: THREE.Vector3; fromPosition: THREE.Vector3; toPosition: THREE.Vector3 } | null = null;
  private pressed: { x: number; y: number } | null = null;
  private observer: ResizeObserver | null = null;
  private dirty = false;
  /** What's lit and drawn, and the one that was there before it, kept for comparing. */
  body: LoadedBody | null = null;
  previous: LoadedBody | null = null;
  comparing = false;
  look: LookName = "toon";
  /** Called after each frame that drew something. */
  onDrawn: () => void = () => {};
  /** Called when a click, not a drag, lands on the body. */
  onPick: (region: RegionId) => void = () => {};

  constructor() {
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.world.add(this.hemisphere);
    this.key.position.set(1.5, 2.5, 3);
    this.world.add(this.key);
    this.world.add(this.camera);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.minDistance = 0.4;
    // The look people will judge the body by. Its key light rides on the camera, as in
    // scripts/render.py, so the terminator stays put while orbiting and only a new body changes it.
    this.looks = makeLooks(this.world, this.camera, this.renderer);
    this.setTheme("light");
    this.controls.addEventListener("change", () => this.invalidate());
    // Whoever takes the camera by hand takes it from any flight under way.
    this.controls.addEventListener("start", () => {
      this.flight = null;
    });
    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointerdown", (event) => {
      this.pressed = event.button === 0 && event.isPrimary ? { x: event.clientX, y: event.clientY } : null;
    });
    canvas.addEventListener("pointerup", (event) => this.released(event));
    requestAnimationFrame(this.loop);
  }

  attach(host: HTMLElement) {
    host.append(this.renderer.domElement);
    this.observer?.disconnect();
    this.observer = new ResizeObserver(() => this.resize(host));
    this.observer.observe(host);
    this.resize(host);
  }

  private resize(host: HTMLElement) {
    const { clientWidth: width, clientHeight: height } = host;
    if (!width || !height) return;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  invalidate() {
    this.dirty = true;
  }

  setTheme(theme: keyof typeof THEMES) {
    const colors = THEMES[theme];
    this.world.background = new THREE.Color(colors.background);
    if (this.grid) {
      this.world.remove(this.grid);
      this.grid.dispose();
    }
    this.grid = new THREE.GridHelper(2, 20, colors.gridCenter, colors.grid);
    this.world.add(this.grid);
    this.invalidate();
  }

  /** The new body takes the old one's place in one go. `keepPrevious` holds the old one for comparing. */
  land(loaded: LoadedBody, keepPrevious: boolean) {
    this.clearPrevious();
    this.looks.off();
    if (this.body) {
      if (keepPrevious) {
        this.previous = this.body;
        this.previous.scene.visible = false;
      } else {
        this.world.remove(this.body.scene);
        disposeBody(this.body);
      }
    }
    this.body = loaded;
    this.world.add(loaded.scene);
  }

  clearPrevious() {
    this.showBefore(false);
    if (!this.previous) return;
    this.world.remove(this.previous.scene);
    disposeBody(this.previous);
    this.previous = null;
  }

  showBefore(held: boolean) {
    if (held === this.comparing || !this.previous || !this.body) return;
    this.comparing = held;
    this.body.scene.visible = !held;
    this.previous.scene.visible = held;
    this.applyLook();
  }

  /** The jaw and the eyes, over whichever pose is shown. Not stored in a body's file. */
  private face: Pose = {};

  setFace(face: Pose) {
    this.face = face;
  }

  /** A check pose by name, or `rotations` for a pose the body's file doesn't carry. */
  applyPose(pose: string, driven = true, rotations?: Record<string, [number, number, number]>) {
    for (const body of [this.body, this.previous]) {
      if (!body) continue;
      const posed = { ...(rotations ?? body.info.checkPoses[pose] ?? {}), ...this.face };
      if (driven) body.setPose(posed);
      else body.setNamedOnly(posed);
    }
    this.invalidate();
  }

  /** Parts and expressions, on what's shown and what it would be compared with. */
  applyInspection(hiddenParts: string[], morphs: Record<string, number>) {
    for (const body of [this.body, this.previous]) {
      if (!body) continue;
      for (const part of body.parts) part.visible = !hiddenParts.includes(part.userData.name);
      body.setMorphs(morphs);
    }
    this.invalidate();
  }

  setLook(look: LookName) {
    this.look = look;
    this.applyLook();
  }

  private applyLook() {
    const shown = this.comparing ? this.previous : this.body;
    if (!shown) return;
    this.looks.show(this.look, shown, [this.hemisphere, this.key]);
    this.invalidate();
  }

  /** The whole standing body, from slightly off the front, at once. */
  frameFirst() {
    if (!this.body) return;
    const height = bodyHeight(this.body);
    const distance = (1.2 * height) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
    const direction = new THREE.Vector3(0.35, 0.08, 1).normalize();
    this.flight = null;
    this.controls.target.set(0, 0.45 * height, 0);
    this.camera.position.copy(this.controls.target).addScaledVector(direction, distance);
    this.controls.update();
  }

  /** Turn around whatever is being looked at, keeping the zoom. */
  view(view: ViewName) {
    const direction = new THREE.Vector3(...DIRECTIONS[view]).normalize();
    const target = this.flight?.toTarget ?? this.controls.target;
    this.fly(target, target.clone().addScaledVector(direction, this.controls.getDistance()));
  }

  private fit(span: number) {
    return span / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
  }

  /** Fit the body, in the pose it's in, keeping the side it's seen from. */
  crop(crop: CropName) {
    const body = this.body;
    if (!body) return;
    const direction = this.camera.position.clone().sub(this.controls.target).normalize();
    const target = new THREE.Vector3();
    let span: number;
    if (crop === "upper") {
      body.mesh.skeleton.getBoneByName("upperChest")?.getWorldPosition(target);
      span = (0.47 * bodyHeight(body)) / Math.min(1, this.camera.aspect);
    } else {
      // The skinned bounds include the active pose, including a crouch.
      body.mesh.computeBoundingBox();
      const bounds = (body.mesh.boundingBox as THREE.Box3).clone().applyMatrix4(body.mesh.matrixWorld);
      const size = bounds.getSize(new THREE.Vector3());
      bounds.getCenter(target);
      span = 1.16 * Math.max(size.y, Math.max(size.x, size.z) / this.camera.aspect);
    }
    this.fly(target, target.clone().addScaledVector(direction, this.fit(span)));
  }

  /** Go to where a region can be judged. The whole-figure regions fit the body from where it is. */
  focus(region: RegionId, { instant = false } = {}) {
    const body = this.body;
    if (!body) return;
    const focus = REGION_FOCUS[region];
    if (!focus) {
      this.crop("full");
      return;
    }
    const target = new THREE.Vector3();
    const point = new THREE.Vector3();
    let found = 0;
    for (const name of focus.bones) {
      const bone = body.mesh.skeleton.getBoneByName(name);
      if (!bone) continue;
      target.add(bone.getWorldPosition(point));
      found += 1;
    }
    if (!found) return;
    target.divideScalar(found);
    target.y += (focus.lift ?? 0) * bodyHeight(body);
    const direction = new THREE.Vector3(...focus.direction).normalize();
    const span = (focus.span * bodyHeight(body)) / Math.min(1, this.camera.aspect);
    this.fly(target, target.clone().addScaledVector(direction, this.fit(span)), instant);
  }

  private fly(toTarget: THREE.Vector3, toPosition: THREE.Vector3, instant = false) {
    if (instant || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.flight = null;
      this.controls.target.copy(toTarget);
      this.camera.position.copy(toPosition);
      this.controls.update();
      return;
    }
    this.flight = {
      start: performance.now(),
      fromTarget: this.controls.target.clone(),
      toTarget: toTarget.clone(),
      fromPosition: this.camera.position.clone(),
      toPosition: toPosition.clone(),
    };
    this.invalidate();
  }

  private released(event: PointerEvent) {
    const from = this.pressed;
    this.pressed = null;
    if (!from || !this.body) return;
    if (Math.hypot(event.clientX - from.x, event.clientY - from.y) > CLICK_PX) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(pointer, this.camera);
    const mesh = this.body.mesh;
    const [hit] = ray.intersectObject(mesh, false);
    if (!hit?.face) return;
    const skinIndex = mesh.geometry.getAttribute("skinIndex");
    const skinWeight = mesh.geometry.getAttribute("skinWeight");
    const weights = new Map<number, number>();
    for (const vertex of [hit.face.a, hit.face.b, hit.face.c]) {
      for (let slot = 0; slot < 4; slot++) {
        const bone = skinIndex.getComponent(vertex, slot);
        weights.set(bone, (weights.get(bone) ?? 0) + skinWeight.getComponent(vertex, slot));
      }
    }
    const [bone] = [...weights.entries()].sort((a, b) => b[1] - a[1])[0];
    const name = mesh.skeleton.bones[bone]?.name ?? "";
    this.onPick(regionOfBone(name, hit.point.z < 0));
  }

  private loop = () => {
    requestAnimationFrame(this.loop);
    if (this.flight) {
      const { start, fromTarget, toTarget, fromPosition, toPosition } = this.flight;
      const t = Math.min(1, (performance.now() - start) / FLIGHT_MS);
      const eased = 1 - (1 - t) ** 3;
      this.controls.target.lerpVectors(fromTarget, toTarget, eased);
      this.camera.position.lerpVectors(fromPosition, toPosition, eased);
      this.controls.update();
      if (t === 1) this.flight = null;
      this.dirty = true;
    }
    if (!this.body || !this.dirty) return;
    this.dirty = false;
    this.looks.update();
    this.renderer.render(this.world, this.camera);
    this.onDrawn();
  };
}
