// The editor's looks besides the file's own: toon.mjs's toon, and three for judging form that change
// only the materials and draw over the body, never its geometry.
//
// silhouette: one flat colour, for proportion and the gaps between the limbs and the trunk.
// matcap: a sculptor's clay lit from the camera, for small bumps, dents and flat spots.
// xray: a see-through skin and the skeleton on top, for whether the joints sit inside the forms.
//
// The matcaps are drawn here, not loaded, so the page needs no image and the light can be read off.

import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import type { BodyMaterial, BodyMesh, LoadedBody } from "../runtime/body.mjs";
import { makeToon } from "../runtime/toon.mjs";

export const LOOKS = ["toon", "standard", "silhouette", "matcap", "xray"] as const;
export type LookName = (typeof LOOKS)[number];
export const isLook = (name: string | undefined): name is LookName => LOOKS.includes(name as LookName);

// Dark enough to read on both themes' backgrounds, and not black, so the toon ink stays its own.
const SILHOUETTE = 0x2f3136;
// The clay, in linear light: a warm light terracotta, the colour sculptors judge skin forms in.
const CLAY: [number, number, number] = [0.62, 0.33, 0.24];
// The X-ray's skin: a pale cool grey, opaque at the edges and nearly clear where it faces the camera.
const GLASS: [number, number, number] = [0.3, 0.4, 0.55];
const GLASS_EDGE = 0.7;
const GLASS_FACE = 0.08;
// The humanoid joints, and the helper bones the driver turns (Half, Twist), smaller and apart.
const JOINT = new THREE.Color(0xff5a1f);
const HELPER = new THREE.Color(0x19b3c4);
const JOINT_PX = 8;
const HELPER_PX = 5;
const BONE_PX = 2;
// The matcap's light, in the camera's frame: from the left and above, as toon.mjs's key.
const MATCAP_KEY = new THREE.Vector3(-0.45, 0.55, 0.7).normalize();
const MATCAP_FILL = new THREE.Vector3(0.6, -0.35, 0.5).normalize();

const isHelper = (name: string) => /(Half|Twist\d)$/.test(name);

const toSrgb = (linear: number) =>
  Math.round(255 * Math.min(1, linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055));

/**
 * A matcap of the given colour: a key from MATCAP_KEY, wrapped so the shade stays readable, a weak fill
 * from below on the other side, a small highlight, and edges darkened so the outline holds.
 */
function drawMatcap(color: [number, number, number], highlight: number) {
  const size = 256;
  const data = new Uint8Array(size * size * 4);
  const normal = new THREE.Vector3();
  const half = new THREE.Vector3(0, 0, 1).add(MATCAP_KEY).normalize();
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const x = ((column + 0.5) / size) * 2 - 1;
      // A texture's first row is its bottom: up in the image is up on screen.
      const y = ((row + 0.5) / size) * 2 - 1;
      const r = Math.hypot(x, y);
      if (r < 1) normal.set(x, y, Math.sqrt(1 - r * r));
      else normal.set(x / r, y / r, 0);
      const key = Math.max(0, (normal.dot(MATCAP_KEY) + 0.35) / 1.35) ** 1.6;
      const fill = 0.12 * Math.max(0, normal.dot(MATCAP_FILL));
      const ambient = 0.08 + 0.1 * normal.z;
      const rim = THREE.MathUtils.lerp(0.55, 1, THREE.MathUtils.smoothstep(normal.z, 0, 0.4));
      const shine = highlight * Math.max(0, normal.dot(half)) ** 36;
      const at = (row * size + column) * 4;
      for (let i = 0; i < 3; i++) data[at + i] = toSrgb((color[i] * (ambient + 0.9 * key + fill) + shine) * rim);
      data[at + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(data, size, size);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

// The X-ray's skin: the matcap's alpha by how squarely each point faces the camera.
function useGlass(shader: THREE.WebGLProgramParametersWithUniforms) {
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <opaque_fragment>",
    /* glsl */ `
      float facing = abs(dot(normal, normalize(vViewPosition)));
      diffuseColor.a *= mix(${GLASS_EDGE.toFixed(3)}, ${GLASS_FACE.toFixed(3)}, smoothstep(0.0, 0.85, facing));
      #include <opaque_fragment>
    `,
  );
}

// Round dots of a size in CSS pixels, each with a dark ring so it reads on the skin and off it.
const DOT_VERTEX = /* glsl */ `
  attribute float size;
  attribute vec3 color;
  uniform float pixelRatio;
  varying vec3 vColor;
  void main() {
    vColor = color;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * pixelRatio;
  }
`;
const DOT_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  void main() {
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;
    float ring = smoothstep(0.62, 0.72, r);
    gl_FragColor = vec4(mix(vColor, vec3(0.05), ring), 1.0);
    #include <colorspace_fragment>
  }
`;

/** The skeleton of one body, drawn over everything and following the pose at every frame. */
class SkeletonOverlay extends THREE.Group {
  private readonly bones: THREE.Bone[];
  private readonly pairs: [THREE.Bone, THREE.Bone][];
  private readonly dots: THREE.Points;
  private readonly lines: LineSegments2;
  private readonly at = new THREE.Vector3();
  private readonly lineMaterial: LineMaterial;
  private readonly dotMaterial: THREE.ShaderMaterial;

  constructor(body: LoadedBody) {
    super();
    // The root is the floor, not a joint.
    this.bones = body.mesh.skeleton.bones.filter((bone) => bone.name !== "root");
    this.pairs = this.bones
      .filter((bone) => !isHelper(bone.name) && bone.parent instanceof THREE.Bone && bone.parent.name !== "root")
      .map((bone) => [bone.parent as THREE.Bone, bone]);
    // Helpers first, so the joints they sit on are drawn over them.
    this.bones.sort((a, b) => Number(isHelper(b.name)) - Number(isHelper(a.name)));

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.bones.length * 3), 3));
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(this.bones.flatMap((bone) => (isHelper(bone.name) ? HELPER : JOINT).toArray())), 3),
    );
    geometry.setAttribute(
      "size",
      new THREE.BufferAttribute(new Float32Array(this.bones.map((bone) => (isHelper(bone.name) ? HELPER_PX : JOINT_PX))), 1),
    );
    this.dotMaterial = new THREE.ShaderMaterial({
      vertexShader: DOT_VERTEX,
      fragmentShader: DOT_FRAGMENT,
      uniforms: { pixelRatio: { value: 1 } },
      depthTest: false,
      depthWrite: false,
    });
    this.dots = new THREE.Points(geometry, this.dotMaterial);
    this.dots.renderOrder = 11;
    this.dots.frustumCulled = false;

    this.lineMaterial = new LineMaterial({ color: JOINT, linewidth: BONE_PX, depthTest: false, depthWrite: false, transparent: true, opacity: 0.85 });
    this.lines = new LineSegments2(new LineSegmentsGeometry(), this.lineMaterial);
    this.lines.renderOrder = 10;
    this.lines.frustumCulled = false;
    this.add(this.lines, this.dots);
  }

  setViewport(width: number, height: number, pixelRatio: number) {
    this.lineMaterial.resolution.set(width, height);
    this.dotMaterial.uniforms.pixelRatio.value = pixelRatio;
  }

  // Added after the body, so its bones' world matrices are this frame's by the time this runs.
  override updateMatrixWorld(force?: boolean) {
    const position = this.dots.geometry.getAttribute("position") as THREE.BufferAttribute;
    this.bones.forEach((bone, i) => {
      this.at.setFromMatrixPosition(bone.matrixWorld);
      position.setXYZ(i, this.at.x, this.at.y, this.at.z);
    });
    position.needsUpdate = true;
    const ends: number[] = [];
    for (const [from, to] of this.pairs) {
      ends.push(...this.at.setFromMatrixPosition(from.matrixWorld).toArray());
      ends.push(...this.at.setFromMatrixPosition(to.matrixWorld).toArray());
    }
    this.lines.geometry.setPositions(ends);
    super.updateMatrixWorld(force);
  }

  dispose() {
    this.dots.geometry.dispose();
    this.dotMaterial.dispose();
    this.lines.geometry.dispose();
    this.lineMaterial.dispose();
  }
}

/**
 * All the looks, one on at a time. `show` puts a look on a body, taking off whatever was on; "standard"
 * is the file's own materials and lights.
 */
export function makeLooks(world: THREE.Scene, camera: THREE.Camera, renderer: THREE.WebGLRenderer) {
  const toon = makeToon(world, camera, renderer);
  const clay = drawMatcap(CLAY, 0.18);
  const glass = drawMatcap(GLASS, 0.3);
  const size = new THREE.Vector2();

  function material(look: LookName, from: BodyMaterial): BodyMaterial {
    if (look === "silhouette") return new THREE.MeshBasicMaterial({ color: SILHOUETTE, side: from.side });
    if (look === "matcap") return new THREE.MeshMatcapMaterial({ matcap: clay, side: from.side });
    const see = new THREE.MeshMatcapMaterial({ matcap: glass, transparent: true, depthWrite: false });
    see.onBeforeCompile = useGlass;
    return see;
  }

  let active: { entries: { mesh: BodyMesh; material: BodyMaterial }[]; skeleton: SkeletonOverlay | null } | null = null;

  function off() {
    toon.off();
    if (!active) return;
    for (const { mesh, material: was } of active.entries) {
      mesh.material.dispose();
      mesh.material = was;
    }
    if (active.skeleton) {
      active.skeleton.removeFromParent();
      active.skeleton.dispose();
    }
    active = null;
  }

  /** `lights` are the file's look's, which toon turns off for its own. The others need none. */
  function show(look: LookName, body: LoadedBody, lights: THREE.Light[]) {
    off();
    if (look === "standard") return;
    if (look === "toon") {
      toon.on(body, lights);
      return;
    }
    const entries = body.meshes.map((mesh) => ({ mesh, material: mesh.material }));
    for (const { mesh, material: was } of entries) mesh.material = material(look, was);
    let skeleton: SkeletonOverlay | null = null;
    if (look === "xray") {
      skeleton = new SkeletonOverlay(body);
      world.add(skeleton);
    }
    active = { entries, skeleton };
  }

  // Before each frame.
  function update() {
    toon.update();
    if (active?.skeleton) {
      renderer.getSize(size);
      active.skeleton.setViewport(size.x, size.y, renderer.getPixelRatio());
    }
  }

  return { show, off, update };
}
