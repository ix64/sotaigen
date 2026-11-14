// The look scripts/render.py calls toon, in three.js: two flat tones with a hard step, and an
// inverted hull for the outline. MeshToonMaterial with its step replaced, no VRM package.
//
// The numbers are render.py's, so the two can be held side by side: SHADE over LIT, TERMINATOR, KEY,
// OUTLINE_PX and OUTLINE_BEHIND. The colours are the file's, so the skin and the parts that share its
// colour read as one, lit and in shade. A material of colour LIT would get render.py's two tones
// exactly. This is an evaluation of whether the page can read like the review sheets; see
// docs/systems/export.md.

import * as THREE from "three";

/** @typedef {import("./body.mjs").BodyMesh} BodyMesh */
/** @typedef {import("./body.mjs").BodyMaterial} BodyMaterial */
/** @typedef {import("./body.mjs").LoadedBody} LoadedBody */

const LIT = [0.86, 0.84, 0.82];
const SHADE = [0.4, 0.4, 0.48];
const TERMINATOR = 0.35;
const INK = new THREE.Color().setRGB(0.02, 0.02, 0.03, THREE.LinearSRGBColorSpace);
const OUTLINE_PX = 2.5;
const OUTLINE_BEHIND = 4.0;
// In the camera's frame: left of it, above it and behind it.
const KEY = new THREE.Vector3(-0.45, 0.55, 0.7).normalize();

// Replaces MeshToonMaterial's lookup. A gradient map's step would do, but it is a shader edge, which
// multisampling leaves jagged, so the step here is one pixel of smoothstep. render.py gets the same
// from its 16 samples of temporal antialiasing.
const STEP = /* glsl */ `
  vec3 getGradientIrradiance(vec3 normal, vec3 lightDirection) {
    float light = dot(normal, lightDirection);
    float blur = fwidth(light) * 0.75;
    return vec3(smoothstep(${TERMINATOR.toFixed(4)} - blur, ${TERMINATOR.toFixed(4)} + blur, light));
  }
`;

/** @param {THREE.WebGLProgramParametersWithUniforms} shader */
function useStep(shader) {
  shader.fragmentShader = shader.fragmentShader.replace("#include <gradientmap_pars_fragment>", STEP);
}

// The step only scales the light, so a tinted shadow comes from light colours: the ambient alone
// gives a material's colour times SHADE / LIT, and ambient plus key gives the colour itself. The
// factor of pi cancels Lambert's.
/** @param {THREE.Camera} camera */
function lights(camera) {
  const ratio = LIT.map((lit, i) => SHADE[i] / lit);
  const ambient = new THREE.AmbientLight(
    new THREE.Color().setRGB(.../** @type {[number, number, number]} */ (ratio.map((r) => r * Math.PI)), THREE.LinearSRGBColorSpace),
    1,
  );
  const key = new THREE.DirectionalLight(
    new THREE.Color().setRGB(.../** @type {[number, number, number]} */ (ratio.map((r) => (1 - r) * Math.PI)), THREE.LinearSRGBColorSpace),
    1,
  );
  // On the camera, as in render.py, so the terminator stays put on screen while orbiting.
  key.position.copy(KEY);
  key.target.position.set(0, 0, 0);
  camera.add(key);
  camera.add(key.target);
  return { ambient, key };
}

// Moves each vertex out along its normal by the width of the line at its depth, then back along the
// view by OUTLINE_BEHIND of those, exactly as render.py moves its hull, but per frame on the GPU.
const OUTLINE_VERTEX = /* glsl */ `
  #include <common>
  #include <morphtarget_pars_vertex>
  #include <skinning_pars_vertex>
  uniform float heightPx;
  uniform float widthPx;
  uniform float behind;
  void main() {
    #include <morphinstance_vertex>
    #include <beginnormal_vertex>
    #include <morphnormal_vertex>
    #include <skinbase_vertex>
    #include <skinnormal_vertex>
    #include <begin_vertex>
    #include <morphtarget_vertex>
    #include <skinning_vertex>
    vec4 place = modelViewMatrix * vec4(transformed, 1.0);
    float metresPerPx = -place.z * 2.0 / (projectionMatrix[1][1] * heightPx);
    float line = widthPx * metresPerPx;
    place.xyz += normalize(normalMatrix * objectNormal) * line;
    place.xyz += normalize(place.xyz) * line * behind;
    gl_Position = projectionMatrix * place;
  }
`;
const OUTLINE_FRAGMENT = /* glsl */ `
  uniform vec3 ink;
  void main() {
    gl_FragColor = vec4(ink, 1.0);
    #include <colorspace_fragment>
  }
`;

/**
 * @param {THREE.Scene} world
 * @param {THREE.Camera} camera
 * @param {THREE.WebGLRenderer} renderer
 */
export function makeToon(world, camera, renderer) {
  const { ambient, key } = lights(camera);
  const size = new THREE.Vector2();
  const outlineMaterial = new THREE.ShaderMaterial({
    vertexShader: OUTLINE_VERTEX,
    fragmentShader: OUTLINE_FRAGMENT,
    side: THREE.BackSide,
    uniforms: {
      heightPx: { value: 1 },
      widthPx: { value: OUTLINE_PX },
      behind: { value: OUTLINE_BEHIND },
      ink: { value: INK },
    },
  });

  /** @param {BodyMesh} mesh */
  function hull(mesh) {
    const line = new THREE.SkinnedMesh(mesh.geometry, outlineMaterial);
    line.bind(mesh.skeleton, mesh.bindMatrix);
    // The same arrays, so that the hull follows the targets without being told.
    line.morphTargetInfluences = mesh.morphTargetInfluences;
    line.morphTargetDictionary = mesh.morphTargetDictionary;
    line.frustumCulled = false;
    return line;
  }

  /** @param {BodyMesh} mesh */
  function toonMaterial(mesh) {
    const from = mesh.material;
    const material = new THREE.MeshToonMaterial({
      color: from.color.clone(),
      transparent: from.transparent,
      opacity: from.opacity,
      alphaTest: from.alphaTest,
      side: from.side,
      depthWrite: from.depthWrite,
    });
    material.onBeforeCompile = useStep;
    return material;
  }

  /**
   * @type {{
   *   entries: { mesh: BodyMesh, material: BodyMaterial }[],
   *   lines: { line: THREE.SkinnedMesh, mesh: BodyMesh }[],
   *   outsideLights: THREE.Light[],
   * } | null}
   */
  let active = null;

  // Turns the look on for a body, keeping what it needs to turn it off again.
  /**
   * @param {LoadedBody} body
   * @param {THREE.Light[]} outsideLights
   */
  function on(body, outsideLights) {
    off();
    const entries = body.meshes.map((mesh) => ({ mesh, material: mesh.material }));
    const lines = [];
    for (const { mesh, material } of entries) {
      mesh.material = toonMaterial(mesh);
      if (mesh === body.mesh || material.userData.sotaigen?.outline) {
        const line = hull(mesh);
        lines.push({ line, mesh });
        body.scene.add(line);
      }
    }
    for (const light of outsideLights) light.visible = false;
    world.add(ambient);
    key.visible = true;
    active = { entries, lines, outsideLights };
  }

  function off() {
    if (!active) return;
    for (const { mesh, material } of active.entries) {
      mesh.material.dispose();
      mesh.material = material;
    }
    for (const { line } of active.lines) line.removeFromParent();
    for (const light of active.outsideLights) light.visible = true;
    world.remove(ambient);
    key.visible = false;
    active = null;
  }

  // Before each frame: the line's width is in pixels, and it follows its part's checkbox.
  function update() {
    if (!active) return;
    outlineMaterial.uniforms.heightPx.value = renderer.getSize(size).y;
    for (const { line, mesh } of active.lines) line.visible = mesh.visible;
  }

  key.visible = false;
  return { on, off, update, isOn: () => active !== null };
}
