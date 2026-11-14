// What the pages leave on window for the browser checks and the console. JSDoc can't extend a global,
// so this one declaration lives here.

interface Window {
  bodyStats: any;
  bodyReady: boolean;
  bodyState: () => any;
  frameMs: (frames?: number) => { ms: number; calls: number; triangles: number };
  checkPreset: (preset: string) => Promise<any>;
  checkParts: (preset: string) => Promise<any>;
  checkReady: boolean;
  editor: any;
  __first: unknown;
}

declare module "*.css";
declare module "gltf-validator";
