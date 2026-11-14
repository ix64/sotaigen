// @ts-nocheck: a Node script, untyped. It is checked here only because the tests import it.
// The one local server, on one port: the pages at their routes (the router decides which page, so any
// path without a file gets the app), the repository's build/ at /build/, and the Python worker that
// makes bodies behind /api/body.
//
//   pnpm run editor [port]       Builds the pages first, then serves build/web/app/.
//   pnpm run editor-dev [port]   Vite inside this server: hot reload, and still the one port.
//
// Listens on 127.0.0.1 only, and answers only a Host of 127.0.0.1, localhost or [::1]; a POST must also
// come from one of its own pages, or from no page. POST /api/body takes { sex, body } and answers
// when that body is ready, from the cache or from the worker, with a report naming its GLB at
// /api/body/<key>.glb. A body too broken to export still has its GLB, for the page to show: the report's
// `broken` says why, and the page refuses to save it.
//
// One body at a time: a request for another kills what is running. It is a real cancel, by process
// group, and a new worker is ready about 0.3 s later.
//
// { sex, body, contact: true } asks for the body with its IPC contact tables, for an export: minutes for a
// body not baked before, and { ok: false, broken } with no bake and no file for a broken one. It has a worker of its own, started on the first such request, so that moving
// a slider can't cancel it, and nor can the page going away: the body is finished and kept either way.
// GET /api/contact says what that worker is on: { job: { key, stage, startedAt } } or { job: null }. A body is kept in build/cache/editor/ under a
// digest of its parameters and the pipeline's version; see BodyCache for how many. The worker holds
// the pipeline it started with, so after changing sotaigen/ or template/, start this again.
//
// POST /api/touch takes { sex, body } and answers when the angles of that body's hand-to-head
// contact poses (sotaigen/rig/touch.py) are solved, at /api/touch/<key>.touch, a JSON file. The page never solves
// them itself. They have a worker of their own too, started on the first such request.
//
// POST /api/blend takes { sex, body } and answers when the body's rigged .blend is ready, at
// /api/blend/<key>.blend, or says why there is none. Blender runs on the server's side only, through
// the worker: no code of blender/ reaches a page. It has a worker of its own too, so that the sliders
// stay free meanwhile, and a cache of its own under build/cache/editor/blend/, so that these bigger
// files can't push a GLB out. GET /api/blend says whether this machine has Blender: { blender }, null
// while that isn't known.
//
// POST /api/mend takes { sex, body } and answers { ok, mend } when it knows which parameter ends a
// broken body's breakage: minutes at worst, a wrap per parameter, so it has a worker of its own, and the
// page going away or moving on kills it. GET /api/mend says how far it is: { job: { key, stage, startedAt } }.
//
// POST /api/params/read takes a parameter file's text and answers what sotaigen/params/file.py reads of
// it, { ok: true, file: { sex, body, base, unknown } }, or { ok: false, problems: [diagnostic] }. It runs
// its own short Python, so a body being made isn't cancelled for it.
//
// All of this is the page's own interface, not a contract: it changes with any version
// (docs/systems/editor.md). What downstream may rely on are the files.

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  createReadStream,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import { pipeline } from "node:stream";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
// The bodies, caches and reports, where sotaigen.paths.build_dir has them. The pages themselves are
// always built into the checkout's build/web/app (web/vite.config.mjs).
export const BUILD = process.env.SOTAIGEN_BUILD ? resolve(process.env.SOTAIGEN_BUILD) : join(ROOT, "build");
const APP = join(ROOT, "build", "web", "app");
// SOTAIGEN_UV names the uv to run Python with, where it isn't the one on PATH.
const UV_PYTHON = [process.env.SOTAIGEN_UV || "uv", "run", "python"];
const STDERR_KEPT = 2000;
// A GLB is about 1 MB, so about 64 MB. Each preset is one body, and that leaves room to wander.
const CACHE_BODIES = 64;
// A .blend is about 3.6 MB, so about 29 MB. An export, not something to wander among.
const BLEND_BODIES = 8;
// A body's contact poses are a few kilobytes.
const TOUCH_BODIES = 64;
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".glb": "model/gltf-binary",
  ".blend": "application/x-blender",
  ".png": "image/png",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
};

function inside(root, wanted) {
  const path = resolve(root, `.${sep}${wanted}`);
  return relative(root, path).startsWith("..") ? null : path;
}

function locate(pathname) {
  if (pathname.startsWith("/build/")) return inside(BUILD, pathname.slice("/build/".length));
  // No extension: a route, for the router in the page to read. A missing asset stays a 404.
  if (!extname(pathname)) return join(APP, "index.html");
  return inside(APP, pathname.slice(1));
}

// What /runtime can show: the bodies uv run sotaigen-dev export wrote, and the stand-in parts of runtime-check.
function exported() {
  const bodies = join(BUILD, "bodies");
  const hasBody = (name) => existsSync(join(bodies, name, "body.glb"));
  const folder = join(BUILD, "reports", "runtime-check", "standin");
  return {
    presets: existsSync(bodies) ? readdirSync(bodies).filter(hasBody).sort() : [],
    standIns: existsSync(folder)
      ? readdirSync(folder)
          .filter((name) => name.endsWith(".glb"))
          .map((name) => name.slice(0, -".glb".length))
          .sort()
      : [],
  };
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonical(value[key])]),
  );
}

// The same parameters in any key order, from the same pipeline, are the same body. With its contact
// tables it is another file.
export function cacheKey(version, sex, body, contact = false) {
  const text = JSON.stringify(canonical(contact ? { version, sex, body, contact: "ipc" } : { version, sex, body }));
  return createHash("sha256").update(text).digest("hex").slice(0, 32);
}

// Finished files on disk: <key><extension> and <key>.json, the report. The least recently used go first
// once there are more than `limit`; a hit counts as use, through the file's modification time.
// Bodies of an older pipeline are never hit again, so they age out the same way.
export class BodyCache {
  constructor(folder, limit = CACHE_BODIES, extension = ".glb") {
    this.folder = folder;
    this.limit = limit;
    this.extension = extension;
    mkdirSync(folder, { recursive: true });
    this.sweep();
  }

  // Half made: <key>.<id>.part<extension>, the .part.tmp.blend Blender saves first, and the blend-*
  // folder body.blend writes Blender's input to. A killed job leaves them behind.
  isHalfMade(name) {
    return name.endsWith(`.part${this.extension}`) || name.endsWith(`.part.tmp${this.extension}`) || name.startsWith("blend-");
  }

  // Only with no job running: what a killed server or a cancelled job left.
  sweep() {
    for (const name of readdirSync(this.folder)) {
      if (this.isHalfMade(name)) rmSync(join(this.folder, name), { force: true, recursive: true });
    }
  }

  filePath(key) {
    return join(this.folder, `${key}${this.extension}`);
  }

  read(key) {
    const file = this.filePath(key);
    const report = join(this.folder, `${key}.json`);
    if (!existsSync(file) || !existsSync(report)) return null;
    const now = new Date();
    utimesSync(file, now, now);
    return JSON.parse(readFileSync(report, "utf8"));
  }

  // The file moves last, so that a body is either all there or not there at all.
  store(key, row, finished) {
    writeFileSync(join(this.folder, `${key}.json`), JSON.stringify(row));
    renameSync(finished, this.filePath(key));
    this.evict();
  }

  evict() {
    const bodies = readdirSync(this.folder)
      .filter((name) => name.endsWith(this.extension) && !this.isHalfMade(name))
      .map((name) => ({ name, time: statSync(join(this.folder, name)).mtimeMs }))
      .sort((a, b) => b.time - a.time);
    for (const { name } of bodies.slice(this.limit)) {
      rmSync(join(this.folder, name), { force: true });
      rmSync(join(this.folder, `${name.slice(0, -this.extension.length)}.json`), { force: true });
    }
  }
}

// The worker process, and the one job it is on. Nothing waits in line: a newer request replaces the
// job, because the slider has moved on and that result is no use to anyone.
export class BodyWorker {
  constructor({ command = [...UV_PYTHON, "-m", "sotaigen_dev.editor.worker"], folder, extension = ".glb" }) {
    this.command = command;
    this.folder = folder;
    this.extension = extension;
    this.nextId = 1;
    this.child = null;
    this.job = null;
    this.closed = false;
    this.stderr = "";
    this.version = null;
    this.blendVersion = null;
    this.blender = null;
  }

  start() {
    const [program, ...args] = this.command;
    // Its own process group, so that the wrap's second process goes when it does.
    const child = spawn(program, args, { cwd: ROOT, detached: true, stdio: ["pipe", "pipe", "pipe"] });
    this.child = child;
    this.stderr = "";
    this.announced = new Promise((done, reject) => {
      this.announce = { done, reject };
    });
    // Whoever asks later still hears it; this only stops an unasked failure from being unhandled.
    this.announced.catch(() => {});
    child.stdin.on("error", () => {});
    child.stderr.on("data", (piece) => {
      process.stderr.write(piece);
      this.stderr = (this.stderr + piece).slice(-STDERR_KEPT);
    });
    createInterface({ input: child.stdout }).on("line", (line) => {
      if (child === this.child) this.heard(line);
    });
    child.on("error", (error) => this.died(child, error.message));
    child.on("exit", (code, signal) => this.died(child, `it exited with ${signal ?? code}`));
  }

  heard(line) {
    let row;
    try {
      row = JSON.parse(line);
    } catch {
      return; // Anything else a library printed.
    }
    if (row.ready) {
      this.version = row.version;
      this.blendVersion = row.blendVersion ?? null;
      this.blender = row.blender ?? null;
      this.announce.done(row.version);
    } else if (row.id === this.job?.id && row.stage) {
      this.job.stage = row.stage;
    } else if (row.id === this.job?.id) {
      this.finish(row);
    }
  }

  died(child, why) {
    if (child !== this.child) return;
    this.child = null;
    const message = `the Python worker is not running: ${why}\n${this.stderr}`.trim();
    this.announce.reject(new Error(message));
    this.job?.reject(new Error(message));
    this.job = null;
  }

  // The pipeline's version, once a worker has said. Rejects when the first one can't start. A worker
  // started after a cancel says it again, so a pipeline changed since then is picked up by the next.
  ready() {
    if (!this.child) this.start();
    return this.version ?? this.announced;
  }

  // A job this client can leave. Asking for what is already running joins it.
  request(key, payload) {
    if (this.job?.key === key) {
      this.job.clients += 1;
    } else {
      this.stop();
      if (!this.child) this.start();
      const id = this.nextId++;
      const job = { id, key, clients: 1, out: join(this.folder, `${key}.${id}.part${this.extension}`), stage: null, startedAt: Date.now() };
      job.done = new Promise((done, reject) => {
        job.resolve = done;
        job.reject = reject;
      });
      this.job = job;
      this.child.stdin.write(`${JSON.stringify({ ...payload, id, out: job.out })}\n`);
    }
    const { job } = this;
    return {
      done: job.done,
      leave: () => {
        job.clients -= 1;
        if (job === this.job && job.clients === 0) {
          this.stop();
          if (!this.closed) this.start(); // Warm for what comes next.
        }
      },
    };
  }

  finish(row) {
    const { job } = this;
    this.job = null;
    job.resolve({ ...row, out: job.out });
  }

  // Cancel what is running, for real. Nothing to do when it is idle: an idle worker is kept.
  stop() {
    const { job } = this;
    if (!job) return;
    this.job = null;
    this.kill();
    rmSync(job.out, { force: true });
    job.resolve({ superseded: true });
  }

  kill() {
    const { child } = this;
    this.child = null;
    if (!child?.pid) return;
    try {
      process.kill(-child.pid, "SIGKILL");
    } catch {
      // Already gone.
    }
  }

  close() {
    this.closed = true;
    this.stop();
    this.kill();
  }
}

function readBody(request) {
  return new Promise((done, reject) => {
    const parts = [];
    request.on("data", (part) => parts.push(part));
    request.on("end", () => done(Buffer.concat(parts).toString()));
    request.on("error", reject);
  });
}

function sendJson(response, row) {
  response.writeHead(200, { "content-type": TYPES[".json"], "cache-control": "no-store" });
  response.end(JSON.stringify(row));
}

// The typical bodies as the page starts from them, named <sex>-<group> as sotaigen.form.anatomy names them.
// Test bodies stay off the page. Plain data, so this works whatever state Python is in.
// Each is a parameter file; youngest first, as sotaigen.form.bodies.TYPICAL_GROUPS orders them.
export const TYPICAL_GROUPS = ["chibi", "compact", "slim", "long", "balanced", "rounded"];
function presets() {
  const bodies = join(ROOT, "sotaigen", "data", "bodies");
  const out = {};
  for (const sex of readdirSync(bodies).sort()) {
    const folder = join(bodies, sex, "typical");
    if (!existsSync(folder)) continue;
    const groups = readdirSync(folder).filter((file) => file.endsWith(".json")).map((file) => file.slice(0, -5));
    groups.sort((a, b) => TYPICAL_GROUPS.indexOf(a) - TYPICAL_GROUPS.indexOf(b));
    for (const group of groups) {
      const { body } = JSON.parse(readFileSync(join(folder, `${group}.json`), "utf8"));
      out[`${sex}-${group}`] = { sex, body };
    }
  }
  return out;
}

// One body's file, from the cache or from the worker, and the answer to the request that asked for it.
// `name` says which field of the answer holds the file's address, and `path` makes that address from
// the cache's key. A job that `stays` is finished even when the client goes.
async function answerWithFile(response, { worker, cache, keyOf, name, path, payload, stays = false }) {
  const started = performance.now();
  if (!worker) {
    sendJson(response, { ok: false, unavailable: true, error: "this server was started without the worker" });
    return;
  }
  try {
    await worker.ready();
  } catch (error) {
    sendJson(response, { ok: false, unavailable: true, error: error.message });
    return;
  }
  const key = keyOf(worker);
  const hit = cache.read(key);
  if (hit) {
    sendJson(response, { ...hit, key, [name]: path(key), cached: true, serverMs: performance.now() - started });
    return;
  }
  if (!worker.job) cache.sweep();
  const job = worker.request(key, payload);
  // The page gave up on it, by moving a slider or by leaving, and nobody else wants this body.
  response.on("close", () => {
    if (!response.writableFinished && !stays) job.leave();
  });
  let row;
  try {
    row = await job.done;
  } catch (error) {
    // It worked before and the next request starts another, so that isn't "no Python".
    sendJson(response, { ok: false, unavailable: worker.version === null, error: error.message });
    return;
  }
  if (row.superseded || !row.ok) {
    sendJson(response, row);
    return;
  }
  const { out, id: _id, ...report } = row;
  // A client that joined this job may already have stored it.
  if (existsSync(out)) cache.store(key, report, out);
  sendJson(response, { ...report, key, [name]: path(key), cached: false, serverMs: performance.now() - started });
}

async function makeBody(workers, cache, request, response) {
  const { sex, body, contact = false } = JSON.parse(await readBody(request));
  await answerWithFile(response, {
    worker: contact ? workers?.contact : workers?.body,
    cache,
    keyOf: (worker) => cacheKey(worker.version, sex, body, contact),
    name: "glb",
    path: (key) => `/api/body/${key}.glb`,
    payload: contact ? { sex, body, contact } : { sex, body },
    // A body with its contact tables took minutes, and it is in the cache for the next ask.
    stays: contact,
  });
}

async function touch(workers, cache, request, response) {
  const { sex, body } = JSON.parse(await readBody(request));
  await answerWithFile(response, {
    worker: workers?.touch,
    cache,
    keyOf: (worker) => cacheKey(worker.version, sex, body),
    name: "touch",
    path: (key) => `/api/touch/${key}.touch`,
    payload: { sex, body, touch: true },
  });
}

async function blend(workers, cache, request, response) {
  const { sex, body } = JSON.parse(await readBody(request));
  await answerWithFile(response, {
    worker: workers?.blend,
    cache,
    keyOf: (worker) => cacheKey(worker.blendVersion, sex, body),
    name: "blend",
    path: (key) => `/api/blend/${key}.blend`,
    payload: { sex, body, blend: true },
  });
}

// No file comes of this, so nothing is kept: the page holds on to what it was told.
async function mend(workers, request, response) {
  const { sex, body } = JSON.parse(await readBody(request));
  const worker = workers?.mend;
  if (!worker) {
    sendJson(response, { ok: false, unavailable: true, error: "this server was started without the worker" });
    return;
  }
  try {
    await worker.ready();
  } catch (error) {
    sendJson(response, { ok: false, unavailable: true, error: error.message });
    return;
  }
  const job = worker.request(cacheKey(worker.version, sex, body), { sex, body, mend: true });
  response.on("close", () => {
    if (!response.writableFinished) job.leave();
  });
  try {
    const { out: _out, id: _id, ...row } = await job.done;
    sendJson(response, row);
  } catch (error) {
    sendJson(response, { ok: false, unavailable: worker.version === null, error: error.message });
  }
}

// Only this machine's own names. Anything else is another site that had its name point here (DNS
// rebinding) and must not read build/ or start bodies.
const LOCAL_HOST = /^(?:127\.0\.0\.1|localhost|\[::1\])(?::\d+)?$/i;

// A request from a page of this server, or from no page at all (a script). A POST from any other page
// gets through without a preflight when it is text/plain, so it is turned away here: it could cancel the
// body being made, or start a contact bake of tens of minutes.
function allowed(request) {
  const host = request.headers.host ?? "";
  if (!LOCAL_HOST.test(host)) return false;
  if (request.method === "GET" || request.method === "HEAD") return true;
  const { origin } = request.headers;
  return origin === undefined || origin.toLowerCase() === `http://${host.toLowerCase()}`;
}

function refuse(response, status, text) {
  response.writeHead(status, { "content-type": "text/plain; charset=utf-8" }).end(text);
}

// The file is opened before anything is said, so a file the cache let go after the check is a 404, and
// one that fails partway, or a client that goes, only ends this response.
function sendFile(response, path) {
  const stream = createReadStream(path);
  stream.once("error", () => {
    if (!response.headersSent) refuse(response, 404, "not found");
  });
  stream.once("open", () => {
    // The client went while the file was opening, as a page does that has moved on to another body.
    if (response.destroyed) {
      stream.destroy();
      return;
    }
    response.writeHead(200, {
      "content-type": TYPES[extname(path)] ?? "application/octet-stream",
      "cache-control": "no-store",
    });
    pipeline(stream, response, () => {});
  });
}

// The one reader of parameter files, in Python. Anything but its answer, such as no Python, is an error.
async function readParameters(request, response) {
  const text = await readBody(request);
  const [program, ...args] = UV_PYTHON;
  // Not spawnSync: the server keeps answering for the page meanwhile.
  const child = spawn(program, [...args, "-m", "sotaigen.params.file"], { cwd: ROOT, stdio: ["pipe", "pipe", "pipe"] });
  const out = [];
  const err = [];
  child.stdout.on("data", (part) => out.push(part));
  child.stderr.on("data", (part) => err.push(part));
  const status = await new Promise((done) => {
    child.on("error", (error) => {
      err.push(Buffer.from(String(error)));
      done(-1);
    });
    child.on("close", done);
    child.stdin.end(text);
  });
  if (status !== 0) {
    sendJson(response, { ok: false, error: `could not read the file: ${Buffer.concat(err).toString().slice(-STDERR_KEPT)}` });
    return;
  }
  sendJson(response, JSON.parse(Buffer.concat(out).toString()));
}

function makeSchema() {
  const [program, ...args] = UV_PYTHON;
  const done = spawnSync(program, [...args, "-m", "sotaigen_dev.editor.schema"], { cwd: ROOT, encoding: "utf8" });
  // The page checks bodies by these rules: a page without them would only fail later and less clearly.
  if (done.status !== 0) throw new Error(`Could not write the page's rules: ${done.error ?? done.stderr}`);
}

// `bodies: false` is for pages that only read files, such as the runtime check: no Python is started.
// `dev` has Vite answer for the pages, on this same server and port.
export async function serve(
  port = 0,
  { command, schema = true, cacheFolder = join(BUILD, "cache", "editor"), bodies: makesBodies = true, dev = false } = {},
) {
  if (schema && makesBodies) makeSchema();
  const cache = new BodyCache(cacheFolder);
  const blendCache = new BodyCache(join(cacheFolder, "blend"), BLEND_BODIES, ".blend");
  const touchCache = new BodyCache(join(cacheFolder, "touch"), TOUCH_BODIES, ".touch");
  const worker = makesBodies ? new BodyWorker({ command, folder: cache.folder }) : null;
  // Not started until a body with contact tables is asked for.
  const contactWorker = makesBodies ? new BodyWorker({ command, folder: cache.folder }) : null;
  // Nor this until a .blend is.
  const blendWorker = makesBodies ? new BodyWorker({ command, folder: blendCache.folder, extension: ".blend" }) : null;
  // Nor this until a contact pose is.
  const touchWorker = makesBodies ? new BodyWorker({ command, folder: touchCache.folder, extension: ".touch" }) : null;
  // Nor this until a refused export looks for what ends the breakage.
  const mendWorker = makesBodies ? new BodyWorker({ command, folder: cache.folder }) : null;
  const workers = makesBodies ? { body: worker, contact: contactWorker, blend: blendWorker, touch: touchWorker, mend: mendWorker } : null;
  // Warm, and if it can't start the first request will say why.
  if (worker) Promise.resolve(worker.ready()).catch(() => {});
  let vite = null;
  const answer = async (request, response) => {
    if (!allowed(request)) {
      refuse(response, 403, "Only this machine's pages may use this server.");
      return;
    }
    let pathname;
    let decoded;
    try {
      ({ pathname } = new URL(request.url, "http://localhost"));
      decoded = decodeURIComponent(pathname);
    } catch {
      refuse(response, 400, "bad path");
      return;
    }
    if (pathname === "/api/body" && request.method === "POST") {
      try {
        await makeBody(workers, cache, request, response);
      } catch (error) {
        sendJson(response, { ok: false, error: String(error) });
      }
      return;
    }
    if (pathname === "/api/touch" && request.method === "POST") {
      try {
        await touch(workers, touchCache, request, response);
      } catch (error) {
        sendJson(response, { ok: false, error: String(error) });
      }
      return;
    }
    if (pathname === "/api/blend" && request.method === "POST") {
      try {
        await blend(workers, blendCache, request, response);
      } catch (error) {
        sendJson(response, { ok: false, error: String(error) });
      }
      return;
    }
    if (pathname === "/api/params/read" && request.method === "POST") {
      try {
        await readParameters(request, response);
      } catch (error) {
        sendJson(response, { ok: false, error: String(error) });
      }
      return;
    }
    if (pathname === "/api/mend" && request.method === "POST") {
      try {
        await mend(workers, request, response);
      } catch (error) {
        sendJson(response, { ok: false, error: String(error) });
      }
      return;
    }
    if (pathname === "/api/mend") {
      const job = mendWorker?.job;
      sendJson(response, { job: job ? { key: job.key, stage: job.stage, startedAt: job.startedAt } : null });
      return;
    }
    if (pathname === "/api/blend") {
      // The worker that is running anyway says, so that asking starts nothing.
      try {
        await worker?.ready();
      } catch {
        // A body asked for will say what is wrong with it.
      }
      sendJson(response, { blender: worker?.blender ?? null });
      return;
    }
    if (pathname === "/api/contact") {
      const job = contactWorker?.job;
      sendJson(response, { job: job ? { key: job.key, stage: job.stage, startedAt: job.startedAt } : null });
      return;
    }
    if (pathname === "/api/presets") {
      sendJson(response, { presets: presets() });
      return;
    }
    if (pathname === "/api/exported") {
      sendJson(response, exported());
      return;
    }
    if (vite && !pathname.startsWith("/build/") && !pathname.startsWith("/api/")) {
      vite.middlewares(request, response);
      return;
    }
    const glb = pathname.match(/^\/api\/body\/([0-9a-f]+)\.glb$/);
    const dotBlend = pathname.match(/^\/api\/blend\/([0-9a-f]+)\.blend$/);
    const solved = pathname.match(/^\/api\/touch\/([0-9a-f]+)\.touch$/);
    const path = glb
      ? cache.filePath(glb[1])
      : dotBlend
        ? blendCache.filePath(dotBlend[1])
        : solved
          ? touchCache.filePath(solved[1])
          : locate(decoded);
    if (!path || !existsSync(path) || !statSync(path).isFile()) {
      const unbuilt = path === join(APP, "index.html");
      response.writeHead(404).end(unbuilt ? "The pages aren't built. Use pnpm run editor, or pnpm run web-build." : "not found");
      return;
    }
    sendFile(response, path);
  };
  // Nothing one request does may take the server down: it would leave its workers running on their own.
  const server = createServer((request, response) => {
    answer(request, response).catch((error) => {
      console.error(error);
      if (!response.headersSent) refuse(response, 500, "server error");
      else response.destroy();
    });
  });
  server.on("close", () => {
    worker?.close();
    contactWorker?.close();
    blendWorker?.close();
    touchWorker?.close();
    mendWorker?.close();
    vite?.close();
  });
  server.worker = worker; // For web/tests/, which kills it to see what the page does.
  server.contactWorker = contactWorker;
  server.blendWorker = blendWorker;
  server.touchWorker = touchWorker;
  server.mendWorker = mendWorker;
  if (dev) {
    // Only here, so that the tests and the plain server don't need Vite installed.
    const { createServer: createVite } = await import("vite");
    vite = await createVite({
      configFile: join(HERE, "vite.config.mjs"),
      server: { middlewareMode: true, hmr: { server } },
    });
  }
  return new Promise((done) => {
    server.listen(port, "127.0.0.1", () => done(server));
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const dev = process.argv.includes("--dev");
  const [port = 8124] = process.argv.slice(2).filter((word) => word !== "--dev");
  const server = await serve(Number(port), { dev });
  const stop = () => {
    server.close(() => process.exit(0));
    server.closeAllConnections();
  };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  console.log(`http://127.0.0.1:${server.address().port}/`);
}
