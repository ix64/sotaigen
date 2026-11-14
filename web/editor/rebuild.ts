// When the page asks for a body: never during a drag, a moment after the last release, only for the
// latest parameters. No browser API here, and only types Node can strip, so that tests/test_editor.py
// can run this under Node.
//
// The phases, as the page shows them:
//   idle        what's on screen is the wanted body
//   adjusting   a slider is held; nothing is asked for until it lets go
//   generating  released, whether the short wait is still running or the request is out
//   invalid     the rules refuse the wanted parameters, so nothing is asked for
//   failed      the request came back with an error

export type Phase = "idle" | "adjusting" | "generating" | "invalid" | "failed";

// Long enough to get from one slider to the next. Getting there late costs little: the body that was
// started is cancelled at the first movement of the next slider.
export const DELAY_MS = 600;

interface Timers {
  setTimeout: (run: () => void, ms: number) => any;
  clearTimeout: (id: any) => void;
}

export class Rebuilder<Parameters, Result> {
  build: (parameters: Parameters, signal: AbortSignal) => Promise<Result>;
  /** Called only for the latest. */
  land: (result: Result, parameters: Parameters) => void;
  /** Called whenever the phase may have changed. */
  notify: () => void;
  delayMs: number;
  timers: Timers;
  phase: Phase = "idle";
  error: Error | null = null;
  holding = false;
  wanted: Parameters | null = null;
  valid = true;
  landed: string | null = null;
  /** Parameters that have landed before, so the cache will answer and the delay would only cost. */
  seen = new Set<string>();
  ticket = 0;
  timer: unknown = undefined;
  controller: AbortController | null = null;
  running: Promise<void> | null = null;
  requests = 0;
  cancelled = 0;
  dropped = 0;

  constructor({
    build,
    land,
    notify = () => {},
    delayMs = DELAY_MS,
    timers = globalThis,
  }: {
    build: (parameters: Parameters, signal: AbortSignal) => Promise<Result>;
    land: (result: Result, parameters: Parameters) => void;
    notify?: () => void;
    delayMs?: number;
    timers?: Timers;
  }) {
    this.build = build;
    this.land = land;
    this.notify = notify;
    this.delayMs = delayMs;
    this.timers = timers;
  }

  // Whatever was on its way belongs to parameters nobody wants any more.
  forgetOlder() {
    this.ticket += 1;
    this.timers.clearTimeout(this.timer);
    if (this.running) this.cancelled += 1;
    this.controller?.abort();
    this.controller = null;
    this.running = null;
  }

  /** The wanted parameters changed. `now` skips the delay: a choice made once, not a slider. */
  want(parameters: Parameters, valid: boolean, { now = false }: { now?: boolean } = {}) {
    this.forgetOlder();
    this.wanted = parameters;
    this.valid = valid;
    this.error = null;
    if (!valid) this.phase = "invalid";
    else if (this.holding) this.phase = "adjusting";
    else this.arm(now);
    this.notify();
  }

  // A slider was pressed. It may not move, so what's running goes on until it does.
  hold() {
    this.holding = true;
    if (this.phase === "generating" && !this.running) {
      this.timers.clearTimeout(this.timer);
      this.phase = "adjusting";
    }
    this.notify();
  }

  release() {
    if (!this.holding) return;
    this.holding = false;
    if (this.phase === "adjusting") this.arm(false);
    this.notify();
  }

  arm(now: boolean) {
    const text = JSON.stringify(this.wanted);
    if (text === this.landed) {
      this.phase = "idle";
      return;
    }
    this.phase = "generating";
    if (now || this.seen.has(text)) {
      this.start();
    } else {
      this.timers.clearTimeout(this.timer);
      this.timer = this.timers.setTimeout(() => this.start(), this.delayMs);
    }
  }

  start() {
    this.timers.clearTimeout(this.timer);
    this.ticket += 1;
    const { ticket } = this;
    const wanted = this.wanted as Parameters;
    const controller = new AbortController();
    this.controller = controller;
    this.requests += 1;
    this.running = this.build(wanted, controller.signal)
      .then((result) => {
        if (ticket !== this.ticket) {
          this.dropped += 1;
          return;
        }
        this.land(result, wanted);
        const text = JSON.stringify(wanted);
        this.landed = text;
        this.seen.add(text);
        this.phase = "idle";
      })
      .catch((error) => {
        if (ticket !== this.ticket) return;
        this.error = error;
        this.phase = "failed";
      })
      .finally(() => {
        if (ticket !== this.ticket) return;
        this.controller = null;
        this.running = null;
        this.notify();
      });
    return this.running;
  }

  // Don't wait out the delay; settles when what was running has come back.
  async flush() {
    if (this.phase === "generating" && !this.running) this.start();
    await this.running;
  }
}
