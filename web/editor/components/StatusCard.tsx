import { Box, ChevronDown, Download, Layers, type LucideIcon, Search, X } from "lucide-react";
import { useState } from "react";
import { type ContactExport, type Editor, type StatusKind, useEditor } from "../store";
import { cn } from "@/lib/utils";
import { say } from "../text";

const DOT: Record<StatusKind, string> = {
  good: "bg-good",
  warn: "bg-warn",
  bad: "bg-bad",
  busy: "bg-primary animate-pulse",
};

// The stages the worker reports; the message is contact_stage_<stage>, and a new one shows as it is.
const STAGES = new Set(["shoulder", "pelvic", "hang"]);

function clockSince(startedAt: number) {
  const seconds = Math.max(0, Math.round((Date.now() - startedAt) / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function contactLine(job: ContactExport) {
  if (job.phase !== "running") return job.messages.length ? job.messages.join("\n") : say("contact_done");
  const clock = clockSince(job.startedAt);
  if (!job.stage) return say("contact_preparing", { clock });
  return say("contact_baking", { stage: STAGES.has(job.stage) ? say(`contact_stage_${job.stage}`) : job.stage, clock });
}

/** An export running beside everything else: how far it is, then how it went. */
function JobNote({
  id,
  Icon,
  running,
  bad,
  text,
  dismissLabel,
  onDismiss,
  stoppable = false,
  action,
}: {
  id: string;
  Icon: LucideIcon;
  running: boolean;
  bad: boolean;
  text: string;
  dismissLabel: string;
  onDismiss: () => void;
  /** Can be put away while it runs, which ends it. */
  stoppable?: boolean;
  action?: { label: string; onClick: () => void };
}) {
  return (
    <div className="pointer-events-auto flex max-w-[32rem] items-start gap-2 rounded-lg border bg-card/95 py-1.5 pr-1.5 pl-3 text-xs shadow-sm backdrop-blur-sm">
      <Icon className={cn("mt-0.5 size-3.5 shrink-0", running ? "animate-pulse text-primary" : "text-muted-foreground")} />
      <span id={id} className={cn("whitespace-pre-wrap tabular-nums", bad ? "text-bad" : "text-muted-foreground")}>
        {text}
      </span>
      {action && (
        <button
          type="button"
          className="shrink-0 pt-0.5 text-muted-foreground underline underline-offset-2 outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
          onClick={action.onClick}
        >
          {action.label}
        </button>
      )}
      {(!running || stoppable) && (
        <button
          type="button"
          aria-label={dismissLabel}
          onClick={onDismiss}
          className="grid size-5 shrink-0 place-items-center rounded-sm text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <X className="size-3" />
        </button>
      )}
    </div>
  );
}

function MendNote({ editor }: { editor: Editor }) {
  const search = useEditor(editor, (state) => state.mendSearch);
  if (!search) return null;
  const running = search.phase === "running";
  let text = search.message?.text ?? "";
  if (running) {
    const clock = clockSince(search.startedAt);
    const [tried, total] = search.stage?.split("/") ?? [];
    text = total ? say("mend_running", { tried, total, clock }) : say("mend_preparing", { clock });
  }
  const panel = search.message?.panel;
  return (
    <JobNote
      id="mendNote"
      Icon={Search}
      running={running}
      bad={search.phase === "failed"}
      text={text}
      dismissLabel={say("mend_dismiss")}
      onDismiss={editor.dismissMendSearch}
      stoppable
      action={panel ? { label: say("status_go_adjust"), onClick: () => editor.setPanel(panel) } : undefined}
    />
  );
}

function GlbRefusal({ editor }: { editor: Editor }) {
  const refusal = useEditor(editor, (state) => state.glbRefusal);
  if (!refusal) return null;
  return <JobNote id="glbNote" Icon={Download} running={false} bad text={refusal} dismissLabel={say("glb_dismiss")} onDismiss={editor.dismissGlbRefusal} />;
}

function ContactJob({ editor }: { editor: Editor }) {
  const job = useEditor(editor, (state) => state.contactExport);
  if (!job) return null;
  return (
    <JobNote
      id="contactNote"
      Icon={Layers}
      running={job.phase === "running"}
      bad={job.messages.length > 0}
      text={contactLine(job)}
      dismissLabel={say("contact_dismiss")}
      onDismiss={editor.dismissContactExport}
    />
  );
}

function BlendJob({ editor }: { editor: Editor }) {
  const job = useEditor(editor, (state) => state.blendExport);
  if (!job) return null;
  const text = job.phase === "running" ? say("blend_running", { clock: clockSince(job.startedAt) }) : (job.message ?? say("blend_done"));
  return (
    <JobNote
      id="blendNote"
      Icon={Box}
      running={job.phase === "running"}
      bad={job.message !== null}
      text={text}
      dismissLabel={say("blend_dismiss")}
      onDismiss={editor.dismissBlendExport}
    />
  );
}

function TouchNote({ editor }: { editor: Editor }) {
  const job = useEditor(editor, (state) => state.touchJob);
  if (!job) return null;
  return (
    <JobNote
      id="touchNote"
      Icon={Box}
      running={job.phase === "running"}
      bad={job.phase !== "running"}
      text={job.phase === "running" ? say("touch_running") : (job.message ?? "")}
      dismissLabel={say("touch_dismiss")}
      onDismiss={editor.dismissTouchJob}
    />
  );
}

/** Where things stand: one line at rest, the reasons whenever there are any, the rest on asking. */
export function StatusCard({ editor }: { editor: Editor }) {
  const status = useEditor(editor, (state) => state.status);
  const failed = useEditor(editor, () => editor.rebuilder.phase === "failed");
  const [asked, setAsked] = useState(false);
  const open = asked || failed;
  return (
    <div className="pointer-events-none absolute inset-x-3 top-3 flex flex-col items-center gap-2">
      <div className="pointer-events-auto max-w-full overflow-hidden rounded-lg border bg-card/95 shadow-sm backdrop-blur-sm">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setAsked(!asked)}
          className="flex h-8 max-w-full items-center gap-2 px-3 outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
        >
          <span aria-hidden className={cn("size-2 shrink-0 rounded-full", DOT[status.kind])} />
          <span id="state" role="status" aria-live="polite" className="truncate">
            {status.line}
          </span>
          <ChevronDown className={cn("size-3.5 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} />
        </button>
        {status.progress !== null && (
          <div className="h-0.5 bg-muted">
            <div className="h-full bg-primary transition-[width] duration-200" style={{ width: `${status.progress * 100}%` }} />
          </div>
        )}
        <div hidden={!open} className="max-w-[26rem] border-t px-3 py-2 text-xs">
          <div id="hint" className="whitespace-pre-wrap text-muted-foreground">
            {status.hint}
          </div>
          <div id="numbers" className={cn("whitespace-pre-wrap tabular-nums", status.numbers && "mt-1.5")}>
            {status.numbers}
          </div>
        </div>
      </div>
      <GlbRefusal editor={editor} />
      <MendNote editor={editor} />
      <ContactJob editor={editor} />
      <BlendJob editor={editor} />
      <TouchNote editor={editor} />
      <ul
        id="problems"
        aria-label={say("status_problems")}
        hidden={status.problems.length === 0}
        className="pointer-events-auto max-h-48 w-full max-w-[32rem] space-y-1 overflow-y-auto rounded-lg border bg-card/95 p-2.5 text-xs shadow-sm backdrop-blur-sm"
      >
        {status.problems.map(({ text, seam, panel }, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: a list rebuilt whole, and lines can repeat.
          <li key={index} className={cn("flex items-baseline justify-between gap-3", seam ? "text-warn" : "text-bad")}>
            <span>{text}</span>
            {panel && (
              <button
                type="button"
                className="shrink-0 text-muted-foreground underline underline-offset-2 outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
                onClick={() => editor.setPanel(panel)}
              >
                {say("status_go_adjust")}
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
