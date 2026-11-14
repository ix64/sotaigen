import { RotateCcw, X } from "lucide-react";
import type { ReactNode } from "react";
import { Segmented } from "@/components/Segmented";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { type Editor, FACE_BONES, type FaceBones, useEditor } from "../store";
import type { LookName } from "../viewport";
import { morphLabel, partLabel, say } from "../text";

function Section({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section className="border-b px-3 py-2.5 last:border-b-0">
      <header className="mb-1 flex h-6 items-center justify-between">
        <h3 className="text-[11px] text-muted-foreground">{title}</h3>
        {action}
      </header>
      {children}
    </section>
  );
}

function Look({ editor }: { editor: Editor }) {
  const look = useEditor(editor, (state) => state.look);
  return (
    <Section title={say("look_title")}>
      <Segmented
        label={say("look_title")}
        idPrefix="look"
        value={look}
        options={[
          { value: "toon", label: say("look_toon") },
          { value: "standard", label: say("look_standard") },
          { value: "silhouette", label: say("look_silhouette"), title: say("look_silhouette_hint") },
          { value: "matcap", label: say("look_matcap"), title: say("look_matcap_hint") },
          { value: "xray", label: say("look_xray"), title: say("look_xray_hint") },
        ]}
        onChange={(next) => editor.setLook(next as LookName)}
      />
    </Section>
  );
}

function Morphs({ editor }: { editor: Editor }) {
  const names = useEditor(editor, (state) => state.morphNames);
  const weights = useEditor(editor, (state) => state.morphs);
  const faceBones = useEditor(editor, (state) => state.faceBones);
  if (!names.length) return null;
  const bones = Object.keys(FACE_BONES) as (keyof FaceBones)[];
  const any = names.some((name) => weights[name]) || bones.some((name) => faceBones[name]);
  return (
    <Section
      title={say("morphs_title")}
      action={
        any ? (
          <Button id="morphsReset" variant="ghost" size="xs" className="text-muted-foreground" onClick={editor.resetMorphs}>
            <RotateCcw /> {say("morphs_reset")}
          </Button>
        ) : undefined
      }
    >
      {names.map((name) => {
        const value = weights[name] ?? 0;
        return (
          <div key={name} className="py-1">
            <div className="flex items-center justify-between">
              <span id={`morph-label-${name}`}>{morphLabel(name)}</span>
              <output className="text-xs text-muted-foreground tabular-nums">{value.toFixed(2)}</output>
            </div>
            <Slider
              id={`morph-${name}`}
              min={0}
              max={1}
              step={0.01}
              value={value}
              origin={0}
              onValueChange={([next]) => editor.setMorph(name, next)}
              thumbProps={{ "aria-labelledby": `morph-label-${name}` }}
            />
          </div>
        );
      })}
      {bones.map((name) => (
        <div key={name} className="py-1">
          <div className="flex items-center justify-between">
            <span id={`face-label-${name}`}>{morphLabel(name)}</span>
            <output className="text-xs text-muted-foreground tabular-nums">{faceBones[name].toFixed(0)}°</output>
          </div>
          <Slider
            id={`face-${name}`}
            min={FACE_BONES[name].min}
            max={FACE_BONES[name].max}
            step={1}
            value={faceBones[name]}
            origin={0}
            onValueChange={([next]) => editor.setFaceBone(name, next)}
            thumbProps={{ "aria-labelledby": `face-label-${name}` }}
          />
        </div>
      ))}
    </Section>
  );
}

function Parts({ editor }: { editor: Editor }) {
  const names = useEditor(editor, (state) => state.partNames);
  const hidden = useEditor(editor, (state) => state.hiddenParts);
  if (!names.length) return null;
  return (
    <Section title={say("parts_title")}>
      <div className="grid grid-cols-1 gap-x-3 gap-y-1 md:grid-cols-2">
        {names.map((name) => (
          <label key={name} htmlFor={`part-${name}`} className="flex h-6 items-center justify-between gap-2">
            {partLabel(name)}
            <Switch
              id={`part-${name}`}
              checked={!hidden.includes(name)}
              onCheckedChange={(on) => editor.setPartVisible(name, on)}
            />
          </label>
        ))}
      </div>
    </Section>
  );
}

function Stats({ editor }: { editor: Editor }) {
  const stats = useEditor(editor, (state) => state.stats);
  if (!stats) return null;
  const rows: [string, string][] = [
    [say("stats_vertices"), String(stats.vertices)],
    [say("stats_triangles"), String(stats.triangles)],
    [say("stats_bones"), String(stats.bones)],
    [say("stats_part_vertices"), say("stats_part_count", { vertices: stats.partVertices, count: stats.parts })],
    [say("stats_file"), `${(stats.bytes / 1e6).toFixed(2)} MB`],
    [say("stats_fetch"), `${stats.fetchMs.toFixed(0)} ms`],
    [say("stats_parse"), `${stats.parseMs.toFixed(0)} ms`],
  ];
  return (
    <Section title={say("stats_title")}>
      <dl id="stats" className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="text-right tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

/**
 * How the body is seen, not what it is: everything here takes effect at once, asks for no new body, and
 * stays out of the parameter file. That's why it isn't with the sliders on the left.
 */
export function Inspector({ editor }: { editor: Editor }) {
  const open = useEditor(editor, (state) => state.inspecting);
  const driven = useEditor(editor, (state) => state.driven);
  if (!open) return null;
  return (
    <aside
      id="inspector"
      aria-label={say("inspector_title")}
      className="absolute inset-y-0 right-0 z-10 flex w-48 flex-col border-l bg-card shadow-sm md:static md:w-64 md:shadow-none"
    >
      <header className="flex h-9 shrink-0 items-center justify-between border-b pr-1.5 pl-3">
        <h2 className="font-medium">{say("inspector_title")}</h2>
        <Button variant="ghost" size="icon-xs" aria-label={say("inspector_close")} onClick={() => editor.setInspecting(false)}>
          <X />
        </Button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <p className="px-3 pt-2 text-xs text-muted-foreground">{say("inspector_note")}</p>
        <Look editor={editor} />
        <Morphs editor={editor} />
        <Parts editor={editor} />
        <Section title={say("rig_title")}>
          <label htmlFor="driven" className="flex h-6 items-center justify-between gap-2">
            {say("rig_driven")}
            <Switch id="driven" checked={driven} onCheckedChange={editor.setDriven} />
          </label>
        </Section>
        <Stats editor={editor} />
      </div>
    </aside>
  );
}
