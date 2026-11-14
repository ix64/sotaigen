import { Slider as SliderPrimitive } from "radix-ui";
import type * as React from "react";
import { cn } from "@/lib/utils";

const THUMB_PX = 16;

/** A stretch of the track to mark, as shares of it. `title` shows on hover, so it stays hoverable. */
export interface SliderZone {
  from: number;
  to: number;
  level: "blocked" | "warned";
  title: string;
}

// Radix keeps the thumb inside the track, so a value's place is not simply its share of the width.
const place = (share: number) => `calc(${share * 100}% + ${(0.5 - share) * THUMB_PX}px)`;

const shareOf = (value: number, min: number, max: number) => Math.min(1, Math.max(0, (value - min) / (max - min)));

/**
 * shadcn's slider, with one change: the bar runs from `origin`, marked with a tick, to the thumb. So a
 * slider says how far it has come from where it started, in whichever direction.
 */
function Slider({
  className,
  value,
  origin,
  min,
  max,
  zones = [],
  thumbProps,
  ...props
}: Omit<React.ComponentProps<typeof SliderPrimitive.Root>, "value" | "min" | "max"> & {
  value: number;
  origin: number;
  min: number;
  max: number;
  zones?: SliderZone[];
  thumbProps?: React.ComponentProps<typeof SliderPrimitive.Thumb>;
}) {
  const here = shareOf(value, min, max);
  const start = shareOf(origin, min, max);
  const low = Math.min(here, start);
  const high = Math.max(here, start);
  const inBlocked = zones.some((zone) => zone.level === "blocked" && zone.from <= here && here <= zone.to);
  return (
    <SliderPrimitive.Root
      data-slot="slider"
      value={[value]}
      min={min}
      max={max}
      className={cn(
        "relative flex h-5 w-full touch-none items-center select-none data-[disabled]:opacity-40",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1 w-full grow overflow-hidden rounded-full bg-muted" />
      {zones.map((zone) => (
        <span
          key={`${zone.level}-${zone.from}`}
          data-zone={zone.level}
          title={zone.title}
          className={cn("absolute h-1.5 rounded-[1px]", zone.level === "blocked" ? "zone-blocked" : "bg-warn/45")}
          style={{ left: place(zone.from), width: `calc(${place(zone.to)} - ${place(zone.from)})` }}
        />
      ))}
      <span
        className="pointer-events-none absolute h-1 rounded-full bg-primary"
        style={{ left: place(low), width: `calc(${place(high)} - ${place(low)})` }}
      />
      <span
        className="pointer-events-none absolute h-2.5 w-0.5 -translate-x-1/2 rounded-full bg-muted-foreground/50"
        style={{ left: place(start) }}
      />
      <SliderPrimitive.Thumb
        data-slot="slider-thumb"
        className={cn(
          "block size-4 rounded-full border-2 bg-card shadow-xs transition-[box-shadow] outline-none hover:ring-4 focus-visible:ring-4 focus-visible:ring-ring/40 data-[disabled]:border-muted-foreground",
          inBlocked ? "border-bad hover:ring-bad/15" : "border-primary hover:ring-primary/15",
        )}
        {...thumbProps}
      />
    </SliderPrimitive.Root>
  );
}

export { Slider };
