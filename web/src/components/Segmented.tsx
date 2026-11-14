import type { ReactNode } from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";

/** One choice out of a few, all in sight, on a second row when a language's words don't fit one. Choosing the one that's on does nothing. */
export function Segmented({
  label,
  value,
  options,
  onChange,
  idPrefix,
  className,
}: {
  label: string;
  value: string;
  options: { value: string; label: ReactNode; title?: string }[];
  onChange: (value: string) => void;
  idPrefix?: string;
  className?: string;
}) {
  return (
    <ToggleGroup
      type="single"
      aria-label={label}
      spacing={1}
      value={value}
      onValueChange={(next) => {
        if (next) onChange(next);
      }}
      className={cn("w-full flex-wrap gap-0.5 rounded-md bg-muted p-0.5", className)}
    >
      {options.map((option) => (
        <ToggleGroupItem
          key={option.value}
          id={idPrefix ? `${idPrefix}-${option.value}` : undefined}
          value={option.value}
          title={option.title}
          className="h-7 flex-auto rounded-[5px] px-2 text-xs font-normal text-muted-foreground hover:bg-transparent hover:text-foreground data-[state=on]:bg-card data-[state=on]:font-medium data-[state=on]:text-foreground data-[state=on]:shadow-xs"
        >
          {option.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
