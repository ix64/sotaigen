import type { ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

/** A tooltip around one control. `keys` names its keyboard shortcut. */
export function Hint({ text, keys, children }: { text: string; keys?: string; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent>
        {text}
        {keys && <kbd className="ml-2 font-sans opacity-70">{keys}</kbd>}
      </TooltipContent>
    </Tooltip>
  );
}
