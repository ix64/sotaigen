import { RouterProvider } from "@tanstack/react-router";
import { createRoot } from "react-dom/client";
import { m } from "@/paraglide/messages.js";
import { getLocale } from "@/paraglide/runtime.js";
import { router } from "./router";
import "./index.css";

// The page's language decides which glyphs Han characters take, and what a screen reader speaks.
document.documentElement.lang = getLocale() === "zh" ? "zh-CN" : "en";
document.title = m.app_title();

createRoot(document.getElementById("root") as HTMLElement).render(<RouterProvider router={router} />);
