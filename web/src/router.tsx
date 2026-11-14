// The pages, all on the one port web/serve.mjs listens on. A page and three.js load only when its route
// is opened. web/runtime's check pages are not here: they are built apart and must not carry the app.

import { createRootRoute, createRoute, createRouter, Link, lazyRouteComponent, Outlet } from "@tanstack/react-router";
import { m } from "@/paraglide/messages.js";

// The address's values may parse as numbers: ?driven=0 is 0. The pages read them as written.
function strings<Key extends string>(keys: readonly Key[], search: Record<string, unknown>) {
  const out: Partial<Record<Key, string>> = {};
  for (const key of keys) {
    if (search[key] !== undefined && search[key] !== null) out[key] = String(search[key]);
  }
  return out;
}

const EDITOR_KEYS = ["body", "pose", "look", "panel"] as const;

export type EditorSearch = Partial<Record<(typeof EDITOR_KEYS)[number], string>> & { inspect?: boolean };

function NotFound() {
  return (
    <div className="grid h-dvh place-items-center text-muted-foreground">
      <p>
        {m.app_not_found()}
        <Link to="/" className="text-primary underline underline-offset-4">
          {m.app_back_to_editor()}
        </Link>
      </p>
    </div>
  );
}

const root = createRootRoute({ component: Outlet, notFoundComponent: NotFound });

const editor = createRoute({
  getParentRoute: () => root,
  path: "/",
  validateSearch: (search: Record<string, unknown>): EditorSearch => ({
    ...strings(EDITOR_KEYS, search),
    inspect: search.inspect ? true : undefined,
  }),
  loaderDeps: ({ search }) => search,
  // Once, on the way in. The editor keeps going while the address follows what it's showing.
  shouldReload: false,
  loader: async ({ deps }) => (await import("../editor/boot")).bootEditor(deps),
  component: lazyRouteComponent(() => import("../editor/EditorRoute"), "EditorRoute"),
});

export const router = createRouter({
  routeTree: root.addChildren([editor]),
  // The page itself says what it's waiting for.
  defaultPendingMs: Number.POSITIVE_INFINITY,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
