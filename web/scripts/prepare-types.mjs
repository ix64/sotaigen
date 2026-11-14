// The generated schema has no parameter types; keep its type-checking copy separate.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const source = new URL("../../build/editor/params.generated.mjs", import.meta.url);
const directory = new URL("../../build/typecheck/", import.meta.url);
mkdirSync(directory, { recursive: true });
writeFileSync(new URL("params.generated.mjs", directory), "// @ts-nocheck\n" + readFileSync(source, "utf8"));
