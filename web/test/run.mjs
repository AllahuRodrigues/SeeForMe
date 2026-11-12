#!/usr/bin/env node
// Unit tests of the web app's decision code (no browser, no new dependencies): each test/*.test.ts is bundled with esbuild
// (already a devDependency, as for parity/run.mjs) into test/.build/ and run with node's built-in test runner.
//
//   cd web && npm test                 # type-check (test/tsconfig.json: the tests and src/), then every test file
//   node test/run.mjs voice gate       # only the files whose name contains one of these words
import { readdirSync, mkdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { build } from "esbuild";

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, ".build");
const only = process.argv.slice(2);
const files = readdirSync(here).filter((f) => f.endsWith(".test.ts") && (!only.length || only.some((w) => f.includes(w)))).sort();
if (!files.length) { console.error("no test files matched"); process.exit(1); }
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
const bundles = [];
for (const f of files) {
  const outfile = resolve(out, f.replace(/\.ts$/, ".mjs"));
  await build({ entryPoints: [resolve(here, f)], bundle: true, format: "esm", platform: "node", target: "node20", outfile, logLevel: "warning",
    define: { "import.meta.env.DEV": "false" } });
  bundles.push(outfile);
}
const r = spawnSync(process.execPath, ["--test", "--test-reporter=spec", ...bundles], { stdio: "inherit" });
process.exit(r.status ?? 1);
