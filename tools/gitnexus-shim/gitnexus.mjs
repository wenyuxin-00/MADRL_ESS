#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const TOP_LEVEL_HELP = `
Shimmed commands:
  detect_changes [--scope unstaged|staged|all|compare] [--base_ref <git-ref>] [--repo <name>] [--json]

Why this shim exists:
  This workspace's npm/npx install path is unstable on Windows, and the cached
  GitNexus CLI version does not expose detect_changes as a top-level command.
  The shim forwards normal commands to the cached GitNexus runtime and calls
  the local backend directly for detect_changes.
`;

async function pathExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function statMtime(targetPath) {
  try {
    const stats = await fs.stat(targetPath);
    return stats.mtimeMs;
  } catch {
    return -1;
  }
}

async function isGitNexusRuntime(runtimeDir) {
  const required = [
    path.join(runtimeDir, "dist", "cli", "index.js"),
    path.join(runtimeDir, "dist", "mcp", "local", "local-backend.js"),
  ];
  for (const filePath of required) {
    if (!(await pathExists(filePath))) {
      return false;
    }
  }
  return true;
}

async function findRuntimeDir() {
  const candidates = [];
  const explicit = process.env.GITNEXUS_RUNTIME_DIR;
  if (explicit) {
    candidates.push({ dir: explicit, priority: 1000, mtimeMs: Number.MAX_SAFE_INTEGER });
  }

  const localPackage = path.join(REPO_ROOT, "node_modules", "gitnexus");
  if (await pathExists(localPackage)) {
    candidates.push({
      dir: localPackage,
      priority: 900,
      mtimeMs: await statMtime(path.join(localPackage, "package.json")),
    });
  }

  const cacheRoot = path.join(
    process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"),
    "npm-cache",
    "_npx",
  );

  if (await pathExists(cacheRoot)) {
    const entries = await fs.readdir(cacheRoot, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      const runtimeDir = path.join(cacheRoot, entry.name, "node_modules", "gitnexus");
      if (!(await isGitNexusRuntime(runtimeDir))) {
        continue;
      }
      candidates.push({
        dir: runtimeDir,
        priority: 100,
        mtimeMs: await statMtime(path.join(runtimeDir, "package.json")),
      });
    }
  }

  candidates.sort((left, right) => {
    if (left.priority !== right.priority) {
      return right.priority - left.priority;
    }
    return right.mtimeMs - left.mtimeMs;
  });

  for (const candidate of candidates) {
    if (await isGitNexusRuntime(candidate.dir)) {
      return candidate.dir;
    }
  }

  throw new Error(
    [
      "Unable to find a usable GitNexus runtime.",
      "The repo-local shim is installed, but no cached or local GitNexus package was found.",
      "Try one of these:",
      "  1. Recreate the cache with a one-time working install of gitnexus",
      "  2. Set GITNEXUS_RUNTIME_DIR to a valid gitnexus package directory",
    ].join("\n"),
  );
}

function print(text) {
  process.stdout.write(String(text).replace(/\s*$/, "") + "\n");
}

function printError(text) {
  process.stderr.write(String(text).replace(/\s*$/, "") + "\n");
}

async function runCli(runtimeDir, args, extraHelpText = "") {
  const cliPath = path.join(runtimeDir, "dist", "cli", "index.js");

  const exitCode = await new Promise((resolve) => {
    const child = spawn(process.execPath, [cliPath, ...args], {
      cwd: REPO_ROOT,
      stdio: "inherit",
      env: process.env,
    });
    child.on("error", () => resolve(1));
    child.on("exit", (code) => resolve(code ?? 1));
  });

  if (exitCode === 0 && extraHelpText) {
    print(extraHelpText);
  }

  process.exit(exitCode);
}

function parseDetectChangesArgs(rawArgs) {
  const params = {};
  let outputJson = false;
  let help = false;

  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];
    switch (arg) {
      case "--scope":
        params.scope = rawArgs[index + 1];
        index += 1;
        break;
      case "--base_ref":
      case "--base-ref":
        params.base_ref = rawArgs[index + 1];
        index += 1;
        break;
      case "--repo":
      case "-r":
        params.repo = rawArgs[index + 1];
        index += 1;
        break;
      case "--json":
        outputJson = true;
        break;
      case "--help":
      case "-h":
        help = true;
        break;
      default:
        if (arg.startsWith("-")) {
          throw new Error(`Unsupported detect_changes option: ${arg}`);
        }
        throw new Error(`Unexpected detect_changes argument: ${arg}`);
    }
  }

  return { params, outputJson, help };
}

async function runDetectChanges(runtimeDir, rawArgs) {
  const { params, outputJson, help } = parseDetectChangesArgs(rawArgs);
  if (help) {
    print(`
Usage: gitnexus detect_changes [--scope unstaged|staged|all|compare] [--base_ref <git-ref>] [--repo <name>] [--json]

Examples:
  gitnexus detect_changes --scope all
  gitnexus detect_changes --scope compare --base_ref main
`);
    process.exit(0);
  }

  const backendUrl = pathToFileURL(
    path.join(runtimeDir, "dist", "mcp", "local", "local-backend.js"),
  ).href;
  const formatterUrl = pathToFileURL(
    path.join(runtimeDir, "dist", "cli", "eval-server.js"),
  ).href;

  const [{ LocalBackend }, { formatDetectChangesResult }] = await Promise.all([
    import(backendUrl),
    import(formatterUrl),
  ]);

  const backend = new LocalBackend();
  const ok = await backend.init();
  if (!ok) {
    printError("GitNexus: No indexed repositories found. Run: gitnexus analyze");
    process.exit(1);
  }

  try {
    const result = await backend.callTool("detect_changes", params);
    if (outputJson) {
      print(JSON.stringify(result, null, 2));
    } else {
      print(formatDetectChangesResult(result));
    }
    process.exit(result?.error ? 1 : 0);
  } finally {
    if (typeof backend.dispose === "function") {
      await backend.dispose();
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  const runtimeDir = await findRuntimeDir();

  if (args.length === 0) {
    await runCli(runtimeDir, ["--help"], TOP_LEVEL_HELP);
    return;
  }

  const [command, ...rest] = args;

  if (command === "detect_changes" || command === "detect-changes") {
    await runDetectChanges(runtimeDir, rest);
    return;
  }

  if ((command === "--help" || command === "-h") && rest.length === 0) {
    await runCli(runtimeDir, ["--help"], TOP_LEVEL_HELP);
    return;
  }

  await runCli(runtimeDir, args);
}

main().catch((error) => {
  printError(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
