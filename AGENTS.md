<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **MADRL_ESS** (2294 symbols, 6901 relationships, 199 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/MADRL_ESS/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/MADRL_ESS/context` | Codebase overview, check index freshness |
| `gitnexus://repo/MADRL_ESS/clusters` | All functional areas |
| `gitnexus://repo/MADRL_ESS/processes` | All execution flows |
| `gitnexus://repo/MADRL_ESS/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## Working Style

- Use first-principles reasoning. Start from the underlying problem and desired outcome, not just the requested implementation.
- Do not assume the user has fully specified the right goal or the best path.
- If the motivation, objective, or success criteria are unclear, pause and ask concise clarifying questions before acting.
- If the goal is clear but the requested path is not the shortest, safest, or highest-leverage approach, say so explicitly and recommend a better alternative.
- Prefer the simplest path that achieves the real goal with clear tradeoffs.
- Be cautious about hidden assumptions, especially when the user sounds confident but key constraints are still missing.

## Codebase Hard Rules

- Optimize first for owner purity, explicit contracts, and low complexity; prefer fewer lines and shorter chains only after those are preserved.
- Treat `owner` at the Python module or package level: one owner should hold one concept's full lifecycle, including input contract, core logic, state or artifacts, and public entrypoint.
- Treat the `mainline` as the actual runtime path behind the current canonical CLI, notebook, or documented entrypoint; history paths, compatibility paths, and backup flows do not count.
- Rule priority is fixed: task constraints and correctness > owner purity > explicit failure without fallback > mainline length > code size > file count.
- Apply the strictest rules to `envs/`, `predictors/`, `controllers/`, `models/`, and `data/loaders/`; allow thin entrypoints in `scripts/`; allow test helpers in `tests/`; treat `configs/` as configuration data, not business-logic wrappers.
- Thin wrappers are only allowed when they have one concrete value: stable API, narrowed type or boundary, naming alignment, protocol isolation, or CLI or notebook entrypoint. Otherwise delete them.
- Do not keep transition layers for old and new contracts together. No deprecated aliases, bridge adapters, auto dual-path dispatch, or hidden migrations.
- When old schema, keys, artifact names, cache packages, or directory layouts are encountered, fail explicitly. No fallback, sibling scan, latest-compatible lookup, or prefix fuzzy matching.
- Compatibility and version checks belong at the first boundary that admits data into an owner, such as a loader, parser, manifest reader, or public owner entrypoint. Do not duplicate compatibility logic across outer wrappers.
- Every contract failure must name the old object that was hit, the new contract that is expected, and the exact entrypoint or notebook that must be rerun.
- Explicit versioning such as `SCHEMA_VERSION` is allowed only as explicit mismatch-to-failure. Silent acceptance of old aliases or old field names is forbidden.
- High-cost reproducible artifacts may remain on disk, but they must be reached only through exact locators such as a manifest, explicit path, or upstream-produced identifier. Never degrade to fuzzy discovery after a miss.
- Validation, debug, diagnostics, and export code must belong to a clear owner, usually in adjacent `validation/`, `reports/`, `artifacts/`, or subsystem-specific export modules, not inside core runtime files.
- Large files are acceptable only when they are still a pure core owner. Files over 800 lines must not be wrappers, compatibility buckets, or `utils` junk drawers; files over 1200 lines must be reviewed as reduction targets.
- `scripts/utils/` must not become a general dumping ground. Shared code stays only when multiple scripts truly depend on the same stable owner.
- Task-specific exceptions are allowed only when written explicitly with scope, affected objects, and exit conditions. Outside that scope, the default hard rules apply.

## Environment

- Default to the `MADRL_ESS` Python environment for all repo work.
- Treat the named Conda environment `MADRL_ESS` as the repository default state.
- For Python, pytest, notebook inspection, and dependency checks, prefer the `MADRL_ESS` interpreter first, for example `C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe` or `conda run -n MADRL_ESS ...`.
- Do not use the system `python` by default for this repository unless the user explicitly asks for it.
- Do not prefer the repo-local `.\.conda\python.exe` as the default anymore; treat it only as a legacy fallback when the named `MADRL_ESS` environment is unavailable or the user explicitly asks for it.
- If a dependency appears missing under the system interpreter, retry with the `MADRL_ESS` interpreter before concluding that the dependency is unavailable.
- When reporting environment issues, describe the status of the `MADRL_ESS` environment first; avoid treating missing packages in the system interpreter as the repo's default state.

## GitNexus Runtime

- For this repository, treat GitNexus CLI and npm-based tooling as permission-sensitive by default.
- If a GitNexus CLI command fails because of sandbox, network, registry, npm-cache, or similar environment issues, proactively rerun it with escalated permissions instead of assuming the repo is at fault.
- Prefer a preinstalled or repo-local `gitnexus` over repeated fresh `npx gitnexus ...` installs when available, because transient npm execution has been unreliable on this machine.
- If `npx gitnexus ...` fails with npm/arborist errors such as `Cannot destructure property 'package' of 'node.target' as it is null.`, treat that as a GitNexus/npm runtime failure and report it explicitly as tooling instability.
- Keep explaining the difference between "the repo/index is stale" and "the CLI/runtime failed before GitNexus could run".
- The agent should proactively request the permissions needed to run GitNexus and npm commands successfully, but must not claim it can bypass the system's approval model or self-grant unrestricted access.
