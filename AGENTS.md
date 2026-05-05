<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **MADRL_ESS** (784 symbols, 2009 relationships, 66 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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

## Encoding Hygiene

- Treat repository text files and notebooks as UTF-8. When editing Chinese comments, markdown, or notebook cells, use UTF-8-aware tooling and verify the result by reading it back with `encoding="utf-8"`.
- Never leave corrupted placeholders or mojibake in comments, markdown, notebook source, or saved notebook outputs. Check for repeated question-mark placeholders, Unicode replacement characters, and obvious mojibake sequences.
- If the exact Chinese wording cannot be recovered, replace the damaged text with concise correct Chinese or plain English. Do not preserve unreadable placeholders.
- After editing notebooks or documentation with non-ASCII text, run `conda run -n MADRL_ESS python -m pytest tests/test_encoding_hygiene.py` before finishing.

## Codebase Hard Rules

- Rule priority is fixed: correctness > explicit contracts > owner purity > minimal code > tests and docs.
- One owner owns one concept end to end, including input contract, core logic, state or artifacts, and public entrypoint.
- Mainline only. Do not keep compatibility paths, deprecated aliases, bridge adapters, auto dual-path dispatch, hidden migrations, or backup flows.
- Fail early at the first data or API boundary. Version checks belong at that boundary and should only produce explicit mismatch failures.
- Use exact paths and contracts only. Do not fall back to sibling scans, latest-compatible lookup, prefix fuzzy matching, or silent old-key acceptance.
- Keep wrappers only when they protect a real boundary: stable API, narrowed type or boundary, naming alignment, protocol isolation, or CLI or notebook entrypoint.
- Inline single-use private helpers when they only wrap one obvious operation and do not protect a real boundary; fewer jumps and more direct code are preferred over naming every step.
- Prefer the shortest code that preserves correctness and explicit contracts. When two designs are equally correct, choose fewer branches, fewer helpers, fewer files, and shorter error messages.
- Use compact code layout for all code writing, modification, and reformatting: keep simple expressions, return statements, function calls, short dict/list literals, and short signatures on one line when they remain readable. Do not introduce unnecessary line breaks just to satisfy broad formatting habits. Keep multiline layout for long error messages, complex nested structures, or expressions that would become hard to scan. Do not compress unrelated statements onto one line.
- Comments must explain business purpose, data meaning, or non-obvious constraints. Do not write comments that merely restate syntax, such as "check condition", "return result", "save intermediate value", or "execute current step".
- Contract failures should identify what was expected and what was received. Do not include recovery instructions unless the caller cannot infer the fix.
- If a wrapper, branch, option, or helper has no current mainline caller and no test asserting its contract, delete it.
- Run GitNexus impact before changing public functions, classes, methods, or shared runtime paths. For private single-use helpers, impact is recommended but not required unless behavior changes.

## Environment

- Default to the `MADRL_ESS` Python environment for all repo work.
- Treat the named Conda environment `MADRL_ESS` as the repository default state.
- For Python, pytest, notebook inspection, and dependency checks, prefer the named `MADRL_ESS` Conda environment through `conda run -n MADRL_ESS python ...`; if that environment is already active, use `python ...`. Do not hard-code a user-profile interpreter path.
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
