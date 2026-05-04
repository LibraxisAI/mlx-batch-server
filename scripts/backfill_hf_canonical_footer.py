#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, PTH123
"""
Backfill the canonical Vibecrafted footer into every README.md across the
LibraxisAI HF org that does not already have one.

The canonical footer is, byte-for-byte:

    ---

    𝚅𝚒𝚋𝚎𝚌𝚛𝚊𝚏𝚝𝚎𝚍. with AI Agents by VetCoders (c)2024-2026 LibraxisAI

(Note the trailing dot after Vibecrafted, the year range, and the
LibraxisAI suffix. Do not paraphrase.)

Patcher is conservative:
- never deletes content
- only appends if `𝚅𝚒𝚋𝚎𝚌𝚛𝚊𝚏𝚝𝚎𝚍` is absent in the body
- preserves trailing whitespace tail; ensures exactly one blank line before
  the horizontal rule and exactly one final newline
- does NOT touch cards that have ANY form of the Vibecrafted footer (even
  non-canonical ones) — those will be normalised in a later, more invasive
  pass

Auth: relies on `hf auth login` having been run.

Usage:
    python3 scripts/backfill_hf_canonical_footer.py            # dry-run
    python3 scripts/backfill_hf_canonical_footer.py --apply    # push
    python3 scripts/backfill_hf_canonical_footer.py --limit 5
    python3 scripts/backfill_hf_canonical_footer.py --only Bielik Qwen
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

ORG = "LibraxisAI"
CANONICAL_FOOTER = (
    "\n---\n\n𝚅𝚒𝚋𝚎𝚌𝚛𝚊𝚏𝚝𝚎𝚍. with AI Agents by VetCoders (c)2024-2026 LibraxisAI\n"
)
COMMIT_MSG = "card: append canonical Vibecrafted footer"

# Conservative detection: any occurrence of the Vibecrafted bytes is enough
# to skip the file. Non-canonical variants will be normalised separately.
FOOTER_MARKER = "𝚅𝚒𝚋𝚎𝚌𝚛𝚊𝚏𝚝𝚎𝚍"


def patch_readme(content: str) -> tuple[str, str]:
    """Return (new_content, change_kind). change_kind is "" if no change."""
    if FOOTER_MARKER in content:
        return content, ""

    # Trim a single trailing newline if present so we control spacing exactly.
    body = content.rstrip("\n")
    new_content = body + "\n" + CANONICAL_FOOTER
    return new_content, "appended-canonical-footer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="actually push (default: dry-run)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="process only first N models"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="substring filter: only modelIds containing any of these",
    )
    args = parser.parse_args()

    api = HfApi()
    print(f"# Listing models for org={ORG}")
    models = list(api.list_models(author=ORG, limit=200))
    print(f"# Found {len(models)} models")

    if args.only:
        models = [m for m in models if any(s in m.modelId for s in args.only)]
        print(f"# Filtered by --only {args.only}: {len(models)} models")

    if args.limit:
        models = models[: args.limit]
        print(f"# Limited to first {args.limit}: {len(models)} models")

    counts = {
        "already_has": 0,
        "patched": 0,
        "no_readme": 0,
        "errors": 0,
    }

    for m in models:
        repo_id = m.modelId
        try:
            readme_path = api.hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            print(f"  [no-readme] {repo_id}")
            counts["no_readme"] += 1
            continue
        except Exception as exc:
            print(f"  [error] {repo_id}: {exc}")
            counts["errors"] += 1
            continue

        with open(readme_path, encoding="utf-8") as fh:
            content = fh.read()

        new_content, kind = patch_readme(content)
        if not kind:
            counts["already_has"] += 1
            continue

        counts["patched"] += 1
        print(f"  [{kind}] {repo_id}")

        if args.apply:
            try:
                api.upload_file(
                    path_or_fileobj=new_content.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=COMMIT_MSG,
                )
                print("    pushed")
            except Exception as exc:
                print(f"    [push-error] {exc}")
                counts["errors"] += 1
                counts["patched"] -= 1

    print()
    print("# Summary")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if not args.apply:
        print()
        print("# Dry-run only. Re-run with --apply to push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
