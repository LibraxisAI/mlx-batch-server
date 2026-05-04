#!/usr/bin/env python3
# ruff: noqa: RUF001, PTH123, PLR0915
"""
Backfill `## Inference tested on` section into every README.md across the
LibraxisAI HF org.

Runs in dry-run mode by default. Pass --apply to actually push.

Patcher is conservative:
- never deletes content
- only adds the `## Inference tested on` section if absent
- inserts before `## Related` if present, else before the Vibecrafted footer,
  else at the end of the file
- skips any model where README already contains `## Inference tested on`
- skips models with no README at all (logs them; we will write fresh cards
  in a separate pass)

Auth: relies on `hf auth login` having been run.

Usage:
    python3 scripts/backfill_hf_inference_section.py            # dry-run
    python3 scripts/backfill_hf_inference_section.py --apply    # push
    python3 scripts/backfill_hf_inference_section.py --limit 5  # only first 5
    python3 scripts/backfill_hf_inference_section.py --only Svetliq Bielik
"""

from __future__ import annotations

import argparse
import re
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

ORG = "LibraxisAI"
INFERENCE_SECTION = (
    "## Inference tested on\n"
    "\n"
    "[`LibraxisAI/mlx-batch-server`](https://github.com/LibraxisAI/mlx-batch-server)\n"
)
COMMIT_MSG = "card: add 'Inference tested on' link to mlx-batch-server"

VIBECRAFTED_FOOTER_RE = re.compile(
    r"\n---\s*\n[^-\n]*𝚅𝚒𝚋𝚎𝚌𝚛𝚊𝚏𝚝𝚎𝚍",
    re.MULTILINE,
)


def patch_readme(content: str) -> tuple[str, str]:
    """Return (new_content, change_kind). change_kind is "" if no change."""
    if "## Inference tested on" in content:
        return content, ""

    section = INFERENCE_SECTION

    if "## Related" in content:
        new_content = content.replace(
            "## Related",
            section + "\n## Related",
            1,
        )
        return new_content, "before-related"

    match = VIBECRAFTED_FOOTER_RE.search(content)
    if match:
        idx = match.start()
        new_content = content[:idx] + "\n" + section + content[idx:]
        return new_content, "before-vibecrafted-footer"

    new_content = content.rstrip() + "\n\n" + section
    return new_content, "appended-eof"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually push to HF (default: dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only first N models",
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
    diffs: list[str] = []

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
        except Exception as exc:  # network, auth, etc.
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
        diffs.append(f"  [{kind}] {repo_id}")
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
