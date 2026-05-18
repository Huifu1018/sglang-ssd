"""Install the source-level SGLANG_GROUP patch into SGLang 0.5.9."""

from __future__ import annotations

import argparse
import json
import sys

from sglang_group.sglang.source_patch import (
    apply_source_integration,
    is_source_integrated,
    resolve_sglang_root,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patch an installed SGLang 0.5.9 package or local source tree so "
            "--speculative-algorithm SGLANG_GROUP is accepted natively and "
            "scheduler output hooks can prefetch SSD proposals after request commits."
        )
    )
    parser.add_argument(
        "--sglang-root",
        help=(
            "Path to the SGLang package root or repository root. Defaults to the "
            "installed sglang package in the current Python environment."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would change.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether the source-level patch is already installed.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    if args.check:
        root = resolve_sglang_root(args.sglang_root)
        integrated = is_source_integrated(root)
        payload = {
            "sglang_root": str(root),
            "source_integrated": integrated,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"sglang_root: {root}")
            print(f"source_integrated: {integrated}")
        raise SystemExit(0 if integrated else 1)

    report = apply_source_integration(args.sglang_root, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return

    print(f"sglang_root: {report.sglang_root}")
    if report.already_integrated:
        print("source_integrated: true")
        print("changed_files: []")
    elif args.dry_run:
        print("source_integrated: false")
        print("dry_run: true")
        for path in report.changed_files:
            print(f"would_change: {path}")
    else:
        print("source_integrated: true")
        for path in report.changed_files:
            print(f"changed: {path}")
        print("backup_suffix: .sglang-group.bak")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
