#!/usr/bin/env python3
"""
Build a compact SOTA comparison report for current MAIFS runs.

Inputs:
  - multi-dataset case3 summary json
  - dataset preparation report json

Outputs:
  - markdown report
  - machine-readable json report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MULTI_SUMMARY = (
    "experiments/results/phase2_patha_case3_multidata/"
    "multi_dataset_case3_comparison_20260304_v2_6sets.json"
)
DEFAULT_PREP_REPORT = "experiments/results/sota_dataset_prep_report_20260304.json"
DEFAULT_OUT_MD = "experiments/results/sota_comparison_20260304.md"
DEFAULT_OUT_JSON = "experiments/results/sota_comparison_20260304.json"


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _find_dataset(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for row in rows:
        if row.get("dataset") == name:
            return row
    raise KeyError(f"dataset not found in summary: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MAIFS vs SOTA comparison report")
    p.add_argument("--multi-summary", default=DEFAULT_MULTI_SUMMARY, type=str)
    p.add_argument("--prep-report", default=DEFAULT_PREP_REPORT, type=str)
    p.add_argument("--out-md", default=DEFAULT_OUT_MD, type=str)
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    multi = _load_json(args.multi_summary)
    prep = _load_json(args.prep_report)
    rows = multi.get("datasets", [])
    overall = multi.get("overall", {})

    opensdi = _find_dataset(rows, "OpenSDI_subset300")
    aigen = _find_dataset(rows, "AI-GenBench_proxy300")

    # External leaderboard snapshots (source/date pinned in final report)
    # NOTE: metric definitions and protocols differ from MAIFS 3-class macro-F1 setup.
    external = {
        "opensdi_leaderboard_2025_11_03": {
            "top_method": "GMMDet",
            "f1_avg": 94.16,
            "reference_url": "https://opensdi.github.io/",
            "snapshot_date": "2025-11-03",
            "other_methods": {
                "DIRE": 91.62,
                "NPR": 88.66,
                "UniversalFakeDetect": 86.95,
                "RINE": 84.44,
            },
        },
        "aigenbench_leaderboard_2025_02_17": {
            "top_method": "CLOSER",
            "avg_accuracy": 95.31,
            "avg_auc": 99.16,
            "avg_eer": 1.97,
            "reference_url": "https://mi-biolab.github.io/ai-genbench/",
            "snapshot_date": "2025-02-17",
            "other_methods": {
                "Msm-detection_acc": 93.18,
                "RIGID_acc": 89.95,
                "NPR_acc": 89.77,
                "DRCT_acc": 87.95,
            },
        },
    }

    maifs_opensdi_best = max(
        float(opensdi["cobra_only_f1_mean"]),
        float(opensdi["daac_only_f1_mean"]),
        float(opensdi["cobra_plus_daac_f1_mean"]),
    )
    maifs_aigen_best = max(
        float(aigen["cobra_only_f1_mean"]),
        float(aigen["daac_only_f1_mean"]),
        float(aigen["cobra_plus_daac_f1_mean"]),
    )

    derived = {
        "maifs_open_sdi_best_macro_f1_3class": maifs_opensdi_best,
        "maifs_aigen_proxy_best_macro_f1_3class": maifs_aigen_best,
        "protocol_mismatch_note": (
            "MAIFS numbers are 3-class macro-F1 on local subset/proxy protocols; "
            "external leaderboard numbers are dataset-specific official protocols "
            "(mostly binary detection and/or different splits)."
        ),
        "naive_gap_vs_leaderboard": {
            "opensdi_best_minus_top_f1avg": maifs_opensdi_best - external["opensdi_leaderboard_2025_11_03"]["f1_avg"],
            "aigen_best_minus_top_avg_accuracy": maifs_aigen_best - external["aigenbench_leaderboard_2025_02_17"]["avg_accuracy"],
        },
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "multi_summary": args.multi_summary,
            "prep_report": args.prep_report,
        },
        "maifs_overall": overall,
        "maifs_datasets": rows,
        "external_snapshots": external,
        "restricted_dataset_checks": prep.get("restricted_dataset_checks", {}),
        "derived": derived,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append("# MAIFS vs SOTA Snapshot (2026-03-04)")
    md_lines.append("")
    md_lines.append("## 1) MAIFS 6-dataset internal comparison (3-class macro-F1)")
    md_lines.append("")
    md_lines.append("| Dataset | COBRA-only | DAAC-only | COBRA+DAAC |")
    md_lines.append("|---|---:|---:|---:|")
    for row in rows:
        md_lines.append(
            "| "
            + str(row["dataset"])
            + " | "
            + f"{_fmt(float(row['cobra_only_f1_mean']))} ± {_fmt(float(row['cobra_only_f1_std']))}"
            + " | "
            + f"{_fmt(float(row['daac_only_f1_mean']))} ± {_fmt(float(row['daac_only_f1_std']))}"
            + " | "
            + f"{_fmt(float(row['cobra_plus_daac_f1_mean']))} ± {_fmt(float(row['cobra_plus_daac_f1_std']))}"
            + " |"
        )
    md_lines.append("")
    md_lines.append(
        "- Overall (60 runs): "
        + f"COBRA={_fmt(float(overall['cobra_only_f1_mean']))}, "
        + f"DAAC={_fmt(float(overall['daac_only_f1_mean']))}, "
        + f"COBRA+DAAC={_fmt(float(overall['cobra_plus_daac_f1_mean']))}"
    )
    md_lines.append("")

    md_lines.append("## 2) External leaderboard snapshots (official pages)")
    md_lines.append("")
    md_lines.append("### OpenSDI leaderboard (2025-11-03 snapshot)")
    md_lines.append("")
    md_lines.append(
        "- Top: GMMDet F1 Avg 94.16 | DIRE 91.62 | NPR 88.66 | UniFD 86.95 | RINE 84.44"
    )
    md_lines.append("- Source: https://opensdi.github.io/")
    md_lines.append("")
    md_lines.append("### AI-GenBench leaderboard (last updated 2025-02-17)")
    md_lines.append("")
    md_lines.append(
        "- Top: CLOSER Acc 95.31 / AUC 99.16 / EER 1.97; "
        "Msm-detection 93.18 / 98.89 / 2.57; NPR 89.77 / 97.08 / 5.42"
    )
    md_lines.append("- Source: https://mi-biolab.github.io/ai-genbench/")
    md_lines.append("")

    md_lines.append("## 3) Comparison caveat (important)")
    md_lines.append("")
    md_lines.append(
        "- MAIFS results here are **3-class macro-F1 on local subset/proxy protocols**."
    )
    md_lines.append(
        "- Leaderboard results are **official protocol metrics** (mostly binary detection and different split design)."
    )
    md_lines.append("- Therefore, direct numeric superiority claims are not statistically valid yet.")
    md_lines.append("")

    md_lines.append("## 4) Restricted dataset readiness")
    md_lines.append("")
    md_lines.append("| Dataset | URL status | Local ready | Note |")
    md_lines.append("|---|---:|---:|---|")
    for key in ("coverage", "columbia", "nist16"):
        item = prep.get("restricted_dataset_checks", {}).get(key, {})
        md_lines.append(
            "| "
            + key
            + " | "
            + str(item.get("http_status"))
            + " | "
            + str(item.get("local_ready"))
            + " | "
            + (str(item.get("error")) if item.get("error") else "")
            + " |"
        )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[saved] json: {out_json}")
    print(f"[saved] md:   {out_md}")


if __name__ == "__main__":
    main()
