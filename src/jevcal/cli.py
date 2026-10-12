from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from .config import load_config
from .data import prepare_all
from .inference import infer_dataset
from .pipeline import run_analysis
from .reporting import regenerate_reports
from .robustness import run_robustness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevcal", description="Reproduce the routing-conditional calibration audit."
    )
    parser.add_argument("--config", default="configs/reproduce.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Prepare and validate both datasets")
    infer = subparsers.add_parser("infer", help="Run resumable OpenJev inference")
    infer.add_argument("--dataset", choices=["civil_comments", "goemotions", "all"], default="all")
    infer.add_argument(
        "--run-name",
        help="Write to artifacts/predictions/runs/<name> instead of the main prediction files",
    )
    infer.add_argument("--prompt-set", help="Prompt set defined under model.prompt_sets")
    infer.add_argument("--model-profile", help="Endpoint/model override under model.profiles")
    analyze = subparsers.add_parser("analyze", help="Run bootstrap audit, controls, tables, plots")
    analyze.add_argument(
        "--quick", action="store_true", help="Use 50 bootstrap samples for smoke tests"
    )
    robustness = subparsers.add_parser(
        "robustness",
        help="Run calibration, binning, spline, permutation, label, prompt/model checks",
    )
    robustness.add_argument("--quick", action="store_true", help="Use small resample counts")
    robustness.add_argument(
        "--strict-prediction-sets",
        action="store_true",
        help="Fail if a configured prompt/model prediction set is missing",
    )
    subparsers.add_parser(
        "report", help="Regenerate all LaTeX tables and figures from saved results"
    )
    strengthen = subparsers.add_parser(
        "strengthen",
        help="CPU-only score balancing, fitted-null, held-out calibration, all-pairs audit",
    )
    strengthen.add_argument("--settings", default="configs/strengthen.yaml")
    strengthen.add_argument("--output-dir", help="Separate resumable output directory")
    strengthen.add_argument(
        "--prediction-set",
        action="append",
        dest="prediction_sets",
        help="Configured robustness prediction-set name; repeat to select several",
    )
    strengthen.add_argument(
        "--stage", action="append", choices=["local", "null", "recalibration", "matrix"]
    )
    strengthen.add_argument("--workers", type=int, default=1)
    strengthen.add_argument(
        "--quick", action="store_true", help="Development only; NOT paper results"
    )
    followup_report = subparsers.add_parser(
        "strengthen-report", help="Rebuild follow-up figures/tables only"
    )
    followup_report.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    if args.command == "prepare":
        result = prepare_all(config)
    elif args.command == "infer":
        datasets = ["civil_comments", "goemotions"] if args.dataset == "all" else [args.dataset]
        result = {
            dataset: asyncio.run(
                infer_dataset(
                    dataset,
                    config,
                    run_name=args.run_name,
                    prompt_set=args.prompt_set,
                    model_profile=args.model_profile,
                )
            )
            for dataset in datasets
        }
    elif args.command == "analyze":
        result = run_analysis(config, quick=bool(args.quick))
    elif args.command == "robustness":
        result = run_robustness(
            config,
            quick=bool(args.quick),
            strict_prediction_sets=bool(args.strict_prediction_sets),
        )
    elif args.command == "report":
        result = regenerate_reports(config)
    elif args.command == "strengthen":
        from .strengthen import run_strengthening

        result = run_strengthening(
            config,
            settings_path=args.settings,
            output_dir=args.output_dir,
            prediction_sets=args.prediction_sets,
            stages=args.stage,
            workers=args.workers,
            quick=bool(args.quick),
        )
    elif args.command == "strengthen-report":
        from .strengthen_reporting import generate_strengthening_report

        result = generate_strengthening_report(
            args.output_dir or Path(config["paths"]["results_dir"]) / "strengthening"
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
