import argparse
import json
from pathlib import Path
import sys

from . import MODEL_IDS
from .manifest import load_manifest
from .report import create_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate pose models on recorded fencing clips")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check inputs without loading model packages")
    validate.add_argument("manifest", type=Path)
    run = commands.add_parser("run", help="Run one model in its own environment/process")
    run.add_argument("manifest", type=Path)
    run.add_argument("--model", choices=MODEL_IDS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--image-size", type=int)
    run.add_argument("--threshold", type=float, default=0.25)
    run.add_argument("--weights", type=Path)
    run.add_argument("--warmup", type=int, default=3)
    report = commands.add_parser("report", help="Create an offline comparison for one clip directory")
    report.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            print(create_report(args.directory))
            return 0
        clips = load_manifest(args.manifest)
        if args.command == "validate":
            print(json.dumps({"valid": True, "clips": [c.id for c in clips]}, indent=2))
            return 0
        if args.warmup < 0:
            raise ValueError("warmup must be non-negative")
        for clip in clips:
            destination = args.output / clip.id / f"{args.model}.json"
            if destination.exists():
                raise FileExistsError(f"Refusing to replace {destination}; choose another output directory")
        from .adapters import ModelAdapter
        from .runner import run_clip

        adapter = ModelAdapter(args.model, device=args.device, image_size=args.image_size,
                               threshold=args.threshold, weights=args.weights)
        for clip in clips:
            print(run_clip(clip, adapter, args.output, warmup=args.warmup), flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
