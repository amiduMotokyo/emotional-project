from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_problem1 import main as audit_problem1
from src.downstream_probe import main as downstream_probe
from src.whisper_text_alignment import main as whisper_text_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--whisper-limit", type=int, default=20)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-max-seconds", type=float, default=8.0)
    args = parser.parse_args()

    print("Step 1/3: coverage, alignment, stream and reproducibility audit")
    audit_problem1(args.config)
    print("Step 2/3: grouped downstream linear probe")
    downstream_probe(args.config)
    if not args.skip_whisper:
        print("Step 3/3: text time alignment with Whisper")
        whisper_text_alignment(
            args.config,
            args.whisper_limit,
            args.whisper_model,
            args.whisper_max_seconds,
        )
    else:
        print("Step 3/3: Whisper text-time validation skipped")


if __name__ == "__main__":
    main()
