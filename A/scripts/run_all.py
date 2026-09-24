from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_audio import main as extract_audio
from src.extract_text import main as extract_text
from src.extract_visual import main as extract_visual
from src.merge_and_export import main as merge_and_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("Step 1/4: audio extraction and alignment")
    extract_audio(args.config, args.overwrite)
    print("Step 2/4: visual extraction and alignment")
    extract_visual(args.config, args.overwrite)
    print("Step 3/4: text extraction")
    extract_text(args.config, args.overwrite)
    print("Step 4/4: merge and export")
    merge_and_export(args.config)
    print(f"All steps complete. Output: {PROJECT_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()

