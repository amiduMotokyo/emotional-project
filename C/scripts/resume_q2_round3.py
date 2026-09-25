"""Serialization-only adapter for the frozen round-three runner.

NumPy boolean eligibility cannot be serialized by the original JSON writer.
Convert that return value to Python bool without changing comparisons, model
training, predictions, selection or any frozen producer source hash.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from C.scripts import run_q2_round3 as experiment


def install_adapter():
    original = experiment.guard
    def guard(*args, **kwargs):
        return bool(original(*args, **kwargs))
    experiment.guard = guard
    path = experiment.OUT/'serialization_adapter.json'
    record = dict(source=str(Path(__file__).resolve()), sha256=experiment.digest(__file__),
                  original_manifest=experiment.digest(experiment.OUT/'manifest.json'),
                  change='Convert numpy.bool_ guard return to Python bool, identical truth value; no numerical or training changes.',
                  reason='Initial select completed predictions but JSON serialization failed before lock or test.')
    if path.exists():
        assert experiment.read(path) == record
    else:
        experiment.write(path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all','prepare','train','text','select','evaluate','report'], default='all')
    args = parser.parse_args()
    run = experiment.Run()
    install_adapter()
    for stage in (['prepare','train','text','select','evaluate','report'] if args.stage == 'all' else [args.stage]):
        getattr(run, stage)()
