"""Run the authorized local Q2 experiment sequentially, logging every stage."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def wait_for_prepare(pid):
    """Wait for an already running Windows preparation process without sending signals."""
    if os.name != 'nt':
        raise ValueError('--prepared-pid is supported only on Windows')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        # The caller must still verify that the preparation manifest exists.
        return
    try:
        while True:
            result = kernel.WaitForSingleObject(handle, 1000)
            if result == 0:
                return
            if result != 258:
                raise OSError('failed to wait for preparation process')
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--aligned', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'C/configs/q2_optimization.json')
    parser.add_argument('--prepared-pid', type=int)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    lock = root / 'pipeline.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    state = dict(pid=os.getpid(), python=sys.executable, output=str(root), completed=[])

    def update(status, stage, **extra):
        state.update(status=status, stage=stage,
                     updated=datetime.datetime.now().astimezone().isoformat(), **extra)
        temp = root / 'pipeline_status.tmp'
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(root / 'pipeline_status.json')

    def run(stage, arguments):
        command = [sys.executable, '-u', *map(str, arguments)]
        update('running', stage, command=command)
        with (logs / f'{stage}.log').open('a', encoding='utf-8') as log:
            log.write('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n')
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} exited with {result.returncode}; see logs/{stage}.log')
        state['completed'].append(stage)

    try:
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        os.environ['PYTHONUNBUFFERED'] = '1'
        update('running', 'waiting_for_prepare')
        if args.prepared_pid:
            wait_for_prepare(args.prepared_pid)
        if not (root / 'prepared.json').exists():
            raise RuntimeError('preparation did not complete; prepared.json is missing')
        base = [ROOT / 'C/scripts/run_q2_optimization.py', '--output', root,
                '--config', args.config.resolve(), '--device', 'cuda']
        for stage in ('preflight', 'pilot', 'hpo', 'features', 'final', 'evaluate'):
            # hpo itself rejects unreliable short-budget screening; no silent fallback.
            run(stage, [*base, '--phase', stage])
        reporter = ROOT / 'C/scripts/report_q2_optimization.py'
        run('report_validation', [reporter, '--output', root, '--split', 'validation'])
        run('test', [*base, '--phase', 'test', '--aligned', args.aligned.resolve()])
        run('report_test', [reporter, '--output', root, '--split', 'test'])
        run('attachment3', [ROOT / 'C/scripts/infer_q2_corrected.py', '--data-root', ROOT / 'data',
                           '--package', root / 'inference', '--output', root / 'attachment3_predictions.csv',
                           '--device', 'cuda'])
        update('complete', 'complete')
    except Exception as error:
        update('failed', state.get('stage'), error=str(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
