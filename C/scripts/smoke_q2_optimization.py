"""Small real-input smoke check; its metrics are NOT experimental results."""
from __future__ import annotations
import gc
import json
import pickle
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from B.src.data import assemble_sample, encode_text, fit_audio_vision_scale, apply_audio_vision_scale
from C.src.q2_protocol import prepare_views, prepare_case, SELECT_CASES, case_name
from C.scripts.run_q2_corrected import train_one, loader_for_case
from C.scripts.run_q2_optimization import session_for, digest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aligned', type=Path, required=True)
    parser.add_argument('--encoder', type=Path)
    parser.add_argument('--legacy-package', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    encoder = args.encoder
    if encoder is None:
        if args.legacy_package is None:
            raise ValueError('provide --encoder or --legacy-package')
        encoder = args.output / 'text_encoder_int8.onnx'
        with zipfile.ZipFile(args.legacy_package) as archive:
            members = [n for n in archive.namelist() if n.endswith('/text_encoder_int8.onnx')]
            if len(members) != 1:
                raise ValueError('expected a single encoder artifact')
            encoder.write_bytes(archive.read(members[0]))
    torch.set_num_threads(1)
    with args.aligned.open('rb') as handle:
        source = pickle.load(handle)
    small = {}
    for split, n in (('train', 8), ('valid', 6)):
        small[split] = {k: np.asarray(source[split][k][:n]).copy() for k in
                        ('text_bert', 'audio', 'vision', 'classification_labels', 'regression_labels')}
    del source
    gc.collect()
    session = session_for(encoder)
    data = {}
    for split, row in small.items():
        text = encode_text(session, row['text_bert'], 'cpu', batch_size=1)
        data[split] = assemble_sample(row['text_bert'], text, row['audio'], row['vision'],
                                      row['classification_labels'], row['regression_labels'])
    scale = fit_audio_vision_scale(data['train'])
    for row in data.values():
        apply_audio_vision_scale(row, scale)
    prepare_views(data['train'], session, args.output / 'views', 1, 20260924)
    cases = [prepare_case(data['valid'], session, args.output / 'valid_cases', *case)
             for case in SELECT_CASES]
    clean = loader_for_case(data['valid'], None)
    loaders = [loader_for_case(data['valid'], p) for p in cases]
    run = train_one('robust_gate', 17, data['train'], clean, loaders,
                    args.output / 'views', args.output, args.device, 1, 2,
                    model_kwargs={'dropout': .25, 'audio_indices': list(range(37)),
                                  'vision_indices': list(range(18))}, fixed_budget=True)
    report = dict(status='passed', purpose='engineering smoke only; not performance evidence',
                  train_rows=8, valid_rows=6, epochs=1, encoder_sha256=digest(encoder),
                  torch=torch.__version__, checkpoint=run['checkpoint'],
                  device=args.device, peak_cuda_bytes=run['history'][0]['peak_cuda_bytes'],
                  seconds=run['history'][0]['seconds'])
    (args.output / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
