"""Preflight or completed-run audit for the third-round experiment."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from C.scripts.run_q2_round3 import *


def preflight(run):
    a, b = TrainData(run.fit, .4), TrainData(run.fit, .4, True)
    for i in range(32):
        x, y = a[i], b[i]
        assert all(torch.equal(x[j], y[j]) for j in (1, 2, 3, 4, 5, 7, 8))
        newly_missing = run.fit['tmask'][i] & ~x[3].numpy()
        assert np.all(y[0].numpy()[newly_missing] == 100)
    seed_all(1122)
    model = TextFusion(True).to(DEVICE).train()
    names = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert names and all(n.startswith(('encoder.layer.4.', 'encoder.layer.5.')) for n in names)
    assert not model.encoder.embeddings.training and not model.encoder.encoder.layer[0].training
    assert model.encoder.encoder.layer[5].training
    batch = next(iter(DataLoader(b, batch_size=4)))
    logits, reg, cls, score = forward(model, batch)
    loss = torch.nn.functional.cross_entropy(logits, cls) + .8*torch.nn.functional.smooth_l1_loss(reg, score)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters() if p.requires_grad)
    return dict(passed=True, trainable_encoder_names=names, loss=float(loss.detach()))


def audit(run):
    adapter = read(OUT/'serialization_adapter.json')
    assert digest(adapter['source']) == adapter['sha256']
    assert digest(OUT/'manifest.json') == adapter['original_manifest']
    states = list((OUT/'state').glob('*.json'))
    for path in states:
        assert run.done(path.stem)
    split = run.split
    groups = np.asarray(split['source_metadata']['train']['group_ids'])
    assert not set(groups[split['fit_indices']]) & set(groups[split['stop_indices']])
    vg = np.asarray(split['source_metadata']['valid']['group_ids'])
    folds = np.asarray(split['valid_fold'])
    for f in range(3):
        assert not set(vg[folds == f]) & set(vg[folds != f])
    model_results = list((OUT/'models').glob('*/result.json'))
    assert len(model_results) == 21
    epochs = 0
    for path in model_results:
        r = read(path)
        history = read(path.parent/'history.json')
        assert len(history) == run.cfg['epochs']
        selected = max(history, key=lambda x: (x['clean']['accuracy'], x['clean']['macro_f1'], -x['clean']['mae']))
        assert r['best_epoch'] == selected['epoch']
        epochs += len(history)
        z = load_npz(path.parent/'stop.npz')
        assert np.array_equal(z['sample_id'], run.stop['sample_id'])
        actual = metrics(z['prob'], z['reg'], run.stop['cls'], run.stop['score'])
        assert actual == r['stop']
    # Independently verify frozen encoder parameters are byte-identical to pretrained weights.
    from safetensors.torch import load_file
    original = load_file(str(MODEL/'model.safetensors'))
    encoder_checked, tuned_changed = 0, 0
    for path in model_results:
        info = read(path)
        if info['kind'] == 'int8':
            continue
        state = torch.load(path.parent/'best.pt', map_location='cpu', weights_only=False)['state_dict']
        for name, value in original.items():
            if 'encoder.'+name not in state:
                continue
            trained = state['encoder.'+name]
            can_change = info['kind'] == 'tuned' and name.startswith(('encoder.layer.4.', 'encoder.layer.5.'))
            if can_change:
                tuned_changed += int(not torch.equal(value, trained))
            else:
                assert torch.equal(value, trained), name
                encoder_checked += 1
    assert encoder_checked > 0 and tuned_changed > 0
    sel = read(OUT/'selection.json')
    for label, record in sel['candidates'].items():
        cache = [load_npz(OUT/'valid_predictions'/f'{n}.npz') for n in record['models']]
        p, r = [np.mean([z[k] for z in cache], 0) for k in ('prob', 'reg')]
        if record['calibrated']:
            for f, bias in enumerate(record['fold_bias']):
                train, held = folds != f, folds == f
                # Fit only complementary groups, then compare stored held-out predictions.
                expected = fit_bias(p[:, train], r[:, train], run.valid['cls'][train], run.valid['score'][train], run.cfg)
                assert expected == bias
            clean_p = p.copy()
            for f, bias in enumerate(record['fold_bias']):
                p[:, folds == f] = apply_bias(clean_p[:, folds == f], bias)
        z = load_npz(OUT/'oof'/f'{label}.npz')
        assert np.allclose(p, z['prob'], atol=1e-7)
        assert metrics(z['prob'], r, run.valid['cls'], run.valid['score']) == record['metrics']
    summary = read(OUT/'test_summary.json')
    assert len(summary['scenarios']) == 64
    for name, reports in summary['scenarios'].items():
        z = load_npz(OUT/'test'/f'{name}.npz')
        for ref in ('baseline', 'selected'):
            assert metrics(z[ref+'_prob'][None], z[ref+'_reg'][None], z['cls'], z['score'])['scenarios'][0] == reports[ref]
    lock = read(OUT/'lock.json')
    assert lock['selection_hash'] == digest(OUT/'selection.json')
    assert summary['lock_hash'] == digest(OUT/'lock.json')
    return dict(passed=True, states=len(states), models=len(model_results), epochs=epochs,
                unchanged_encoder_tensors=encoder_checked, changed_top_layer_tensors=tuned_changed,
                validation_candidates=len(sel['candidates']), test_scenarios=64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    run = Run()
    result = audit(run) if args.audit else preflight(run)
    write(OUT/('audit.json' if args.audit else 'preflight.json'), result)
    print(result)
