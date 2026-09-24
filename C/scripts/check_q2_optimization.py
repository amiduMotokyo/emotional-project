"""Synthetic integration checks only: no contest-data performance claims."""
from __future__ import annotations
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from B.src.fusion import Fusion
from C.src.intelligent_search import parameter_search, feature_search
from C.src.q2_protocol import _encode_with_masks, prepare_views
from C.scripts.run_q2_corrected import train_one, loader_for_case, load_checkpoint
from C.scripts.run_q2_optimization import Runner, pilot, hpo, features, finalize, evaluate_grid
from C.scripts.report_q2_optimization import report as paper_report
from types import SimpleNamespace


def synthetic(n=24):
    rng = np.random.default_rng(123)
    mask = np.ones((n, 50), bool)
    mask[:, 8:] = False
    data = {key: rng.normal(size=(n, 50, width)).astype(np.float32)
            for key, width in [('text', 384), ('audio', 74), ('vision', 35)]}
    data.update(tmask=mask.copy(), amask=mask.copy(), vmask=mask.copy(),
                token_ids=np.where(mask, 200, 0).astype(np.int64), attention=mask.copy(),
                cls=np.arange(n) % 3, score=np.linspace(-2, 2, n, dtype=np.float32))
    return data


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def run(self, _, inputs):
        self.calls.append({k: v.copy() for k, v in inputs.items()})
        return [np.repeat(inputs['input_ids'][..., None], 384, axis=2).astype(np.float32) / 1000]


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_budget_and_shared_initialization(self):
        def score(p):
            return -(p['dropout'] - .3) ** 2
        de = parameter_search(score, 'de', population=6, generations=2)
        random = parameter_search(score, 'random', population=6, generations=2)
        self.assertEqual(len(de), 18)
        self.assertEqual(de[:6], random[:6])
        for row in de:
            self.assertTrue(.1 <= row['parameters']['dropout'] <= .5)

    def test_feature_quotas(self):
        relevance = np.linspace(0, 1, 109)
        redundancy = np.zeros((109, 109))
        for method in ('bpso', 'random', 'topk'):
            for q, counts in ((.5, (37, 18)), (.75, (56, 27))):
                r = feature_search(relevance, redundancy, method, q, population=4, updates=3)
                self.assertEqual((len(r['audio_indices']), len(r['vision_indices'])), counts)
                self.assertEqual(len(set(r['audio_indices'])), counts[0])
                self.assertEqual(r['evaluations'], 1 if method == 'topk' else 16)

    def test_mask_and_legacy_checkpoint_compatibility(self):
        default = Fusion()
        model = Fusion(audio_indices=[0, 1], vision_indices=[0]).eval()
        model.load_state_dict(default.state_dict(), strict=True)
        data = synthetic(2)
        args = [torch.from_numpy(data[k]) for k in ('text', 'audio', 'vision', 'tmask', 'amask', 'vmask')]
        with torch.no_grad():
            expected = model(*args)[0]
            args[1][:, :, 2:] += 1000
            args[2][:, :, 1:] -= 1000
            actual = model(*args)[0]
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_unk_precedes_encoding(self):
        data = synthetic(2)
        encoder = FakeEncoder()
        masks = np.stack([data[k] for k in ('tmask', 'amask', 'vmask')], axis=1)
        masks[:, 0, 2:4] = False
        _encode_with_masks(encoder, data, masks)
        self.assertEqual(len(encoder.calls), 2)
        for call in encoder.calls:
            self.assertEqual(call['input_ids'].shape, (1, 50))
            self.assertTrue((call['input_ids'][0, 2:4] == 100).all())
            self.assertTrue(call['attention_mask'][0, 2:4].all())
        self.assertTrue((data['token_ids'][:, 2:4] == 200).all())

    def test_exact_cpu_resume_and_saved_feature_mask(self):
        self.check_resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_exact_cuda_resume_and_saved_feature_mask(self):
        self.check_resume('cuda')

    def check_resume(self, device):
        data = synthetic()
        loader = loader_for_case(data, None)
        hp = dict(learning_rate=.001, weight_decay=.01, lambda_reg=.8)
        kwargs = dict(dropout=.3, audio_indices=[0, 1], vision_indices=[0, 2])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            view = root / 'views'
            prepare_views(data, FakeEncoder(), view, views=2, seed=123)
            common = ('robust_gate', 17, data, loader, [loader] * 3, view)
            whole = train_one(*common, root / 'whole', device, 2, 3,
                              hyperparameters=hp, model_kwargs=kwargs, fixed_budget=True)
            first = train_one(*common, root / 'split', device, 1, 3,
                              hyperparameters=hp, model_kwargs=kwargs, fixed_budget=True)
            second = train_one(*common, root / 'resumed', device, 2, 3,
                               hyperparameters=hp, model_kwargs=kwargs, fixed_budget=True,
                               resume=Path(first['last_checkpoint']))
            x = torch.load(whole['last_checkpoint'], weights_only=False)
            y = torch.load(second['last_checkpoint'], weights_only=False)
            for key in x['state_dict']:
                torch.testing.assert_close(x['state_dict'][key], y['state_dict'][key], rtol=0, atol=0)
            self.assertEqual(whole['terminal'], second['terminal'])
            restored = load_checkpoint(Path(second['checkpoint']), 'cpu')
            self.assertEqual(restored.audio_channels.sum().item(), 2)
            self.assertEqual(restored.vision_channels.sum().item(), 2)

    def test_pipeline_on_synthetic_data(self):
        data = synthetic(12)
        c = json.loads((ROOT / 'C/configs/q2_optimization.json').read_text())
        c.update(low_epochs=1, full_epochs=1, search_population=4, search_generations=1,
                 promote=1, feature_population=4, feature_updates=1, views=1, final_seeds=[101])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            prepared = root / 'prepared'
            views = prepared / 'views'
            prepare_views(data, FakeEncoder(), views, 1, 123)
            (prepared / 'valid_cases').mkdir()
            masks = np.stack([data[k] for k in ('tmask', 'amask', 'vmask')], axis=1)
            for mode in ('text', 'audio', 'vision'):
                np.savez_compressed(prepared / 'valid_cases' / f'{mode}_30_middle.npz',
                                    text=data['text'], masks=masks)
            # These are placeholders only for final copy mechanics, never used as encoders.
            (prepared / 'text_encoder_int8.onnx').write_bytes(b'synthetic-test-placeholder')
            np.savez(prepared / 'audio_vision_normalization.npz', dummy=np.ones(1))
            r = Runner(SimpleNamespace(output=root, device='cpu'), c, data, data)
            pilot(r)
            hpo(r)
            features(r)
            finalize(r)
            report = json.loads((root / 'final.json').read_text())
            self.assertEqual(set(report), {f'E{i}' for i in range(6)})
            self.assertTrue((root / 'inference/model.pt').exists())
            self.assertTrue((root / 'locked_selection.json').exists())
            with patch('C.scripts.run_q2_optimization.session_for', return_value=FakeEncoder()):
                evaluate_grid(r)
            grid = json.loads((root / 'validation_grid.json').read_text())
            self.assertEqual(len(grid['rows']), 6 * 64)
            exported = paper_report(root, 'validation')
            self.assertTrue((exported / 'E0_101_confusion.png').exists())
            self.assertTrue((exported / 'coherent_missing_mae.pdf').exists())
            self.assertTrue((exported / 'paired_delta_vs_E0.csv').exists())
            # Never silently publish partial or duplicated evaluations.
            grid['rows'].append(grid['rows'][0])
            (root / 'validation_grid.json').write_text(json.dumps(grid))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                paper_report(root, 'validation')


if __name__ == '__main__':
    unittest.main(verbosity=2)
