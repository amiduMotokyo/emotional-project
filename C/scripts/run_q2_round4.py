"""Round-four architecture comparison using the frozen shared round-three engine."""
import sys
import time
import argparse
import pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from C.scripts import run_q2_round3 as engine
from C.src.q2_structural import build
from C.scripts.run_q2_optimization import read,write,digest
import numpy as np
import torch

OUT=ROOT/'C/outputs/q2_round4_structure_v1'
CONFIG=ROOT/'C/configs/q2_round4_structure.json'
engine.OUT=OUT
engine.CFG=CONFIG


def average_metrics(rows):
    return dict(scenarios=[{k:float(np.mean([r['scenarios'][s][k] for r in rows]))
               for k in ('accuracy','macro_f1','mae','pearson')} for s in range(len(rows[0]['scenarios']))])


class Experiment(engine.Run):
    def __init__(self):
        super().__init__()
        code={str(p):digest(p) for p in [Path(__file__),ROOT/'C/src/q2_structural.py',ROOT/'C/scripts/check_q2_round4.py',
              ROOT/'docs/C/experiments/第四轮结构优化实验方案.md']}
        extra=dict(code=code,base_manifest=digest(OUT/'manifest.json'),
                   note='Reuse frozen engine in a separate process; OUT/CFG and training factory explicitly bound to round four. No old artifacts modified.')
        path=OUT/'implementation.json'
        if path.exists():assert read(path)==extra,'Round-four implementation changed'
        else:write(path,extra)
        self.run_hash=digest(path)

    def train(self):
        assert self.done('prepare')
        original=engine.Fusion
        try:
            for family in self.cfg['families']:
                engine.Fusion=lambda:build(family)
                for seed in self.cfg['seeds']:
                    self.train_one(f'{family}_{seed}',dict(self.cfg['recipe'],architecture=family),seed)
        finally:engine.Fusion=original

    def load(self,name):
        assert self.done(name)
        obj=torch.load(OUT/'models'/name/'best.pt',map_location='cpu',weights_only=False)
        model=build(obj['recipe']['architecture'])
        model.load_state_dict(obj['state_dict'])
        return model.to(engine.DEVICE).eval()

    def select(self):
        if self.done('select'):return
        start=time.perf_counter();families={};files=[]
        for family in self.cfg['families']:
            results=[]
            for seed in self.cfg['seeds']:
                name=f'{family}_{seed}';path=OUT/'valid_predictions'/f'{name}.npz'
                if not self.done('valid_'+name):
                    t=time.perf_counter();model=self.load(name)
                    p,r=engine.panel(model,self.valid,self.valid_cases)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,prob=p,reg=r,sample_id=self.valid['sample_id'])
                    self.finish('valid_'+name,[path],t);del model
                z=engine.load_npz(path)
                results.append(engine.metrics(z['prob'],z['reg'],self.valid['cls'],self.valid['score']))
                files.append(path)
            families[family]=dict(mean=average_metrics(results),seeds=results,
                accuracy_sd=float(np.std([engine.key(r)[0] for r in results],ddof=1)),
                parameters=read(OUT/'models'/f"{family}_{self.cfg['seeds'][0]}"/'result.json')['trainable'])
        baseline=families['baseline']['mean']
        for name,r in families.items():
            r['eligible']=bool(engine.guard(r['mean'],baseline,self.cfg))
            r['per_seed_accuracy_delta']=[engine.key(a)[0]-engine.key(b)[0] for a,b in zip(r['seeds'],families['baseline']['seeds'])]
        eligible=[n for n in families if families[n]['eligible']]
        selected=sorted(eligible,key=lambda n:(-engine.key(families[n]['mean'])[0],n!='baseline',families[n]['parameters'],n))[0]
        write(OUT/'selection.json',dict(families=families,selected=selected,
            rule='Mean of three independent single-model metrics, NOT ensemble probabilities. Internal validation.'))
        write(OUT/'lock.json',dict(selected=selected,seeds=self.cfg['seeds'],run_hash=self.run_hash,
            selection_hash=digest(OUT/'selection.json'),implementation_hash=digest(OUT/'implementation.json'),
            note='No calibration or ensemble. Evaluate baseline and selected family for all three seeds, never select seed on test.'))
        self.finish('select',[OUT/'selection.json',OUT/'lock.json',*files],start)
        print('LOCKED',selected,engine.key(families[selected]['mean'])[0],flush=True)

    def evaluate(self):
        assert self.done('select')
        if self.done('evaluate'):return
        start=time.perf_counter();locked=read(OUT/'lock.json')
        aligned=next((ROOT/'data').glob('*/aligned_50.pkl'))
        assert digest(aligned)==read(engine.OLD/'prepared.json')['signature']['source']['aligned_pickle']
        with aligned.open('rb') as f:original=pickle.load(f)['test']
        session=engine.session_for(engine.OLD/'prepared/text_encoder_int8.onnx')
        text=engine.encode_text(session,original['text_bert'],'cpu',batch_size=1)
        data=engine.assemble_sample(original['text_bert'],text,original['audio'],original['vision'],original['classification_labels'],original['regression_labels'])
        data['sample_id']=np.array([f'test:{i}' for i in range(len(data['cls']))])
        engine.apply_audio_vision_scale(data,self.scale)
        names=list(dict.fromkeys(['baseline',locked['selected']]))
        models={f'{n}_{s}':self.load(f'{n}_{s}') for n in names for s in self.cfg['seeds']}
        reports={};files=[];texts={}
        for scenario in [None]+[(m,r,p) for m in engine.MODES for r in engine.RATES for p in engine.POSITIONS]:
            name='clean' if scenario is None else engine.case_name(*scenario)[:-4]
            path=OUT/'test'/f'{name}.json';predpath=path.with_suffix('.npz')
            if not self.done('test_'+name):
                t=time.perf_counter();masks=engine.masks_for(data,scenario);key=masks[:,0].tobytes()
                if key not in texts:texts[key]=data['text'] if np.array_equal(masks[:,0],data['tmask']) else engine._encode_with_masks(session,data,masks)
                stats={};arrays={}
                for family in names:
                    rows=[]
                    for seed in self.cfg['seeds']:
                        mid=f'{family}_{seed}';p,r=engine.predict(models[mid],data,masks,texts[key])
                        rows.append(engine.metrics(p[None],r[None],data['cls'],data['score']))
                        arrays[mid+'_prob']=p;arrays[mid+'_reg']=r
                    stats[family]=dict(mean=average_metrics(rows)['scenarios'][0],seeds=[x['scenarios'][0] for x in rows],
                        accuracy_sd=float(np.std([engine.key(x)[0] for x in rows],ddof=1)))
                write(path,stats)
                np.savez_compressed(predpath,**arrays,cls=data['cls'],score=data['score'],sample_id=data['sample_id'])
                self.finish('test_'+name,[path,predpath],t)
            reports[name]=read(path);files.extend([path,predpath])
        write(OUT/'test_summary.json',dict(selected=locked['selected'],lock_hash=digest(OUT/'lock.json'),scenarios=reports,
              averages={n:{k:float(np.mean([r[n]['mean'][k] for r in reports.values()])) for k in ('accuracy','macro_f1','mae','pearson')} for n in names}))
        self.finish('evaluate',[OUT/'test_summary.json',*files],start)

    def report(self):
        assert self.done('evaluate')
        sel=read(OUT/'selection.json');test=read(OUT/'test_summary.json')
        lines=['# 第四轮结构优化实验结果','','状态：训练、选型、锁定及64条件复评完成。', '',
              f"最终选择：**{sel['selected']}**。指标为三个独立单模型的均值，未集成、未校准。",'',
              '| 结构 | 参数量 | valid Accuracy±样本SD | 干净宏F1 | 缺失平均宏F1 | 通过约束 |',
              '|---|---:|---:|---:|---:|---|']
        for n,r in sel['families'].items():
            s=r['mean']['scenarios']
            lines.append(f"| {n} | {r['parameters']} | {s[0]['accuracy']:.4f} ± {r['accuracy_sd']:.4f} | {s[0]['macro_f1']:.4f} | {np.mean([v['macro_f1'] for v in s[1:]]):.4f} | {r['eligible']} |")
        lines+=['','| 锁定后复评结构 | 干净 Accuracy±SD | 干净宏F1 | 64条件平均Accuracy | 64条件平均宏F1 | 平均MAE |',
                '|---|---:|---:|---:|---:|---:|']
        for n,a in test['averages'].items():
            r=test['scenarios']['clean'][n];s=r['mean']
            lines.append(f"| {n} | {s['accuracy']:.4f} ± {r['accuracy_sd']:.4f} | {s['macro_f1']:.4f} | {a['accuracy']:.4f} | {a['macro_f1']:.4f} | {a['mae']:.4f} |")
        seconds=sum(read(OUT/'state'/f'{n}_{s}.json')['seconds'] for n in self.cfg['families'] for s in self.cfg['seeds'])
        lines+=['',f'18个模型、324个epoch。训练及stop评估共{seconds/60:.2f}分钟；不含开发、选型和测试。', '',
                '配方及方法定义见[实验方案](第四轮结构优化实验方案.md)。valid用于选型，属于内部验证；test历史上已被查看，本轮仅为锁定后复评。',
                '退化约束仅覆盖开发面板，不能保证所有缺失条件。分解头只是概率参数化变化，不能凭结构名称断言中性识别会变好。',
                '主结论必须与参数匹配对照、种子波动一起判断；本轮未自动替换正式提交模型。审计结果见outputs中的audit.json。','']
        (ROOT/'docs/C/experiments/第四轮结构优化实验结果.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','prepare','train','select','evaluate','report'],default='all');a=p.parse_args()
    run=Experiment()
    for stage in (['prepare','train','select','evaluate','report'] if a.stage=='all' else [a.stage]):getattr(run,stage)()
