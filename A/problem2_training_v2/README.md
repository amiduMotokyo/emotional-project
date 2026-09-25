# 问题二 v2：多模型集成与阈值校准

## 数据协议

严格使用附件2 `aligned_50.pkl` 的官方划分：

- `train` 3395条：训练、连续缺失增强；
- `valid` 728条：早停、模型选择、集成权重和类别偏置选择；
- `test` 727条：全部模型和阈值冻结后，仅做最终留出评估。

test没有参与训练、缺失增强、早停、集成权重或分类阈值选择。

## 模型与改进

v2没有继续只使用单个模型，而是保留同一组缺失感知三模态编码结构，并增加：

1. 五个随机种子模型；
2. 两个模型使用更大的128维隐层容量；
3. 对五个模型的验证集概率求平均；
4. 在valid上搜索类别偏置，使Accuracy最大化；
5. 最终强度继续使用“中性为0、正负号与预测极性一致”的一致性解码。

最终类别偏置为：

```text
Negative: +0.0250
Neutral:  -0.2250
Positive:  0.0000
```

## 最终指标

验证集：

| 指标 | 结果 |
| --- | ---: |
| Accuracy | **0.6511** |
| Macro-F1 | 0.6181 |
| MAE | 0.5609 |
| Pearson | 0.6789 |

最终留出test：

| 指标 | 结果 |
| --- | ---: |
| Accuracy | 0.6864 |
| Macro-F1 | 0.6174 |
| MAE | 0.5972 |
| Pearson | 0.7080 |

## 附件3结果

使用同一个五模型集成和同一类别偏置推理附件3的30条样本：

```text
Negative: 5
Neutral:  6
Positive: 19
```

平均预测强度为 `0.2192`，平均门控为文本 `0.7381`、音频 `0.1414`、视觉 `0.1205`。

## v2.1：十模型加权集成

在五个基础模型之外，又训练了五个不同容量或缺失增强强度的模型：

```text
seed47: hidden_dim=128, missing_probability=0.6
seed48: hidden_dim=128, missing_probability=1.0
seed49: hidden_dim=96,  missing_probability=0.5
seed50: hidden_dim=160, missing_probability=0.8
seed51: hidden_dim=128, missing_probability=0.7
```

然后在valid上联合搜索十个模型的集成权重和类别偏置：

```text
weights = softmax(raw_weights)
p_ensemble = sum(weights_m * p_m)
p_final = softmax(log(p_ensemble) + bias)
```

最终valid结果：

| 指标 | 结果 |
| --- | ---: |
| Accuracy | **0.6635** |
| Macro-F1 | 0.6481 |
| MAE | 0.5450 |
| Pearson | 0.6851 |

对应test结果：

| 指标 | 结果 |
| --- | ---: |
| Accuracy | 0.6644 |
| Macro-F1 | 0.6282 |
| MAE | 0.6159 |
| Pearson | 0.6856 |

十模型权重和偏置保存在 `reports/加权集成结果.json`，val/test逐样本预测保存在
`reports/加权集成验证集预测.csv` 和 `reports/加权集成测试集预测.csv`。

需要明确：这一版把valid Accuracy提高到了66%以上，但权重是在728条valid上
搜索的，存在valid过拟合风险。等权五模型版本的test Accuracy为0.6864，
十模型加权版本的test Accuracy为0.6644。两者代表不同的选择策略。

## 主要输出

- `ensemble_q2.py`：五模型集成、阈值校准、缺失网格和中文图表入口；
- `infer_attachment3_v2.py`：附件3五模型集成推理；
- `model_lib.py`：模态编码、缺失增强和单模型训练实现；
- `reports/集成结果.json`：validation、test和63种缺失条件结果；
- `reports/类别偏置.json`：valid选择的类别偏置；
- `reports/集成验证集预测.csv`：728条验证集预测；
- `reports/集成测试集预测.csv`：727条test预测；
- `reports/缺失网格_验证集.csv`：验证集63种缺失条件；
- `reports/缺失网格_测试集.csv`：test集63种缺失条件；
- `reports/附件3预测_五模型集成.csv`：附件3的30条预测；
- `reports/附件3预测汇总.json`：附件3预测汇总；
- `reports/单模型验证结果.csv`：五个单模型和集成前的验证结果；
- `figures/`：全部中文图表。

## 复现命令

```powershell
& 'D:\Anaconda\envs\spam_classifier\python.exe' `
  'C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\ensemble_q2.py'

& 'D:\Anaconda\envs\spam_classifier\python.exe' `
  'C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\infer_attachment3_v2.py'
```
