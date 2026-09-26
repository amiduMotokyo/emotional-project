# 问题二最终单模型

默认模型为32维A4，seed 43，第3轮。仅加载 checkpoints/width32_seed43_cal.pt，不进行模型集成。分类输出使用 softmax(log(p)+[0.375,0.275,0]) 的最大概率类别；回归强度保留原输出。

## 文件

- attachment3_predictions.csv：附件三30条样本的正式单模型输出。
- checkpoints/width32_seed43_cal.pt：所选模型参数。
- checkpoints/normalization.npz：仅由训练集统计的标准化参数。
- results/selection_frozen.json：固定模型与偏置。
- protocol.json：结构、训练与数据处理配置。
- src：数据处理、冻结编码器获取、模型、训练、偏置搜索和预测代码。
- requirements.txt：问题二所需主要依赖；requirements.server.lock.txt记录实验服务器环境。
- 实验报告.md：验证集、测试集及缺失场景评估。
- results/verification.json：从附件三原始文件重新编码后的预测一致性检查。

## 从赛题文件复现专项预测

使用Python 3.11建立环境：
python3 -m venv env
env/bin/python -m pip install -r requirements.txt

将MOSEI_DATA_ROOT设置为赛题数据总目录，例如：
export MOSEI_DATA_ROOT=/path/to/competition_data

数据总目录保留赛题附件目录结构。原始PKL不需要放进本提交包。脚本只读取对齐版本，排除路径含“未对齐”的专项文件。

获取冻结通用编码器：
env/bin/python src/fetch_pretrained.py

编码器为bert-base-uncased，固定revision为86b5e0934494bd15c9632b12f734a8a67f723594。冻结编码器约440 MB，不包含在本包中；第一次获取需要网络。已下载时可直接将相同版本放到models/bert-base-uncased。该包不是含所有依赖的完全离线包。

仅处理附件三并加载已训练参数推理：
env/bin/python src/prepare.py --inference-only
env/bin/python src/run_single.py

输出为attachment3_predictions.csv。无需旧项目缓存、其他候选模型或附件四。CPU也可运行；不同设备/库版本可能产生微小浮点差异。

## 重训所选配置

env/bin/python src/prepare.py
env/bin/python src/train_search.py
env/bin/python src/run_single.py --evaluate

完整数据准备使用附件二train/valid/test及附件三。只用train更新模型权重和统计标准化参数；valid选择轮次与偏置，test只用于固定模型后的评价。训练20轮，seed43，维度32，4头单层Transformer，Dropout0.2，AdamW学习率0.0008，batch128，分类权重0.5，一致性权重0.3。

上述重训针对已经选中的单模型配置，不重新进行先前10组配置的完整搜索。保存校准后最优检查点为width32_seed43_cal.pt；正式输出仍使用selection_frozen.json中的固定偏置。重训结果如因环境而改变，不应根据测试表现重新调偏置。

## 数据处理

统一使用aligned_50接口：文本50×768、音频50×74、视觉50×35。文本由text_bert经冻结BERT重编码；CLS、SEP、填充不参与池化，UNK标记为不可用。音频和视觉全零行按不可用处理。训练集有效观测计算均值、标准差，标准化裁剪到[-10,10]。

训练沿用六组固定局部缺失视图和完整/缺失双视图监督；文本缺失在BERT前替换UNK再重新编码，以防被遮挡词通过上下文泄漏。验证选模使用未额外制造缺失的完整输入；缺失类型、比例、位置的评估另行报告。

## 评价口径

模型是先前搜索中校准后验证Acc最高的单模型。验证分数包含模型与偏置选择效应，不是独立泛化成绩。原测试集曾用于此前实验的结果报告，本次不将其描述为全新未接触数据。分类偏置调整决策边界，不保证概率置信度已校准；分类类别与回归强度独立输出。
