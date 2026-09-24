# C智能优化与B修订协议对接及运行说明

日期：2026-09-24。当前状态：第一轮本机myenv实验已于22:56完成，包括搜索、六组三种子重训、验证／独立测试、图表和附件3预测。最终按验证规则选择E0，未取得全面性能提升；结果见[论文草稿](../paper/第一轮智能优化实验论文草稿.md)及[Word导出](../paper/第一轮智能优化实验论文草稿.docx)。原6项工程检查、CUDA精确续训及真实小样本GPU冒烟均通过。下文保留准备和启动过程的历史记录，不将冒烟指标作论文结果。

## 1. 已冻结的共同协议

复用[输入一致性修订实验](q2_input_consistent_experiments.md)的Fusion门控结构、384维MiniLM文本、74维音频、35维视觉。文本在编码前把新增缺失词元替换成[UNK]、保留attention，同一int8 ONNX逐条编码；不沿用旧文本向量。音视频归一化与截断仅用train拟合。准备缓存可直接读取原始aligned_50.pkl的train/valid，也可读取B已有未标准化的train.npz/valid.npz重新编码。

训练增强继承概率0.8、七种非空模态组合、比例10%/20%/30%/50%、8组固定视图与轮换规则。区间取三模态基础有效位置并集的原始跨度，不压缩内部空洞，也不等于每模态实际有效点比例。最终网格是七种组合×10%/30%/50%×首/中/尾共63条件，另列完整条件。

损失为平方根类别频率加权CE＋lambda×SmoothL1；AdamW、batch64、梯度裁剪1；学习率恒定。选模和搜索均最大化四条件（完整＋T/A/V各中段30%缺失）的平均`Macro-F1−0.15×MAE＋0.05×Pearson`，采用原始回归头。最终另报符号一致性解码的MAE/Pearson，不能混用两种输出的指标。

DE/BPSO采用[主设计](../research/2026-09-24_智能优化参数与特征选择实验设计.md)的搜索机制和保底约束。区别于队友35轮早停历史实验，新对照统一6轮筛选、18轮复核，关闭早停；需pilot判断是否合理。因此历史分数不直接填入E0。E0是默认参数的缺失增强门控模型，E1随机调参，E2 DE，E3随机子集搜索，E4单变量排序，E5 BPSO。

## 2. 实现与兼容范围

| 文件 | 对接内容 |
|---|---|
| [B/src/fusion.py](../../../B/src/fusion.py) | 可选dropout、audio_indices、vision_indices；默认仍为原模型，state_dict兼容旧权重 |
| [C/scripts/run_q2_corrected.py](../../../C/scripts/run_q2_corrected.py) | 原train_one新增可选超参数、模型参数、固定预算、恢复路径；保留旧默认流程 |
| [C/src/intelligent_search.py](../../../C/src/intelligent_search.py) | 固定预算DE、同预算随机、过滤统计、BPSO、随机子集与top-k |
| [实验入口](../../../C/scripts/run_q2_optimization.py) | prepare/preflight/pilot/hpo/features/final/evaluate/test分阶段运行 |
| [配置](../../../C/configs/q2_optimization.json) | 唯一可执行配置；准备前调整，准备后不静默修改 |
| [附件3入口](../../../C/scripts/infer_q2_corrected.py) | 加载checkpoint中的模型参数与通道掩码，旧checkpoint仍可读取 |

固定通道掩码在Fusion内部投影前应用，原始74/35维输入不变。掩码索引存入checkpoint的model_kwargs，恢复训练和推理自动使用相同索引；不修改原始缓存，不声称置零能减少密集计算量。

最优权重保存在`checkpoints/robust_gate_<seed>.pt`；累计训练终点保存在`*_last.pt`。续训使用后者，保存AdamW、Python/NumPy/Torch/CUDA随机状态、完整历史及最佳权重。CPU已验证分段续训与连续训练逐参数一致；CUDA确定性仍须在实际服务器验证。

## 3. 本地验证与依赖

本次在项目隔离环境`.venv`安装Python3.10对应的torch2.7.0+cpu、NumPy2.2.5、scikit-learn1.6.1、onnxruntime1.20.1，用于工程验证。队友服务器的CUDA环境可直接复用；本地这个环境尚不支持GPU训练，不据CPU冒烟耗时估算服务器工时。

```powershell
.venv/Scripts/python C/scripts/check_q2_optimization.py
```

通过项：DE/随机请求预算和初始候选一致；BPSO/随机/top-k满足逐模态配额；旧state_dict可严格加载；被删除通道的变化不影响输出；[UNK]替换在单条编码前发生；精确续训、掩码恢复；合成数据完整pilot→hpo→features→final流程。使用临时目录，无赛题test读取。

真实输入检查使用原始附件2的前8条train和6条valid，编码器从现有历史提交包中读取，仅测试兼容性；该编码器哈希为`b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21`。没有将历史模型分数当成新实验结果。报告在`C/outputs/optimization_smoke/smoke_report.json`，完整入口如下：

```powershell
.venv/Scripts/python C/scripts/smoke_q2_optimization.py --aligned "data/附件2-数据集特征文件/aligned_50.pkl" --legacy-package C/outputs/submission/question2_submission.zip --output C/outputs/optimization_smoke
```

服务器正式运行须核对其修订实验采用的编码器哈希。若不同，使用服务器实际编码器重新准备本轮缓存；本轮内部所有方法必须相同。

## 4. 正式运行顺序

先查看配置，确认分配给C的GPU时段。以下命令已验证入口解析；正式全量准备和训练尚未执行。所有命令从项目根目录运行，`--output`必须使用新的C/outputs子目录，不覆盖队友实验。

本地直接使用原始附件启动准备，无需先运行B的全量缓存脚本：

```powershell
.venv/Scripts/python C/scripts/run_q2_optimization.py --phase prepare --aligned "data/附件2-数据集特征文件/aligned_50.pkl" --encoder C/outputs/optimization_smoke/text_encoder_int8.onnx --output C/outputs/q2_optimization_v1 --device cpu
.venv/Scripts/python C/scripts/run_q2_optimization.py --phase preflight --output C/outputs/q2_optimization_v1 --device cpu
.venv/Scripts/python C/scripts/run_q2_optimization.py --phase pilot --output C/outputs/q2_optimization_v1 --device cpu
```

在已有CUDA环境的服务器中改用`python`及`--device cuda`。也可将prepare的`--aligned`替换为`--cache /实际路径/cache`，该目录必须包含B接口生成的未标准化train.npz和valid.npz，以及token_ids、attention等字段；重新编码以避免旧缓存口径混入。`--cache`与`--aligned`二选一，ONNX路径必须为实际存在的文件。

prepare生成干净特征、train拟合的标准化参数、8组输入损坏视图和三个选模条件；`prepared.json`记录源文件、编码器、配置、源码及生成文件哈希。preflight核对哈希、维度、样本数及CUDA是否可用。原始pickle会整体反序列化，但只使用train/valid生成准备产物，test不参与统计、选择或训练。准备中断留下半成品时须换新输出目录，脚本不自动删除或复用不完整视图。

**先检查pilot.json再启动搜索。** 四配置的短/长训练排名只是小样本诊断。`low_budget_screening_allowed=false`时，hpo拒绝短训筛选。需在新配置中设置low_epochs=full_epochs或重新选择预算，用新输出目录准备；脚本不会看结果后自动给某一种优化器加预算。若估算时间超过可用GPU时间70%，在正式搜索前统一缩减DE/随机的代数和最终重复次数。默认840轮是保守账面预算，缓存去重可节省实际训练；估时含每轮四条件验证，不含准备和最终网格。

pilot通过且预算可承担后，依次运行：

```powershell
.venv/Scripts/python C/scripts/run_q2_optimization.py --phase hpo --output C/outputs/q2_optimization_v1 --device cpu
.venv/Scripts/python C/scripts/run_q2_optimization.py --phase features --output C/outputs/q2_optimization_v1 --device cpu
.venv/Scripts/python C/scripts/run_q2_optimization.py --phase final --output C/outputs/q2_optimization_v1 --device cpu
.venv/Scripts/python C/scripts/run_q2_optimization.py --phase evaluate --output C/outputs/q2_optimization_v1 --device cpu
```

服务器同样将解释器和device改为其CUDA环境。若采用自定义配置，每个阶段都传相同的`--config`；配置改变会触发清单不匹配。同一实验目录只允许一个进程写入，不并行运行两个阶段。

## 5. 输出与选择规则

- `pilot.json`：四配置结果、6/18轮排名关联、预算估计。低/高预算关联不佳时不能继续默认筛选。
- `trials/<配置哈希>/spec.json、result.json、checkpoints/`：参数、预算、seed、时间、逐轮四场景指标、best/last权重。OOM和非有限损失记失败并占请求，不自动改变batch size；接口级错误直接停止以便修复。
- `hpo.json`：两个搜索器各24个请求、完整预算复核及回退状态。共享初始候选和重复候选可缓存，分别记各方法逻辑预算；实际消耗按唯一trial另计，不能把逻辑预算说成实测墙钟。
- `filter_statistics.npz、features.json`：仅train拟合的相关性/冗余、每方法每比例的子集与真实模型复核；过滤分数不是预测指标。
- `final.json`：E0—E5分别用三个固定种子重训的完整验证指标、均值与样本标准差；只有一个种子时std为null。seed=17用于搜索，最终三个seed独立于搜索seed。
- `locked_selection.json`：完整性能保底条件内，按三个种子的平均验证选择分数选组，再选该组验证分数最高的种子用于交付，与B的选择方式一致。不是根据test选模型。
- `validation_grid.json`：6组×3种子×64条件，各自保留raw/coherent指标。此表含参与选模的valid，不是独立测试成绩。
- `inference/`：选定model.pt、原始74/35维标准化参数和ONNX编码器，可调用现有附件3入口。

附件3推理在锁定选择之后运行：

```powershell
.venv/Scripts/python C/scripts/infer_q2_corrected.py --data-root data --package C/outputs/q2_optimization_v1/inference --output C/outputs/q2_optimization_v1/attachment3_predictions.csv --device cpu
```

不要用旧的分析/打包脚本自动推断本轮输出布局；这些脚本原本面向三架构修订实验。本轮图表使用下面的独立导出入口，最终提交压缩包仍需在正式结果完成后整理。

## 6. 独立test与当前限制

test阶段仅在final及选择锁定之后允许运行，可直接传原始附件2的`--aligned`，只评价其727条test。若准备阶段也用原始pickle，脚本要求源文件哈希一致。也支持`--test-cache`替代，但必须是未标准化的B批次schema，包含token_ids、attention、三个mask、audio、vision、cls、score及text；text会用本轮ONNX重编码。**不能将附件3的q2_*.npz当成test缓存**。

```powershell
.venv/Scripts/python C/scripts/run_q2_optimization.py --phase test --aligned "data/附件2-数据集特征文件/aligned_50.pkl" --output C/outputs/q2_optimization_v1 --device cpu
```

test使用train标准化、同一63条件和完整条件，raw/coherent分别报告；不更新锁定选择。生成test_grid.json后脚本阻止本目录继续优化或重复测试。中途中断会保留complete=false，需要人工检查完整性，不自动反复打开test用于开发。完整归档前核查所有请求/失败和预算、最终协议、编码器哈希及64条件覆盖。

本次不新增架构对比，不实现A1完整场景目标消融；因此不能声称已经证明缺失感知目标比完整目标更优。特征置零不代表实际加速，单次搜索也不能证明DE/BPSO在所有种子下稳定优于随机搜索。

## 7. 对照B成果的论文产物准备检查

2026-09-24追加检查：训练与搜索入口已具备；补齐了逐样本预测留存、完整性检查、论文表格和PNG/PDF图形导出。正式性能实验尚未运行，不能将导出流程的合成测试图用作结果。

| B已有的结果／论文需要 | 本轮对应输出 |
|---|---|
| 完整输入Accuracy、Macro-F1、MAE、Pearson | `summary.csv`的clean，按E0—E5分别报跨种子均值和样本标准差 |
| 最终交付单模型结果 | `selected_model.csv`，明确组别、seed、raw/coherent，不与跨种子均值混淆 |
| 各类F1、混淆矩阵 | `per_class_f1.csv`；每组代表seed的混淆矩阵CSV与PNG/PDF |
| 真实强度—预测强度散点图 | 每组代表seed各导出raw/coherent两版 |
| 缺失分类曲线、误差曲线 | T/A/V三面板，同图对比E0—E5，位置平均后计算种子间标准差 |
| 全部63缺失条件与最差条件 | `grid.csv`逐条件逐seed；`summary.csv`按模态组合、比例、位置及最差Macro-F1汇总 |
| 优化提升是否稳定 | `paired_delta_vs_E0.csv`，相同seed相减后汇总；三次重训不自动等于显著提升 |
| 智能优化过程证据 | DE/随机搜索候选请求—最优选择分数图；BPSO/随机/top-k过滤目标图，两种目标不混画 |
| 特征与计算预算 | `configurations.csv`含参数、通道索引和保留数；`unique_trial_costs.csv`与`requests.csv`区分实测trial和缓存请求 |
| 可追溯预测与图表 | `predictions/<split>/<group>_<seed>_clean.npz`保存样本ID、真实标签、分类概率、raw/coherent强度；`provenance.json`记录来源哈希 |
| 附件3专项结果 | 锁定模型后使用第5节入口生成30条CSV；无标签，不计算准确率 |

每组代表seed按该组**验证选择分数**确定，独立test图也使用相同规则，不按test成绩挑seed。63个条件先在每个seed内平均，再对seed求标准差，不把63个条件充当独立重复。表中差值为本方法减E0，MAE负差值表示改善。发生回退时可能有多个组实际配置相同，应结合hpo/features.json和配置表说明，不冒称算法带来改进。

完成evaluate之后运行（服务器换成其Python解释器）：

```powershell
.venv/Scripts/python C/scripts/report_q2_optimization.py --output C/outputs/q2_optimization_v1 --split validation
```

完成锁定的test阶段之后，单独导出独立测试结果：

```powershell
.venv/Scripts/python C/scripts/report_q2_optimization.py --output C/outputs/q2_optimization_v1 --split test
```

结果分别放在实验目录的`paper/validation/`和`paper/test/`。导出仅读取已有结果，不重新训练或打开原始test数据；拒绝未完成、缺条件、重复条件、各组seed不一致、预测样本身份/标签不一致及预测与汇总Accuracy/MAE不一致的记录。旧版网格若没有逐样本预测文件不能直接导图；本轮应从更新后的代码开始准备。图表仅添加了matplotlib依赖，见`requirements_optimization.txt`。论文定稿时将选定图及来源说明归档到`docs/C/paper/assets/`，不要只引用被Git忽略的outputs。

**与B历史结果的比较边界**：61.68%是B修订协议下选定单模型的验证准确率，其训练最多35轮、早停7轮。本轮正式对照为同协议、同预算、同种子的E0，不能把61.68%直接写成E0成绩，也不能将旧版图或旧文献变体分数混入同一公平对照表。若论文要声称直接超过B历史最终模型，仍需核对原始数据和编码器哈希，并说明训练预算及单模型／跨种子统计差异。

**开跑前尚待实机验证**：服务器编码器哈希、CUDA环境和GPU时间窗口；完整prepare/preflight；pilot的6/18轮排序与工时诊断。这些尚未完成，所以当前结论是“工程上具备进入全量准备和pilot的条件”，不是“正式结果已经验证”。六项合成检查已扩展到六组×64条件评估、逐样本留存、实际PNG/PDF导出及重复记录拒绝；真实小样本冒烟沿用第3节的已验证结果，不据此推断正式效果。

## 8. 本机myenv运行记录（取代服务器优先安排）

用户确认在本机运行，并指定现有`myenv`。解释器为`C:/Users/ken/miniconda3/envs/myenv/python.exe`，GPU实测为RTX 4070 Laptop GPU、8188 MiB。当前终端没有conda命令，因此执行时使用myenv解释器的绝对路径；未继续使用项目`.venv`，也没有修改系统默认Python。

实测环境：Python3.10.10、PyTorch2.7.0+cu128、CUDA12.8、NumPy1.26.4、scikit-learn1.6.1、ONNX Runtime1.20.1、matplotlib3.10.1。原typing_extensions4.7.1缺少TypeIs，导致PyTorch导入失败；本次升级为4.16.0后CUDA矩阵运算通过。pip还报告环境中旧spaCy等包的依赖冲突，这些包不参与本实验，未为此改动其他依赖。此处记录实际myenv环境，不宣称与第3节CPU环境或B服务器逐包一致，不直接用requirements文件覆盖现有环境。

- 六项原工程检查在myenv通过；额外CUDA连续2轮与1轮＋恢复1轮逐参数一致，通道掩码恢复通过。
- GPU真实小样本冒烟报告：`C/outputs/optimization_smoke_myenv_cuda/smoke_report.json`；峰值Torch分配显存20,124,160字节，仅8条训练样本，不能代替完整batch64显存实测。
- 正式输出目录：`C/outputs/q2_optimization_local_myenv_v1/`；环境快照为`environment.json`，阶段日志在`logs/`。
- 编码器沿用本地已核验哈希`b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21`，来自已有历史包；尚未获得B服务器编码器哈希作外部核对。本轮E0—E5统一使用该编码器与重新生成的输入级缺失缓存，因此内部对照口径一致；对B历史成绩的比较限制仍然有效。
- 数据准备和后续CPU计算使用OMP/MKL线程数4；ONNX单样本编码仍使用CPU，显卡在准备阶段空闲正常。没有改动样本数量、搜索预算、batch size或实验目标。

[本机串行入口](../../../C/scripts/run_local_q2_pipeline.py)在本次已经启动的prepare进程结束后检查prepared.json，依次执行preflight→pilot→hpo→features→final→evaluate→验证图表→锁定test→test图表→附件3推理。每个阶段使用同一myenv解释器，失败立即停止并记录原因；hpo沿用pilot可靠性限制，不自动绕过。后台运行隐藏窗口，`pipeline_status.json`记录PID、当前阶段、已完成阶段及错误，`pipeline.lock`防止重复启动。不要向同一个输出目录另起第二个训练进程。

查看本次进度：

```powershell
Get-Content -Encoding UTF8 C/outputs/q2_optimization_local_myenv_v1/pipeline_status.json
Get-Content -Encoding UTF8 C/outputs/q2_optimization_local_myenv_v1/logs/prepare.log -Tail 10
```

此记录只确认启动与已完成的工程验证。正式耗时、完整batch显存、pilot排序和优化提升必须以随后生成的真实日志为准；若pilot拒绝筛选，需检查日志后在新配置／目录调整共同预算，不能声称搜索已经完成。

全量准备于本机完成，preflight确认3395/728样本、384/74/35维度及文件哈希通过。第一组pilot已完成18轮；batch64、每轮四条件验证时，实测最慢轮次约3.143秒，Torch峰值分配显存36.113 MiB（不含全部驱动／上下文开销）。这说明当前小型融合模型在8GB卡上有显存余量；总预算仍以四组pilot完成后的pilot.json为准。

四组pilot现已全部完成：6/18轮终点选择分数的Spearman为1.0，前两名重合2/2，允许按原配置启动短轮次筛选；这只是四配置诊断，不是普遍有效性的证明。按最慢轮次估算840轮训练（含每轮四条件验证）约0.733小时，即44分钟，尚未计入全网格准备、评估和绘图，且这是整套账面训练预算而非精确剩余时间。串行入口已经进入hpo，后续状态请查实时JSON，暂不报告优化完成或性能提升。
