# C：智能优化算法、参数优化与特征选择文献

检索日期：2026-09-24。当前方向已由用户明确调整为智能优化算法驱动的参数优化与特征选择。连续局部缺失是评价场景，自适应采样不再作为主方法。

截至2026-09-24，已通读目录内六篇PDF，见[阅读笔记](../../2026-09-24_智能优化文献阅读笔记.md)与[实验方案](../../2026-09-24_智能优化参数与特征选择实验设计.md)。未复现论文代码，未验证本赛题效果。上轮5份PDF下载成功、BOPSO请求失败；本次发现目录已有BOPSO本地文件并完成解析，保留原文件名与失败请求历史。

## 优先阅读与下载

| 优先级 | 论文、作者与出处 | 本地文件与来源 | 对应工作 |
|---|---|---|---|
| 1 | **DEHB: Evolutionary Hyberband for Scalable, Robust and Efficient Hyperparameter Optimization**；Noor Awad, Neeratyoy Mallik, Frank Hutter；IJCAI 2021，2147–2153 | [本地PDF，7页](DEHB_IJCAI_2021.pdf) · [官方页面](https://www.ijcai.org/proceedings/2021/296) · [官方PDF](https://www.ijcai.org/proceedings/2021/0296.pdf) | 差分进化与分预算候选评估相结合，参数优化主参考 |
| 2 | **Particle Swarm Optimization for Feature Selection in Classification: A Multi-Objective Approach**；Bing Xue, Mengjie Zhang, Will N. Browne；IEEE Transactions on Cybernetics 43(6)，1656–1671，2013 | [本地作者版，16页](MOPSO_Feature_Selection_2013_author.pdf) · [DOI](https://doi.org/10.1109/TSMCB.2012.2227469) · [作者PDF](https://homepages.ecs.vuw.ac.nz/~xuebing/Papers/SMC-B.pdf) | 多目标PSO搜索特征子集，平衡分类表现与特征数量 |
| 3 | **A Multi-Objective Particle Swarm Optimisation for Filter Based Feature Selection in Classification Problems**；Bing Xue, Liam Cervante, Lin Shang, Will N. Browne, Mengjie Zhang；Connection Science 24(2–3)，91–116，2012 | [本地作者手稿，27页](Filter_MOPSO_2012_author_manuscript.pdf) · [DOI](https://doi.org/10.1080/09540091.2012.737765) · [作者PDF](https://homepages.ecs.vuw.ac.nz/~xuebing/Papers/ConnectionScience.pdf) | 用互信息/熵等过滤式评价降低特征子集搜索成本 |
| 4 | **Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization**；Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, Ameet Talwalkar；JMLR 18(185)，1–52，2018 | [本地PDF，52页](Hyperband_JMLR_2018.pdf) · [期刊页面](https://www.jmlr.org/beta/papers/v18/16-558.html) · [官方PDF](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf) | 早停与资源分配基础，控制三天内的搜索成本 |
| 5 | **BOHB: Robust and Efficient Hyperparameter Optimization at Scale**；Stefan Falkner, Aaron Klein, Frank Hutter；ICML 2018，PMLR 80，1437–1446 | [本地PDF，10页](BOHB_ICML_2018.pdf) · [官方页面](https://proceedings.mlr.press/v80/falkner18a.html) · [官方PDF](https://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) | 贝叶斯优化与预算分配结合，可作非进化搜索对照；不是粒子群算法 |
| 6 | **Sentiment classification via improved feature selection using Boolean operator-based particle swarm optimization**；Harish Dutt Sharma等；Scientific Reports 15，38923，2025 | [本地PDF，19页](s41598-025-22894-3.pdf) · [出版社全文](https://www.nature.com/articles/s41598-025-22894-3) · [PDF下载入口](https://www.nature.com/articles/s41598-025-22894-3.pdf) | 文本情感分类应用参考；复现疑点见阅读笔记 |

### 版本说明

- DEHB官方页面和下载PDF标题使用“Hyberband”拼写；检索时也能见到“Hyperband”。两者对应同一项工作，DOI为10.24963/ijcai.2021/296。
- 2013 PSO文件是作者主页链接的提前出版版本，首页提示最终卷期页码尚未编排；引用使用正式期刊元数据。
- 2012过滤式PSO文件由作者[论文列表](https://homepages.ecs.vuw.ac.nz/Users/BingXue/PublicationsList)直接链接，但本地PDF带2015构建日期、占位卷期/DOI且无正式作者栏，因此标记为作者手稿，不冒充出版社定稿；引用年份与DOI按正式记录。
- Hyperband按下载的期刊PDF首页“Published 4/18”记录为2018，不混用2016预印本年份。
- BOPSO上轮出版社PDF接口返回非PDF响应；本次阅读时目录已有`s41598-025-22894-3.pdf`，已解析19页并补记校验值，不将其来源记为本次自动下载成功。
- [下载清单](download_manifest.json)记录来源、下载状态、页数、文件SHA-256及首页摘录。论文仅作研究参考，不进入最终提交包。

## 初步筛选结论

DEHB最直接对应“智能优化算法做超参数搜索”；2013多目标PSO和2012过滤式PSO分别提供包装式性能权衡与较低成本特征筛选的参考。Hyperband/BOHB帮助设定有限预算及合理对照。2025 BOPSO研究的是高维文本情感分类，不是CMU-MOSEI三模态局部缺失任务，其指标不能直接移用。

前五篇主要是通用优化方法，不声称它们已解决本赛题。把优化算法放入本项目后，适应度应包含固定缺失面板上的分类/回归表现，特征选择还需考虑保留维数与各模态覆盖。此为项目适配设想，并非这些论文已经给出的结论。

阅读全文后的首版方案采用固定预算DE与过滤式BPSO预选，具体搜索边界、通道粒度和预算见[实验设计](../../2026-09-24_智能优化参数与特征选择实验设计.md)。代码已接入B修订协议并通过工程检查，真实搜索尚未运行，须先pilot计时。这是基于文献的项目改造，不是完整DEHB/MOPSO/BOPSO复现。

## 接入项目时的边界

1. C实现外层候选生成、预算分配、适应度评估、参数优化和特征选择，调用B唯一维护的模型进行训练/验证。梯度训练仍是评价候选的内部步骤，不要求C重新设计融合网络。
2. 先区分连续/离散超参数和二进制特征掩码。特征通道选择不等于时间片段遮挡，也不等于问题1重新提取特征。
3. 768维文本嵌入不是768个独立词义；音频74维也不能臆造物理含义。分组选择若采用统计聚类，只能从train拟合分组规则。
4. 过滤式筛选只在train上计算统计量；包装式搜索以valid适应度选候选。test和附件3/4不得进入搜索或预算分配决策。反复使用valid会产生选择偏差，控制试验数并保留独立test。
5. 比较默认参数、等预算随机搜索和选定智能优化算法；特征选择比较全特征、等数量随机子集和优化子集。最终候选需要重新训练/确认，不能把冻结模型上直接置零的敏感度试验当成特征选择全部结论。
6. 先完成参数优化，再评估特征选择的增量；不立即同时搜索全部特征位和大量模型参数。三天预算下不同时完整复现六篇。

旧的三篇缺失鲁棒论文继续作为任务背景，见[原文献目录](../README.md)；旧自适应采样实验设计已暂停。
