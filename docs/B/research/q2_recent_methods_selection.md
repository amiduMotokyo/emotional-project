# 第二题近期方法甄选与赛题适配

状态：文献筛选和三种方法实验已完成；实验结果见 [近期方法对比](../experiments/q2_recent_methods.md)。负责人：B。

## 题目约束与当前基线

赛题要求对独立视频片段的文本、音频、视觉局部连续时段缺失进行三分类极性和连续强度预测，分析缺失模态、位置、时长。附件2只能用 train 学参数、valid 选模型，附件3为无标签最终推理。当前项目使用 aligned_50.pkl，经冻结 MiniLM 得到每位置384维文本表示；音频74维、视觉35维。训练3395条、验证728条，GPU为 RTX 2080 Ti 11GB。现有三种子门控基线及老论文架构对比见 [已有实验](../experiments/q2_literature_comparison.md)。

这意味着论文使用全模态缺失、对话图、多情绪类别、其他原始特征或 BERT 端到端训练得到的公开数值均不能直接与本题分数比较。新方法须使用相同缓存、双任务损失、连续缺失策略、验证选模和27条件评估。适配实现不能称为论文的精确复现。

## 候选与取舍

| 方法 | 论文机制 | 与题目关系 | 工程状态与取舍 |
| --- | --- | --- | --- |
| [DPDF-LQ，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.571/) | 局部路径与全局动态查询融合 | 对齐序列和片段级预测匹配；本身无专门缺失建模 | [作者代码](https://github.com/ZhouMiaoGX/DPDF-LQ)已公开；适合作为新融合对照。采用冻结特征、掩码注意力、双任务头的轻量适配。 |
| [EBMC，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html) | 模态增强、专家与实例级可信度调权 | 目标是弱模态和缺失鲁棒性 | [作者代码](https://github.com/kangverse/EBMC)有序列缺失率实验，但主实验为整模态缺失；采用共享/特有表示、跨模态补充和可靠性专家的轻量适配。 |
| [CMAD，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html) | 完整输入教师向缺失输入学生转移特征与相关性 | 完整训练样本可提供教师目标，适合比较蒸馏是否提高局部缺失稳定性 | [作者代码](https://github.com/YetZzzzzz/CMAD)已公开；适配版使用现有干净门控权重作教师，同构学生并蒸馏预测、特征与样本间相似性。 |
| [RECAP，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39349) | 两阶段时序恢复、信息分解与融合 | 对局部缺失有吸引力 | [作者代码](https://github.com/Taylor-HHT/RECAP-MSA)采用非对齐特征、可微调BERT和24GB GPU；本项目只有11GB且当前可比缓存是冻结对齐特征。列为拓展实验，完整原版无法直接进入同一实验协议。 |
| [T²DR，ACL Findings 2025](https://aclanthology.org/2025.findings-acl.452/) | 模态内部细粒度缺失注意力与整模态补全 | 概念非常贴合 | 论文包含重新编码原始模态和其他数据实验；[作者仓库](https://github.com/LH019/T2DR)当前仅写代码待发布，暂不作为复现基线。 |
| [EASE，ACL Findings 2026](https://aclanthology.org/2026.findings-acl.260/) | 概率补全和跨视角预测一致性 | 可借鉴一致性损失 | 完整框架工程量较大，待验证前三项后再决定是否扩展。 |
| [FUSE-Net，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Factorize_Reconstruct_Enhance_A_Unified_Framework_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf) | 共享、特有、噪声子空间与动态融合 | 主要针对融合与噪声，非连续局部缺失 | 可作为论文相关工作，短期实验优先级低于前三项。 |
| [HyperEF，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_Beyond_Missing_Modalities_Hypergraph_Conditioned_Diffusion_for_Uncertainty-Aware_Multimodal_Emotion_CVPR_2026_paper.html) | 对话超图与扩散恢复 | 依赖跨话语上下文 | 本题以独立片段为单位，不能直接移植对话图。 |

题目文档还列有 [P-RMF，ACL 2025](https://aclanthology.org/2025.acl-long.1075/) 的不确定性代理融合、[CmIR，ACL 2026](https://aclanthology.org/2026.acl-long.2119/) 的因果不变表示、[EMOE，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html) 的模态专家、[CaReFlow，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Mai_CaReFlow_Cyclic_Adaptive_Rectified_Flow_for_Multimodal_Fusion_CVPR_2026_paper.html) 的模态分布映射。其机制与本题有关，但当前实验优先验证有明确可比对照和可实现输入路径的前三项。题目文献中的 [Locate and Explain，ACL 2026](https://aclanthology.org/2026.acl-long.2012/) 处理对话情绪原因，更适合问题3相关工作，不能把其原因话语直接当作本题视频关键帧。

## 最小验证与失败条件

先用 DPDF-LQ 适配版验证“局部与全局融合”是否改善完整输入；在同样缺失增强下比较 DPDF-LQ、EBMC、CMAD。每种设置使用三个种子，统一 Accuracy、Macro-F1、MAE、Pearson 和27种缺失条件；按原项目的验证集选择分数选权重。附件3不参与训练和选择。已完成的结果显示三种近期方法均未超过当前门控基线的 Macro-F1；CMAD适配版的 Pearson 和 MAE 较有优势，但没有全面胜出。详见实验记录。

若新方法的三种子均值没有超过现有门控基线，或仅完整输入提高而缺失下下降，应如实报告其适用范围。若附加机制使训练明显过拟合或对文本缺失无改善，不把结构复杂度本身当成效果证据。所有论文机制归属原作者；本项目只声称赛题输入与缺失方案下的适配、比较和观察。
