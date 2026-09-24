# 缺失鲁棒任务背景：参考论文

检索与下载日期：2026-09-24。方向已调整：C当前重点为[智能优化算法、参数优化与特征选择](intelligent_optimization/README.md)，新文献保存在该子目录。以下三篇作为任务背景保留，自适应采样方案已暂停，尚未实现或验证。

这些论文分别提供一致性训练、细粒度缺失建模和难度自适应蒸馏的参考，不意味着它们已经验证本项目拟议的连续区间采样策略。

| 论文 | 本地文件 | 官方下载地址 | 阅读重点 |
|---|---|---|---|
| MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis（TACL 2023） | [PDF](MissModal_TACL_2023.pdf) | [ACL Anthology PDF](https://aclanthology.org/2023.tacl-1.94.pdf) | 完整与缺失表示的一致性约束，作为训练策略基线参考 |
| T²DR: A Two-Tier Deficiency-Resistant Framework for Incomplete Multimodal Learning（Findings of ACL 2025） | [PDF](T2DR_Findings_ACL_2025.pdf) | [ACL Anthology PDF](https://aclanthology.org/2025.findings-acl.452.pdf) | 模态内部细粒度缺失、模态间缺失，以及原文缺失设置与赛题连续区间缺失的区别 |
| CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities（ICCV 2025） | [PDF](CMAD_ICCV_2025.pdf) | [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf) | 模态难度、自适应加权与课程学习；重点区分原方法和拟议采样调度 |

建议阅读顺序：MissModal → T²DR → CMAD。设计自适应策略时重点回看 CMAD。

已完成的[全文阅读笔记](../2026-09-24_三篇缺失鲁棒论文阅读笔记.md)与[主方向实验设计](../2026-09-24_连续缺失难度自适应采样实验设计.md)见上级目录。注意：CMAD正文的MAR权重在预热结束时计算一次，并不是逐轮更新的情景采样器。

## 已暂停的历史方向

在附件2训练集上按缺失模态、连续区间位置和长度构造情景，依据训练损失统计调整采样概率，并保留随机探索。对照普通训练、均匀连续缺失训练和自适应连续缺失训练，使用相同评估掩码与指标。附件3仅用于最终推理，不参与策略选择。

需要进一步核查上述论文的方法与缺失协议，并通过消融与重复实验判断收益；不据此声称学术首次。
