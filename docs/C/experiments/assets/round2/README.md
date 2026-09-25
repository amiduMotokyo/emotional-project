# 第二轮实验的可携带汇总证据

这些文件摘自`C/outputs/q2_round2_ensemble_v1/`，不含原始特征或逐样本标签，便于在未共享大体积outputs时核对论文表格。

- `crossfit_summary.json`：两库R0—R7折外指标、搜索种子结果及请求数。
- `bootstrap.json`：2000次视频簇配对bootstrap区间。
- `lock.json`：预设规则的逐项判定及锁定方法。
- `locked_grid_summary.json`：valid/test各64条件的平均S与clean指标、原始网格文件哈希。
- `audit.json`：真实产物、预算、分组和预测重建审计结果。
- `timings.json`、`environment_versions.json`：成功主流程计时和环境版本。

原始预测、搜索轨迹、模型参数及完整逐条件表仍在outputs。`lock.json`、`audit.json`中的绝对路径或哈希对应原运行环境，不意味着这些权重已打入正式提交包。
