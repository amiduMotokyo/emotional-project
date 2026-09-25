# 第四轮结果归档

来自 `C/outputs/q2_round4_structure_v1/` 的已审计产物，由 `C/scripts/report_q2_round4.py` 归档。`source_manifest.json`保存各源文件SHA-256。

- `manifest.json`：数据、缓存、共享源码、配置、环境。
- `implementation.json`：第四轮模型、编排、检查脚本与训练前方案的哈希。
- `preflight.json`：缺失边界、概率分解、有限梯度、帧序响应与参数匹配检查。
- `selection.json`：六组结构的三种子单模型验证均值、SD及退化约束。
- `lock.json`：测试前锁定的wide_control与三个种子，未按测试挑选种子。
- `test_summary.json`：64条件复评的逐种子和平均指标。
- `analysis.json`：逐类指标、结构对照差值、学习曲线摘要与视频配对bootstrap。
- `audit.json`：18模型、324 epoch、103个完成记录及64条件指标的重算审计。

所有均值先计算各独立模型指标再平均，不是概率集成。test历史已查看过，本轮结果是锁定后复评；统计区间条件于固定的三种子已训练模型，不含整个训练/选型流程的不确定性。完整权重、逐样本预测和训练历史留在outputs中，没有复制原始数据。

环境与运行入口见上两级实验方案。运行`run_q2_round4.py --stage all`可从完成记录恢复；再次生成详细分析需运行`report_q2_round4.py`。共享引擎初始化仍要求第三轮环境和本地预训练缓存的校验依赖，完整路径见manifest；全精度权重不参与本轮前向。
