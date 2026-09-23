# 项目协作入口

本文件适用于整个项目，用户明确指令优先。详细工作流程已整理为 `$emotional-project-collaboration` Skill。

## 使用Skill

本项目的开发、需求修改、创新调研、实验、论文及跨模块集成任务，应读取并使用：

[emotional-project-collaboration / SKILL.md](.agents/skills/emotional-project-collaboration/SKILL.md)

Skill包含模块路由、文件归档、数据边界、接口交接、文档模板与完成检查。按任务读取其中链接的参考文件，不必每次加载全部参考资料。

此Skill随项目保存在 `.agents/skills/`，组员获取完整项目后，在项目根目录打开Codex即可发现，无需安装到个人技能目录。共享或提交代码时包含整个 `.agents/` 目录。若技能列表尚未刷新，重启Codex；也可通过上面的相对路径直接读取。若暂不可用，遵循以下核心规则和项目内的[协作约定](docs/项目结构与协作约定.md)。

## 核心目录规则

- A：特征、对齐、证据定位；B：数据接口、融合、解释；C：缺失鲁棒训练、优化、评估。代码分别放在 `A/`、`B/`、`C/` 下的 `src/`、`configs/`、`scripts/`，产物放各自 `outputs/`。
- 各自产生的需求、创新调研、实验说明和论文放在 `docs/<负责人>/requirements/`、`research/`、`experiments/`、`paper/`；`docs/`根目录保留公共题目、总计划与约定。
- `data/`原始附件共享读取，不改写、搬迁或复制到个人目录；自生成特征写入个人outputs。
- 共享实现保留在所属负责人目录，通过接口复用；跨目录修改按任务必要范围执行，维护调用方兼容，不额外引入审批流程。
- 个人建议不能替代赛题原文；跨组已确定结论同步公共约定。不得编造实验、验证或实现状态。
- 论文合并稿放 `docs/A/paper/`，最终提交包放 `C/outputs/submission/`。
