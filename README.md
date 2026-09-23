# 多模态情感预测项目

三人按模块协作，代码按负责人归档，工作文档按负责人归入 `docs/`。原始数据共享一份，公共接口保持一个维护来源。

## 目录结构

```text
项目根目录/
├── .agents/skills/           # 随项目共享、由Codex发现的项目Skill
├── AGENTS.md                 # 后续开发与AI协作遵循的目录规则
├── README.md                 # 项目导航
├── data/                     # 赛题原始附件，共享读取，保留现有路径
├── A/                        # 计算机视觉：特征、对齐、证据定位
│   ├── src/                  # 模块实现
│   ├── configs/              # 配置文件
│   ├── scripts/              # 运行入口
│   └── outputs/              # 自生成特征、日志、图表等运行产物
├── B/                        # 多模态：数据接口、融合基线、解释
│   └── src/ configs/ scripts/ outputs/
├── C/                        # 优化：缺失策略、鲁棒训练、统一评估
│   └── src/ configs/ scripts/ outputs/
└── docs/
    ├── 原有题目与三天计划      # 公共基准文档，保留原文件名和位置
    ├── 项目结构与协作约定.md  # 归档规则、接口归属、变更流程
    ├── A/                    # A的工作文档
    │   ├── requirements/     # 需求理解、修改建议、任务拆解
    │   ├── research/         # 文献、创新点、方案对比
    │   ├── experiments/      # 实验记录与结果分析
    │   └── paper/            # 论文草稿与排版材料
    ├── B/                    # 与docs/A相同的分类
    └── C/                    # 与docs/A相同的分类
```

## 负责人和共享模块

| 人员 | 实现目录 | 文档目录 | 模块 |
|---|---|---|---|
| A | [A](A/) | [docs/A](docs/A/) | M2特征核查、M3时间映射；M8统稿 |
| B | [B](B/) | [docs/B](docs/B/) | M1数据接口、M4融合基线、M6解释 |
| C | [C](C/) | [docs/C](docs/C/) | M5缺失鲁棒训练、M7评估；M8打包 |

共用不代表复制：C调用B的数据接口和融合基线，B调用A的证据定位，B/C使用C维护的评估逻辑。各负责人在自己的实现目录维护这些模块。

## 使用入口

- [项目协作Skill](.agents/skills/emotional-project-collaboration/SKILL.md)
- [三天模块分工与论文写作计划](docs/三天模块分工与论文写作计划.md)
- [项目结构与协作约定](docs/项目结构与协作约定.md)
- [A工作说明](A/README.md)、[B工作说明](B/README.md)、[C工作说明](C/README.md)

当前目录结构已建立，尚未接入训练与特征提取代码；问题1现有代码交接后放入A目录。推荐从项目根目录运行各组脚本，通过配置或根据脚本位置解析数据路径，避免依赖个人机器的绝对路径。

## 共享Skill

获取项目时保留 `.agents/` 目录，在项目根目录打开Codex，即可发现 `emotional-project-collaboration`。可用 `$emotional-project-collaboration` 显式调用，也可在匹配的项目任务中按需选用。若未显示，重启Codex后检查；无需复制到每个人的全局技能目录。

Skill元数据会被发现，完整规则在选用时加载，并非每次对话自动载入全部内容。此目录未被项目 `.gitignore` 忽略；使用Git共享时需将其一并提交。data被忽略，应单独分发。

加载位置依据：[OpenAI官方Skill文档](https://learn.chatgpt.com/docs/build-skills)。
