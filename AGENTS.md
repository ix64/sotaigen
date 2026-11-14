# 给代理的说明

这份说明对所有在本仓库工作的代理都适用，包括 Claude 和 Codex。这里只放入口和最重要的规则，细节在 `docs/`，不要在这里再写一份。

公共技能正文在 [skills/sotaigen](skills/sotaigen/SKILL.md)。`.agents/skills/sotaigen` 和 `.claude/skills/sotaigen` 都通过仓库内的相对软链接指向它；`CLAUDE.md` 只引用本文件。工具入口不另存一份规则或技能正文，个人配置不提交。

## 先读

先读 [docs/README.md](docs/README.md) 按任务选文档，再读 [docs/DECISIONS.md](docs/DECISIONS.md) 的待决定事项。不要默认通读全部文档。

- 环境和命令：[WORKFLOW](docs/WORKFLOW.md)；协作与隔离：[agents](docs/agents.md)；合并检查：[MERGING](docs/MERGING.md)。
- 改实现：读 [ARCHITECTURE](docs/ARCHITECTURE.md) 和相关 `parts/`、`systems/` 文档；改骨骼、权重或姿势变形前先读该部位的[动态解剖卡](docs/motion/README.md)，没有就先写。
- 改对外接口：先读[对外约定](docs/reference/README.md)，冻结部分的变动按兼容政策处理。

## 要问用户的

- `docs/DECISIONS.md` 里待决定的问题，以及新的设计取舍。
- 观感好不好。这是主要判断标准，指标通过不等于好看，要出图给用户看。
- `uv run sotaigen-dev compare` 报出的退化是否认可。
- 推送、创建远端、打标签、公开任何内容。

任务的先后可以自己按依赖和文件冲突安排，不用问。用户说暂停，就不再派发新任务。

## 不要做

- 放宽测试来换通过。要改测试，先说明它为什么错。`skip` 和 `xfail` 都算失败。
- 引入任何具体角色的参数、网格或肖像。本工具不依赖任何下游项目。
- 退回逐层填截面的做法。
- 在主仓库目录里做多步操作。用自己的 worktree 和分支。
