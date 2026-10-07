# 语料来源与修复记录

核对与修复日期：2026-10-07。两份公开 JSONL 共 6,515 条，已实际修正来源链接、定位和许可证字段，并同步检索索引与评估标注。块编号、标题、章节、正文及向量保持不变；这次是元数据修复。

## 已核实的来源快照与许可证

| 来源 | 块数 | 核对的上游提交 | 许可证及证据 |
| --- | ---: | --- | --- |
| Being09/game-design-wiki | 712 | `715c2e765402b664e0aad126e26c8fec068493a0` | [MIT，LICENSE](https://github.com/Being09/game-design-wiki/blob/715c2e765402b664e0aad126e26c8fec068493a0/LICENSE) |
| diedie23/Game-Knowledge-Base | 621 | `4171d38daa61f033015e69e4e477db625155e051` | 核对快照未发现许可证声明 |
| Thaelith/open-game-mechanics-dataset | 2,676 | `da3b1ce634613f3f5580205e9e6abdbd074d6cca` | [数据为 CC0-1.0，LICENSE](https://github.com/Thaelith/open-game-mechanics-dataset/blob/da3b1ce634613f3f5580205e9e6abdbd074d6cca/LICENSE) |
| lsc1414/Game_Num_Basics_And_Calc | 1,809 | `7a3b6c23a246ef81d10bde4e7bf555cddfa8274c` | [MIT，LICENSE](https://github.com/lsc1414/Game_Num_Basics_And_Calc/blob/7a3b6c23a246ef81d10bde4e7bf555cddfa8274c/LICENSE) |
| zsc/gamedev_at_home | 609 | `ad83e2b52e334d75ae317ddda042c098dda5212b` | 核对快照未发现许可证声明 |
| tigermkiiiddd/senior-game-designer | 88 | `89ee5915f34df9094f977626162a12c2e96cb85d` | [README 声明 MIT](https://github.com/tigermkiiiddd/senior-game-designer/blob/89ee5915f34df9094f977626162a12c2e96cb85d/README.md)，未发现独立 LICENSE 文件 |

机制数据原先误标为 MIT，已改为 CC0-1.0；上游区分数据许可证与工具脚本的 MIT 许可证。数值资料和高级设计资料的 TBD 已改为有原文证据的 MIT。合计 MIT 2,609 条、CC0-1.0 2,676 条、未声明 1,230 条。

未声明的记录使用 `license: null`，同时保存 `provenance.license.status: not_declared`。有声明的记录保存 SPDX 值、声明类型、固定版本证据 URL 和证据 SHA-256。未声明不是一种许可证，不能通过填字符串推定授权。

## 实际修复的字段

| 项目 | 修复结果 |
| --- | --- |
| 5,894 条相对路径、621 条站点首页 | 6,515 条均改为具体来源文件的 GitHub 固定版本链接，含来源范围 |
| 原来 5,182 条 0/0 行号 | 换为可核实的片段行或明确的 JSON 记录／文章范围 |
| 来源版本与文件指纹 | 每条保存仓库、40 位提交、文件路径及 SHA-256；文件内容与提交的 Git blob 校验一致 |
| 许可证错误与 TBD | 更正为 MIT／CC0-1.0；确实未声明的来源保存明确状态 |
| 评估标注和实际索引 | 41 个标注块的链接／范围同步修复；五个现有索引更新元数据，向量指纹不变 |

界面、工具证据与 Markdown 导出区分三种定位：

- **3,822 条 matched_text**：896 条逐字匹配，2,373 条忽略空白／BOM 后匹配，553 条另忽略符号、组合及格式标记后匹配。行号属于原始 Markdown 或 HTML，不是切分后文本。重复匹配记录出现次数，链接定位第一处。
- **2,676 条 json_record**：逐条重建既有结构化字段投影并验证正文一致，定位整条 JSON 记录；中文标签和列表整理不属于逐字摘录。
- **17 条 document**：确认来源文章，但既有 HTML 片段与快照正文仍有差异，只保存文章范围并标记 `excerpt_not_verified_against_snapshot`。文章范围不能解释成片段精确行号，应通过原始资料核对约束和上下文。

空别名和空匹配词列表本身有效，不为凑齐字段生成别名或关键词。历史抓取时间无法追溯，不编造 dump_date；verified_at 是本次核对日期。上表是此次核对的来源快照，不声称已恢复原始抓取提交。

## 复核与更新

[scripts/repair_corpus_metadata.py](../scripts/repair_corpus_metadata.py) 使用 knowledge/ 中已存在的公开来源副本，默认只读计划，不下载资料、不执行上游代码，也不调用模型。目录名与脚本 SOURCES 配置一致；复核本次结果应使用上表提交。

```powershell
./.venv/Scripts/python.exe -X utf8 scripts/repair_corpus_metadata.py --verified-at 2026-10-07
./.venv/Scripts/python.exe -X utf8 scripts/repair_corpus_metadata.py --verified-at 2026-10-07 --apply --index data/game_knowledge_bge_combined_index.sqlite
```

可重复传入 --index 同步其他本地索引。脚本核对 Git blob、来源身份与片段后，备份到 .runtime/metadata-backup-*，在副本上验证索引正文及向量一致性，再替换文件；失败恢复输入。同一来源与核对日期的重复运行结果一致。若正文、标题或章节变化，拒绝以旧向量执行元数据修复，应重新构建索引。

完整修复统计与来源证据见 [结果快照](../evals/reference_results/corpus-metadata-2026-10-07.json)。修复后的检索重新评估，见 [实验报告](experiments.md)；10 月 6 日的快照保留为修复前历史记录。
