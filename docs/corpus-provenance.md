# 语料来源与字段核对

核对日期：2026-10-06。实际逐行解析 `data/game_knowledge_chunks.jsonl` 与 `data/new_knowledge_chunks.jsonl`，统计 `source_collection`、`collection_label`、`license` 及字段缺失情况；没有访问上游网站、私有资料或会话数据库。下表许可证字符串来自原数据声明，不是本次审查确认的法律结论，也不能据此推断具体使用或再分发已获授权。

## 来源集合与许可证原值

| 文件 | source_collection（collection_label 同值） | 块数 | license 原值 |
| --- | --- | ---: | --- |
| game_knowledge_chunks.jsonl | game_design_wiki | 712 | MIT |
| game_knowledge_chunks.jsonl | game_knowledge_base | 621 | Not specified by source |
| new_knowledge_chunks.jsonl | open_game_mechanics_dataset | 2,676 | MIT |
| new_knowledge_chunks.jsonl | game_num_basics | 1,809 | TBD |
| new_knowledge_chunks.jsonl | gamedev_at_home | 609 | TBD |
| new_knowledge_chunks.jsonl | senior_game_designer | 88 | TBD |

第一个文件 1,333 块，第二个文件 5,182 块；总计 6,515 块，`chunk_id` 唯一值也是 6,515。许可证原值合计：MIT 3,388；Not specified by source 621；TBD 2,506。后两类共 3,127 块，表示授权信息尚未明确；TBD 不能当作许可证。

集合对应的上游项目名称取自仓库 README：Being09/game-design-wiki、diedie23/Game-Knowledge-Base、Thaelith/open-game-mechanics-dataset、lsc1414/Game_Num_Basics_And_Calc、zsc/gamedev_at_home、tigermkiiiddd/senior-game-designer。本次没有独立核验它们的许可证、权利归属或当前版本。

## 缺失、空值与占位值

两个文件的每条记录都有这 13 个顶层字段：`chunk_id`、`title`、`alternate_titles`、`source_url`、`section_path`、`start_line`、`end_line`、`text`、`license`、`matched_terms`、`collection_label`、`classification_reason`、`source_collection`。逐字段检查均为：键缺失 0、null 0、空字符串 0。字段存在不意味着信息完整或正确。

| 核对项 | game_knowledge_chunks.jsonl | new_knowledge_chunks.jsonl |
| --- | ---: | ---: |
| alternate_titles 为空列表 | 1,333 | 2,506 |
| matched_terms 为空列表 | 833 | 2,506 |
| start_line/end_line 同为 0 | 0 | 5,182 |
| source_url 不以 http:// 或 https:// 开头 | 712 | 5,182 |

空列表表示未提供别名或匹配词，不等于正文缺失。第二个文件的行号全部为 `0/0`，没有可直接使用的原文行范围；不能伪造行号引用。共 5,894 条 `source_url` 是相对路径形式的资料标识，不可直接当网页链接；另外 621 条虽是 HTTP(S) 地址，其可访问性与是否精确定位片段未核验。

所有记录均未提供 `dump_date`、上游提交版本或抓取时间字段，因此无法从这两个 JSONL 确定来源版本。上述统计不验证来源内容的真实性、切分质量或回答质量。

## 使用与复核边界

展示依据时保留集合、标题、章节、来源标识及 chunk_id；HTTP(S) 来源才展示为网页入口。需要对外发布语料、商用或再分发时，应逐来源核对实际版本的许可证文本、署名要求及内容权利；授权未明确的记录需进一步确认。本页不补写未知许可证，也不把“公开可访问”解释成“可以任意使用”。
