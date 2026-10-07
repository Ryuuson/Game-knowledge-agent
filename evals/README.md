# 检索回归评估

`retrieval_cases.jsonl` 包含 40 条 AI 辅助构造案例：30 条游戏正例、10 条非游戏问题。六个来源各有五条正例。案例未经独立人工审核，不是盲测；标签只列选定的支持块，未穷举所有相关资料。

## 案例格式

| 字段 | 含义 |
| --- | --- |
| `case_id` / `query` | 案例编号与问题 |
| `expected_domain` | 游戏或非游戏语境 |
| `expected_chunk_ids` | 正例的支持块编号；负例为空 |
| `source_collections` / `rationale` | 来源集合与标注理由 |
| `evidence` | 来源信息及语料中的逐字片段 |
| `dataset_kind` / `provenance` | 案例构造方式和审核状态 |
| `annotation_scope` | 标注覆盖范围 |

来源范围由语料的 `provenance.line_scope` 区分片段、JSON 记录或文章，见 [知识库来源](../docs/corpus-provenance.md)。评估前核对语料、索引和标注的一致性。

## 运行

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_retrieval.py --backend all --top-k 5 --output .runtime/retrieval_evaluation.json
```

| 参数 | 用途 |
| --- | --- |
| `--backend` | `bge`、`keyword`、`hybrid` 或 `all` |
| `--top-k` | 每次返回数量，1–10，默认 5 |
| `--cases` / `--corpus` / `--index-path` | 指定案例、语料和索引 |
| `--threshold` / `--high-confidence` | 覆盖默认阈值 |
| `--output` | JSON 输出路径，同时生成同名 Markdown |

BGE 只使用本地缓存模型，评估不需要聊天 API。退出码 0 表示运行有效，1 表示输入、模型或降级等失败，2 表示参数错误。BGE／混合模式发生降级时不输出该模式的平均成绩。

## 指标

- `raw` 关闭语义阈值，保留生产领域规则；`gated` 单独执行带阈值的检索和候选补位。
- Recall@k：前 k 个结果命中的标注块数除以该案例的全部标注块数。
- MRR@k：首个标注块的名次倒数，未命中为 0。
- nDCG@k：使用二元相关性和对数位置折损。
- Hit rate：至少命中一个标注块的正例比例。

排名指标对 30 条正例宏平均。非游戏正确拒答率以 10 条负例为分母，要求没有结果且状态为 `out_of_domain` 或 `no_evidence`；游戏误拒率以 30 条正例为分母。

每种模式先预热一次，再按固定顺序执行案例。比较检索延迟应使用正例查询的 p50／p95；全体延迟包括领域拒答短路。进程 RSS 和峰值包含模型、索引与共享缓存，并非单模式独占内存。

## 结果文件

JSON 保存输入指纹、配置、环境、逐案例状态、命中编号、分数和耗时。运行前后检查输入及代码变化。实验结果见 [完整数据](reference_results/2026-10-07.json) 和 [结果说明](../docs/experiments.md)。
