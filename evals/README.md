# 离线检索回归评估

`retrieval_cases.jsonl` 是 **AI辅助人工构造回归集**。原有问题保留并经真实语料核对，本次补齐、改写和原文片段核对由 AI 执行；没有独立人工审核，也不是盲测或独立测试集。不要以此宣称“人工审核通过”“用户盲测通过”或无偏泛化能力。该集可用于发现回归、比较当前实现，不能用于独立校准生产阈值。

当前共 40 条：30 条有相关性标注的游戏问题，10 条明确非游戏问题。6 个来源各有 5 条正例：`game_design_wiki`、`game_knowledge_base`、`open_game_mechanics_dataset`、`game_num_basics`、`senior_game_designer`、`gamedev_at_home`。包含中文、英文、场景改写、多块证据，以及与游戏资料共享术语的非游戏请求。非游戏案例中包含两条未照搬当前领域路由提示词的请求，漏拒必须如实报告。

## 案例与语料核验

每行包含 `case_id`、`query`、`expected_domain`、`expected_chunk_ids`、`source_collections`、`rationale`、`dataset_kind`、`provenance`、`annotation_scope` 与 `evidence`。每个正例 ID 配有语料文件、来源集合、标题、章节、来源范围、固定版本链接和语料中的逐字片段。2026-10-07 已同步修复来源元数据；语料的 `provenance.line_scope` 区分匹配片段、结构化 JSON 记录及完整文章，后两种范围不能解释为逐字摘录行，详见 [语料修复记录](../docs/corpus-provenance.md)。

脚本以只读方式打开 SQLite 索引，核对所有索引块与两份 JSONL 语料的 ID 集合、正文和来源字段，并检查每条标注的来源及片段实际存在。默认索引目前包含 6,515 个块。此核验能发现 ID 失效、来源写错、片段虚构或索引陈旧；**不能代替独立人员判断语义相关性**。标签只列出选定的支持块，不保证穷举全部相关块，跨来源也可能存在等价证据。未标注块按不相关计分，因此这些分数可能低估语义检索的实际相关性。商店素材等涉及外部平台的案例只检查已有语料定位，不证明其中要求现在仍有效。

更新语料时先阅读真实内容，再更新 ID、来源和原文片段；不要根据检索结果反向挑标签来提高分数。新增正例必须有标签，拒答案例必须没有 ranking positives，保留来源与非游戏覆盖。`provenance.independent_human_review`、`provenance.blind_test` 当前必须是 `false`。

## 运行

从项目根目录运行，使用项目 `.venv`，无需加载 `.env`，也不需要模型 API 密钥：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_retrieval.py --backend keyword --top-k 5 --output .runtime/retrieval_keyword.json
```

`--backend` 支持 `bge`、`keyword`、`hybrid`、`all`，默认 `all`；`--top-k` 默认为 5，范围 1–10。`--output PATH` 同时生成同名 JSON 和 Markdown 文件，建议指定 `.json`。可用 `--cases`、`--index-path`、`--corpus PATH [PATH ...]` 指定输入。`--threshold` 和 `--high-confidence` 是显式覆盖选项；未指定时使用 `RetrievalService` 的默认值，并将实际生效值写入报告。输出路径不能覆盖输入文件。

全模式评估可能超过 30 秒。在 Windows 上可隐藏启动并保存日志：

```powershell
New-Item -ItemType Directory -Path .runtime -Force | Out-Null
$evalPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
$evalRoot = (Get-Location).Path
$evalProcess = Start-Process -FilePath $evalPython -ArgumentList @('-X', 'utf8', 'scripts/evaluate_retrieval.py', '--backend', 'all', '--top-k', '5', '--output', '.runtime/retrieval_evaluation.json') -WorkingDirectory $evalRoot -WindowStyle Hidden -RedirectStandardOutput "$evalRoot\.runtime\retrieval-evaluation.stdout.log" -RedirectStandardError "$evalRoot\.runtime\retrieval-evaluation.stderr.log" -PassThru
$evalProcess.Id
```

读取完成后的 JSON `status` 和各模式 `valid`，不要仅以进程已结束判断成功。前台调用的退出码：完整有效运行为 `0`，输入核验失败、模型故障、模式变化或降级为 `1`；命令行参数错误为 `2`。`passed` 表示运行有效，不代表召回率达到质量门槛；当前不设置或暗示质量达标阈值。

BGE 使用 `BAAI/bge-small-zh-v1.5`，强制 `local_files_only=True`、CPU、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，关闭 Hub 遥测。只使用本地已有缓存，不下载模型，不调用在线模型 API。缺缓存时如实失败。脚本只导入独立检索服务，禁止导入 `Agent`，不初始化聊天模型或会话数据库，不读取或修改 `.env`。报告不写任意环境变量、异常原文或语料全文，异常仅保存类别，非白名单 `reason` 被替换。

## 指标口径

脚本通过 `RetrievalService(mode=..., index_path=..., embedder=...)` 调用服务，使用 `hits`、`status`、`mode`、`fallback`、`elapsed_ms`、`reason` 和 `to_dict()`，没有复制生产排序或领域规则。

- **raw**：`search(query, top_k=k, apply_threshold=False)`，关闭分数阈值；领域路由仍由生产服务执行，不绕过非游戏拒答。
- **gated**：另一次 `search(query, top_k=k, apply_threshold=True)`，使用生产阈值和候选补位行为。不能通过过滤 raw 的前 k 个结果模拟，因为较深候选可能补位。keyword 使用生产关键词行为，没有余弦阈值，不能将 BM25 或融合分数解释成语义置信度。
- **Recall@k**：前 k 个结果中命中的标注块数 / 该案例全部标注块数。
- **MRR@k**：首个相关块排名的倒数，没有命中则为 0。JSON 字段为 `mrr`。
- **nDCG@k**：二元相关性，`DCG = Σ relevant(rank) / log2(rank+1)`，理想 DCG 取前 `min(k, 标签数)` 个全相关结果；没有命中则为 0。JSON 字段为 `ndcg`。
- **Hit rate**：至少命中一个标注块的正例比例。

上述排名指标只对 30 条正例做逐案例宏平均；raw/gated 分开，也分别输出各来源结果及每案例命中列表。重复返回同一块不能提高分数，会使评估失败。

拒答的分母独立：10 条负例的 `correct_refusal_rate`，要求没有 hits 且状态为 `out_of_domain` 或 `no_evidence`；同时报告有 hits 的负例数和负例状态计数。正例的 `false_refusal_rate` 单独以 30 条正例为分母。仅有空结果、却带其他状态，不算正确拒答。检索没有命中标注块但返回其他候选，属于 ranking miss，不能当成拒答。

任何一次 BGE/hybrid 预热、raw 或 gated 搜索出现 `fallback=True`、返回模式改变、dense 模式状态变成 `keyword_only` 或异常，该模式整体 `failed`、`valid=false`，raw/gated 平均指标为 `null`，总运行退出码为 1。不会把关键词降级计为 BGE 成绩。`all` 模式仍可保存其他有效模式的报告。失败详情与已经观察到的案例保留用于诊断。输入或参与评估的源代码在运行中改变，也会取消全部模式平均指标。

## 延迟、内存与可复现信息

每个模式先用首个正例预热一次 raw 搜索，单独记录预热结果并从 p50/p95 样本排除。之后按固定案例顺序先 raw 后 gated，每阶段每问题一次，分别汇总全体、正例和负例的 `elapsed_ms`。p50/p95 使用线性插值。这是暖态、无并发、无重复采样的 CPU 服务延迟；生产领域拒答的短路延迟也计入全体统计。因此比较检索性能优先看 `ranking_cases`，不能把含大量快速拒答的全体 p50 当成纯检索耗时。模型与索引首次加载包含在预热，不包含在暖态 p50/p95；`wall_time_ms` 包括预热与案例运行。

内存单位为 bytes，Windows 使用 `GetProcessMemoryInfo` 的 resident working set，Linux 使用 `/proc/self/statm` 和 `getrusage`；不可测平台明确返回 `null`。记录模式预热前与案例结束后的 RSS、两者之差及**整个进程生命周期的峰值 RSS**。它包含 Python、原生 CPU 模型、向量、词项索引与分配器，既不是仅 Python 堆，也不是精确的单模式分配量；峰值不能当作单模式峰值。`all` 顺序运行并共享 BGE 模型，后续模式基线可能已有模型内存，结束时的 RSS 差值可为负。没有测量 GPU 显存。如需隔离内存比较，分别运行单模式、使用独立进程；也不要将进程峰值差值称作模型内存。

JSON/Markdown 都保存案例文件 hash、各语料文件 hash 与组合 hash、索引文件 hash、只读索引内容清单 hash、配置 hash、参与排序/路由/评估的源文件 hash，以及 Python/依赖版本、OS、架构、CPU 数量、执行路径、设备和离线模型加载方式。配置包含实际阈值、top-k 与模式，报告保留每案例状态、命中 ID、分数、fallback、耗时和经净化的 reason。开始与结束再次比对输入/代码 hash，发现变化会失败。报告不会枚举环境变量或秘密；这些 hash 不能替代随机种子、多次采样、独立数据集或人工审核。

## 测试

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_evaluation.py -q -p no:cacheprovider --basetemp .runtime/evaluation-pytest
```

测试覆盖指标分母与折损、拒答分组、真实服务 embedding 故障、后续 gated 降级、候选补位、语料/索引/片段核验、hash 与运行中输入变更、报告落盘和失败退出、离线模型参数以及导入无 Agent/model/db 副作用。实际回归集核验使用项目已有语料和索引，不访问网络；单元测试不加载 BGE 权重。
