# 游戏知识助手

面向游戏设计资料的本地知识助手，支持机制、数值、经济系统和制作流程的资料检索，以及带来源的连续对话。

- **离线检索**：BGE 语义检索、BM25 关键词检索、RRF 混合检索。
- **来源展示**：查看原始片段、章节与来源链接。
- **连续对话**：LangGraph 工具调用、SQLite 历史保存、上下文摘要。
- **对话导出**：将回答与来源导出为 Markdown。

知识库包含 6 个公开来源、6,515 个知识块。应用默认进入离线检索模式；启用对话助手需要配置聊天服务。

## 快速开始

需要 Python 3.10。Windows PowerShell：

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

浏览器打开 `http://127.0.0.1:8501`。BM25 可以直接使用仓库中的 JSONL 语料。

使用 BGE 语义检索前构建索引：

```powershell
.\.venv\Scripts\python.exe -X utf8 build_bge_combined_index.py
```

首次构建下载 `BAAI/bge-small-zh-v1.5`；运行时使用本地缓存。正文变化后使用 `--overwrite` 重建索引。

Linux/macOS 可创建虚拟环境、安装 `requirements.txt`，再运行：

```bash
python -m streamlit run UI.py --server.address 127.0.0.1
```

## 聊天配置

在 `.env` 中配置兼容 OpenAI 接口的聊天服务：

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your_chat_model
RETRIEVAL_MODE=bge
```

`RETRIEVAL_MODE` 支持 `bge`、`keyword`、`hybrid`。可选配置 `METASO_API_KEY` 启用联网搜索，`VISION_*` 指定 OCR 模型。离线检索不需要这些接口。

## 项目结构

| 路径 | 功能 |
| --- | --- |
| `UI.py` | Streamlit 界面 |
| `Agent.py` | LangGraph 对话与工具编排 |
| `game_agent/` | 配置、提示词、文件访问与来源展示 |
| `wiki_corpus/` | 语料、向量索引与检索策略 |
| `conversation_*.py` | 历史存储与上下文管理 |
| `evals/` | 检索回归案例与实验结果 |
| `tests/` | 离线测试 |

## 测试与评估

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\verify.ps1
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_retrieval.py --backend all --top-k 5 --output .runtime/retrieval_evaluation.json
```

测试使用临时数据库并阻断网络连接；[GitHub Actions](https://github.com/Ryuuson/Game-knowledge-agent/actions) 执行离线测试。检索评估使用本地模型与索引，不需要聊天 API。

40 条 AI 辅助构造回归案例上的结果如下。案例未经独立人工审核，标签未穷举所有相关资料，因此分数表示标注块召回，不能解释为回答准确率。

| 策略 | Recall@5 | 阈值后 Recall@5 | 非游戏拒答 | 游戏误拒 |
| --- | ---: | ---: | ---: | ---: |
| BGE | 46.7% | 40.0% | 10/10 | 4/30 |
| BM25 | 70.0% | 70.0% | 8/10 | 0/30 |
| BGE + BM25 / RRF | 63.3% | 48.3% | 10/10 | 4/30 |

实验环境、指标口径和失败案例见 [检索实验](docs/experiments.md)。

## 使用范围

应用面向本地单用户。聊天与摘要会将问题及必要上下文发送给配置的服务；凭据、会话、笔记和模型缓存保存在本机。

语义服务不可用时，界面明确显示关键词降级。来源编号核验只能确认引用来自检索结果；设计结论仍需结合原始资料与具体项目判断。资料来源及许可声明见 [知识库来源](docs/corpus-provenance.md)。

## 文档

- [使用示例](docs/demo.md)
- [系统架构](docs/architecture.md)
- [检索实验](docs/experiments.md)
- [知识库来源](docs/corpus-provenance.md)
