# Game Knowledge Agent

一个面向游戏设计工作的本地知识 Agent。它能从游戏机制、数值设计、AI、制作流程等资料中检索证据，结合对话模型给出可继续追问的回答。

基于 LangGraph 构建，支持会话持久化、长对话摘要压缩、文件与笔记工具、OCR，以及可选的联网搜索。

## 能做什么

- 用本地游戏设计知识库回答机制、经济、数值、战斗、AI、原型、外包与制作流程问题
- 保存并切换历史会话；对长对话自动压缩上下文，完整聊天记录仍会保留
- 读取本地 Markdown 或文本资料、保存笔记、识别图片文字
- 本地资料不足时，可选调用 Metaso 联网搜索
- 聊天模型使用任意 OpenAI-compatible API，不绑定特定厂商

## 快速开始

需要：Windows、Python 3.10，以及一个 OpenAI-compatible 聊天模型的 API 配置。

### 1. 安装依赖

```powershell
.\scripts\setup.ps1
```

首次运行会创建 `.venv`、安装依赖，并在缺少配置时创建 `.env`。

### 2. 配置聊天模型

打开 `.env`，填写下面三项：

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your_chat_model
```

### 3. 构建索引并启动

```powershell
.\.venv\Scripts\python.exe build_bge_combined_index.py
.\scripts\run.ps1
```

第一次构建会下载 BGE 模型并生成本地索引；之后只需运行第二条命令。

## 配置

| 配置项 | 是否必填 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 是 | 聊天模型 API Key |
| `LLM_BASE_URL` | 是 | 聊天模型的 OpenAI-compatible 地址 |
| `LLM_MODEL` | 是 | 聊天模型名称 |
| `RAG_BACKEND=bge` | 否 | 默认值。本地 BGE 检索，无需 embedding API |
| `RAG_BACKEND=ark` | 否 | 使用 Ark 检索，需另行配置 Ark 凭据和匹配的索引 |
| `METASO_API_KEY` | 否 | 开启联网搜索；留空时 Agent 仍可使用本地知识库 |
| `VISION_*` | 否 | 为 OCR 指定单独的视觉模型；留空时使用聊天模型 |

原有 `ARK_*` 配置仍可兼容使用，但新配置优先使用通用的 `LLM_*` 变量。

## 知识库与索引

仓库包含两份已切分好的游戏知识块：

- `data/game_knowledge_chunks.jsonl`：早期游戏 wiki 资料，1,333 个 chunks
- `data/new_knowledge_chunks.jsonl`：扩充资料，5,182 个 chunks

`build_bge_combined_index.py` 会将它们合并为 6,515 个 BGE 向量，写入本地 SQLite 索引。这个索引是生成物，不会提交到 Git；修改语料后可重建：

```powershell
.\.venv\Scripts\python.exe build_bge_combined_index.py --overwrite
```

不要混用不同 embedding 模型生成的索引和查询编码器。默认 BGE 索引只能配合 BGE 查询；Ark 索引同理。

## 常见问题

**启动提示找不到 BGE 索引？**

运行一次 `build_bge_combined_index.py`。新克隆的项目不附带生成好的 SQLite 索引。

**不使用 Ark 或 Metaso 能运行吗？**

可以。默认 BGE 检索完全在本地运行；Metaso 只是联网搜索的可选能力。聊天模型仍需要你自己的 OpenAI-compatible API。

**想改用 Ark 检索？**

将 `.env` 中的 `RAG_BACKEND` 改为 `ark`，并配置 Ark embedding 凭据与匹配索引。没有匹配索引时，请保持 BGE 默认设置。

## 边界

这是一个本地应用，不是托管服务。仓库不会包含 API Key、聊天记录、虚拟环境或生成的 SQLite 索引。
