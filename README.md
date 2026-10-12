# 游戏知识助手

基于 RAG 的本地游戏设计知识助手，支持检索游戏机制、数值与经济系统相关资料，并结合来源进行多轮对话。内置 5,285 个知识块，来自 4 个公开资料库。

本地检索无需 API，返回资料片段和来源；AI 对话需要配置聊天服务。

## 功能

- **本地检索**：支持 BM25 关键词检索、BGE 语义检索和混合检索。
- **AI 对话**：结合资料回答问题，支持继续追问。
- **引用追溯**：查看回答引用的原文、章节和来源链接。
- **历史记录**：保存、恢复会话，导出对话与来源为 Markdown。

## 效果展示

例如，搜索“游戏经济 水龙头 污水槽 消耗 产出 通胀”，可以找到《虚拟经济系统》的相关章节，展开片段并打开原文。

![检索结果与来源](docs/assets/screenshots/sources.png)

对话与引用展示（示例回答，非模型实际输出）。

![对话与引用展示示例](docs/assets/screenshots/chat-example.png)

完整操作和问答示例见 [使用示例](docs/demo.md)。

## 快速开始

需要 Python 3.10。在 Windows PowerShell 中运行：

```powershell
py -3.10 --version
.\scripts\setup.ps1
.\scripts\run.ps1
```

浏览器打开 `http://127.0.0.1:8501`，选择 **BM25 关键词检索**即可查找资料。

没有 `py` 启动器时，安装脚本也支持 PATH 中的 Python 3.10 或 `-PythonExecutable` 指定路径。其他平台、脚本执行策略和安装故障见 [安装说明](docs/installation.md)。

### 聊天配置

在安装脚本创建的 `.env` 中填写兼容 OpenAI 接口的聊天服务：

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your_chat_model
RETRIEVAL_MODE=keyword
```

重启应用后切换到“对话助手”。聊天和摘要会将问题、检索片段及必要历史发送给配置的服务，费用按服务商规则计收；会话记录保存在本机。应用面向本地单用户。

### 可选配置

BGE 和混合检索需要额外安装模型依赖并构建本地索引，步骤见 [安装说明](docs/installation.md#安装与启动)。构建完成后，`RETRIEVAL_MODE` 可设为 `bge` 或 `hybrid`。

`METASO_API_KEY` 启用联网搜索，`VISION_*` 指定 OCR 模型。配置项见 [.env.example](.env.example)。

## 技术实现

界面使用 Streamlit，LangGraph 编排模型与工具调用，SQLite 保存会话和向量索引。检索服务由界面、对话助手和评估脚本共用；详细流程见 [系统架构](docs/architecture.md)。

## 文档与许可

- [安装与常见问题](docs/installation.md)
- [使用示例](docs/demo.md)
- [系统架构](docs/architecture.md)
- [检索评估：方法、结果与复现](docs/experiments.md)
- [知识库来源](docs/corpus-provenance.md)

项目原创代码和文档采用 [MIT](LICENSE)。第三方语料及其引文保留各自来源许可，见 [第三方声明](THIRD_PARTY_NOTICES.md)。
