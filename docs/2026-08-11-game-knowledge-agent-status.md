# Game Knowledge Agent 当前状态

更新时间：2026-08-11

## 定位

项目名称为 **Game Knowledge Agent**。OpenViking 是其上下文后端，负责游戏知识资源检索和长期记忆；LangGraph 负责 Agent 编排，Streamlit 负责可见聊天界面。本地 SQLite 不再承担游戏知识检索或长期记忆存储职责。

## 已完成的 Agent 修改

### 游戏知识检索

- 移除了原本基于 BGE/SQLite 的游戏知识检索运行链路及其回退逻辑。
- 新增 `openviking_client.py`，以结构化错误封装 OpenViking 的检索、内容读取、文件系统、资源导入和 Session 接口。
- 新增 `game_knowledge.py`，并在 `Agent.py` 暴露 `search_game_knowledge` 和 `describe_game_knowledge` 工具。
- 检索范围固定为 `viking://resources/game-knowledge/`，只请求 OpenViking 的 L2 内容；结果 URI 去重，并受 `OPENVIKING_EVIDENCE_CHAR_BUDGET`（默认 12000 字符）限制。
- 对明显非游戏领域的问题，先要求确认游戏语境；OpenViking 不可用时明确返回后端错误，不能静默改用联网搜索。
- “知识库有什么内容”一类问题必须调用 `describe_game_knowledge`。该工具只报告可读取来源及实际资源节点数，禁止模型虚构教材数、主题范围或内容质量。

### 长期记忆

- 新增 `memory.py`，将长期记忆改为 OpenViking 用户记忆命名空间，而非自定义本地目录或 SQLite 表。
- 每次模型调用前，根据最后一条用户消息检索相关记忆；只接受 `preferences`、`entities` 和 `events` 三类受管记忆。
- 每轮可见回答已经渲染并持久化后，才执行尽力而为的记忆提取和 Session commit，不影响对用户的回复。
- 提取器只保留稳定偏好、已确认项目事实/约束、明确决策和持续任务；过滤密钥、密码、令牌、原始对话、小聊、猜测和一次性问题。
- Streamlit 侧栏现在可以刷新、查看和删除长期记忆。“清空当前聊天”只新建可见会话，不删除长期记忆。

### 会话与 UI

- SQLite 继续保存 LangGraph checkpoint、会话 ID、标题和完整可见聊天记录；不再用作游戏资料或长期记忆检索库。
- 上下文裁剪按完整用户回合进行，避免截断工具调用序列。
- 历史会话由 `conversation_store.py` 管理；旧 checkpoint 会以“历史会话”形式惰性登记，再由首条用户消息生成标题。

### 知识导入与配置

- 新增 `ingestion.py`，定义六个稳定目标 URI 的公开来源清单，并提供 `ingest` 与 `verify` 命令。
- 导入目标固定在 `viking://resources/game-knowledge/<source>`；单个来源失败不会中断后续来源的处理。
- 更新 `.env.example`、`requirements.txt` 和 README，说明 OpenViking 连接、认证、超时和证据预算配置，以及本地启动和导入流程。

## 当前资源状态

OpenViking 本地任务记录中，以下 GitHub 来源已成功完成导入：

| 来源 | 任务状态 | 导入文件数 | 资源 URI |
| --- | --- | ---: | --- |
| `Game-Knowledge-Base` | `success` | 482 | `viking://resources/game-knowledge/Game-Knowledge-Base` |
| `Game_Num_Basics_And_Calc` | `success` | 318 resources | `viking://resources/game-knowledge/Game_Num_Basics_And_Calc` |
| `game-design-wiki` | `success` | 83 | `viking://resources/game-knowledge/game-design-wiki` |
| `gamedev_at_home` | `success` | 15 | `viking://resources/game-knowledge/gamedev_at_home` |
| `open-game-mechanics-dataset` | `success` | 311 | `viking://resources/game-knowledge/open-game-mechanics-dataset` |
| `senior-game-designer` | `success` | 43 | `viking://resources/game-knowledge/senior-game-designer` |

任务记录没有错误。`game-design-wiki` 能导入的原因是它是可公开访问的 GitHub 仓库，OpenViking 的 Git 导入器成功克隆并解析了该仓库；它没有使用特殊白名单或绕过限制。

`Game_Num_Basics_And_Calc` 已通过临时归档导入：其仓库 `.gitignore` 的 Windows 反斜杠模式会使 OpenViking 0.4.13 的 Git 解析器异常。`Game-Knowledge-Base` 已改为使用对应的公开 GitHub 仓库导入，避免网站根地址缺少 sitemap/feed 的限制。

## 已知问题与限制

- `ingestion.py verify` 目前对已成功的 `gamedev_at_home` 和 `senior-game-designer` 仍可能报“缺失”。直接读取 OpenViking 内容已证明两者实际存在，因此这是校验代码的假阴性，尚待修正。
- `Game_Num_Basics_And_Calc` 的源仓库仍含 OpenViking 0.4.13 无法解析的 Windows `.gitignore` 规则；需要保留临时归档导入作为兼容性方案，或等待上游修复该规则/解析器兼容性。
- 导入成功不等同于所有语义向量均已建立。任务记录显示成功来源的语义队列已处理；嵌入队列记录为零项，需要在实际检索质量验收中继续观察。
- OpenViking 服务是运行时前置条件。知识检索不可用时，Agent 会显式报错；长期记忆捕获失败只写日志，不会阻塞已生成的回复。
- 联网搜索仍保留为单独的 Metaso 工具，只应在本地资料不足或用户明确需要外部最新信息时使用，不应替代 OpenViking 故障。

## 已执行验证

以下检查已在本次 V2 修改后通过：

- `python -m unittest discover -s tests -q`：9 项单元/契约测试通过。
- `python -m py_compile Agent.py UI.py openviking_client.py game_knowledge.py memory.py ingestion.py conversation_context.py conversation_store.py`：通过。
- `git diff --check`：通过。
- Streamlit `AppTest`：`UI.py` 可启动，未出现应用异常。
- OpenViking 健康检查：本地服务返回 HTTP 200。

## 工作区与提交状态

- 当前 V2 实现仍在工作区，尚未提交。
- 当前 HEAD：`9fd257c docs: define Game Knowledge Agent v2 architecture`。
- 本次状态文档也处于未提交状态。
- 未执行 push，也没有重写该提交之前的历史。
