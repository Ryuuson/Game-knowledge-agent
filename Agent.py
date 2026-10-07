"""一个最小的 LangGraph 工具调用 Agent。

流程：START -> llm -> tools -> llm ... -> END
模型决定是否调用工具；ToolNode 负责执行工具调用。
"""

import os
import sqlite3
import requests
import base64
import mimetypes
from threading import Lock
from uuid import uuid4

from pathlib import Path
from typing import Annotated, Literal
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime
from langchain_core.messages import AIMessageChunk, ToolMessage
from conversation_context import (
    estimate_message_tokens,
    remove_internal_summary,
    split_complete_user_turns,
    fit_model_context,
)
from conversation_store import ConversationStore
from wiki_corpus.domain_signals import classify_query_domain
from wiki_corpus.vector_search import load_index, rank_chunks

from game_agent.settings import ROOT, Settings
from game_agent import local_files
from game_agent.presentation import validate_citations
from wiki_corpus.retrieval import RetrievalService, format_evidence

load_dotenv(ROOT / ".env")
settings = Settings.from_env()


def _setting(primary: str, legacy: str, default: str | None = None) -> str | None:
    """Prefer provider-neutral settings while preserving existing Ark setups."""
    return os.getenv(primary) or os.getenv(legacy) or default


LLM_API_KEY = _setting("LLM_API_KEY", "ARK_API_KEY")
LLM_BASE_URL = settings.base_url or "http://127.0.0.1:9/v1"
LLM_MODEL = settings.model or "offline-not-configured"

model = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY or "offline-not-configured",
    base_url=LLM_BASE_URL,
    temperature=0,
    timeout=settings.request_timeout,
    max_retries=1,
)

# 视觉模型：仅用于 ocr_image，与主 Agent 模型解耦。
# 优先使用 VISION_*；为兼容已有配置，未填写时才复用 Ark 视觉/主模型。
vision_model = ChatOpenAI(
    model=os.getenv("VISION_MODEL") or os.getenv("ARK_VISION_MODEL") or LLM_MODEL,
    api_key=os.getenv("VISION_API_KEY") or (os.getenv("ARK_API_KEY") if os.getenv("ARK_VISION_MODEL") else LLM_API_KEY) or "offline-not-configured",
    base_url=os.getenv("VISION_BASE_URL") or (os.getenv("ARK_BASE_URL") if os.getenv("ARK_VISION_MODEL") else LLM_BASE_URL),
    temperature=0,
    timeout=settings.request_timeout,
    max_retries=1,
)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
NOTES_DIR = Path(__file__).parent / "notes"
IMAGE_DIR = Path(__file__).parent / "images"

TEXT_SUFFIXES = {".md", ".txt"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_READ_SIZE = 100 * 1024  # read_document 单次全文读取上限，超大文件用 read_document_section 分段
MAX_IMAGE_SIZE = 4 * 1024 * 1024  # ocr_image 单张图片上限；base64 后约 5.3 MB，需留足模型输入余量

LOCAL_DOCUMENT_ROOTS = {
    "game_design_wiki": "game-design-wiki",
    "game_num_basics": "Game_Num_Basics_And_Calc",
    "senior_game_designer": "senior-game-designer",
    "gamedev_at_home": "gamedev_at_home",
}


def _local_knowledge_path(hit: dict) -> str | None:
    """Translate a chunk source identifier into a safe readable local path."""
    collection = str(hit.get("collection_label", ""))
    source = str(hit.get("source_url", ""))
    root_name = LOCAL_DOCUMENT_ROOTS.get(collection)
    prefix = f"{collection}/"
    provenance_path = (hit.get("provenance") or {}).get("source_path")
    if not root_name or not (provenance_path or source.startswith(prefix)):
        return None

    relative_path = Path(provenance_path or source.removeprefix(prefix))
    if relative_path.is_absolute() or ".." in relative_path.parts or ":" in str(relative_path):
        return None
    candidate = Path(root_name) / relative_path
    resolved = (KNOWLEDGE_DIR / candidate).resolve()
    if (
        not resolved.is_relative_to((KNOWLEDGE_DIR / root_name).resolve())
        or resolved.suffix.lower() not in TEXT_SUFFIXES
    ):
        return None
    return candidate.as_posix()

# ---------- 语义检索（RAG）相关 ----------
RAG_BACKEND = os.getenv("RAG_BACKEND", "bge").lower()
ARK_INDEX_PATH = Path(__file__).parent / "data" / "game_knowledge_combined_index.sqlite"
BGE_INDEX_PATH = Path(__file__).parent / "data" / "game_knowledge_bge_combined_index.sqlite"
SEMANTIC_TOP_K = 3
ARK_EMBEDDING_MODEL = os.getenv("ARK_EMBEDDING_MODEL", "ep-20260805175555-j5hff")
BGE_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
if RAG_BACKEND == "ark":
    GAME_INDEX_PATH = ARK_INDEX_PATH
    EXPECTED_INDEX_MODEL = f"ark:{ARK_EMBEDDING_MODEL}"
    # Ark 严格评估校准：低于 0.51 不返回；0.51-0.63 交给外层 LLM 核对游戏语境。
    SEMANTIC_THRESHOLD = 0.51
    HIGH_CONFIDENCE_THRESHOLD = 0.63
elif RAG_BACKEND == "bge":
    GAME_INDEX_PATH = BGE_INDEX_PATH
    EXPECTED_INDEX_MODEL = BGE_EMBEDDING_MODEL
    # BGE 合并索引经开发集和 holdout 校准：低于 0.62 不返回；0.62-0.67 待确认。
    SEMANTIC_THRESHOLD = 0.62
    HIGH_CONFIDENCE_THRESHOLD = 0.67
else:
    raise ValueError("RAG_BACKEND 只支持 ark 或 bge")
# 模块级缓存：embedding 客户端和索引只在首次调用时加载一次，之后复用。
# 因为 Agent 是长运行进程，工具会被多次调用，不能每次重新加载模型/打开库。
_cached_embedder = None
_cached_index = None
_embedder_lock = Lock()
_index_lock = Lock()


def classify_game_retrieval(score: float) -> str:
    """Classify the top retrieval score for the outer Agent's routing decision."""
    if score < SEMANTIC_THRESHOLD:
        return "no_evidence"
    if score < HIGH_CONFIDENCE_THRESHOLD:
        return "ambiguous"
    return "high_confidence"


def _domain_signal_response(query: str) -> str | None:
    """Avoid retrieving game evidence for a high-specificity non-game query."""
    signals = classify_query_domain(query)
    if signals.classification != "clear_non_game":
        return None
    matched = "、".join(signals.non_game_signals)
    return (
        f"检索状态：领域待确认（非游戏信号：{matched}）。"
        "不要使用本地游戏知识直接回答。若用户实际在问游戏内系统，请先请用户补充游戏语境；"
        "否则应使用联网搜索，并说明回答来自联网搜索。"
    )


def _get_embedder():
    """懒加载当前检索后端的 embedding 客户端。"""

    global _cached_embedder
    if _cached_embedder is not None:
        return _cached_embedder

    with _embedder_lock:
        if _cached_embedder is None:
            if RAG_BACKEND == "bge":
                from sentence_transformers import SentenceTransformer

                _cached_embedder = SentenceTransformer(
                    BGE_EMBEDDING_MODEL, local_files_only=True
                )
            else:
                from wiki_corpus.ark_multimodal_embeddings import ArkMultimodalTextEmbedder

                api_key = os.getenv("ARK_API_KEY")
                if not api_key:
                    raise RuntimeError("未配置 ARK_API_KEY，无法执行 Ark 语义检索。")
                _cached_embedder = ArkMultimodalTextEmbedder(
                    api_key=api_key,
                    model=ARK_EMBEDDING_MODEL,
                    base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                )
    return _cached_embedder


def _get_index():
    """懒加载游戏知识索引，并在进程内缓存。"""
    global _cached_index
    if _cached_index is not None:
        return _cached_index

    with _index_lock:
        if _cached_index is None:
            if not GAME_INDEX_PATH.is_file():
                return None
            connection = sqlite3.connect(
                f"file:{GAME_INDEX_PATH.as_posix()}?mode=ro", uri=True
            )
            try:
                _cached_index = load_index(connection)
            finally:
                connection.close()
    return _cached_index

def _resolve_knowledge_path(filename: str) -> Path:
    return local_files.resolve_document(KNOWLEDGE_DIR, filename)

@tool
def list_files(
    source: Literal["knowledge", "images"] = "knowledge",
    recursive: bool = True
) -> str:
    """列出 knowledge 中的学习资料，或 images 中可供识别的图片。
    默认递归列出所有子目录；设置 recursive=False 仅显示顶层文件。
    """
    sources = {
        "knowledge": (KNOWLEDGE_DIR, TEXT_SUFFIXES, "资料文件"),
        "images": (IMAGE_DIR, IMAGE_SUFFIXES, "图片"),
    }
    directory, suffixes, label = sources[source]

    if not directory.is_dir():
        return f"{source} 目录不存在。"

    files = []
    if recursive:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in suffixes:
                rel_path = file_path.relative_to(directory).as_posix()
                files.append(rel_path)
    else:
        files = sorted(
            fp.name for fp in directory.iterdir()
            if fp.is_file() and fp.suffix.lower() in suffixes
        )

    return "\n".join(files) if files else f"没有找到{label}。"

@tool
def read_document(filename: str) -> str:
    """读取 knowledge 目录中的 Markdown 或文本文件（可含子目录）。"""
    return local_files.read_document(KNOWLEDGE_DIR, filename, max_bytes=MAX_READ_SIZE)


@tool
def search_documents(query: str) -> str:
    """在 knowledge 目录中按关键词搜索，返回文件名和真实行号。"""
    return local_files.search_documents(KNOWLEDGE_DIR, query)


def _search_index(query: str, top_k: int) -> str:
    """执行游戏知识库的向量检索。"""
    query = query.strip()
    if not query:
        return "查询内容不能为空。"

    domain_response = _domain_signal_response(query)
    if domain_response:
        return domain_response

    index = _get_index()
    if index is None:
        build_command = (
            "build_bge_combined_index.py"
            if RAG_BACKEND == "bge"
            else "build_merged_game_index.py"
        )
        return f"游戏知识索引不存在，请先运行 {build_command} 构建索引。"

    chunks, vectors, model_name = index
    if model_name != EXPECTED_INDEX_MODEL:
        return (
            f"游戏知识索引与当前 {RAG_BACKEND} embedding 配置不匹配。"
            "请切换到匹配的 RAG_BACKEND，或重建对应索引后再检索。"
        )
    try:
        if RAG_BACKEND == "bge":
            query_vector = _get_embedder().encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            query_vector = _get_embedder().encode(query)
    except Exception as error:
        return "语义检索编码失败，请检查 embedding 配置。"

    hits = rank_chunks(query_vector, chunks, vectors, top_k=top_k)
    # 过滤掉低于下限的结果，并把边界结果交给外层 LLM 结合语境判断。
    hits = [hit for hit in hits if hit.get("score", 0.0) >= SEMANTIC_THRESHOLD]
    if not hits:
        return f"知识库中没有与“{query}”相关的内容。"

    confidence = classify_game_retrieval(float(hits[0]["score"]))
    if confidence == "ambiguous":
        results = [
            "检索状态：待确认。候选内容与问题相近，但请先根据用户问题和会话上下文确认是否明确在问游戏领域；不要把游戏资料直接用于其他行业。"
        ]
    else:
        results = ["检索状态：高相关。可基于以下游戏知识回答。"]
    for position, hit in enumerate(hits, start=1):
        source = hit.get("source_url") or hit.get("title", "未知来源")
        score = hit.get("score", 0.0)
        text = hit.get("text", "").strip()
        title = hit.get("title", "未知标题")
        section = hit.get("section_path") or "文章开头"
        collection = hit.get("collection_label") or "game_knowledge"
        local_path = _local_knowledge_path(hit)
        readable_path = f"\n可读取文件：{local_path}" if local_path else ""
        results.append(
            f"[{position}] 相似度={score:.3f} 来源集合：{collection}\n"
            f"标题：{title}\n章节：{section}\n来源：{source}{readable_path}\n{text}"
        )

    return "\n\n".join(results)


_retrieval_service = RetrievalService(BGE_INDEX_PATH, mode=settings.retrieval_mode)


@tool(response_format="content_and_artifact")
def search_game_knowledge(query: str, top_k: int = SEMANTIC_TOP_K) -> tuple[str, dict]:
    """检索游戏设计、机制、数值和制作流程，返回带来源编号的证据。"""
    if RAG_BACKEND == "ark":
        return _search_index(query, top_k), {"hits": [], "mode": "ark", "status": "legacy"}
    try:
        result = _retrieval_service.search(query, top_k=top_k)
    except (ValueError, OSError, sqlite3.Error):
        return "检索失败，请确认问题长度、top_k 为 1–10，以及本地语料可读取。", {"hits": [], "mode": settings.retrieval_mode, "status": "error"}
    for hit in result.hits:
        hit["local_path"] = _local_knowledge_path(hit)
    return format_evidence(result), result.to_dict()


@tool
def save_note(content: str) -> str:
    """将学习笔记保存到 notes 目录，仅在用户明确要求保存时调用。"""
    return local_files.save_note(NOTES_DIR, content)


@tool
def read_document_section(filename: str, start_line: int, end_line: int) -> str:
    """读取 knowledge 中文件的指定行范围，从 1 开始，最多 120 行。"""
    return local_files.read_section(KNOWLEDGE_DIR, filename, start_line, end_line)


@tool
def ocr_image(filename: str) -> str:
    """识别 images 目录中图片里的文字、公式和表格。参数只接受图片文件名。"""
    if Path(filename).name != filename:
        return "文件名不合法。请将图片放到 images 目录后只传文件名。"

    image_path = IMAGE_DIR / filename
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        return "只支持 PNG、JPG、JPEG 和 WEBP 图片。"
    if not image_path.is_file():
        return f"没有找到图片：{filename}。请放入 images 目录。"
    if image_path.stat().st_size > MAX_IMAGE_SIZE:
        return f"图片超过 {MAX_IMAGE_SIZE // (1024 * 1024)} MB，请先压缩或裁剪。"

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")

    try:
        response = vision_model.invoke([
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "请忠实识别图片中的全部可见文字、公式和表格。保留原有层级与顺序；看不清的内容标为[无法辨认]，不要补写或解释。",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
            ]),
        ])
    except Exception as error:
        if "only support text messages" in str(error):
            return "当前视觉模型只支持文本消息，无法识别图片。请将 ARK_VISION_MODEL 配置为视觉对话模型。"
        return "OCR 调用失败，请检查视觉模型配置和连接。"

    return str(response.content)


@tool
def metaso_search(
    query: str,
    scope: Literal["webpage", "document", "scholar"] = "webpage",
    detail: Literal["standard", "concise"] = "standard",
) -> str:
    """搜索互联网资料。scope 可选网页、文档、学术；detail 为标准片段或短片段。"""

    api_key = os.getenv("METASO_API_KEY")
    if not api_key:
        return "未配置 METASO_API_KEY，无法联网搜索。"

    scope_labels = {
        "webpage": "网页",
        "document": "文档",
        "scholar": "学术资料",
    }
    result_keys = {
        "webpage": "webpages",
        "document": "documents",
        "scholar": "scholars",
    }
    if scope not in scope_labels:
        return "scope 只允许 webpage、document 或 scholar。"
    if detail not in {"standard", "concise"}:
        return "detail 只允许 standard 或 concise。"

    try:
        response = requests.post(
            "https://metaso.cn/api/v1/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "scope": scope,
                "size": "5",
                "includeSummary": False,
                "conciseSnippet": detail == "concise",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        return "联网搜索请求失败，请检查搜索配置和连接。"
    except ValueError:
        return "联网搜索返回了无法解析的内容。"

    if data.get("errCode"):
        return f"秘塔搜索失败：{data.get('errMsg', '未知错误')}"

    sources = data.get(result_keys[scope], [])
    if not sources:
        return f"秘塔{scope_labels[scope]}搜索没有找到结果。"

    results = []
    for index, source in enumerate(sources, start=1):
        metadata = []
        if source.get("authors"):
            metadata.append(f"作者：{source['authors']}")
        if source.get("date"):
            metadata.append(f"日期：{source['date']}")

        results.append(
            f"{index}. {source.get('title', '无标题')}\n"
            f"链接：{source.get('link', '无链接')}\n"
            f"摘要：{source.get('snippet', '无摘要')}"
            + (f"\n{'；'.join(metadata)}" if metadata else "")
        )

    return "\n\n".join(results)


@tool
def metaso_reader(url: str) -> str:
    """读取一个网页链接的 Markdown 正文。优先读取 metaso_search 返回的链接。"""
    if not url.startswith(("https://", "http://")):
        return "链接必须以 http:// 或 https:// 开头。"

    api_key = os.getenv("METASO_API_KEY")
    if not api_key:
        return "未配置 METASO_API_KEY，无法联网读取网页。"

    try:
        response = requests.post(
            "https://metaso.cn/api/v1/reader",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={"url": url, "output": "markdown"},
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        return "联网读取请求失败，请检查搜索配置和连接。"
    except ValueError:
        return "联网读取返回了无法解析的内容。"

    if data.get("errCode"):
        return f"秘塔网页读取失败：{data.get('errMsg', '未知错误')}"

    markdown = data.get("markdown")
    if not markdown:
        return "秘塔未返回网页正文。"

    return f"标题：{data.get('title', '无标题')}\n链接：{data.get('url', url)}\n\n{str(markdown)[:14000]}"

tools = [
    list_files,
    read_document,
    read_document_section,
    search_documents,
    search_game_knowledge,
    save_note,
    ocr_image,
    metaso_search,
    metaso_reader,
]

from game_agent.prompts import SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT, INTERNAL_MEMORY_PROMPT

model_with_tools = model.bind_tools(tools)

CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "12000"))
RECENT_USER_TURNS = int(os.getenv("RECENT_USER_TURNS", "4"))
SUMMARY_TOKEN_BUDGET = int(os.getenv("SUMMARY_TOKEN_BUDGET", "1500"))


class AgentState(TypedDict):
    # add_messages 会把每个节点返回的新消息追加到已有对话历史。
    messages: Annotated[list[BaseMessage], add_messages]


def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    labels = {
        "human": "用户",
        "ai": "助手",
        "tool": "工具",
    }
    parts = []
    for message in messages:
        message_type = getattr(message, "type", "消息")
        label = labels.get(message_type, message_type)
        content = message.content if isinstance(message.content, str) else str(message.content)
        parts.append(f"{label}：{content}")
    return "\n\n".join(parts)


def _summarize_messages(existing_summary: str, messages: list[BaseMessage]) -> str:
    summary_input = _format_messages_for_summary(messages)
    if existing_summary:
        summary_input = f"已有摘要：\n{existing_summary}\n\n新增历史：\n{summary_input}"
    if len(summary_input) > 16000:
        summary_input = summary_input[:16000] + "\n[旧历史过长，本次摘要输入已截断，未展示内容不得推断。]"
    response = model.bind(max_tokens=SUMMARY_TOKEN_BUDGET).with_config(tags=["internal_summary"]).invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=summary_input),
        ]
    )
    return str(response.content).strip()


def _model_context(
    messages: list[BaseMessage],
    thread_id: str,
) -> tuple[str, list[BaseMessage]]:
    stored_summary = context_store.get_summary(thread_id)
    covered_count = min(
        stored_summary.covered_message_count if stored_summary else 0,
        len(messages),
    )
    summary_text = stored_summary.summary if stored_summary else ""
    unsummarized_messages = messages[covered_count:]
    context_tokens = estimate_message_tokens(
        [SystemMessage(content=SYSTEM_PROMPT), *unsummarized_messages]
    )
    if summary_text:
        context_tokens += estimate_message_tokens([HumanMessage(content=summary_text)])

    if context_tokens > CONTEXT_TOKEN_BUDGET:
        split = split_complete_user_turns(
            messages,
            recent_user_turns=RECENT_USER_TURNS,
        )
        new_covered_count = len(split.messages_to_summarize)
        messages_to_summarize = messages[covered_count:new_covered_count]
        if messages_to_summarize:
            try:
                new_summary = _summarize_messages(summary_text, messages_to_summarize)
            except Exception:
                new_summary = ""
            if new_summary:
                summary_text = new_summary
                covered_count = new_covered_count
                context_store.save_summary(thread_id, summary_text, covered_count)

    recent = messages[covered_count:]
    if estimate_message_tokens([SystemMessage(content=SYSTEM_PROMPT), *recent]) > CONTEXT_TOKEN_BUDGET:
        # A failed summary must not send the entire history to the next request.
        recent = split_complete_user_turns(messages, recent_user_turns=RECENT_USER_TURNS).recent_messages
    return summary_text, recent


def call_model(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage]]:
    """让模型根据当前消息决定直接回答，还是请求工具调用。"""
    # 每次动态构造系统提示并附上当前日期时间：
    # 模型本身没有"时钟"，若不注入日期，问"今天是周几"就只能瞎猜或联网。
    # 代价是每次调用都拼一次 SYSTEM_PROMPT，但对本地个人 Agent 可接受。
    if not settings.chat_ready:
        raise RuntimeError("聊天模型尚未配置，请设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL；也可先使用离线检索演示。")
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if user_messages and len(str(user_messages[-1].content)) > settings.max_input_chars:
        raise ValueError("输入超过 6000 字符，请缩短问题。")
    # Count tool rounds in the current turn, rather than all persisted history.
    current = split_complete_user_turns(state["messages"], recent_user_turns=1).recent_messages
    calls = sum(bool(getattr(m, "tool_calls", [])) for m in current)
    if calls >= settings.max_tool_rounds:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="本轮已达到工具调用上限，尚未得到可靠答案。请缩小问题范围后重试。") ]}
    now = datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    system_content = (
        SYSTEM_PROMPT
        + f"\n\n当前日期时间：{now:%Y-%m-%d %H:%M}，星期{weekday}。"
    )
    thread_id = str(config.get("configurable", {}).get("thread_id", "default"))
    summary_text, recent_messages = _model_context(state["messages"], thread_id)
    model_messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    if summary_text:
        model_messages.append(
            SystemMessage(
                content=f"{INTERNAL_MEMORY_PROMPT}\n\n<internal_memory>\n{summary_text}\n</internal_memory>"
            )
        )
    response = model_with_tools.invoke(fit_model_context(recent_messages, fixed=model_messages,
                                                        token_budget=CONTEXT_TOKEN_BUDGET))
    if summary_text and isinstance(response.content, str):
        response = response.model_copy(
            update={"content": remove_internal_summary(response.content, summary_text)}
        )
    if not getattr(response, "tool_calls", []) and isinstance(response.content, str):
        text, unknown = validate_citations(response.content, recent_messages)
        if unknown:
            response = response.model_copy(update={"content": text})
    return {"messages": [response]}


def route_after_model(state: AgentState) -> Literal["tools", "__end__"]:
    """模型产生 tool_calls 时执行工具；没有工具请求时结束流程。"""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", []) else END


builder = StateGraph(AgentState)
builder.add_node("llm", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", route_after_model, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")

database_path = settings.database_path
database_path.parent.mkdir(parents=True, exist_ok=True)
context_store = ConversationStore(database_path)
connection = sqlite3.connect(database_path, check_same_thread=False)
checkpointer = SqliteSaver(connection)
checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)

# 工具提示语映射
TOOL_START_MSGS = {
    "list_files": "正在翻阅资料目录...",
    "read_document": "正在打开文档...",
    "read_document_section": "正在阅读指定段落...",
    "search_documents": "正在检索资料...",
    "search_game_knowledge": "正在检索游戏知识库...",
    "save_note": "正在记录笔记...",
    "ocr_image": "正在识别图片文字...",
    "metaso_search": "正在网上搜索...",
    "metaso_reader": "正在阅读网页...",
}
TOOL_END_MSGS = {
    "list_files": "目录列好了",
    "read_document": "内容拿到了",
    "read_document_section": "段落读完了",
    "search_documents": "搜到相关线索",
    "search_game_knowledge": "游戏知识检索完成",
    "save_note": "笔记已保存",
    "ocr_image": "文字识别好了",
    "metaso_search": "了然于心",
    "metaso_reader": "网页内容抓到了",
}
DEFAULT_START = "正在处理..."
DEFAULT_END = "处理完成"

def create_config(thread_id: str | None = None):
    """创建带 thread_id 的 config。"""
    if thread_id is None:
        thread_id = f"demo_{uuid4().hex}"
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.max_tool_rounds * 2 + 4}

if __name__ == "__main__":
    thread_id = "demo"
    config = create_config(thread_id)

    while True:
        user_input = input("😎：").strip()
        if user_input.lower() == "/clear":
            thread_id = f"demo_{uuid4().hex}"
            config = create_config(thread_id)
            print("会话已重置。")
            continue
        if user_input.lower() in {"exit", "quit", "退出"}:
            break
        if not user_input:
            continue

        print("🤖外装代脑：")
        shown_tool_calls = set()       # 记录已打印开始提示的 tool_call_id
        active_tool_calls = {}         # {tool_call_id: tool_name}
        streaming_text = False         # 是否正在流式输出模型文本

        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            # ---------- 处理模型文本内容（逐 token）----------
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                if not streaming_text:
                    streaming_text = True
                print(chunk.content, end="", flush=True)

            # ---------- 检测工具调用开始 ----------
            if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    tc_id = tc.get("id")
                    tc_name = tc.get("name", "")
                    if tc_id and tc_id not in shown_tool_calls:
                        shown_tool_calls.add(tc_id)
                        active_tool_calls[tc_id] = tc_name or "未知"
                        # 暂停文本流，换行显示工具提示
                        if streaming_text:
                            print()
                            streaming_text = False
                        start_msg = TOOL_START_MSGS.get(tc_name, DEFAULT_START)
                        print(f"🔧 {start_msg}")

            # ---------- 工具执行结束 ----------
            if isinstance(chunk, ToolMessage):
                tc_id = chunk.tool_call_id
                if tc_id in active_tool_calls:
                    tool_name = active_tool_calls.pop(tc_id)
                    end_msg = TOOL_END_MSGS.get(tool_name, DEFAULT_END)
                    if streaming_text:
                        print()
                        streaming_text = False
                    print(f"✨ {end_msg}")

        # 流结束，如果最后是文本内容，补充换行
        if streaming_text:
            print()
