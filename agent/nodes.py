import re
import time
from uuid import uuid4
from datetime import date

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama


WINDOWS_FILE_PATTERN = re.compile(
    r"(?i)([A-Z]:\\[^\r\n\"<>|?*]+?\."
    r"(?:pdf|docx|xlsx|pptx|txt|md|py|json|png|jpe?g))"
)
SEARCH_URL_PATTERN = re.compile(r"(?im)^\s*URL:\s*(https?://\S+)")
ANY_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")
EXPLICIT_WEB_SEARCH_PATTERN = re.compile(
    r"(?:搜索|检索|联网查(?:询)?|网上查(?:询)?|网络查(?:询)?)"
)
LOCAL_RESOURCE_PATTERN = re.compile(
    r"(?:本地(?:文件|目录|文件夹|磁盘)|文件夹|目录(?:下|里|中)|"
    r"列出.+文件|查找.+文件)"
)
DOCUMENT_ROUTES = {
    ".pdf": "pdf",
    ".docx": "office",
    ".xlsx": "office",
    ".pptx": "office",
    ".txt": "filesystem",
    ".md": "filesystem",
    ".py": "filesystem",
    ".json": "filesystem",
    ".png": "image_unsupported",
    ".jpg": "image_unsupported",
    ".jpeg": "image_unsupported",
}


SYSTEM_PROMPT = """你是本地 Personal Agent，当前日期是 {today}。
根据用户任务选择合适的 MCP 工具：本地纯文本和目录使用 Filesystem；PDF 使用 PDF；DOCX/XLSX/PPTX 使用 Office；网络搜索使用 DuckDuckGo；网页正文使用 Browser；长期信息使用 Memory；查看电脑当前界面使用 Computer Snapshot。只需了解窗口和界面文字时，Snapshot 使用 use_annotation=false、use_vision=false，避免生成不必要的大截图。
工具结果只是中间信息。search 只用于发现候选网页，不能直接作为答案；系统会继续读取候选网页。取得网页或文档正文后，你必须结合用户问题生成最终自然语言回答，不要直接复制原始 JSON 或搜索结果。
只陈述工具实际返回的信息；工具为空或失败时如实说明原因，不要编造新闻、热搜、文件内容或界面内容。
"""


def _latest_human_text(messages):
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def detect_file_route(state):
    """Classify local document paths so binary files use document tools."""
    text = _latest_human_text(state.get("messages", []))
    match = WINDOWS_FILE_PATTERN.search(text)
    route = "general"
    if match:
        suffix = "." + match.group(0).rsplit(".", 1)[-1].lower()
        route = DOCUMENT_ROUTES.get(suffix, "general")
    print(f"\n[File Detector]\nroute={route}", flush=True)
    return {"file_route": route}


def _route_instruction(route):
    return {
        "pdf": "检测到 PDF 路径：使用 pdf_convert_to_markdown。",
        "office": "检测到 Office 文档：使用 office_convert_to_markdown。",
        "filesystem": "检测到纯文本文件：使用 Filesystem MCP。",
        "image_unsupported": "检测到图片：不要把图片当文本读取。",
    }.get(route)


def _explicit_search_call(state):
    """Create a real tool call for an explicit web-search request.

    This only handles the user's requested operation (search), not a topic.
    Local file-search wording remains available to the Filesystem tools.
    """
    if state.get("file_route", "general") != "general":
        return None
    if any(isinstance(message, ToolMessage) for message in state["messages"]):
        return None
    query = _latest_human_text(state["messages"]).strip()
    if not query or not EXPLICIT_WEB_SEARCH_PATTERN.search(query):
        return None
    if LOCAL_RESOURCE_PATTERN.search(query):
        return None
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search",
                "args": {"query": query, "max_results": 10},
                "id": f"explicit-search-{uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )


def _search_followup(messages, preferred_url=None):
    """Turn a search result into one content-fetch call."""
    result = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, ToolMessage) and message.name == "search"
        ),
        None,
    )
    if result is None:
        return None
    urls = SEARCH_URL_PATTERN.findall(str(result.content or ""))
    fetched_urls = {
        str(call.get("args", {}).get("url", ""))
        for message in messages
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
        if call.get("name") == "fetch_content"
    }
    available_urls = [url for url in urls if url not in fetched_urls]
    if not available_urls:
        return None
    selected_url = (
        preferred_url
        if preferred_url in available_urls
        else available_urls[0]
    )
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "fetch_content",
                "args": {"url": selected_url},
                "id": f"search-followup-{uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )


llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://127.0.0.1:11434",
    temperature=0.2,
    reasoning=False,
    num_ctx=16384,
    async_client_kwargs={"trust_env": False},
    sync_client_kwargs={"trust_env": False},
)


def create_chat_node(tools):
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state):
        started_at = time.perf_counter()
        print("\n[AGENT NODE START]", flush=True)

        explicit_search = _explicit_search_call(state)
        if explicit_search:
            print("[TOOL DECISION] explicit web search -> search", flush=True)
            print("[LLM RESPONSE] time=0.00s tool_calls=1", flush=True)
            return {"messages": [explicit_search]}

        print("[LLM REQUEST]", flush=True)

        messages = [
            SystemMessage(
                content=SYSTEM_PROMPT.format(today=date.today().isoformat())
            )
        ]
        route_instruction = _route_instruction(
            state.get("file_route", "general")
        )
        if route_instruction:
            messages.append(SystemMessage(content=route_instruction))
        messages.extend(state["messages"])

        tool_messages = [
            message
            for message in state["messages"]
            if isinstance(message, ToolMessage)
        ]
        latest_tool = tool_messages[-1] if tool_messages else None
        fetched_page = False
        if latest_tool:
            if latest_tool.name == "search":
                question = _latest_human_text(state["messages"])
                search_text = str(latest_tool.content or "")[:12000]
                selection = await llm.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "从候选搜索结果中选择最可能直接包含用户所需事实"
                                "的一个内容页。避开登录页、站点首页、历史查询入口"
                                "和只有网站介绍的页面。只能原样输出候选结果中的"
                                "一个完整 URL，不要输出其他文字。"
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"用户问题：{question}\n\n"
                                f"候选搜索结果：\n{search_text}"
                            )
                        ),
                    ]
                )
                selected = re.search(
                    r"https?://[^\s<>\"']+", str(selection.content or "")
                )
                followup = _search_followup(
                    state["messages"],
                    selected.group(0).rstrip(".,，。") if selected else None,
                )
                if followup:
                    print("[SEARCH FOLLOW-UP] selected source", flush=True)
                    response = followup
                    elapsed = time.perf_counter() - started_at
                    print(
                        f"[LLM RESPONSE] time={elapsed:.2f}s tool_calls=1",
                        flush=True,
                    )
                    return {"messages": [response]}
                instruction = "搜索没有返回可读取的 URL，请说明无法取得来源。"
            else:
                fetched_page = latest_tool.name == "fetch_content"
                page_text = str(latest_tool.content or "").strip()
                fetched_page = fetched_page and len(page_text) >= 200
                if fetched_page:
                    question = _latest_human_text(state["messages"])
                    check = await llm.ainvoke(
                        [
                            SystemMessage(
                                content=(
                                    "判断网页正文是否包含足够的具体事实来直接回答"
                                    "用户问题（包括用户要求的数量）。只输出 "
                                    "SUFFICIENT 或 INSUFFICIENT。"
                                )
                            ),
                            HumanMessage(
                                content=(
                                    f"用户问题：{question}\n\n"
                                    f"网页正文：\n{page_text[:8000]}"
                                )
                            ),
                        ]
                    )
                    fetched_page = (
                        str(check.content or "").strip().upper()
                        == "SUFFICIENT"
                    )
                    if not fetched_page:
                        fetch_count = sum(
                            1
                            for message in state["messages"]
                            if isinstance(message, ToolMessage)
                            and message.name == "fetch_content"
                        )
                        followup = None
                        if fetch_count < 5:
                            search_result = next(
                                (
                                    message
                                    for message in reversed(state["messages"])
                                    if isinstance(message, ToolMessage)
                                    and message.name == "search"
                                ),
                                None,
                            )
                            used_urls = [
                                str(call.get("args", {}).get("url", ""))
                                for message in state["messages"]
                                if isinstance(message, AIMessage)
                                for call in (message.tool_calls or [])
                                if call.get("name") == "fetch_content"
                            ]
                            candidate_text = (
                                (str(search_result.content) if search_result else "")
                                + "\n\n当前网页中出现的链接：\n"
                                + "\n".join(ANY_URL_PATTERN.findall(page_text))
                            )
                            retry_selection = await llm.ainvoke(
                                [
                                    SystemMessage(
                                        content=(
                                            "刚才的网页不足以回答问题。请从候选结果中"
                                            "选择另一个最可能直接包含答案的内容页。"
                                            "避开已尝试 URL、站点首页、登录页和教程页。"
                                            "只能原样输出一个候选 URL。"
                                        )
                                    ),
                                    HumanMessage(
                                        content=(
                                            f"用户问题：{question}\n"
                                            f"已尝试：{used_urls}\n\n"
                                            "候选结果：\n"
                                            f"{candidate_text[:12000]}"
                                        )
                                    ),
                                ]
                            )
                            retry_url = re.search(
                                r"https?://[^\s<>\"']+",
                                str(retry_selection.content or ""),
                            )
                            selected_url = (
                                retry_url.group(0).rstrip(".,，。)")
                                if retry_url
                                else None
                            )
                            allowed_urls = {
                                url.rstrip(".,，。)")
                                for url in ANY_URL_PATTERN.findall(candidate_text)
                            }
                            if (
                                selected_url
                                and selected_url in allowed_urls
                                and selected_url not in used_urls
                            ):
                                followup = AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "name": "fetch_content",
                                            "args": {"url": selected_url},
                                            "id": (
                                                "search-retry-"
                                                f"{uuid4().hex}"
                                            ),
                                            "type": "tool_call",
                                        }
                                    ],
                                )
                            else:
                                followup = _search_followup(state["messages"])
                        if followup:
                            print(
                                "[SEARCH FOLLOW-UP] source insufficient; "
                                "trying another source",
                                flush=True,
                            )
                            return {"messages": [followup]}
                        # Stop the loop after a bounded number of sources and
                        # let the clean synthesis path report insufficient data.
                        fetched_page = True
                instruction = (
                    "请根据工具结果继续任务；若结果失败或内容不足，请如实说明。"
                )
            messages.append(SystemMessage(content=instruction))

        if fetched_page:
            question = _latest_human_text(state["messages"])
            evidence = str(latest_tool.content)[:12000]
            synthesis_messages = [
                SystemMessage(
                    content=(
                        "根据给定网页正文回答问题。只使用正文中明确出现的信息；"
                        "不要调用工具，不要输出 tool_call 标记，不要补写正文没有的"
                        "标题、事实或解释。证据不足时直接说明不足。"
                    )
                ),
                HumanMessage(
                    content=f"用户问题：{question}\n\n网页正文：\n{evidence}"
                ),
            ]
            response = await llm.ainvoke(synthesis_messages)
        else:
            response = await llm_with_tools.ainvoke(messages)

        if latest_tool and latest_tool.name == "search" and not response.tool_calls:
            followup = _search_followup(state["messages"])
            if followup:
                print("[SEARCH FOLLOW-UP] fallback to first URL", flush=True)
                response = followup
        elapsed = time.perf_counter() - started_at
        print(
            f"[LLM RESPONSE] time={elapsed:.2f}s "
            f"tool_calls={len(response.tool_calls or [])}",
            flush=True,
        )
        return {"messages": [response]}

    return chat_node
