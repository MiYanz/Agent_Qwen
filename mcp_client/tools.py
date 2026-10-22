# 适配层：将 MCP 的工具说明和 JSON Schema 转换成 LangChain StructuredTool。
# 模型只看到工具定义；工具真正执行时，闭包把请求交给已有 MCPManager，并把返回内容转成文字。
"""把 MCP 工具包装成 LangChain 工具，保留参数检查、结果转换和异常处理。"""


import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from langchain_core.tools import StructuredTool


CONFIG_PATH = Path(__file__).with_name("config.json")
# 配置路径相对于当前模块定位，不依赖用户从哪个目录启动程序。
WINDOWS_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
BINARY_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".xlsx", ".pptx"}
OFFICE_DOCUMENT_SUFFIXES = {".docx", ".xlsx", ".pptx"}


# 将输出转换为当前终端可表示的文字；编码失败时用反斜杠转义显示。
# 这是日志可读性处理，不改变随后返回给模型的工具内容。
def _safe_print(value=""):
    """安全地打印日志，尽可能保证日志能够正常输出，避免因为终端编码问题导致程序报错。"""
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(
            text.encode(encoding, errors="backslashreplace").decode(encoding),
            flush=True,
        )


# 优先提取 MCP 文本块，再尝试结构化字段；图片块不会被当作文本加入上下文。
# 返回字符串供工具包装器处理，尚未构造成 LangGraph 消息。
def _result_to_text(result):
    """把 MCP 返回的不同形式的数据统一转换成字符串，让大模型能够读取。"""
    # getattr 提供默认值，兼容不同 MCP SDK 版本中可能不存在的属性。
    content = getattr(result, "content", None)
    # 列表推导式只收集带 text 属性的内容块，图片等非文本块不会混入模型上下文。
    texts = [
        item.text
        for item in (content or [])
        if getattr(item, "text", None) is not None
    ]
    if texts:
        return "\n".join(texts)

    # 兼容 SDK 不同命名约定；优先使用真实字段，不假定所有服务都只返回 content 文本块。
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, default=str)
    return "" if content is not None else str(result)


# 把服务、工具和失败原因编码成统一 JSON 字符串，让模型及子 Agent 能识别 failed。
def _failure_payload(server_name, tool_name, error):
    """统一生成失败结果，当工具执行失败时，将错误整理成统一的 JSON 字符串。"""
    return json.dumps(
        {
            "server": server_name,
            "tool": tool_name,
            "status": "failed",
            "error": str(error),
        },
        ensure_ascii=False,
    )


# 根据实际错误原因分类网页失败，并删除 DNS 失败后的误导性通用私网说明。
# 这里只改善错误数据，不会重试网络请求或放宽地址权限。
def _web_failure_payload(text):
    """上游将 DNS 失败也包装为 Refusing；按真实原因分类，去除误导性私网说明。"""
    lowered = text.lower()
    if any(marker in lowered for marker in (
        "could not resolve host", "getaddrinfo", "name or service not known",
        "temporary failure in name resolution", "dns-over-https failed",
    )):
        code = "dns_resolution_failed"
        error = "DNS 解析失败，目标地址校验未完成，尚未取得网页正文。无法据此判断远端是否拒绝请求或目标是否为私网地址。"
    elif "non-public address" in lowered:
        code = "non_public_address_blocked"
        error = "目标解析到非公网地址，已由原有 SSRF 安全策略阻止访问。"
    else:
        code = "web_fetch_failed"
        error = text[:360]
    payload = json.loads(_failure_payload("duckduckgo-search", "fetch_content", error))
    payload["error_code"] = code
    return json.dumps(payload, ensure_ascii=False)


# 将 file URI 解码成本机路径，netloc 非空时兼容 UNC 形式；返回 Path，暂不读取内容。
def _local_path_from_file_uri(uri):
    """将文件 URI 转换成 Python 可以使用的本地路径，并兼容 UNC 网络路径。"""
    parsed = urlparse(uri)
    path = url2pathname(unquote(parsed.path))
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        path = f"//{parsed.netloc}{path}"
    return Path(path)


# 本地输入会检查存在性、扩展名和可读性，再转换成绝对 file URI。
# http/https/data 地址直接交还转换工具；此函数并不检查这些远程地址的文件类型或公网属性。
def _normalise_document_uri(value, allowed_suffixes, tool_label):
    """检查传入的文档地址是否合法；如果是本地文件路径，就验证文件后将它转换成 file:// URI。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_label} requires a non-empty uri")

    uri = value.strip().strip('"')
    # Windows 盘符路径与 URI 语法不同，先识别盘符，避免把 D: 当作远程协议。
    if WINDOWS_PATH_PATTERN.match(uri):
        local_path = Path(uri)
    elif uri.lower().startswith("file:"):
        local_path = _local_path_from_file_uri(uri)
    elif urlparse(uri).scheme in {"http", "https", "data"}:
        return uri
    else:
        raise ValueError(f"{tool_label} requires a local path or URL")

    if not local_path.is_file():
        raise FileNotFoundError(f"文件不存在: {local_path}")
    if local_path.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"文件类型不匹配: {local_path}")
    if not os.access(local_path, os.R_OK):
        raise PermissionError(f"文件不可读: {local_path}")
    return local_path.resolve().as_uri()


# 复制参数后按实际服务及原始工具名执行专门校验，避免修改模型提供的原字典。
# 普通文件读取拒绝二进制文档；PDF/Office 转换使用上面的路径规范化流程。
def _prepare_arguments(server_name, tool_name, arguments):
    """根据当前要调用的 MCP Server 和工具，对参数进行必要的校验和转换。"""
    # dict() 创建浅拷贝，包装层的路径转换不会直接修改调用方传入的字典。
    prepared = dict(arguments)
    if server_name == "filesystem":
        # 网页地址不能交给本地文件服务；也拦截模型误拼成 D:\https:\host 的形式。
        # 原有 MCP 根目录权限和二进制文件校验仍会继续执行。
        candidates = [prepared.get(key) for key in ("path", "source", "destination")]
        candidates.extend(prepared.get("paths", []) if isinstance(prepared.get("paths"), list) else [])
        for candidate in candidates:
            if candidate and re.search(r"(?:^|[\\/])(?:https?|ftp):[\\/]", str(candidate), re.IGNORECASE):
                raise ValueError("Filesystem 只接受本地路径，网页地址应委托 Research Agent")
    if server_name == "filesystem" and tool_name in {
        "read_file",
        "read_text_file",
        "read_multiple_files",
    }:
        candidates = prepared.get("paths") or [prepared.get("path")]
        if isinstance(candidates, str):
            candidates = [candidates]
        for candidate in candidates:
            if (
                candidate
                and Path(str(candidate)).suffix.lower()
                in BINARY_DOCUMENT_SUFFIXES
            ):
                raise ValueError(
                    "二进制文档不能作为文本读取，请使用 PDF 或 Office MCP"
                )

    if server_name == "pdf" and tool_name == "convert_to_markdown":
        prepared["uri"] = _normalise_document_uri(
            prepared.get("uri"), {".pdf"}, "pdf_convert_to_markdown"
        )
    if server_name == "office" and tool_name == "convert_to_markdown":
        prepared["uri"] = _normalise_document_uri(
            prepared.get("uri"),
            OFFICE_DOCUMENT_SUFFIXES,
            "office_convert_to_markdown",
        )
    return prepared


# 闭包固定 manager、服务及工具名，生成 StructuredTool 真正执行时 await 的协程函数。
def _make_tool_coroutine(manager, server_name, tool_name):
    """为某个 MCP 工具创建调用函数，LangChain 需要一个可执行的函数来代表每个工具，而项目可能有几十个不同的 MCP 工具。如果每个工具都手写一份调用代码，会非常冗余。"""

    # kwargs 来自工具调用；依次校验参数、请求 MCP、转换结果并整理失败信息。
    # 日志只记录参数键和结果长度；返回的是工具内容，ToolMessage 由外层 ToolNode 创建。
    async def call_tool(**kwargs):
        # **kwargs 接收 StructuredTool 按 Schema 解析后的关键字参数。
        started_at = time.perf_counter()
        _safe_print(
            f"\n[TOOL CALL] server={server_name} tool={tool_name} "
            f"argument_keys={sorted(kwargs)}"
        )
        try:
            arguments = _prepare_arguments(server_name, tool_name, kwargs)
            # await 等待 MCP 返回；此处不会使用 asyncio.run 嵌套事件循环。
            result = await manager.call_tool(
                tool_name, arguments, server_name=server_name
            )
            text = _result_to_text(result)
            if (
                server_name == "computer"
                and tool_name == "Snapshot"
                and len(text) > 8000
            ):
                text = text[:8000] + "\n...[Snapshot output truncated]"
            elapsed = time.perf_counter() - started_at
            _safe_print(
                f"[TOOL RESULT] tool={tool_name} time={elapsed:.2f}s chars={len(text)}"
            )
            if (server_name == "duckduckgo-search" and tool_name == "fetch_content"
                    and (text.lstrip().startswith("Error:") or getattr(result, "isError", False)
                         or getattr(result, "is_error", False))):
                return _web_failure_payload(text)
            # 不同 SDK 版本可能使用 isError 或 is_error，因此同时兼容两种命名。
            if getattr(result, "isError", False) or getattr(
                result, "is_error", False
            ):
                return _failure_payload(server_name, tool_name, text)
            if not text.strip():
                return _failure_payload(
                    server_name, tool_name, "工具没有返回内容"
                )
            return text
        except Exception as exc:
            traceback.print_exc()
            return _failure_payload(
                server_name,
                tool_name,
                f"{type(exc).__name__}: {exc}",
            )

    # 返回协程函数本身，稍后由 ToolNode 调用，而不是在加载阶段执行。
    return call_tool


# 从管理器的已发现工具构建 LangChain 列表，应用 includeTools 和展示名前缀。
# Memory 不进入通用工具列表；metadata 保留真实来源供专业 Agent 做白名单筛选。
async def load_mcp_tools(manager):
    """发现并加载所有可用工具"""
    mcp_tools = await manager.get_tools()
    tools = []
    names = set()

    for server_name, mcp_tool in mcp_tools:
        # Memory MCP 由 MemoryManager 统一调用，避免模型绕过去重、敏感信息和删除确认策略。
        # Server 仍由 MCPManager 启动并保持连接，只是不作为通用 Tool 暴露给 LLM。
        # MemoryManager 承担长期记忆的去重和确认策略，因此不能把原始 Memory 工具暴露给任意模型。
        if server_name == "memory":
            continue
        config = manager.servers[server_name]
        included = config.get("includeTools")
        if included and mcp_tool.name not in included:
            continue

        prefix = config.get("toolPrefix")
        name = f"{prefix}_{mcp_tool.name}" if prefix else mcp_tool.name
        # 展示名必须唯一，否则模型工具请求无法明确对应到正确的服务包装器。
        if name in names:
            raise ValueError(f"Duplicate MCP tool name: {name}")
        names.add(name)

        # MCP SDK 不同版本的 Schema 属性命名不同，这里做兼容读取。
        schema = getattr(mcp_tool, "input_schema", None)
        if schema is None:
            schema = getattr(mcp_tool, "inputSchema", None)
        description = " ".join(
            part
            for part in (
                f"[MCP server: {server_name}]",
                config.get("descriptionPrefix", ""),
                mcp_tool.description or "",
            )
            if part
        )
        if server_name == "duckduckgo-search" and mcp_tool.name == "search":
            description += (
                " Returns candidate URLs for discovery. The agent will read a "
                "candidate page before answering."
            )
        if server_name == "duckduckgo-search" and mcp_tool.name == "fetch_content":
            description += " Reads the actual content of a discovered URL."
        if server_name == "pdf" and mcp_tool.name == "convert_to_markdown":
            description += (
                " Accepts an absolute Windows PDF path in the uri argument; "
                "the wrapper converts it to a file URI."
            )
        if server_name == "office" and mcp_tool.name == "convert_to_markdown":
            description += (
                " Accepts an absolute Windows DOCX, XLSX or PPTX path in the "
                "uri argument; the wrapper converts it to a file URI."
            )

        # coroutine 参数声明这是异步工具，ToolNode 会 await 它而不是同步调用。
        tools.append(
            # coroutine 指向异步闭包，args_schema 描述请求结构，metadata 记录来源。
            # 这一步只创建工具定义；真正的调用发生在图中的 ToolNode。
            StructuredTool.from_function(
                coroutine=_make_tool_coroutine(
                    manager, server_name, mcp_tool.name
                ),
                name=name,
                description=description,
                args_schema=schema
                or {"type": "object", "properties": {}},
                metadata={
                    "mcp_server": server_name,
                    "mcp_tool": mcp_tool.name,
                },
            )
        )
    return tools
