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
WINDOWS_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
BINARY_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".xlsx", ".pptx"}
OFFICE_DOCUMENT_SUFFIXES = {".docx", ".xlsx", ".pptx"}


def _safe_print(value=""):
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(
            text.encode(encoding, errors="backslashreplace").decode(encoding),
            flush=True,
        )


def _result_to_text(result):
    content = getattr(result, "content", None)
    texts = [
        item.text
        for item in (content or [])
        if getattr(item, "text", None) is not None
    ]
    if texts:
        return "\n".join(texts)

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, default=str)
    return "" if content is not None else str(result)


def _failure_payload(server_name, tool_name, error):
    return json.dumps(
        {
            "server": server_name,
            "tool": tool_name,
            "status": "failed",
            "error": str(error),
        },
        ensure_ascii=False,
    )


def _local_path_from_file_uri(uri):
    parsed = urlparse(uri)
    path = url2pathname(unquote(parsed.path))
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        path = f"//{parsed.netloc}{path}"
    return Path(path)


def _normalise_document_uri(value, allowed_suffixes, tool_label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_label} requires a non-empty uri")

    uri = value.strip().strip('"')
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


def _prepare_arguments(server_name, tool_name, arguments):
    prepared = dict(arguments)
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


def _make_tool_coroutine(manager, server_name, tool_name):
    async def call_tool(**kwargs):
        started_at = time.perf_counter()
        _safe_print(
            f"\n[TOOL CALL] server={server_name} tool={tool_name} "
            f"arguments={json.dumps(kwargs, ensure_ascii=False, default=str)}"
        )
        try:
            arguments = _prepare_arguments(server_name, tool_name, kwargs)
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
                f"[TOOL RESULT] tool={tool_name} time={elapsed:.2f}s\n{text}"
            )
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

    return call_tool


async def load_mcp_tools(manager):
    mcp_tools = await manager.get_tools()
    tools = []
    names = set()

    for server_name, mcp_tool in mcp_tools:
        config = manager.servers[server_name]
        included = config.get("includeTools")
        if included and mcp_tool.name not in included:
            continue

        prefix = config.get("toolPrefix")
        name = f"{prefix}_{mcp_tool.name}" if prefix else mcp_tool.name
        if name in names:
            raise ValueError(f"Duplicate MCP tool name: {name}")
        names.add(name)

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

        tools.append(
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
