# 真实 MCP 连续调用测试：搜索、读取网页、转换指定 PDF，并检查缺失文件的错误结果。
# 默认 PDF 是用户磁盘中的现有文件；运行前需确认路径和网络环境适合测试。
import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH, load_mcp_tools


DEFAULT_PDF = Path(r"D:\BaiduSyncdisk\Lab\02-晨会\2026.10.13.pdf")
# 默认使用真实本地 PDF，也可以通过 --pdf 参数替换测试文件。


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


# pdf_path 来自命令行或默认真实路径；连续搜索、抓网页并转换该 PDF。
# 最后用不存在的同目录文件检查失败 JSON；这一测试不是纯离线或纯临时数据测试。
async def main(pdf_path):
    """验证联网搜索、网页正文获取、PDF 转换和错误返回结构。"""
    async with MCPManager(CONFIG_PATH) as manager:
        tools = {tool.name: tool for tool in await load_mcp_tools(manager)}

        # 同一个 Manager 连续调用多次，检查长生命周期 Session 的稳定性。
        for query in (
            "OpenAI latest news",
            "今天中国新闻",
            "今天的新浪微博热搜",
        ):
            result = await tools["search"].ainvoke(
                {"query": query, "max_results": 3}
            )
            print(f"\n[SEARCH] {query}\n{result}")
            assert '"status": "failed"' not in result, result
            assert "URL:" in result, result

        fetched = await tools["fetch_content"].ainvoke(
            {"url": "https://example.com", "max_length": 1000}
        )
        print(f"\n[FETCH]\n{fetched}")
        assert '"status": "failed"' not in fetched, fetched

        converted = await tools["pdf_convert_to_markdown"].ainvoke(
            {"uri": str(pdf_path)}
        )
        print(f"\n[PDF] {pdf_path}\n{converted[:1000]}")
        assert '"status": "failed"' not in converted, converted
        assert converted.strip(), "PDF conversion returned empty content."

        missing = await tools["pdf_convert_to_markdown"].ainvoke(
            {"uri": str(pdf_path.with_name("missing-test-file.pdf"))}
        )
        # 这组断言按测试当前约定检查错误字段；它不是生产代码里自动补齐字段的逻辑。
        error = json.loads(missing)
        assert error["status"] == "failed"
        assert {
            "server",
            "tool",
            "status",
            "error",
            "suggestion",
        } <= error.keys()
        print(f"\n[EXPECTED ERROR]\n{missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()
    asyncio.run(main(args.pdf))
