import asyncio
import os
from pathlib import Path
from zipfile import ZipFile

from docx import Document

from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


async def main():
    """创建真实 DOCX，并验证 ZIP 结构和可读文字。"""
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    output_path = desktop / "Local-Personal-Agent-Office-MCP-Test.docx"

    async with MCPManager(CONFIG_PATH) as manager:
        result = await manager.call_tool(
            "create_docx",
            {
                "output_path": str(output_path),
                "title": "Local Personal Agent",
                "paragraphs": ["测试文档", "这是Office MCP生成测试。"],
                "numbered_list": ["Test", "MCP", "Agent"],
            },
            server_name="office-writer",
        )
    print(result)

    # 先检查文件存在和 ZIP 完整性，再使用 python-docx 做高层解析验证。
    assert output_path.exists() and output_path.stat().st_size > 0
    assert ZipFile(output_path).testzip() is None
    with ZipFile(output_path) as package:
        assert "[Content_Types].xml" in package.namelist()
        assert "word/document.xml" in package.namelist()

    document = Document(output_path)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    for expected in ("Local Personal Agent", "测试文档", "Test", "MCP", "Agent"):
        assert expected in text
    print(f"Verified DOCX: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
