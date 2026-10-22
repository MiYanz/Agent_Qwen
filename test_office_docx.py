# Office Writer 的真实写入测试：在桌面生成 DOCX，再检查 ZIP 结构和文档内容。
# 此测试会产生用户可见文件，不能作为没有外部副作用的普通导入检查。
import asyncio
import os
from pathlib import Path
from zipfile import ZipFile

from docx import Document

from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


# 在用户桌面实际生成固定名称的 DOCX，再用 zipfile 与 python-docx 读取验证。
# 如果已存在同名文件可能被覆盖，重复运行前应确认允许写入这个固定桌面路径。
async def main():
    """创建真实 DOCX，并验证 ZIP 结构和可读文字。"""
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    # 固定输出名用于后续验证，也意味着重复运行可能覆盖同名桌面文件。
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
