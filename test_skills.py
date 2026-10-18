import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.nodes import create_skill_node
from skills.loader import SkillLoader


PROJECT_SKILLS = Path(__file__).resolve().parent / "skills"
# 使用测试文件所在目录定位真实 Skill，避免依赖启动时的当前目录。


def _write_skill(directory, folder, text, raw=False):
    """在临时目录创建测试 Skill；raw=True 时允许写入非法编码字节。"""
    skill_dir = directory / folder
    skill_dir.mkdir()
    path = skill_dir / "SKILL.md"
    if raw:
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def test_discovery_and_failures():
    """确认合法 Skill 可发现，而单个格式、重名或编码错误会被隔离。"""
    # TemporaryDirectory 离开 with 块时自动清理测试目录和其中的文件。
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _write_skill(
            root,
            "valid",
            "---\nname: valid\ndescription: valid skill\n---\nBody",
        )
        _write_skill(root, "broken", "No frontmatter")
        _write_skill(
            root,
            "missing-description",
            "---\nname: missing-description\n---\nBody",
        )
        _write_skill(
            root,
            "invalid-name",
            "---\nname: ../outside\ndescription: invalid\n---\nBody",
        )
        _write_skill(
            root,
            "duplicate-a",
            "---\nname: duplicate\ndescription: first\n---\nA",
        )
        _write_skill(
            root,
            "duplicate-b",
            "---\nname: duplicate\ndescription: second\n---\nB",
        )
        _write_skill(root, "bad-encoding", b"\xff\xfe\x00", raw=True)

        loader = SkillLoader(root)
        loader.discover()
        names = {item["name"] for item in loader.catalog()}
        assert names == {"valid", "duplicate"}
        assert loader.load("valid").endswith("Body")

        try:
            loader.load("../outside")
        except KeyError:
            pass
        else:
            raise AssertionError("unknown/path-traversal skill was accepted")

    print("SKILL_DISCOVERY_AND_FAILURES=PASS")


async def test_load_failure_isolated():
    """模拟发现后文件被删除，确认加载失败不会中断 Agent。"""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _write_skill(
            root,
            "ppt",
            (
                "---\nname: ppt-summary\n"
                "description: summarize pptx\n---\nBody"
            ),
        )
        loader = SkillLoader(root)
        loader.discover()
        (root / "ppt" / "SKILL.md").unlink()
        result = await create_skill_node(loader)(
            {
                "messages": [HumanMessage(content=r"总结 D:\demo.pptx")],
                "file_route": "office",
            }
        )
        assert result["active_skill"] == ""
        assert result["skill_instruction"] == ""
    print("SKILL_LOAD_FAILURE_ISOLATED=PASS")


async def test_obvious_matching():
    """验证 PPT 和新闻两个内置 Skill 的稳定匹配路径。"""
    loader = SkillLoader(PROJECT_SKILLS)
    loader.discover()
    node = create_skill_node(loader)

    ppt = await node(
        {
            "messages": [
                HumanMessage(
                    content=(
                        r"总结 D:\BaiduSyncdisk\Lab\02-晨会\2026.9.7.pptx"
                    )
                )
            ],
            "file_route": "office",
        }
    )
    news = await node(
        {
            "messages": [HumanMessage(content="搜索今天10条重要新闻")],
            "file_route": "general",
        }
    )
    assert ppt["active_skill"] == "ppt-summary"
    assert "office_convert_to_markdown" in ppt["skill_instruction"]
    assert news["active_skill"] == "news-search"
    assert "fetch_content" in news["skill_instruction"]
    print("SKILL_MATCHING=PASS")


async def main():
    test_discovery_and_failures()
    await test_load_failure_isolated()
    await test_obvious_matching()


if __name__ == "__main__":
    asyncio.run(main())

