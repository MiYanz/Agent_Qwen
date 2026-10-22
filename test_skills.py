# Skill 离线测试：在临时目录构造合法、损坏和重复的 SKILL.md。
# 明确的 PPT/新闻匹配路径不需要调用真实模型；测试还模拟发现后文件消失的情况。
import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.nodes import create_skill_node
from skills.loader import SkillLoader


PROJECT_SKILLS = Path(__file__).resolve().parent / "skills"
# 使用测试文件所在目录定位真实 Skill，避免依赖启动时的当前目录。


# 把测试文字或故意损坏的字节写入指定临时子目录，返回值不参与测试流程。
def _write_skill(directory, folder, text, raw=False):
    """在临时目录创建测试 Skill；raw=True 时允许写入非法编码字节。"""
    skill_dir = directory / folder
    skill_dir.mkdir()
    path = skill_dir / "SKILL.md"
    if raw:
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


# 混合合法与非法 Skill，断言坏文件被隔离、重名仅保留一个、路径式名称不能读取。
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
        # 这里故意制造无法按 UTF-8 解码的文件，期望隔离该文件而不是让全部 Skill 发现失败。
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


# 先发现再删除临时文件，验证节点返回空 Skill 而不是中断主图。
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
        result = await create_skill_node(loader, agent_name="file")(
            {
                "messages": [HumanMessage(content=r"总结 D:\demo.pptx")],
                "file_route": "office",
            }
        )
        assert result["active_skill"] == ""
        assert result["skill_instruction"] == ""
    print("SKILL_LOAD_FAILURE_ISOLATED=PASS")


# 用实际内置 Skill 的元数据验证明确 PPT/新闻规则，不请求真实 Ollama 选择。
async def test_obvious_matching():
    """验证 PPT 和新闻两个内置 Skill 的稳定匹配路径。"""
    loader = SkillLoader(PROJECT_SKILLS)
    loader.discover()
    file_node = create_skill_node(loader, agent_name="file")
    research_node = create_skill_node(loader, agent_name="research")

    ppt = await file_node(
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
    news = await research_node(
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


# 顺序执行同步发现测试和两个异步节点测试；临时文件生命周期由各测试自行管理。
async def main():
    test_discovery_and_failures()
    await test_load_failure_isolated()
    await test_obvious_matching()


if __name__ == "__main__":
    asyncio.run(main())

