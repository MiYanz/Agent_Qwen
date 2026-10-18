import re
from dataclasses import dataclass
from pathlib import Path


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# 限制名称格式既方便模型准确输出，也避免把路径分隔符混入 Skill 名称。


@dataclass(frozen=True)
class SkillInfo:
    """Skill 元数据；启动时只保存名称、描述和受控文件路径。"""

    # @dataclass 自动生成初始化和比较方法；frozen=True 防止发现后被意外改写。

    name: str
    description: str
    path: Path


class SkillLoader:
    """发现并按需读取项目 skills 目录中的 SKILL.md。"""

    def __init__(self, skills_dir):
        # resolve() 转成绝对路径，后续可以可靠判断 SKILL.md 是否仍位于该目录内。
        self.skills_dir = Path(skills_dir).resolve()
        # 字典以 Skill name 为键，可以直接检查重名并按名称加载。
        self._skills = {}

    def discover(self):
        """扫描直接子目录；单个 Skill 损坏不会影响其他 Skill。"""
        discovered = {}
        if not self.skills_dir.is_dir():
            print(
                f"[Skill Loader] directory not found: {self.skills_dir}",
                flush=True,
            )
            self._skills = discovered
            return list(discovered.values())

        # glob("*/SKILL.md") 只扫描一层直接子目录，不会在任意深度递归查找。
        for skill_file in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                text = skill_file.read_text(encoding="utf-8")
                metadata = self._parse_frontmatter(text)
                name = metadata.get("name", "").strip()
                description = metadata.get("description", "").strip()
                if not SKILL_NAME_PATTERN.fullmatch(name):
                    raise ValueError(
                        "name must use lowercase letters, numbers, or hyphens"
                    )
                if not description:
                    raise ValueError("description is required")
                if name in discovered:
                    raise ValueError(f"duplicate skill name: {name}")
                discovered[name] = SkillInfo(
                    name=name,
                    description=description,
                    path=skill_file.resolve(),
                )
                print(f"[Skill Loader] discovered: {name}", flush=True)
            # 单个文件的路径、编码或格式错误只记录日志，不中断其他 Skill。
            except (OSError, UnicodeError, ValueError) as exc:
                print(
                    f"[Skill Loader] skipped {skill_file}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        self._skills = discovered
        return list(discovered.values())

    @staticmethod
    def _parse_frontmatter(text):
        """解析仅包含简单 key: value 的 YAML 风格头部，不引入新依赖。"""
        # staticmethod 不接收 self：解析过程只依赖传入文本，与 Loader 实例状态无关。
        lines = text.lstrip("\ufeff").splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("missing opening frontmatter delimiter")
        try:
            # next() 从生成器表达式中取得第一个结束分隔符的位置。
            end = next(
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            )
        except StopIteration as exc:
            raise ValueError("missing closing frontmatter delimiter") from exc

        metadata = {}
        for line in lines[1:end]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"invalid metadata line: {line}")
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
        return metadata

    def catalog(self):
        """返回供模型选择的精简目录，不包含 Skill 正文。"""
        # 列表推导式只暴露名称和描述，避免每轮把全部操作说明放进上下文。
        return [
            {"name": skill.name, "description": skill.description}
            for skill in self._skills.values()
        ]

    def load(self, name):
        """只允许读取已经发现且仍位于 skills 根目录内的文件。"""
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"Skill 不存在: {name}")

        # 读取前再次 resolve 并检查父目录，防止文件被替换为指向目录外的路径。
        path = skill.path.resolve()
        if self.skills_dir not in path.parents or path.name != "SKILL.md":
            raise ValueError(f"Skill 路径越界: {path}")
        return path.read_text(encoding="utf-8")

