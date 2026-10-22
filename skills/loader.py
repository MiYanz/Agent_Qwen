# Skill 文件加载层：启动时读文件建立名称和描述目录，命中任务后再加载完整说明用于提示。
# 返回的 Skill 元数据供 nodes.py 选择，正文则作为该轮模型的额外操作说明。
import re
from dataclasses import dataclass
from pathlib import Path


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# 限制名称格式既方便模型准确输出，也避免把路径分隔符混入 Skill 名称。


# 保存 name/description 与源文件路径，供发现结果和按需读取复用；不保存完整正文。
@dataclass(frozen=True)
class SkillInfo:
    """Skill 元数据；启动时只保存名称、描述和受控文件路径。"""

    # @dataclass 自动生成初始化和比较方法；frozen=True 防止发现后被意外改写。

    name: str
    description: str
    path: Path


# 管理某个根目录下的 Skill 目录缓存，供主图 skill_detector 使用。
# 发现与加载分离，可以先提供小目录给模型，再只注入选中的一份完整说明。
class SkillLoader:
    """发现并按需读取项目 skills 目录中的 SKILL.md。"""

    # 告诉程序去哪里寻找 Skill，并准备一个字典保存发现的 Skill。
    # 把根路径转为绝对路径并创建空目录缓存，防止启动目录不同影响后续定位。
    def __init__(self, skills_dir):
        # resolve() 转成绝对路径，后续可以可靠判断 SKILL.md 是否仍位于该目录内。
        self.skills_dir = Path(skills_dir).resolve()
        # 字典以 Skill name 为键，可以直接检查重名并按名称加载。
        self._skills = {}

    # 扫描 skills 目录下的直接子目录，寻找其中的 SKILL.md 文件。
    # 扫描一层子目录，隔离单个坏文件；每次重新构建缓存，返回已接受的 SkillInfo 列表。
    def discover(self):
        """扫描直接子目录；单个 Skill 损坏不会影响其他 Skill。"""
        # 先构建新的目录结果，扫描完成再替换缓存，避免复用上一次已经删除的 Skill 条目。
        discovered = {}
        if not self.skills_dir.is_dir():
            print(
                f"[Skill Loader] directory not found: {self.skills_dir}",
                flush=True,
            )
            self._skills = discovered
            return list(discovered.values())

        # glob("*/SKILL.md") 只扫描一层直接子目录，不会在任意深度递归查找。
        # 排序使重名处理结果稳定；同名 Skill 后遇到的那份会被当作单个坏文件跳过。
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

    # 从 SKILL.md 中提取 Skill 的名称、描述等元数据。
    # 只解析受支持的简单 key: value 头部，不是通用 YAML 解析器。
    # 缺少起止分隔符或某行不含冒号会报错，再由 discover 隔离该文件。
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

        # 只支持每行一个冒号分隔的键值；split 的 1 保留值内后续冒号，不尝试完整 YAML 语法。
        metadata = {}
        for line in lines[1:end]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"invalid metadata line: {line}")
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
        return metadata

    # 把已经发现的 Skill 整理成一份精简目录，供大模型判断当前任务适合哪个 Skill。
    # 为模型生成新的名称/描述字典列表，隐藏路径与正文，降低选择阶段的上下文开销。
    def catalog(self):
        """返回供模型选择的精简目录，不包含 Skill 正文。"""
        # 列表推导式只暴露名称和描述，避免每轮把全部操作说明放进上下文。
        return [
            {"name": skill.name, "description": skill.description}
            for skill in self._skills.values()
        ]

    # 读取指定 Skill 的完整内容
    # 按已发现名称读取完整 UTF-8 文件；未知名称拒绝，读取前再次验证解析后的路径。
    # 文件被删除、改名或替换成越界链接时向调用方抛出错误，由图节点决定如何降级。
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

