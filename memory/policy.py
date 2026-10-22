# 纯文本记忆策略：识别显式命令，并保守判断哪些用户内容适合自动保存。
# 这里不调用模型、不读写记忆文件；返回候选数据，交给 MemoryManager 决定后续操作。
import hashlib
import re

from memory.models import MemoryCandidate, MemoryCommand


# 这些模式在模块导入时预编译，后续每轮只做匹配；规则保守但不是完整的语义分类器。
SENSITIVE_PATTERN = re.compile(
    r"(?:密码|口令|验证码|身份证|银行卡|信用卡|私钥|助记词|"
    r"api[_ -]?key|access[_ -]?token|secret|password)",
    re.IGNORECASE,
)
EXPLICIT_SAVE_PATTERN = re.compile(
    r"^(?:请)?记住[，,:：\s]*(.+)$", re.DOTALL
)
CONFIRMED_SENSITIVE_PATTERN = re.compile(
    r"^确认保存敏感信息[，,:：\s]*(.+)$", re.DOTALL
)
LIST_PATTERN = re.compile(
    r"(?:你记住了哪些|查看|列出|显示).{0,8}(?:关于我的信息|长期记忆|记忆)"
)
DELETE_PATTERN = re.compile(
    r"^(?:请)?(?:忘记|删除)(?:我之前|之前)?(?:关于)?(.+?)(?:的)?(?:长期)?记忆?[。.!！]?$"
)
UPDATE_PATTERN = re.compile(
    r"^(?:请)?把我之前关于(.+?)(?:的偏好|的信息|的记忆)?改成[，,:：\s]*(.+)$",
    re.DOTALL,
)
STABLE_PREFERENCE_PATTERN = re.compile(
    r"(?:以后|今后|默认|我喜欢|我偏好|我的习惯|请始终|总是).{2,}", re.DOTALL
)
STABLE_PROFILE_PATTERN = re.compile(
    r"(?:我的.{1,30}(?:是|为)|我(?:现在)?使用|我的.+路径).{1,}", re.DOTALL
)
ONE_OFF_TASK_PATTERN = re.compile(
    r"^(?:搜索|检索|查一下|查看今天|总结|分析|读取|打开|列出)", re.IGNORECASE
)


# 去掉用于比较的标点和空白并转小写；原始内容仍保留在候选记录中。
def normalise_text(text):
    """只用于比较和生成键，不改变最终保存给用户看的原文。"""
    return re.sub(r"[\s，。！？、,.!?:：;；]+", "", str(text)).lower()


# 使用预编译关键词模式做保守敏感信息识别，返回布尔值，不代表完整的隐私检测。
def contains_sensitive_information(text):
    return bool(SENSITIVE_PATTERN.search(str(text)))


# 优先为常见项目配置和偏好使用稳定键；同一键的新内容可以替换旧值。
# 其他事实根据主题或全文的 SHA-256 摘要生成键，不把原文直接用作实体标识。
def _derived_key(content):
    """为常见稳定事实生成固定键，使新值能够替换旧值。"""
    text = normalise_text(content)
    if "agent" in text and ("项目路径" in text or "项目在" in text):
        return "project.agent_path"
    if "qwen" in text or "本地模型" in text:
        return "project.local_model"
    if "python" in text and ("逐行" in text or "解释" in text):
        return "preference.python_explanation"
    if "注释" in text and ("中文" in text or "英文" in text):
        return "preference.code_comments"
    if "新闻" in text and ("格式" in text or "总结" in text):
        return "preference.news_summary"
    if any(word in text for word in ("ppt", "pdf", "文档")) and "格式" in text:
        return "preference.document_summary"

    # “我的某项是某值”优先按主语生成稳定键，使值变化后仍能找到同一主题。
    # 没有这类主语时才对完整规范化文本取摘要。
    subject = re.search(r"我的(.{1,24}?)(?:是|为)", text)
    if subject:
        digest_source = subject.group(1)
    else:
        digest_source = text
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
    return f"user.fact.{digest}"


# 接收已决定考虑保存的原文及来源标记，清理首尾后判断类别并生成 MemoryCandidate。
# 空内容或超过长度上限返回 None，调用方据此放弃写入。
def candidate_from_content(content, source="explicit"):
    """把已确认值得保存的文本转换为带冲突键的候选记忆。"""
    cleaned = str(content).strip().strip("。")
    if not cleaned or len(cleaned) > 1000:
        return None
    # 类别用于检索和展示；同一条内容先按偏好判断，再考虑项目配置，最后归入普通事实。
    if STABLE_PREFERENCE_PATTERN.search(cleaned):
        kind = "preference"
    elif any(word in normalise_text(cleaned) for word in ("项目", "路径", "qwen", "模型")):
        kind = "project"
    else:
        kind = "fact"
    return MemoryCandidate(
        key=_derived_key(cleaned),
        kind=kind,
        content=cleaned,
        source=source,
    )


# 按固定顺序识别清空、查看、更新、删除及保存命令，返回结构化命令或 None。
# 这里的 confirmed 只记录明确措辞；实际敏感信息策略与写入仍在 MemoryManager 中执行。
def parse_memory_command(text):
    """识别需要立即、可验证执行的显式记忆管理命令。"""
    cleaned = str(text).strip()
    # 完整确认语句先于一般清空意图判断，否则确认本身也会被当作尚未确认的请求。
    if cleaned == "确认清空所有长期记忆":
        return MemoryCommand(action="clear", confirmed=True)
    if "清空所有长期记忆" in cleaned:
        return MemoryCommand(action="clear", confirmed=False)
    if LIST_PATTERN.search(cleaned):
        return MemoryCommand(action="list")

    # 更新命令要同时提取旧主题和新内容；删除命令只需要用于匹配的主题。
    update = UPDATE_PATTERN.match(cleaned)
    if update:
        return MemoryCommand(
            action="update",
            query=update.group(1).strip(),
            content=update.group(2).strip(),
        )
    delete = DELETE_PATTERN.match(cleaned)
    if delete:
        return MemoryCommand(action="delete", query=delete.group(1).strip())

    # 明确的敏感信息确认单独识别，再处理普通“记住”语句；这里不会自动确认任何敏感内容。
    confirmed = CONFIRMED_SENSITIVE_PATTERN.match(cleaned)
    if confirmed:
        return MemoryCommand(
            action="save", content=confirmed.group(1).strip(), confirmed=True
        )
    save = EXPLICIT_SAVE_PATTERN.match(cleaned)
    if save:
        return MemoryCommand(action="save", content=save.group(1).strip())
    return None


# 只对没有疑问、没有一次性任务意图且不含敏感词的稳定偏好或资料生成候选。
# 返回 None 是正常的“无需保存”，不能把每句聊天内容都自动变成长久事实。
def automatic_memory_candidate(user_text):
    """保守提取稳定偏好/配置；普通问题、搜索和文档任务不会进入长期记忆。"""
    text = str(user_text).strip()
    if (
        not text
        or "?" in text
        or "？" in text
        or ONE_OFF_TASK_PATTERN.search(text)
        or contains_sensitive_information(text)
    ):
        return None
    # 只有确实匹配稳定偏好或资料措辞才继续生成候选；普通陈述并不必然值得长期保存。
    if not (
        STABLE_PREFERENCE_PATTERN.search(text)
        or STABLE_PROFILE_PATTERN.search(text)
    ):
        return None
    return candidate_from_content(text, source="automatic_policy")
