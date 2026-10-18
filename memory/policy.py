import hashlib
import re

from memory.models import MemoryCandidate, MemoryCommand


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


def normalise_text(text):
    """只用于比较和生成键，不改变最终保存给用户看的原文。"""
    return re.sub(r"[\s，。！？、,.!?:：;；]+", "", str(text)).lower()


def contains_sensitive_information(text):
    return bool(SENSITIVE_PATTERN.search(str(text)))


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

    subject = re.search(r"我的(.{1,24}?)(?:是|为)", text)
    if subject:
        digest_source = subject.group(1)
    else:
        digest_source = text
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
    return f"user.fact.{digest}"


def candidate_from_content(content, source="explicit"):
    """把已确认值得保存的文本转换为带冲突键的候选记忆。"""
    cleaned = str(content).strip().strip("。")
    if not cleaned or len(cleaned) > 1000:
        return None
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


def parse_memory_command(text):
    """识别需要立即、可验证执行的显式记忆管理命令。"""
    cleaned = str(text).strip()
    if cleaned == "确认清空所有长期记忆":
        return MemoryCommand(action="clear", confirmed=True)
    if "清空所有长期记忆" in cleaned:
        return MemoryCommand(action="clear", confirmed=False)
    if LIST_PATTERN.search(cleaned):
        return MemoryCommand(action="list")

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

    confirmed = CONFIRMED_SENSITIVE_PATTERN.match(cleaned)
    if confirmed:
        return MemoryCommand(
            action="save", content=confirmed.group(1).strip(), confirmed=True
        )
    save = EXPLICIT_SAVE_PATTERN.match(cleaned)
    if save:
        return MemoryCommand(action="save", content=save.group(1).strip())
    return None


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
    if not (
        STABLE_PREFERENCE_PATTERN.search(text)
        or STABLE_PROFILE_PATTERN.search(text)
    ):
        return None
    return candidate_from_content(text, source="automatic_policy")
