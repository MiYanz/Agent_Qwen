from pathlib import Path
from zipfile import BadZipFile, ZipFile

from docx import Document
from mcp.server.mcpserver import MCPServer


server = MCPServer(
    "office-writer",
    instructions=(
        "Create valid Microsoft Word .docx files. Paths must be absolute and "
        "must use the .docx extension."
    ),
)


def _validated_output_path(output_path: str) -> Path:
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        raise ValueError("output_path must be an absolute path")
    if path.suffix.lower() != ".docx":
        raise ValueError("output_path must end with .docx")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _verify_docx(path: Path) -> None:
    try:
        with ZipFile(path, "r") as package:
            corrupt_member = package.testzip()
            if corrupt_member:
                raise ValueError(f"corrupt OOXML member: {corrupt_member}")
            members = set(package.namelist())
    except BadZipFile as exc:
        raise ValueError("generated file is not a valid OOXML ZIP package") from exc

    required_members = {"[Content_Types].xml", "word/document.xml"}
    missing = required_members - members
    if missing:
        raise ValueError(f"generated DOCX is missing: {sorted(missing)}")

    # Parsing it again catches malformed document XML or package relationships.
    Document(path)


@server.tool(
    description=(
        "Create a real Microsoft Word DOCX document with a title, paragraphs, "
        "numbered list, and bullet list. Returns the verified absolute path."
    )
)
def create_docx(
    output_path: str,
    title: str,
    paragraphs: list[str] | None = None,
    numbered_list: list[str] | None = None,
    bullet_list: list[str] | None = None,
) -> str:
    path = _validated_output_path(output_path)
    document = Document()
    document.add_heading(title, level=0)

    for paragraph in paragraphs or []:
        document.add_paragraph(paragraph)
    for item in numbered_list or []:
        document.add_paragraph(item, style="List Number")
    for item in bullet_list or []:
        document.add_paragraph(item, style="List Bullet")

    document.save(path)
    _verify_docx(path)
    return f"DOCX created and verified: {path}"


if __name__ == "__main__":
    server.run(transport="stdio")
