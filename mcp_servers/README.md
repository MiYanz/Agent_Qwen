# MCP server runtime

This directory contains project-local runtime dependencies for MCP servers.

- `@playwright/mcp@0.0.83`: headless, isolated Microsoft Edge browser automation.
- `@modelcontextprotocol/server-memory@1.0.0`: JSONL knowledge-graph memory.
- `markitdown-mcp==0.0.1a7`: launched through `uv` for PDF and Office conversion.
- `office_writer_server.py`: project-local MCP server using `python-docx` to create
  valid Word Office Open XML (`.docx`) documents and verify the saved package.
- `windows-mcp==0.8.5`: launched through `uv` with only `DisplayInventory`,
  `Screenshot`, and `Snapshot` enabled.

The Computer MCP configuration is intentionally observation-only. Mouse, keyboard,
PowerShell, registry, application launch, and filesystem mutation tools are not
loaded. Add them to the `--tools` allowlist only after defining a human approval
policy and preferably running the server in a dedicated Windows VM.
