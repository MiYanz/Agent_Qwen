# Long-term memory

The Memory MCP server stores durable knowledge in `memory.jsonl` as entities,
relations, and observations. It is not a raw chat-history database.

The agent should store only explicit, durable user information such as research
interests and stable preferences. Secrets, credentials, transient requests, and
unverified inferences must not be stored.
