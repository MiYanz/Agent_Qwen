# Agent_Qwen

基于 Qwen3-8B、LangGraph、MCP 和 Gradio 的本地 Personal Agent。

## 安装与启动

```powershell
conda activate qwen_agent
python -m pip install -r requirements.txt
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。页面会显示当前会话 ID；复制该 ID，重启
应用后填回并点击“恢复会话”，即可从 SQLite Checkpointer 恢复短期聊天状态。

长期记忆使用自然语言管理，例如：

- `记住，我的 Agent 项目路径是 D:\46-Agent。`
- `你记住了哪些关于我的信息？`
- `把我之前关于代码注释的偏好改成中英文结合。`
- `忘记我之前关于新闻总结格式的偏好。`

长期记忆保存在本地 Memory MCP 中；会话状态和个人记忆数据不会提交到 Git。
