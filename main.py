import os
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Healthcare Fraud Detection - MCP Agent")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Healthcare Fraud Detection - MCP Agent</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 60px auto; padding: 0 24px; background: #f5f0e8; color: #1a1612; }
    h1 { color: #1A4A8A; border-bottom: 3px solid #1A4A8A; padding-bottom: 12px; }
    h2 { color: #1A6B6B; }
    .badge { background: #1A4A8A; color: white; padding: 4px 12px; border-radius: 4px; font-size: 13px; margin: 4px; display: inline-block; }
    .box { background: white; border: 1px solid #ddd; padding: 16px; border-radius: 6px; margin: 12px 0; }
    code { background: #e8e0d0; padding: 2px 8px; border-radius: 3px; font-size: 13px; }
    pre { background: white; border: 1px solid #ddd; padding: 16px; border-radius: 6px; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>Healthcare Fraud Detection</h1>
  <p>
    <span class="badge">Google Lab Challenge</span>
    <span class="badge">MCP + ADK</span>
    <span class="badge">Gemini 2.0 Flash</span>
  </p>
  <p>An AI agent built with <strong>Google ADK</strong> that connects to a live Healthcare Claims database via <strong>MCP</strong> to detect fraud in real time.</p>
  <h2>Health Check</h2>
  <div class="box">GET <code>/health</code> — returns service status</div>
  <p style="color:#888; font-size:13px; margin-top:40px;">
    Chandrasekaran P · BITS Pilani M.Tech AI/ML · March 2026 ·
    <a href="https://github.com/sekaranpc/google-lab-mcp-challenge">GitHub</a>
  </p>
</body>
</html>
"""

@app.get("/health")
async def health():
    return {"status": "ok", "service": "healthcare-fraud-mcp-agent"}

def _run_mcp():
    from mcp_server import mcp
    mcp.run(transport="sse", host="0.0.0.0", port=8081)

if __name__ == "__main__":
    mcp_thread = threading.Thread(target=_run_mcp, daemon=True)
    mcp_thread.start()
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
