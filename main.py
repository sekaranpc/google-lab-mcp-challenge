import os
import json
import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio

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
    pre { background: white; border: 1px solid #ddd; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
    button { background: #1A4A8A; color: white; border: none; padding: 10px 24px; border-radius: 4px; cursor: pointer; font-size: 14px; margin-top: 8px; }
    button:hover { background: #1A6B6B; }
    textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; font-family: Arial; }
    #response { white-space: pre-wrap; font-size: 13px; color: #1a1612; }
  </style>
</head>
<body>
  <h1>🏥 Healthcare Fraud Detection</h1>
  <p>
    <span class="badge">Google AI Academy</span>
    <span class="badge">MCP + ADK</span>
    <span class="badge">Gemini 2.0 Flash</span>
  </p>
  <p>An AI agent built with <strong>Google ADK</strong> that connects to a live Healthcare Claims database via <strong>MCP</strong> to detect fraud in real time.</p>

  <h2>Try the Agent</h2>
  <div class="box">
    <textarea id="query" rows="3">Analyse the top risky providers and give me a fraud report.</textarea>
    <button onclick="runAgent()">Run Agent</button>
    <div id="status" style="color:#888; font-size:13px; margin-top:8px;"></div>
  </div>
  <div class="box" id="responseBox" style="display:none;">
    <strong>Agent Response:</strong>
    <div id="response" style="margin-top:12px;"></div>
  </div>

  <h2>API Endpoints</h2>
  <div class="box">
    <p><code>GET /health</code> — Health check</p>
    <p><code>POST /run</code> — Run the fraud detection agent</p>
    <p><code>GET /mcp/tools</code> — List available MCP tools</p>
  </div>

  <h2>Sample Queries</h2>
  <div class="box">
    <p>• "Analyse the top risky providers and give me a fraud report."</p>
    <p>• "What is the overall fraud rate in the claims database?"</p>
    <p>• "Get fraud flags for provider PRV0001."</p>
    <p>• "Show me the fraud features for the riskiest provider."</p>
  </div>

  <p style="color:#888; font-size:13px; margin-top:40px;">
    Chandrasekaran P · BITS Pilani M.Tech AI/ML · March 2026 ·
    <a href="https://github.com/sekaranpc/google-lab-mcp-challenge">GitHub</a>
  </p>

  <script>
    async function runAgent() {
      const query = document.getElementById("query").value;
      const status = document.getElementById("status");
      const responseBox = document.getElementById("responseBox");
      const response = document.getElementById("response");
      status.textContent = "Agent is thinking... please wait.";
      responseBox.style.display = "none";
      try {
        const res = await fetch("/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query })
        });
        const data = await res.json();
        response.textContent = data.response || JSON.stringify(data, null, 2);
        responseBox.style.display = "block";
        status.textContent = "Done!";
      } catch (e) {
        status.textContent = "Error: " + e.message;
      }
    }
  </script>
</body>
</html>
"""

@app.get("/health")
async def health():
    return {"status": "ok", "service": "healthcare-fraud-mcp-agent"}

@app.get("/mcp/tools")
async def mcp_tools():
    return {
        "tools": [
            "fraud_summary_stats",
            "top_risky_providers",
            "compute_fraud_features",
            "detect_fraud_flags",
            "query_provider_claims",
            "get_claim_details"
        ]
    }

@app.post("/run")
async def run_agent(request: Request):
    body = await request.json()
    query = body.get("query", "Give me a fraud summary report.")
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types as genai_types
        from agent import root_agent

        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="healthcare_fraud_agent",
            user_id="demo_user",
            session_id="session_001"
        )

        runner = Runner(
            agent=root_agent,
            app_name="healthcare_fraud_agent",
            session_service=session_service
        )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=query)]
        )

        final_response = ""
        async for event in runner.run_async(
            user_id="demo_user",
            session_id="session_001",
            new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text

        return {"status": "ok", "query": query, "response": final_response}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

def _run_mcp():
    from mcp_server import mcp
    mcp.run(transport="sse", host="0.0.0.0", port=8081)

if __name__ == "__main__":
    mcp_thread = threading.Thread(target=_run_mcp, daemon=True)
    mcp_thread.start()
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
