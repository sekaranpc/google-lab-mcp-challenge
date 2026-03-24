import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset, SseServerParams

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8081/sse")

root_agent = LlmAgent(
    name        = "healthcare_fraud_agent",
    model       = "gemini-2.0-flash",
    description = "An AI agent that detects healthcare claims fraud by connecting to a live claims database via MCP tools.",
    instruction = """
You are a healthcare fraud detection specialist with deep expertise in
banking-domain pattern recognition applied to medical claims analysis.

You have access to 6 MCP tools connected to a live Healthcare Claims Database:
1. fraud_summary_stats        - Get overall fraud statistics
2. top_risky_providers        - Rank providers by composite risk score
3. compute_fraud_features     - Compute 13 banking-inspired ML features
4. detect_fraud_flags         - Get rule-based flags (HIGH/MEDIUM/LOW)
5. query_provider_claims      - Fetch all claims for a provider
6. get_claim_details          - Look up a specific claim by ID

WORKFLOW:
1. Start with fraud_summary_stats to establish baseline context
2. Use top_risky_providers to identify high-risk providers
3. For flagged providers, call compute_fraud_features for detailed analysis
4. Call detect_fraud_flags to surface specific rule violations
5. Synthesise findings into a clear fraud intelligence report

Always quote specific numbers from the tools and explain findings in plain language.
""",
    tools = [
        McpToolset(
            connection_params=SseServerParams(url=MCP_SERVER_URL)
        )
    ],
)
