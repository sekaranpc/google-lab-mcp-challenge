"""
MCP Client Demo: AI Agent calling Healthcare Fraud MCP Server
=============================================================
Demonstrates how an AI agent (or test harness) connects to the
MCP server and invokes fraud-detection tools over stdio transport.

Run the server first:
    python mcp_healthcare_fraud_server.py

Then run this client:
    python mcp_client_demo.py
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["mcp_healthcare_fraud_server.py"],
)


async def run_demo():
    print("=" * 60)
    print("  Healthcare Fraud MCP — Agent Demo")
    print("=" * 60)

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            # ── 1. List available tools ──────────────────────────────────
            tools = await session.list_tools()
            print(f"\n[MCP] {len(tools.tools)} tools discovered:")
            for t in tools.tools:
                print(f"  · {t.name}")

            # ── 2. Overall fraud stats ───────────────────────────────────
            print("\n[TOOL] fraud_summary_stats")
            res = await session.call_tool("fraud_summary_stats", {})
            data = json.loads(res.content[0].text)
            s = data["summary"]
            print(f"  Total claims   : {s['total_claims']}")
            print(f"  Fraud claims   : {s['fraud_claims']}  ({s['fraud_rate_pct']}%)")
            print(f"  Avg billed     : ${s['avg_billed']:,}")
            print(f"  Providers      : {s['total_providers']}")

            # ── 3. Top risky providers ────────────────────────────────────
            print("\n[TOOL] top_risky_providers (top 5)")
            res = await session.call_tool("top_risky_providers", {"top_n": 5, "min_claims": 10})
            data = json.loads(res.content[0].text)
            for p in data["top_providers"]:
                print(f"  {p['provider_id']}  risk={p['risk_score']:.4f}"
                      f"  fraud_rate={p['fraud_rate']:.1%}"
                      f"  claims={p['claim_count']}")

            # ── 4. Feature profile of top provider ───────────────────────
            top_pid = data["top_providers"][0]["provider_id"]
            print(f"\n[TOOL] compute_fraud_features → {top_pid}")
            res = await session.call_tool("compute_fraud_features", {"provider_id": top_pid})
            feats = json.loads(res.content[0].text)
            for k, v in feats.items():
                print(f"  {k:<30} : {v}")

            # ── 5. Fraud flags for top provider ───────────────────────────
            print(f"\n[TOOL] detect_fraud_flags → {top_pid}")
            res = await session.call_tool("detect_fraud_flags", {"provider_id": top_pid})
            data = json.loads(res.content[0].text)
            print(f"  {data['flag_count']} flags found")
            for f in data["flags"][:5]:
                print(f"  [{f['severity']}] {f['flag_type']} — {f['flag_detail']}")

    print("\n[Done] MCP session closed.")


if __name__ == "__main__":
    asyncio.run(run_demo())
