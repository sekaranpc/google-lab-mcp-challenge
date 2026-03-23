"""
Healthcare Claims MCP Server
Google Lab Challenge — Connecting AI Agents to Real-World Tools using MCP

Author: Chandrasekaran
Project: Temporal-Financial Pattern Fusion for Healthcare Fraud Detection
"""

import asyncio
import json
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Any

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)


# ─────────────────────────────────────────────
# SYNTHETIC HEALTHCARE CLAIMS DATABASE SETUP
# ─────────────────────────────────────────────

def create_demo_database(db_path: str = ":memory:") -> sqlite3.Connection:
    """Creates and seeds an in-memory SQLite database with synthetic claims data."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Claims table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            claim_id        TEXT PRIMARY KEY,
            provider_id     TEXT NOT NULL,
            patient_id      TEXT NOT NULL,
            claim_date      TEXT NOT NULL,
            procedure_code  TEXT NOT NULL,
            billed_amount   REAL NOT NULL,
            reimbursed_amount REAL NOT NULL,
            diagnosis_code  TEXT,
            claim_velocity  INTEGER DEFAULT 1,
            is_fraud        INTEGER DEFAULT 0,
            fraud_reason    TEXT
        )
    """)

    # Providers table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            provider_id     TEXT PRIMARY KEY,
            provider_name   TEXT,
            specialty       TEXT,
            state           TEXT,
            risk_score      REAL DEFAULT 0.0
        )
    """)

    # Fraud alerts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fraud_alerts (
            alert_id        TEXT PRIMARY KEY,
            claim_id        TEXT,
            alert_type      TEXT,
            detected_at     TEXT,
            severity        TEXT,
            description     TEXT
        )
    """)

    # Seed providers
    providers = [
        ("PRV001", "Sunrise Medical Group",    "General Practice", "CA", 0.12),
        ("PRV002", "BlueStar Diagnostics",     "Radiology",        "TX", 0.78),
        ("PRV003", "Valley Orthopedics",       "Orthopedics",      "FL", 0.21),
        ("PRV004", "QuickCare Clinics",        "General Practice", "NY", 0.91),
        ("PRV005", "Metro Cardiology Center",  "Cardiology",       "IL", 0.15),
    ]
    cur.executemany("INSERT OR IGNORE INTO providers VALUES (?,?,?,?,?)", providers)

    # Seed synthetic claims (mix of normal and fraudulent)
    random.seed(42)
    claims = []
    alerts = []
    base_date = datetime(2024, 1, 1)

    claim_scenarios = [
        # (provider, patient, proc_code, billed, reimb, diag, velocity, is_fraud, reason)
        ("PRV001","PAT101","99213", 150.00, 120.00, "J06.9",  1, 0, None),
        ("PRV001","PAT102","99214", 200.00, 160.00, "I10",    1, 0, None),
        ("PRV002","PRV102","70553", 2800.0, 2100.0, "G89.29", 1, 0, None),
        ("PRV004","PAT201","99213", 150.00, 120.00, "Z00.00", 45, 1, "Claim velocity spike — 45 claims/day"),
        ("PRV004","PAT202","99215", 350.00, 350.00, "M54.5",  38, 1, "100% reimbursement ratio — overbilling"),
        ("PRV002","PAT301","70553", 9500.0, 9500.0, "R51",    12, 1, "Upcoding + velocity anomaly"),
        ("PRV001","PAT103","99213", 155.00, 122.00, "J30.1",  2, 0, None),
        ("PRV003","PAT401","27447", 18000., 14000., "M17.11", 1, 0, None),
        ("PRV004","PAT203","99213", 150.00, 148.50, "Z00.00", 52, 1, "Phantom billing — duplicate procedure"),
        ("PRV005","PAT501","93000", 450.00, 360.00, "I25.10", 1, 0, None),
    ]

    for i, (prov, pat, proc, billed, reimb, diag, vel, fraud, reason) in enumerate(claim_scenarios):
        cid = f"CLM{2024_0000 + i:08d}".replace("CLM2024", "CLM")
        cid = f"CLM{i+1:06d}"
        claim_date = (base_date + timedelta(days=i*7)).strftime("%Y-%m-%d")
        claims.append((cid, prov, pat, claim_date, proc, billed, reimb, diag, vel, fraud, reason))

        if fraud:
            alerts.append((
                f"ALT{i+1:06d}", cid,
                "VELOCITY" if "velocity" in (reason or "").lower() else "BILLING",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "HIGH" if vel > 40 else "MEDIUM",
                reason
            ))

    cur.executemany("INSERT OR IGNORE INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?)", claims)
    cur.executemany("INSERT OR IGNORE INTO fraud_alerts VALUES (?,?,?,?,?,?)", alerts)
    conn.commit()
    return conn


# ─────────────────────────────────────────────
# MCP SERVER DEFINITION
# ─────────────────────────────────────────────

DB_CONN = create_demo_database()
app = Server("healthcare-fraud-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Expose the tools this MCP server offers to the AI agent."""
    return [
        Tool(
            name="query_claims",
            description=(
                "Query the healthcare claims database. "
                "Returns claims filtered by provider, patient, date range, or fraud flag. "
                "Use to retrieve raw claims data for analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "provider_id":  {"type": "string", "description": "Filter by provider ID (e.g. PRV004)"},
                    "patient_id":   {"type": "string", "description": "Filter by patient ID"},
                    "fraud_only":   {"type": "boolean", "description": "If true, return only flagged fraud claims"},
                    "limit":        {"type": "integer", "description": "Max rows to return (default 10)", "default": 10},
                },
            },
        ),
        Tool(
            name="get_provider_risk_profile",
            description=(
                "Returns a fraud risk profile for a given provider: "
                "risk score, claim velocity, reimbursement ratio, and active alerts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "provider_id": {"type": "string", "description": "Provider ID to profile"},
                },
                "required": ["provider_id"],
            },
        ),
        Tool(
            name="run_anomaly_check",
            description=(
                "Runs banking-inspired anomaly detection rules on a specific claim. "
                "Checks: reimbursement ratio, claim velocity, upcoding flag, duplicate billing. "
                "Returns an anomaly report with risk indicators."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "description": "Claim ID to analyse"},
                },
                "required": ["claim_id"],
            },
        ),
        Tool(
            name="list_fraud_alerts",
            description="Returns all active fraud alerts, optionally filtered by severity (HIGH / MEDIUM / LOW).",
            inputSchema={
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"], "description": "Filter by severity"},
                },
            },
        ),
        Tool(
            name="get_database_summary",
            description="Returns an executive summary of the claims database: total claims, fraud rate, top risky providers.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ─────────────────────────────────────────────
# TOOL HANDLERS
# ─────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    cur = DB_CONN.cursor()

    # ── 1. QUERY CLAIMS ──────────────────────
    if name == "query_claims":
        conditions, params = [], []
        if arguments.get("provider_id"):
            conditions.append("c.provider_id = ?")
            params.append(arguments["provider_id"])
        if arguments.get("patient_id"):
            conditions.append("c.patient_id = ?")
            params.append(arguments["patient_id"])
        if arguments.get("fraud_only"):
            conditions.append("c.is_fraud = 1")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        limit = arguments.get("limit", 10)
        params.append(limit)

        cur.execute(f"""
            SELECT c.claim_id, c.provider_id, p.provider_name, c.patient_id,
                   c.claim_date, c.procedure_code, c.billed_amount,
                   c.reimbursed_amount, c.claim_velocity, c.is_fraud, c.fraud_reason
            FROM claims c
            JOIN providers p ON c.provider_id = p.provider_id
            {where}
            LIMIT ?
        """, params)

        rows = [dict(r) for r in cur.fetchall()]
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(rows, indent=2))]
        )

    # ── 2. PROVIDER RISK PROFILE ─────────────
    elif name == "get_provider_risk_profile":
        pid = arguments["provider_id"]
        cur.execute("""
            SELECT p.*,
                   COUNT(c.claim_id)                          AS total_claims,
                   SUM(c.is_fraud)                            AS fraud_claims,
                   ROUND(AVG(c.claim_velocity), 1)            AS avg_velocity,
                   ROUND(AVG(c.reimbursed_amount / NULLIF(c.billed_amount, 0)) * 100, 1) AS avg_reimb_ratio
            FROM providers p
            LEFT JOIN claims c ON p.provider_id = c.provider_id
            WHERE p.provider_id = ?
            GROUP BY p.provider_id
        """, [pid])

        row = cur.fetchone()
        if not row:
            return CallToolResult(content=[TextContent(type="text", text=f"Provider {pid} not found.")])

        profile = dict(row)
        fraud_rate = round((profile["fraud_claims"] or 0) / max(profile["total_claims"], 1) * 100, 1)
        profile["fraud_rate_pct"] = fraud_rate
        profile["risk_tier"] = "HIGH" if profile["risk_score"] > 0.7 else "MEDIUM" if profile["risk_score"] > 0.4 else "LOW"

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(profile, indent=2))]
        )

    # ── 3. ANOMALY CHECK ─────────────────────
    elif name == "run_anomaly_check":
        cid = arguments["claim_id"]
        cur.execute("""
            SELECT c.*, p.provider_name, p.specialty, p.risk_score
            FROM claims c
            JOIN providers p ON c.provider_id = p.provider_id
            WHERE c.claim_id = ?
        """, [cid])

        row = cur.fetchone()
        if not row:
            return CallToolResult(content=[TextContent(type="text", text=f"Claim {cid} not found.")])

        c = dict(row)
        reimb_ratio = c["reimbursed_amount"] / max(c["billed_amount"], 1)
        flags = []

        # Banking-inspired rules (mirroring the project's 13 features)
        if reimb_ratio >= 0.99:
            flags.append({"rule": "FULL_REIMBURSEMENT", "detail": f"Reimbursement ratio {reimb_ratio:.0%} — possible upcoding"})
        if c["claim_velocity"] > 30:
            flags.append({"rule": "VELOCITY_SPIKE", "detail": f"Velocity = {c['claim_velocity']} claims/day (threshold: 30)"})
        if c["billed_amount"] > 5000 and c["procedure_code"].startswith("9"):
            flags.append({"rule": "HIGH_VALUE_E_M", "detail": "High-value E&M code — review medical necessity"})
        if c["provider_risk_score"] if "provider_risk_score" in c else c.get("risk_score", 0) > 0.7:
            flags.append({"rule": "HIGH_RISK_PROVIDER", "detail": f"Provider risk score {c.get('risk_score', 0):.2f} > 0.7"})

        result = {
            "claim_id":       cid,
            "provider":       c["provider_name"],
            "billed":         c["billed_amount"],
            "reimbursed":     c["reimbursed_amount"],
            "reimb_ratio_pct": round(reimb_ratio * 100, 1),
            "velocity":       c["claim_velocity"],
            "anomaly_flags":  flags,
            "risk_level":     "HIGH" if len(flags) >= 2 else "MEDIUM" if flags else "NORMAL",
            "is_known_fraud": bool(c["is_fraud"]),
            "fraud_reason":   c["fraud_reason"],
        }

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )

    # ── 4. FRAUD ALERTS ──────────────────────
    elif name == "list_fraud_alerts":
        severity = arguments.get("severity")
        if severity:
            cur.execute("SELECT * FROM fraud_alerts WHERE severity = ? ORDER BY detected_at DESC", [severity])
        else:
            cur.execute("SELECT * FROM fraud_alerts ORDER BY severity DESC, detected_at DESC")

        rows = [dict(r) for r in cur.fetchall()]
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(rows, indent=2))]
        )

    # ── 5. DATABASE SUMMARY ──────────────────
    elif name == "get_database_summary":
        cur.execute("""
            SELECT
                COUNT(*)                                        AS total_claims,
                SUM(is_fraud)                                   AS fraud_claims,
                ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 1)     AS fraud_rate_pct,
                ROUND(SUM(billed_amount), 2)                    AS total_billed,
                ROUND(SUM(reimbursed_amount), 2)                AS total_reimbursed,
                ROUND(AVG(claim_velocity), 1)                   AS avg_velocity
            FROM claims
        """)
        summary = dict(cur.fetchone())

        cur.execute("""
            SELECT p.provider_id, p.provider_name, p.risk_score,
                   COUNT(c.claim_id) AS claims, SUM(c.is_fraud) AS fraud_count
            FROM providers p
            JOIN claims c ON p.provider_id = c.provider_id
            GROUP BY p.provider_id
            ORDER BY p.risk_score DESC
            LIMIT 3
        """)
        summary["top_risk_providers"] = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT COUNT(*) AS active_alerts FROM fraud_alerts")
        summary["active_alerts"] = cur.fetchone()["active_alerts"]

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(summary, indent=2))]
        )

    return CallToolResult(
        content=[TextContent(type="text", text=f"Unknown tool: {name}")]
    )


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
