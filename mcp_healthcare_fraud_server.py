"""
MCP Server: Healthcare Claims Fraud Detection
=============================================
Connects AI Agents to a Healthcare Claims Database via the
Model Context Protocol (MCP), exposing fraud-detection tools
that an LLM agent can invoke in real-time.

Challenge Track : Google Lab — Connect AI Agents to Real-World Tools using MCP
Author          : Chandrasekaran P  (BITS Pilani M.Tech AI/ML)
Protocol        : MCP (Model Context Protocol) — stdio transport
Dependencies    : mcp, sqlite3 (stdlib), pandas, scikit-learn
"""

import sqlite3
import json
import random
import math
from datetime import datetime, timedelta
from typing import Any

# ── MCP SDK ──────────────────────────────────────────────────────────────────
# Install: pip install mcp
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ═════════════════════════════════════════════════════════════════════════════
# 1.  IN-MEMORY HEALTHCARE CLAIMS DATABASE  (synthetic — for demo purposes)
# ═════════════════════════════════════════════════════════════════════════════

def _create_synthetic_db() -> sqlite3.Connection:
    """
    Creates an in-memory SQLite database populated with synthetic
    healthcare claims data that mirrors the project's 10 000-claim dataset.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ── Schema ──────────────────────────────────────────────────────────────
    cur.executescript("""
        CREATE TABLE providers (
            provider_id     TEXT PRIMARY KEY,
            provider_name   TEXT,
            specialty       TEXT,
            state           TEXT,
            enrolled_since  TEXT
        );

        CREATE TABLE claims (
            claim_id            TEXT PRIMARY KEY,
            provider_id         TEXT,
            patient_id          TEXT,
            claim_date          TEXT,
            procedure_code      TEXT,
            diagnosis_code      TEXT,
            billed_amount       REAL,
            approved_amount     REAL,
            reimbursed_amount   REAL,
            claim_type          TEXT,     -- inpatient / outpatient
            chronic_conditions  INTEGER,  -- count of chronic flags
            is_fraud            INTEGER   -- 1 = fraud, 0 = legitimate
        );

        CREATE TABLE fraud_flags (
            flag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id    TEXT,
            flag_type   TEXT,
            flag_detail TEXT,
            severity    TEXT   -- HIGH / MEDIUM / LOW
        );
    """)

    # ── Seed synthetic data ──────────────────────────────────────────────────
    random.seed(42)
    specialties = ["Cardiology", "Orthopedics", "Neurology", "General Practice", "Oncology"]
    states = ["CA", "TX", "NY", "FL", "IL"]
    procedures = ["99213", "99214", "99215", "27447", "93000", "70553", "71046"]
    diagnoses  = ["I10", "E11", "M17", "Z00", "J18", "C34", "I21"]

    providers = []
    for i in range(1, 51):
        pid = f"PRV{i:04d}"
        providers.append((
            pid,
            f"Dr. Provider {i}",
            random.choice(specialties),
            random.choice(states),
            f"201{random.randint(0,9)}-0{random.randint(1,9)}-01"
        ))
    cur.executemany(
        "INSERT INTO providers VALUES (?,?,?,?,?)", providers
    )

    base_date = datetime(2022, 1, 1)
    claim_rows, flag_rows = [], []

    for i in range(1, 1001):
        cid        = f"CLM{i:06d}"
        provider   = random.choice(providers)[0]
        patient    = f"PAT{random.randint(1, 500):05d}"
        days_off   = random.randint(0, 730)
        claim_date = (base_date + timedelta(days=days_off)).strftime("%Y-%m-%d")
        billed     = round(random.uniform(200, 15000), 2)
        approved   = round(billed * random.uniform(0.4, 0.95), 2)
        reimbursed = round(approved * random.uniform(0.7, 1.0), 2)
        chronic    = random.randint(0, 5)
        ctype      = random.choice(["inpatient", "outpatient"])
        is_fraud   = 1 if random.random() < 0.19 else 0   # ~19 % fraud rate

        claim_rows.append((
            cid, provider, patient, claim_date,
            random.choice(procedures), random.choice(diagnoses),
            billed, approved, reimbursed, ctype, chronic, is_fraud
        ))

        # Inject banking-inspired fraud flags for fraudulent claims
        if is_fraud:
            ratio = billed / approved if approved > 0 else 0
            if ratio > 2.5:
                flag_rows.append((cid, "HIGH_REIMBURSEMENT_RATIO",
                    f"Billed/Approved ratio = {ratio:.2f}", "HIGH"))
            if chronic == 0 and ctype == "inpatient":
                flag_rows.append((cid, "CHRONIC_INPATIENT_MISMATCH",
                    "Inpatient claim with zero chronic conditions", "MEDIUM"))
            if billed > 10000:
                flag_rows.append((cid, "LARGE_CLAIM_AMOUNT",
                    f"Billed ${billed:,.2f} exceeds threshold", "MEDIUM"))

    cur.executemany(
        "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", claim_rows
    )
    cur.executemany(
        "INSERT INTO fraud_flags (claim_id, flag_type, flag_detail, severity) VALUES (?,?,?,?)",
        flag_rows
    )
    conn.commit()
    return conn


DB_CONN = _create_synthetic_db()


# ═════════════════════════════════════════════════════════════════════════════
# 2.  BANKING-INSPIRED FEATURE ENGINEERING  (mirrors Day-3 notebook logic)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_provider_features(provider_id: str) -> dict:
    """
    Computes 13 banking-inspired fraud-detection features for a provider,
    directly mirroring the Isolation Forest feature set from Day 3.
    """
    cur = DB_CONN.cursor()
    rows = cur.execute("""
        SELECT billed_amount, approved_amount, reimbursed_amount,
               claim_type, chronic_conditions, is_fraud
        FROM   claims
        WHERE  provider_id = ?
    """, (provider_id,)).fetchall()

    if not rows:
        return {}

    n = len(rows)
    billed     = [r["billed_amount"]     for r in rows]
    approved   = [r["approved_amount"]   for r in rows]
    reimbursed = [r["reimbursed_amount"] for r in rows]
    fraud_ct   = sum(r["is_fraud"] for r in rows)
    inpatient  = sum(1 for r in rows if r["claim_type"] == "inpatient")
    chronic_0  = sum(1 for r in rows if r["chronic_conditions"] == 0)

    avg_bill   = sum(billed) / n
    avg_reimb  = sum(reimbursed) / n
    reimb_ratio = avg_reimb / avg_bill if avg_bill > 0 else 0

    variance   = sum((x - avg_bill) ** 2 for x in billed) / n
    std_bill   = math.sqrt(variance)

    overpay_ct = sum(1 for a, r in zip(approved, reimbursed) if r > a)
    high_val   = sum(1 for b in billed if b > 5000)

    return {
        "provider_id"              : provider_id,
        "claim_count"              : n,
        "avg_billed_amount"        : round(avg_bill, 2),
        "avg_reimbursed_amount"    : round(avg_reimb, 2),
        "reimbursement_ratio"      : round(reimb_ratio, 4),
        "claim_velocity"           : n,                            # proxy
        "inpatient_ratio"          : round(inpatient / n, 4),
        "chronic_zero_ratio"       : round(chronic_0  / n, 4),
        "std_billed_amount"        : round(std_bill, 2),
        "overpayment_flag_count"   : overpay_ct,
        "high_value_claim_count"   : high_val,
        "fraud_claim_count"        : fraud_ct,
        "fraud_rate"               : round(fraud_ct / n, 4),
        "risk_score"               : round(
            (fraud_ct / n) * 0.4 +
            (1 - reimb_ratio) * 0.3 +
            (high_val / n) * 0.3, 4
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MCP SERVER DEFINITION
# ═════════════════════════════════════════════════════════════════════════════

server = Server("healthcare-fraud-mcp")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Declare all tools the MCP server exposes to the AI agent."""
    return [

        types.Tool(
            name="get_claim_details",
            description=(
                "Retrieve full details of a single healthcare claim by claim ID. "
                "Returns billed/approved/reimbursed amounts, procedure codes, "
                "diagnosis codes, claim type, and fraud label."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "Claim ID in format CLM000001"
                    }
                },
                "required": ["claim_id"]
            }
        ),

        types.Tool(
            name="query_provider_claims",
            description=(
                "Fetch all claims submitted by a specific provider. "
                "Supports optional date-range filtering. "
                "Useful for velocity analysis and pattern detection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "provider_id": {
                        "type": "string",
                        "description": "Provider ID in format PRV0001"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional start date YYYY-MM-DD"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional end date YYYY-MM-DD"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 20)",
                        "default": 20
                    }
                },
                "required": ["provider_id"]
            }
        ),

        types.Tool(
            name="compute_fraud_features",
            description=(
                "Compute 13 banking-inspired fraud-detection features for a provider: "
                "reimbursement ratio, claim velocity, overpayment flags, high-value claim count, "
                "chronic-condition mismatch ratio, and a composite risk score. "
                "These features feed the Isolation Forest anomaly model."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "provider_id": {
                        "type": "string",
                        "description": "Provider ID to analyse"
                    }
                },
                "required": ["provider_id"]
            }
        ),

        types.Tool(
            name="detect_fraud_flags",
            description=(
                "Return all rule-based fraud flags for a claim or provider. "
                "Flags are categorised as HIGH / MEDIUM / LOW severity and include: "
                "HIGH_REIMBURSEMENT_RATIO, CHRONIC_INPATIENT_MISMATCH, LARGE_CLAIM_AMOUNT."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "Specific claim ID to check (optional)"
                    },
                    "provider_id": {
                        "type": "string",
                        "description": "Provider ID — returns all flags across provider claims (optional)"
                    }
                }
            }
        ),

        types.Tool(
            name="top_risky_providers",
            description=(
                "Return the top-N providers ranked by composite risk score. "
                "Combines fraud rate, reimbursement anomalies, and high-value claim density. "
                "Mirrors the Isolation Forest anomaly ranking output."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of providers to return (default 10)",
                        "default": 10
                    },
                    "min_claims": {
                        "type": "integer",
                        "description": "Minimum claim count to include provider (default 5)",
                        "default": 5
                    }
                }
            }
        ),

        types.Tool(
            name="fraud_summary_stats",
            description=(
                "Return overall fraud statistics for the entire claims database: "
                "total claims, fraud count, fraud rate, average billed amounts, "
                "and breakdown by claim type."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Route tool calls from the AI agent to the appropriate handler."""

    cur = DB_CONN.cursor()

    # ── Tool: get_claim_details ──────────────────────────────────────────────
    if name == "get_claim_details":
        claim_id = arguments["claim_id"]
        row = cur.execute(
            "SELECT * FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        if not row:
            result = {"error": f"Claim {claim_id} not found"}
        else:
            result = dict(row)

    # ── Tool: query_provider_claims ──────────────────────────────────────────
    elif name == "query_provider_claims":
        provider_id = arguments["provider_id"]
        limit       = arguments.get("limit", 20)
        start_date  = arguments.get("start_date")
        end_date    = arguments.get("end_date")

        query = "SELECT * FROM claims WHERE provider_id = ?"
        params: list = [provider_id]
        if start_date:
            query  += " AND claim_date >= ?"
            params.append(start_date)
        if end_date:
            query  += " AND claim_date <= ?"
            params.append(end_date)
        query += f" LIMIT {int(limit)}"

        rows = cur.execute(query, params).fetchall()
        result = {
            "provider_id" : provider_id,
            "claim_count" : len(rows),
            "claims"      : [dict(r) for r in rows]
        }

    # ── Tool: compute_fraud_features ─────────────────────────────────────────
    elif name == "compute_fraud_features":
        result = _compute_provider_features(arguments["provider_id"])
        if not result:
            result = {"error": f"Provider {arguments['provider_id']} not found"}

    # ── Tool: detect_fraud_flags ─────────────────────────────────────────────
    elif name == "detect_fraud_flags":
        claim_id    = arguments.get("claim_id")
        provider_id = arguments.get("provider_id")

        if claim_id:
            rows = cur.execute(
                "SELECT * FROM fraud_flags WHERE claim_id = ?", (claim_id,)
            ).fetchall()
        elif provider_id:
            rows = cur.execute("""
                SELECT ff.* FROM fraud_flags ff
                JOIN   claims c ON ff.claim_id = c.claim_id
                WHERE  c.provider_id = ?
            """, (provider_id,)).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM fraud_flags LIMIT 50"
            ).fetchall()

        result = {
            "flag_count" : len(rows),
            "flags"      : [dict(r) for r in rows]
        }

    # ── Tool: top_risky_providers ────────────────────────────────────────────
    elif name == "top_risky_providers":
        top_n      = arguments.get("top_n", 10)
        min_claims = arguments.get("min_claims", 5)

        provider_ids = [
            r["provider_id"]
            for r in cur.execute(
                "SELECT provider_id, COUNT(*) as cnt FROM claims "
                "GROUP BY provider_id HAVING cnt >= ?", (min_claims,)
            ).fetchall()
        ]

        scored = []
        for pid in provider_ids:
            feats = _compute_provider_features(pid)
            if feats:
                scored.append(feats)

        scored.sort(key=lambda x: x["risk_score"], reverse=True)
        result = {
            "top_providers": scored[:top_n]
        }

    # ── Tool: fraud_summary_stats ────────────────────────────────────────────
    elif name == "fraud_summary_stats":
        stats = cur.execute("""
            SELECT
                COUNT(*)                          AS total_claims,
                SUM(is_fraud)                     AS fraud_claims,
                ROUND(AVG(is_fraud)*100, 2)       AS fraud_rate_pct,
                ROUND(AVG(billed_amount), 2)      AS avg_billed,
                ROUND(AVG(reimbursed_amount), 2)  AS avg_reimbursed,
                COUNT(DISTINCT provider_id)       AS total_providers,
                COUNT(DISTINCT patient_id)        AS total_patients
            FROM claims
        """).fetchone()

        by_type = cur.execute("""
            SELECT claim_type,
                   COUNT(*) AS count,
                   SUM(is_fraud) AS fraud_count,
                   ROUND(AVG(is_fraud)*100,2) AS fraud_rate_pct
            FROM claims
            GROUP BY claim_type
        """).fetchall()

        result = {
            "summary"        : dict(stats),
            "by_claim_type"  : [dict(r) for r in by_type]
        }

    else:
        result = {"error": f"Unknown tool: {name}"}

    return [types.TextContent(
        type = "text",
        text = json.dumps(result, indent=2)
    )]


# ═════════════════════════════════════════════════════════════════════════════
# 4.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
