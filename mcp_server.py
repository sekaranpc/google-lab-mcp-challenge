import sqlite3
import json
import random
import math
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP

def _create_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE providers (
            provider_id    TEXT PRIMARY KEY,
            provider_name  TEXT,
            specialty      TEXT,
            state          TEXT
        );
        CREATE TABLE claims (
            claim_id           TEXT PRIMARY KEY,
            provider_id        TEXT,
            patient_id         TEXT,
            claim_date         TEXT,
            procedure_code     TEXT,
            billed_amount      REAL,
            approved_amount    REAL,
            reimbursed_amount  REAL,
            claim_type         TEXT,
            chronic_conditions INTEGER,
            is_fraud           INTEGER
        );
        CREATE TABLE fraud_flags (
            flag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id    TEXT,
            flag_type   TEXT,
            flag_detail TEXT,
            severity    TEXT
        );
    """)
    random.seed(42)
    specialties = ["Cardiology","Orthopedics","Neurology","General Practice","Oncology"]
    states = ["CA","TX","NY","FL","IL"]
    procedures = ["99213","99214","99215","27447","93000"]
    providers = [(f"PRV{i:04d}", f"Dr. Provider {i}", random.choice(specialties), random.choice(states)) for i in range(1,51)]
    cur.executemany("INSERT INTO providers VALUES (?,?,?,?)", providers)
    base = datetime(2022,1,1)
    claims, flags = [], []
    for i in range(1,1001):
        cid = f"CLM{i:06d}"
        pid = random.choice(providers)[0]
        billed = round(random.uniform(200,15000),2)
        approved = round(billed*random.uniform(0.4,0.95),2)
        reimb = round(approved*random.uniform(0.7,1.0),2)
        chronic = random.randint(0,5)
        ctype = random.choice(["inpatient","outpatient"])
        is_fraud = 1 if random.random()<0.19 else 0
        date = (base+timedelta(days=random.randint(0,730))).strftime("%Y-%m-%d")
        claims.append((cid,pid,f"PAT{random.randint(1,500):05d}",date,random.choice(procedures),billed,approved,reimb,ctype,chronic,is_fraud))
        if is_fraud:
            ratio = billed/approved if approved>0 else 0
            if ratio>2.5:
                flags.append((cid,"HIGH_REIMBURSEMENT_RATIO",f"Billed/Approved={ratio:.2f}","HIGH"))
            if chronic==0 and ctype=="inpatient":
                flags.append((cid,"CHRONIC_INPATIENT_MISMATCH","Inpatient with zero chronic conditions","MEDIUM"))
            if billed>10000:
                flags.append((cid,"LARGE_CLAIM_AMOUNT",f"Billed ${billed:,.2f}","MEDIUM"))
    cur.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?)", claims)
    cur.executemany("INSERT INTO fraud_flags (claim_id,flag_type,flag_detail,severity) VALUES (?,?,?,?)", flags)
    conn.commit()
    return conn

DB = _create_db()

def _provider_features(provider_id):
    cur = DB.cursor()
    rows = cur.execute("SELECT billed_amount,approved_amount,reimbursed_amount,claim_type,chronic_conditions,is_fraud FROM claims WHERE provider_id=?",(provider_id,)).fetchall()
    if not rows:
        return {}
    n = len(rows)
    billed = [r["billed_amount"] for r in rows]
    reimbursed = [r["reimbursed_amount"] for r in rows]
    fraud_ct = sum(r["is_fraud"] for r in rows)
    inpatient = sum(1 for r in rows if r["claim_type"]=="inpatient")
    chronic_0 = sum(1 for r in rows if r["chronic_conditions"]==0)
    avg_bill = sum(billed)/n
    avg_reimb = sum(reimbursed)/n
    reimb_ratio = avg_reimb/avg_bill if avg_bill else 0
    variance = sum((x-avg_bill)**2 for x in billed)/n
    high_val = sum(1 for b in billed if b>5000)
    overpay = sum(1 for r in rows if r["reimbursed_amount"]>r["approved_amount"])
    return {
        "provider_id":provider_id,"claim_count":n,
        "avg_billed_amount":round(avg_bill,2),
        "avg_reimbursed_amount":round(avg_reimb,2),
        "reimbursement_ratio":round(reimb_ratio,4),
        "inpatient_ratio":round(inpatient/n,4),
        "chronic_zero_ratio":round(chronic_0/n,4),
        "std_billed_amount":round(math.sqrt(variance),2),
        "overpayment_count":overpay,
        "high_value_claim_count":high_val,
        "fraud_count":fraud_ct,
        "fraud_rate":round(fraud_ct/n,4),
        "risk_score":round((fraud_ct/n)*0.4+(1-reimb_ratio)*0.3+(high_val/n)*0.3,4),
    }

mcp = FastMCP(name="healthcare-fraud-mcp")

@mcp.tool()
def get_claim_details(claim_id: str) -> str:
    """Retrieve full details of a single healthcare claim by claim ID."""
    row = DB.cursor().execute("SELECT * FROM claims WHERE claim_id=?",(claim_id,)).fetchone()
    return json.dumps(dict(row) if row else {"error":f"{claim_id} not found"},indent=2)

@mcp.tool()
def query_provider_claims(provider_id: str, limit: int = 20) -> str:
    """Fetch all claims submitted by a specific provider."""
    rows = DB.cursor().execute("SELECT * FROM claims WHERE provider_id=? LIMIT ?",(provider_id,limit)).fetchall()
    return json.dumps({"provider_id":provider_id,"claim_count":len(rows),"claims":[dict(r) for r in rows]},indent=2)

@mcp.tool()
def compute_fraud_features(provider_id: str) -> str:
    """Compute 13 banking-inspired fraud-detection features for a provider."""
    result = _provider_features(provider_id)
    return json.dumps(result if result else {"error":f"Provider {provider_id} not found"},indent=2)

@mcp.tool()
def detect_fraud_flags(claim_id: str = "", provider_id: str = "") -> str:
    """Return rule-based fraud flags for a claim or provider. Severity: HIGH/MEDIUM/LOW."""
    cur = DB.cursor()
    if claim_id:
        rows = cur.execute("SELECT * FROM fraud_flags WHERE claim_id=?",(claim_id,)).fetchall()
    elif provider_id:
        rows = cur.execute("SELECT ff.* FROM fraud_flags ff JOIN claims c ON ff.claim_id=c.claim_id WHERE c.provider_id=?",(provider_id,)).fetchall()
    else:
        rows = cur.execute("SELECT * FROM fraud_flags LIMIT 50").fetchall()
    return json.dumps({"flag_count":len(rows),"flags":[dict(r) for r in rows]},indent=2)

@mcp.tool()
def top_risky_providers(top_n: int = 10, min_claims: int = 5) -> str:
    """Return top-N providers ranked by composite risk score."""
    cur = DB.cursor()
    pids = [r["provider_id"] for r in cur.execute("SELECT provider_id,COUNT(*) AS cnt FROM claims GROUP BY provider_id HAVING cnt>=?",(min_claims,)).fetchall()]
    scored = [f for pid in pids if (f:=_provider_features(pid))]
    scored.sort(key=lambda x:x["risk_score"],reverse=True)
    return json.dumps({"top_providers":scored[:top_n]},indent=2)

@mcp.tool()
def fraud_summary_stats() -> str:
    """Return overall fraud statistics across the entire claims database."""
    cur = DB.cursor()
    stats = cur.execute("SELECT COUNT(*) AS total_claims,SUM(is_fraud) AS fraud_claims,ROUND(AVG(is_fraud)*100,2) AS fraud_rate_pct,ROUND(AVG(billed_amount),2) AS avg_billed,ROUND(AVG(reimbursed_amount),2) AS avg_reimbursed,COUNT(DISTINCT provider_id) AS total_providers,COUNT(DISTINCT patient_id) AS total_patients FROM claims").fetchone()
    by_type = cur.execute("SELECT claim_type,COUNT(*) AS count,SUM(is_fraud) AS fraud_count,ROUND(AVG(is_fraud)*100,2) AS fraud_rate_pct FROM claims GROUP BY claim_type").fetchall()
    return json.dumps({"summary":dict(stats),"by_claim_type":[dict(r) for r in by_type]},indent=2)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("MCP_PORT", 8081))
    mcp.run(transport="sse", host="0.0.0.0", port=port)

def run_server():
    import os
    port = int(os.environ.get("MCP_PORT", 8081))
    mcp.run(transport="sse", port=port)
