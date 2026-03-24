---
title: Healthcare Fraud MCP Agent
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Healthcare Fraud Detection — MCP Agent

**Google Lab Challenge: Connect AI Agents to Real-World Tools using MCP**

An AI agent built with **Google ADK** that connects to a live Healthcare Claims database via **MCP** to detect fraud in real time.

## Architecture
```
User Query → ADK LlmAgent (Gemini 2.0 Flash) → McpToolset → MCP Server → Claims DB → Fraud Report
```

## MCP Tools
- fraud_summary_stats — Overall fraud statistics
- top_risky_providers — Rank providers by risk score
- compute_fraud_features — 13 banking-inspired ML features
- detect_fraud_flags — Rule-based flags HIGH/MEDIUM/LOW
- query_provider_claims — Fetch provider claims
- get_claim_details — Look up a specific claim

**Author:** Chandrasekaran P · BITS Pilani M.Tech AI/ML · March 2026
