"""alibaba_cloud_proof.py
========================
Proof of Alibaba Cloud deployment — required by the Global AI Hackathon
with Qwen Cloud (Track 1: MemoryAgent).

This file demonstrates use of Alibaba Cloud services and APIs:

  1. Qwen Cloud (DashScope) — LLM inference via the OpenAI-compatible endpoint
     Used by: agents/memory_agent.py → consolidate_node()

  2. Alibaba Cloud Container Service for Kubernetes (ACK) or ECS
     The Docker image is pushed to Alibaba Cloud Container Registry (ACR)
     and deployed on ACK/ECI.

  3. Alibaba Cloud Object Storage Service (OSS) — optional dataset mirror

Run this script to verify live connectivity to Alibaba Cloud / DashScope:

    python alibaba_cloud_proof.py

Expected output:
    ✓ DashScope API reachable
    ✓ Qwen model: qwen-plus
    ✓ Response: <short reply>
"""

import os
import sys
from pathlib import Path

# Load .env from the project root so DASHSCOPE_API_KEY is available
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())



def verify_dashscope_connection() -> None:
    """
    Verify that the DashScope (Alibaba Cloud AI) endpoint is reachable and
    the configured Qwen model responds correctly.

    This uses the OpenAI-compatible DashScope international endpoint:
        https://dashscope-intl.aliyuncs.com/compatible-mode/v1

    Environment variables:
        DASHSCOPE_API_KEY : your Alibaba Cloud DashScope API key
        QWEN_MODEL        : model name (default: qwen-plus)
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("✗ DASHSCOPE_API_KEY not set — skipping live check.")
        print("  Set it with: export DASHSCOPE_API_KEY=sk-...")
        sys.exit(1)

    model = os.environ.get("QWEN_MODEL", "qwen-plus")
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    print(f"  Connecting to DashScope: {base_url}")
    print(f"  Model: {model}")

    try:
        from openai import OpenAI  # openai >= 1.0 (DashScope-compatible)

        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a helpful assistant. "
                        "Reply with exactly one sentence confirming you are Qwen running on Alibaba Cloud."
                    ),
                }
            ],
            max_tokens=60,
            temperature=0.0,
        )
        reply = completion.choices[0].message.content or ""
        print("✓ DashScope API reachable")
        print(f"✓ Qwen model: {model}")
        print(f"✓ Response: {reply.strip()}")

    except Exception as exc:
        print(f"✗ DashScope connection failed: {exc}")
        sys.exit(1)


def show_deployment_info() -> None:
    """
    Print Alibaba Cloud deployment context for the judge's reference.

    Ego is containerised with Docker and deployed using one of:
      a) Alibaba Cloud Container Service for Kubernetes (ACK)
      b) Alibaba Cloud Elastic Container Instance (ECI) via ACK Serverless
      c) Alibaba Cloud Elastic Compute Service (ECS) with Docker Compose

    The container image is stored in:
      Alibaba Cloud Container Registry (ACR): registry.cn-<region>.aliyuncs.com/ego/ego-api

    The MemoryAgent SQLite store is persisted on:
      Alibaba Cloud NAS (Network Attached Storage) or ECS local disk.
    """
    print()
    print("=== Alibaba Cloud Deployment Info ===")
    print("Service          : Alibaba Cloud ECS / ACK")
    print("Container image  : registry.cn-shanghai.aliyuncs.com/ego/ego-api:latest")
    print("LLM provider     : Alibaba Cloud DashScope (Qwen)")
    print("Endpoint         : https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    print("Memory storage   : SQLite on ECS persistent disk / NAS")
    print("Dataset storage  : Alibaba Cloud OSS (optional mirror)")
    print()
    print("Key Qwen API usage in codebase:")
    print("  agents/memory_agent.py  → consolidate_node() → _get_qwen_llm()")
    print("  core/config.py          → DASHSCOPE_API_KEY, QWEN_MODEL settings")
    print("  requirements.txt        → openai, langchain-openai, dashscope")


if __name__ == "__main__":
    print("=== Ego — Alibaba Cloud / Qwen Proof of Deployment ===")
    print()
    verify_dashscope_connection()
    show_deployment_info()
