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
    Actively verify each Alibaba Cloud deployment claim instead of printing
    hardcoded strings.

    Checks performed:
      1. ACR image — HTTP manifest query to Alibaba Cloud Container Registry
      2. Codebase patterns — scan source files for Qwen/DashScope usage
      3. requirements.txt — confirm dashscope, openai, langchain-openai are listed
      4. Installed packages — verify they are importable in the current venv
    """
    import importlib.util
    import urllib.request
    import urllib.error
    import urllib.parse
    import base64
    from pathlib import Path

    root = Path(__file__).parent
    passed = 0
    failed = 0

    def ok(label: str, detail: str = "") -> None:
        nonlocal passed
        passed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  ✓ {label}{suffix}")

    def fail(label: str, detail: str = "") -> None:
        nonlocal failed
        failed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  ✗ {label}{suffix}")

    print()
    print("=== Alibaba Cloud Deployment Verification ===")

    # ------------------------------------------------------------------
    # 1. ACR image — query the registry manifest endpoint
    # ------------------------------------------------------------------
    print()
    print("[ 1 / 4 ]  Alibaba Cloud Container Registry (ACR)")
    registry   = os.environ.get("ACR_REGISTRY", "")
    namespace  = os.environ.get("ACR_NAMESPACE", "")
    acr_user   = os.environ.get("ACR_USER", "")
    acr_pass   = os.environ.get("ACR_PASSWORD", "")   # fixed password set in ACR console (not a GitHub PAT)
    image_tag  = "latest"
    image_repo = "ego-api"

    if not registry:
        fail("ACR_REGISTRY not set in .env — cannot verify image")
    elif not acr_pass:
        fail("ACR_PASSWORD not set in .env — add your ACR fixed password")
    else:
        full_image = f"{registry}/{namespace}/{image_repo}:{image_tag}"
        print(f"  Registry : {registry}")
        print(f"  Image    : {namespace}/{image_repo}:{image_tag}")

        import json as _json
        import re as _re
        import subprocess as _sp

        # ── Helper: get a Bearer token via the ACR OAuth dance ────────────
        def _get_token(scope: str) -> str:
            probe = urllib.request.Request(f"https://{registry}/v2/")
            www_auth = ""
            try:
                urllib.request.urlopen(probe, timeout=10)
            except urllib.error.HTTPError as _e:
                if _e.code == 401:
                    www_auth = _e.headers.get("Www-Authenticate", "")
            realm   = _re.search(r'realm="([^"]+)"',   www_auth).group(1)
            service = _re.search(r'service="([^"]+)"', www_auth).group(1)
            turl = (
                f"{realm}"
                f"?service={urllib.parse.quote(service)}"
                f"&scope={urllib.parse.quote(scope)}"
            )
            treq = urllib.request.Request(turl)
            creds = base64.b64encode(f"{acr_user}:{acr_pass}".encode()).decode()
            treq.add_header("Authorization", f"Basic {creds}")
            with urllib.request.urlopen(treq, timeout=10) as tr:
                d = _json.loads(tr.read())
            return d.get("token") or d.get("access_token", "")

        # ── Part A: tags/list — confirms the tag exists in the registry ───
        try:
            token = _get_token(f"repository:{namespace}/{image_repo}:pull")
            tags_url = f"https://{registry}/v2/{namespace}/{image_repo}/tags/list"
            treq = urllib.request.Request(tags_url)
            treq.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(treq, timeout=10) as tr:
                tags_data = _json.loads(tr.read())
            tags = tags_data.get("tags") or []
            if image_tag in tags:
                ok(f"Tag '{image_tag}' confirmed in ACR tags/list", f"all tags: {tags}")
            else:
                fail(f"Tag '{image_tag}' not found in ACR", f"available: {tags}")
        except urllib.error.HTTPError as exc:
            fail("ACR tags/list HTTP error", f"{exc.code} {exc.reason}")
        except Exception as exc:
            fail("ACR tags/list failed", str(exc))

        # ── Part B: docker manifest inspect — fetches the full manifest ───
        try:
            # Login first so docker has valid credentials in its store
            login_result = _sp.run(
                ["docker", "login", "--username", acr_user,
                 "--password-stdin", registry],
                input=acr_pass, capture_output=True, text=True, timeout=30,
            )
            if login_result.returncode != 0:
                raise RuntimeError(f"docker login failed: {login_result.stderr.strip()}")

            inspect_result = _sp.run(
                ["docker", "manifest", "inspect", full_image],
                capture_output=True, text=True, timeout=30,
            )
            if inspect_result.returncode == 0:
                # Parse the manifest to extract digest + platform info
                try:
                    mf = _json.loads(inspect_result.stdout)
                    schema = mf.get("schemaVersion", "?")
                    media  = mf.get("mediaType", mf.get("config", {}).get("mediaType", ""))
                    ok(
                        "docker manifest inspect succeeded",
                        f"schemaVersion={schema}  mediaType={media.split('.')[-1]}",
                    )
                except _json.JSONDecodeError:
                    ok("docker manifest inspect succeeded", inspect_result.stdout[:80].strip())
            else:
                fail("docker manifest inspect failed", inspect_result.stderr.strip()[:120])
        except FileNotFoundError:
            fail("docker not found — install Docker to enable manifest check")
        except Exception as exc:
            fail("docker manifest inspect error", str(exc)[:120])

    # ------------------------------------------------------------------
    # 2. Codebase patterns — scan for Qwen/DashScope references
    # ------------------------------------------------------------------
    print()
    print("[ 2 / 4 ]  Qwen API usage in codebase")
    patterns = [
        ("agents/memory_agent.py", "_get_qwen_llm"),
        ("agents/memory_agent.py", "consolidate_node"),
        ("core/config.py",         "DASHSCOPE_API_KEY"),
        ("core/config.py",         "QWEN_MODEL"),
    ]
    for rel_path, token in patterns:
        fpath = root / rel_path
        if not fpath.exists():
            fail(f"{rel_path}  ·  {token}", "file not found")
        elif token in fpath.read_text(encoding="utf-8", errors="replace"):
            ok(f"{rel_path}  ·  {token}")
        else:
            fail(f"{rel_path}  ·  {token}", "token not found in file")

    # ------------------------------------------------------------------
    # 3. requirements.txt — confirm packages are declared
    # ------------------------------------------------------------------
    print()
    print("[ 3 / 4 ]  requirements.txt declarations")
    req_file = root / "requirements.txt"
    required_pkgs = ["openai", "langchain-openai", "dashscope"]
    if not req_file.exists():
        for pkg in required_pkgs:
            fail(f"{pkg} in requirements.txt", "file not found")
    else:
        req_text = req_file.read_text(encoding="utf-8").lower()
        for pkg in required_pkgs:
            if pkg.lower() in req_text:
                ok(f"{pkg} listed in requirements.txt")
            else:
                fail(f"{pkg} listed in requirements.txt", "missing")

    # ------------------------------------------------------------------
    # 4. Installed packages — importable in current environment
    # ------------------------------------------------------------------
    print()
    print("[ 4 / 4 ]  Packages installed in current Python environment")
    import_map = {
        "openai":          "openai",
        "langchain-openai": "langchain_openai",
        "dashscope":       "dashscope",
    }
    for display_name, import_name in import_map.items():
        if importlib.util.find_spec(import_name) is not None:
            ok(f"{display_name} importable ({import_name})")
        else:
            fail(f"{display_name} not importable ({import_name})", "run: pip install " + display_name)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    total = passed + failed
    print(f"=== Result: {passed}/{total} checks passed {'✓' if failed == 0 else '✗'} ===")


if __name__ == "__main__":
    print("=== Ego — Alibaba Cloud / Qwen Proof of Deployment ===")
    print()
    verify_dashscope_connection()
    show_deployment_info()
