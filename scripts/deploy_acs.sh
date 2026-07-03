#!/usr/bin/env bash
# =============================================================================
# deploy_acs.sh — Deploy Ego to Alibaba Cloud Container Service
# =============================================================================
# Usage:
#   ./scripts/deploy_acs.sh [OPTIONS]
#
# Options:
#   --region <region>       ACR/ECS region  (default: cn-shanghai)
#   --namespace <ns>        ACR namespace   (default: ego)
#   --ecs-ip <ip>           ECS public IP   (required for ECS mode)
#   --mode ecs|ack          Deployment mode (default: ecs)
#   --tag <tag>             Image tag       (default: latest)
#   --push-only             Only push image, skip remote deploy
#   -h, --help              Show this help
#
# Prerequisites:
#   1. Alibaba Cloud CLI (aliyun) installed and configured:
#        aliyun configure set --profile default --access-key-id <AK> \
#            --access-key-secret <SK> --region cn-shanghai
#   2. Docker logged in to ACR:
#        docker login registry.<REGION>.aliyuncs.com
#   3. .env file present with DASHSCOPE_API_KEY set
#   4. (ECS mode) SSH key pair configured on the ECS instance
# =============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
REGION="ap-southeast-1"
NAMESPACE="ego"
MODE="ecs"
TAG="latest"
ECS_IP=""
PUSH_ONLY=false
ACR_REGISTRY=""        # filled in from REGION below

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --region)     REGION="$2"; shift 2 ;;
    --namespace)  NAMESPACE="$2"; shift 2 ;;
    --ecs-ip)     ECS_IP="$2"; shift 2 ;;
    --mode)       MODE="$2"; shift 2 ;;
    --tag)        TAG="$2"; shift 2 ;;
    --push-only)  PUSH_ONLY=true; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,2\}//' | head -30
      exit 0 ;;
    *) error "Unknown argument: $1"; exit 1 ;;
  esac
done

ACR_REGISTRY="registry.${REGION}.aliyuncs.com"
API_IMAGE="${ACR_REGISTRY}/${NAMESPACE}/ego-api:${TAG}"
FRONTEND_IMAGE="${ACR_REGISTRY}/${NAMESPACE}/ego-frontend:${TAG}"

# ── Load .env ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "${PROJECT_ROOT}/.env" | grep '=' | xargs)
  ok "Loaded .env"
else
  warn ".env not found — make sure DASHSCOPE_API_KEY is in the environment"
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  error "DASHSCOPE_API_KEY is not set. Add it to .env or export it."
  exit 1
fi

# ── 1. Build images ───────────────────────────────────────────────────────────
echo ""
info "=== Step 1: Building Docker images ==="

info "Building API image…"
docker build \
  --target runtime \
  --tag "ego-api:${TAG}" \
  --tag "${API_IMAGE}" \
  "${PROJECT_ROOT}"
ok "API image built → ${API_IMAGE}"

info "Building frontend image…"
docker build \
  --file "${PROJECT_ROOT}/frontend.Dockerfile" \
  --tag "ego-frontend:${TAG}" \
  --tag "${FRONTEND_IMAGE}" \
  "${PROJECT_ROOT}"
ok "Frontend image built → ${FRONTEND_IMAGE}"

# ── 2. Push to ACR ───────────────────────────────────────────────────────────
echo ""
info "=== Step 2: Pushing to Alibaba Cloud Container Registry ==="
info "Registry: ${ACR_REGISTRY}"

# Check if already logged in; prompt if not
if ! docker info 2>/dev/null | grep -q "Username"; then
  warn "Not logged in to Docker. Attempting ACR login…"
  docker login "${ACR_REGISTRY}"
fi

docker push "${API_IMAGE}"
ok "Pushed → ${API_IMAGE}"

docker push "${FRONTEND_IMAGE}"
ok "Pushed → ${FRONTEND_IMAGE}"

if [[ "${PUSH_ONLY}" == "true" ]]; then
  ok "--push-only requested. Done."
  exit 0
fi

# ── 3. Deploy ─────────────────────────────────────────────────────────────────
echo ""
info "=== Step 3: Deploying (mode=${MODE}) ==="

case "${MODE}" in
  # ── ECS + Docker Compose ─────────────────────────────────────────────────
  ecs)
    if [[ -z "${ECS_IP}" ]]; then
      error "--ecs-ip is required in ECS mode."
      error "Example: ./scripts/deploy_acs.sh --mode ecs --ecs-ip 47.xx.xx.xx"
      exit 1
    fi

    SSH_USER="${ECS_SSH_USER:-root}"
    SSH_KEY="${ECS_SSH_KEY:-~/.ssh/id_rsa}"
    REMOTE_DIR="${ECS_REMOTE_DIR:-/opt/ego}"

    info "Target: ${SSH_USER}@${ECS_IP}:${REMOTE_DIR}"

    # Copy updated compose file and .env to ECS
    info "Syncing docker-compose.yml and .env to ECS…"
    # Generate a compose override that pins images to the ACR versions
    COMPOSE_OVERRIDE=$(cat <<EOF
# Auto-generated by deploy_acs.sh — do not edit manually
services:
  api:
    image: ${API_IMAGE}
    environment:
      - DASHSCOPE_API_KEY=\${DASHSCOPE_API_KEY}
      - QWEN_MODEL=\${QWEN_MODEL:-qwen-plus}
      - LLM_MODEL=\${QWEN_MODEL:-qwen-plus}
      - EMBEDDING_MODEL=\${EMBEDDING_MODEL:-all-MiniLM-L6-v2}
      - HF_HOME=/app/scratch/cache/huggingface
  frontend:
    image: ${FRONTEND_IMAGE}
EOF
)

    # Upload files
    scp -i "${SSH_KEY}" \
      "${PROJECT_ROOT}/docker-compose.yml" \
      "${PROJECT_ROOT}/.env" \
      "${SSH_USER}@${ECS_IP}:${REMOTE_DIR}/"

    # Write override and pull+restart on the remote
    ssh -i "${SSH_KEY}" "${SSH_USER}@${ECS_IP}" bash -s <<REMOTE
set -e
cd "${REMOTE_DIR}"

echo "Logging in to ACR…"
docker login ${ACR_REGISTRY}

echo "Writing compose override…"
cat > docker-compose.override.yml <<'OVERRIDE'
${COMPOSE_OVERRIDE}
OVERRIDE

echo "Pulling latest images…"
docker compose pull

echo "Restarting services…"
docker compose up -d --remove-orphans

echo "Health check…"
sleep 10
curl -sf http://localhost:8000/health && echo "✓ API healthy"
REMOTE

    ok "Deployed to ECS ${ECS_IP}"
    ;;

  # ── ACK (Kubernetes) ─────────────────────────────────────────────────────
  ack)
    info "Deploying to Alibaba Cloud Container Service for Kubernetes (ACK)…"

    # Ensure kubectl is configured (aliyun ack get-credentials must have run)
    if ! kubectl cluster-info &>/dev/null; then
      error "kubectl cannot reach the cluster. Run:"
      error "  aliyun cs GET /clusters/<cluster-id>/user_config | jq -r .config > ~/.kube/config"
      exit 1
    fi

    # Apply manifests from deploy/ack/
    MANIFESTS_DIR="${PROJECT_ROOT}/deploy/ack"
    if [[ ! -d "${MANIFESTS_DIR}" ]]; then
      error "ACK manifests not found at ${MANIFESTS_DIR}. Did you run 'make ack-manifests'?"
      exit 1
    fi

    # Patch image tag in manifests on the fly
    info "Patching image tags in manifests → ${TAG}…"
    kubectl set image deployment/ego-api \
      ego-api="${API_IMAGE}" --namespace=ego 2>/dev/null || true
    kubectl set image deployment/ego-frontend \
      ego-frontend="${FRONTEND_IMAGE}" --namespace=ego 2>/dev/null || true

    kubectl apply -f "${MANIFESTS_DIR}/" --namespace=ego
    kubectl rollout status deployment/ego-api --namespace=ego --timeout=120s
    ok "Rollout complete"
    ;;

  *)
    error "Unknown mode '${MODE}'. Use --mode ecs or --mode ack."
    exit 1
    ;;
esac

echo ""
ok "=== Ego deployed to Alibaba Cloud (${MODE}) ==="
echo ""
echo "  API image    : ${API_IMAGE}"
echo "  Frontend img : ${FRONTEND_IMAGE}"
echo "  Region       : ${REGION}"
if [[ "${MODE}" == "ecs" ]]; then
  echo "  ECS IP       : ${ECS_IP}"
  echo "  Health check : http://${ECS_IP}:8000/health"
fi
