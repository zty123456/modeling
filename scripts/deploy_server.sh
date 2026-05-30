#!/usr/bin/env bash
# Deploy ZRT-Sim: systemd service + firewall + force restart.
#
# Usage (Linux server, run as root or with sudo):
#   sudo bash scripts/deploy_server.sh [install] [PORT]
#
# Modes:
#   (no args)         Assume env is ready. Only sanity-check, write/refresh
#                     systemd unit, open firewall, force-restart service.
#   install           Same as above PLUS install OS packages + venv + pip deps.
#                     Use this on the first run or after editing this script.
#
# Examples:
#   sudo bash scripts/deploy_server.sh install        # first time
#   sudo bash scripts/deploy_server.sh                # subsequent redeploys
#   sudo bash scripts/deploy_server.sh install 8002   # install + custom port
#   sudo bash scripts/deploy_server.sh 8002           # redeploy + custom port
#
# Flags via env vars:
#   FORCE=1           Re-run install steps even if stamp says they're fresh.
#                     Only meaningful with `install`.
#   SKIP_FULL_DEPS=1  Skip torch/transformers/networkx/openpyxl — /estimate
#                     and /trace will fail, only /search + /health work.
#                     Only meaningful with `install`.
#
# Service is restarted every run regardless of mode — port conflicts auto-cleaned.

set -euo pipefail

DO_INSTALL=0
PORT="8001"
for arg in "$@"; do
  case "$arg" in
    install)         DO_INSTALL=1 ;;
    [1-9]*[0-9])     PORT="$arg" ;;
    -h|--help)       sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//;$d'; exit 0 ;;
    *)               echo "[!] Unknown arg: $arg (expected 'install' or a port number)"; exit 2 ;;
  esac
done

SERVICE_NAME="zrt-sim"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-server"
RUN_USER="${SUDO_USER:-$USER}"
FORCE="${FORCE:-0}"
SKIP_FULL_DEPS="${SKIP_FULL_DEPS:-0}"

PIP_STAMP="$VENV_DIR/.deploy-stamp"
APT_STAMP="/var/lib/zrt-sim-apt.stamp"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ $EUID -ne 0 ]]; then
  echo "[!] Re-run with sudo: sudo bash $0 $*"
  exit 1
fi

log() { printf "\033[1;36m[deploy]\033[0m %s\n" "$*"; }
skip() { printf "\033[1;33m[skip]\033[0m   %s\n" "$*"; }
warn() { printf "\033[1;31m[!]\033[0m      %s\n" "$*"; }

MODE_LABEL=$([[ $DO_INSTALL -eq 1 ]] && echo "install+deploy" || echo "deploy-only")
log "Project: $PROJECT_DIR | Port: $PORT | User: $RUN_USER | Mode: $MODE_LABEL"

# 1) System packages — skip if apt cache refreshed < 7 days ago -----------------
need_sys_install() {
  [[ "$FORCE" == "1" ]] && return 0
  [[ ! -f "$APT_STAMP" ]] && return 0
  local age=$(( $(date +%s) - $(stat -c %Y "$APT_STAMP") ))
  [[ $age -gt 604800 ]]   # > 7 days
}

if [[ $DO_INSTALL -eq 1 ]]; then
  if need_sys_install; then
    log "[1/5] Installing system packages..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -y
      apt-get install -y python3 python3-venv python3-pip iproute2
    elif command -v dnf >/dev/null 2>&1; then
      dnf install -y python3 python3-pip iproute
    elif command -v yum >/dev/null 2>&1; then
      yum install -y python3 python3-pip iproute
    fi
    touch "$APT_STAMP"
  else
    skip "[1/5] System packages — apt stamp fresh (< 7 days)."
  fi
else
  # Deploy-only mode: assume packages are present, just verify.
  missing=()
  command -v python3 >/dev/null || missing+=("python3")
  command -v ss      >/dev/null || missing+=("ss (iproute2)")
  if [[ ${#missing[@]} -gt 0 ]]; then
    warn "[1/5] Missing system packages: ${missing[*]}"
    warn "      Re-run with: sudo bash $0 install ${PORT}"
    exit 1
  fi
  skip "[1/5] System packages — OK (python3, ss)."
fi

# 2) venv (lazy) ---------------------------------------------------------------
if [[ $DO_INSTALL -eq 1 ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    log "[2/5] Creating venv at $VENV_DIR ..."
    sudo -u "$RUN_USER" python3 -m venv "$VENV_DIR"
    sudo -u "$RUN_USER" "$VENV_DIR/bin/pip" install --upgrade pip wheel
  else
    skip "[2/5] venv exists at $VENV_DIR"
  fi
else
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    warn "[2/5] venv missing at $VENV_DIR"
    warn "      Re-run with: sudo bash $0 install ${PORT}"
    exit 1
  fi
  skip "[2/5] venv — OK ($VENV_DIR)."
fi

# 3) Python deps — skip if nothing relevant has changed since last install ----
# Dropped: onnx (HTTP API never imports it), pandas (only training_search_util
# CLI uses it). Only the four packages /estimate + /trace actually need are
# installed here.
FULL_PKGS=(
  "torch>=2.0.0"
  "transformers>=4.36.0"
  "networkx>=3.0"
  "openpyxl>=3.1.0"
)

need_pip_install() {
  [[ "$FORCE" == "1" ]] && return 0
  [[ ! -f "$PIP_STAMP" ]] && return 0
  [[ "$PROJECT_DIR/server/requirements.txt" -nt "$PIP_STAMP" ]] && return 0
  # Bump deploy_server.sh (e.g. version pins) → triggers reinstall.
  [[ "${BASH_SOURCE[0]}" -nt "$PIP_STAMP" ]] && return 0
  return 1
}

if [[ $DO_INSTALL -eq 1 ]]; then
  if need_pip_install; then
    log "[3/5] Installing Python deps..."
    sudo -u "$RUN_USER" "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/server/requirements.txt"
    if [[ "$SKIP_FULL_DEPS" != "1" ]]; then
      log "      Installing ${FULL_PKGS[*]} (slow on first run)..."
      sudo -u "$RUN_USER" "$VENV_DIR/bin/pip" install "${FULL_PKGS[@]}"
    else
      skip "      SKIP_FULL_DEPS=1 — /trace will fail without torch."
    fi
    sudo -u "$RUN_USER" touch "$PIP_STAMP"
  else
    skip "[3/5] Python deps — unchanged since last install."
  fi
else
  if [[ ! -x "$VENV_DIR/bin/uvicorn" ]]; then
    warn "[3/5] uvicorn not found in venv"
    warn "      Re-run with: sudo bash $0 install ${PORT}"
    exit 1
  fi
  # Warn (don't fail) if the heavy deps look absent — /trace will break but
  # /estimate + /health can still serve.
  if ! "$VENV_DIR/bin/python" -c "import torch" 2>/dev/null; then
    warn "[3/5] torch not importable — /trace will fail (run install if needed)."
  else
    skip "[3/5] Python deps — uvicorn + torch present."
  fi
fi

# 4) systemd unit — only rewrite + restart if content changed ------------------
NEW_UNIT="$(cat <<EOF
[Unit]
Description=ZRT-Sim FastAPI server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR/python
ExecStart=$VENV_DIR/bin/uvicorn server.main:app --host 0.0.0.0 --port $PORT --workers 1
Restart=always
RestartSec=5
StartLimitIntervalSec=0
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF
)"

unit_changed=0
if [[ ! -f "$UNIT_FILE" ]] || [[ "$(cat "$UNIT_FILE")" != "$NEW_UNIT" ]]; then
  log "[4/5] Writing $UNIT_FILE ..."
  printf "%s\n" "$NEW_UNIT" > "$UNIT_FILE"
  systemctl daemon-reload
  unit_changed=1
else
  skip "[4/5] systemd unit unchanged."
fi

systemctl enable "$SERVICE_NAME" >/dev/null 2>&1 || true

# Force restart: stop the unit, kill ANY leftover process still holding $PORT
# (e.g. a manual `uvicorn` started before systemd took over, or a stale child
# that didn't exit cleanly), then start fresh. No more "address already in
# use" errors on redeploy.
leftover_pids() {
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltnp "sport = :$PORT" 2>/dev/null \
      | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null | sort -u
  fi
}

log "      Stopping any running instance..."
systemctl stop "$SERVICE_NAME" 2>/dev/null || true

pids="$(leftover_pids || true)"
if [[ -n "$pids" ]]; then
  log "      Killing leftover processes on port $PORT: $(echo $pids | tr '\n' ' ')"
  kill $pids 2>/dev/null || true
  sleep 1
  still="$(leftover_pids || true)"
  if [[ -n "$still" ]]; then
    log "      SIGKILL survivors: $(echo $still | tr '\n' ' ')"
    kill -9 $still 2>/dev/null || true
    sleep 1
  fi
fi

log "      Starting service..."
systemctl start "$SERVICE_NAME"

# 5) Firewall ------------------------------------------------------------------
log "[5/5] Ensuring firewall allows ${PORT}/tcp ..."
fw_handled=0
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q "Status: active"; then
  ufw allow "${PORT}/tcp" >/dev/null
  echo "      ufw: ${PORT}/tcp allowed."
  fw_handled=1
fi
if command -v firewall-cmd >/dev/null 2>&1 && systemctl is-active --quiet firewalld; then
  if ! firewall-cmd --query-port="${PORT}/tcp" >/dev/null 2>&1; then
    firewall-cmd --permanent --add-port="${PORT}/tcp" >/dev/null
    firewall-cmd --reload >/dev/null
    echo "      firewalld: ${PORT}/tcp added."
  else
    echo "      firewalld: ${PORT}/tcp already open."
  fi
  fw_handled=1
fi
if [[ $fw_handled -eq 0 ]] && command -v iptables >/dev/null 2>&1; then
  if iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null; then
    echo "      iptables: rule already present."
  else
    iptables -I INPUT -p tcp --dport "$PORT" -j ACCEPT
    if command -v netfilter-persistent >/dev/null 2>&1; then
      netfilter-persistent save >/dev/null
    elif [[ -d /etc/iptables ]] && command -v iptables-save >/dev/null 2>&1; then
      iptables-save > /etc/iptables/rules.v4
    fi
    echo "      iptables: rule added."
  fi
  fw_handled=1
fi
[[ $fw_handled -eq 0 ]] && echo "      No active firewall detected."

# SELinux: idempotent label
if command -v getenforce >/dev/null 2>&1 && [[ "$(getenforce)" == "Enforcing" ]] \
   && command -v semanage >/dev/null 2>&1; then
  if ! semanage port -l 2>/dev/null | awk '/^http_port_t/{print $3}' | tr ',' '\n' \
        | grep -qw "$PORT"; then
    semanage port -a -t http_port_t -p tcp "$PORT" 2>/dev/null || \
      semanage port -m -t http_port_t -p tcp "$PORT" 2>/dev/null || true
    echo "      SELinux: port $PORT labeled http_port_t."
  fi
fi

# Health + report --------------------------------------------------------------
sleep 2
if "$VENV_DIR/bin/python" - <<EOF >/dev/null 2>&1
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:$PORT/health", timeout=3).read()
except Exception:
    sys.exit(1)
EOF
then
  health="OK"
else
  health="FAILED — run: journalctl -u $SERVICE_NAME -n 80"
fi

echo
echo "== Done =="
echo "  /health    : $health"
echo "  Status     : systemctl status $SERVICE_NAME"
echo "  Live logs  : journalctl -u $SERVICE_NAME -f"
echo
echo "  Access from Windows on the same LAN — pick the right IP below:"
# Print every non-loopback IPv4 with its interface so multi-NIC machines
# don't mislead you into the wrong (e.g. Docker bridge) address.
if command -v ip >/dev/null 2>&1; then
  ip -o -4 addr show scope global \
    | awk -v port="$PORT" '{ split($4,a,"/"); printf "    %-10s http://%s:%s/\n", $2, a[1], port }'
else
  for ip in $(hostname -I 2>/dev/null); do
    printf "    %s http://%s:%s/\n" "" "$ip" "$PORT"
  done
fi
echo
echo "  Verify reachability from Windows PowerShell:"
echo "    Test-NetConnection <ip-above> -Port $PORT"
echo
echo "  Cloud VM? Open the same port in your provider's security group."
