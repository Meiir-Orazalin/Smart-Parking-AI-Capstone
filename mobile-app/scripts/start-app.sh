#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${EXPO_PORT:-8081}"

find_lan_ip() {
  local ip=""

  if command -v route >/dev/null 2>&1; then
    local iface
    iface="$(route get default 2>/dev/null | awk '/interface: / { print $2; exit }')"
    if [ -n "${iface:-}" ] && command -v ipconfig >/dev/null 2>&1; then
      ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
    fi
  fi

  if [ -z "$ip" ] && command -v hostname >/dev/null 2>&1; then
    ip="$(hostname -I 2>/dev/null | awk '{ print $1 }' || true)"
  fi

  printf '%s' "$ip"
}

LAN_IP="$(find_lan_ip)"

if [ -n "$LAN_IP" ]; then
  export REACT_NATIVE_PACKAGER_HOSTNAME="$LAN_IP"
  echo "Using REACT_NATIVE_PACKAGER_HOSTNAME=$REACT_NATIVE_PACKAGER_HOSTNAME"
else
  echo "Could not determine LAN IP automatically. Expo will use its default networking settings."
fi

cd "$ROOT_DIR"
exec node ./node_modules/expo/bin/cli start --go --lan --port "$PORT" -c
