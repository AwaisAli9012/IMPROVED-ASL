#!/usr/bin/env bash
# Share the running app (python -m signbridge) on a public https URL. Needs outbound port 7844.
# With CLOUDFLARE_TUNNEL_TOKEN in .env it runs your named tunnel (fixed address); without it, a temporary one.
cd "$(dirname "$0")/.." || exit 1
[ -f .env ] && set -a && . ./.env && set +a
BIN="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
  exec "$BIN" tunnel --no-autoupdate --protocol "${PROTOCOL:-http2}" run --token "$CLOUDFLARE_TUNNEL_TOKEN"
fi
exec "$BIN" tunnel --no-autoupdate --protocol "${PROTOCOL:-http2}" --url "http://127.0.0.1:${PORT:-5000}"
