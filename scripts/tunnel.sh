#!/usr/bin/env bash
# Share the running app (python -m signbridge) on a temporary public https URL. Needs outbound port 7844.
exec "${CLOUDFLARED:-$HOME/.local/bin/cloudflared}" tunnel --no-autoupdate --protocol "${PROTOCOL:-http2}" --url "http://127.0.0.1:${PORT:-5000}"
