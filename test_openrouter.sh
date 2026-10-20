#!/usr/bin/env bash
set -a
source .env
set +a

curl -s https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax/minimax-m3.1-flash:free",
    "messages": [{"role": "user", "content": "diga ok"}]
  }'
echo
