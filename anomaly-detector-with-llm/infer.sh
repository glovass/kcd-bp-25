curl -v \
  -X POST \
  "http://localhost:8080/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @./input.json | jq

