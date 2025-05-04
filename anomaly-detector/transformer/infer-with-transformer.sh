curl -X POST \
  "http://localhost:8080/v2/models/detector/infer" \
  -H 'Content-Type: application/json' \
  -d @./log-inference-input.json 