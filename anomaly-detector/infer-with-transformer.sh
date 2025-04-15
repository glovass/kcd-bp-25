curl -v \
  -H "Content-Type: application/json" \
  -d @./log-inference-input.json \
  http://localhost:8083/v2/models/detector/infer
