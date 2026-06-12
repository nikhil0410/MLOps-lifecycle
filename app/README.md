# FastAPI model serving

Endpoints
- `GET /health` — returns status
- `POST /predict` — accepts JSON payload: `{ "instances": [ {feature1: value, feature2: value, ...}, ... ] }`

Example `curl`:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '
{
  "instances": [
    {"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}
  ]
}'
```

Docker build & run

```bash
# Build image (from repo root)
docker build -t heart-disease-api:latest .

# Run container
docker run -p 8000:8000 --rm heart-disease-api:latest
```
