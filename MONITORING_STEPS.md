# Step 8 Monitoring and Logging Guide

This guide explains how Step 8 is implemented for the assignment using:

- application request logging
- a Prometheus-compatible `/metrics` endpoint
- Kubernetes scrape annotations for local monitoring

The goal is to demonstrate simple but practical monitoring for the deployed API.

## What Step 8 Requires

The assignment asks for:

1. logging of API requests
2. simple monitoring using Prometheus, Grafana, or API metrics/log dashboards

This implementation covers both requirements in a lightweight way.

## What Was Added

### 1. API request logging

The FastAPI app now logs:

- HTTP method
- request path
- response status code
- request duration
- prediction batch size
- prediction errors

This helps demonstrate operational observability even without a full external logging stack.

### 2. Metrics endpoint

The app now exposes:

```text
/metrics
```

This endpoint returns Prometheus-format metrics.

### 3. Prometheus-compatible counters and latency metrics

The API publishes metrics for:

- total requests
- request latency
- total prediction records processed

### 4. Kubernetes scrape annotations

The Kubernetes deployment includes annotations so Prometheus can scrape metrics from the pod:

- `prometheus.io/scrape: "true"`
- `prometheus.io/path: "/metrics"`
- `prometheus.io/port: "8000"`

## Endpoints Available

After Step 8, the application exposes:

- `/health`
- `/predict`
- `/metrics`

## How to Verify Logging

After deploying the application in Kubernetes, trigger a few API calls and inspect pod logs:

```bash
kubectl get pods
kubectl logs <pod-name>
```

You should see request log lines showing:

- method
- path
- status code
- duration

Prediction requests should also log batch size.

## How to Verify Metrics

Use port-forwarding if needed:

```bash
kubectl port-forward svc/heart-disease-api-service 8000:80
```

Then open:

```bash
curl http://127.0.0.1:8000/metrics
```

You should see Prometheus-formatted metric output.

Look for metric names such as:

- `heart_disease_api_requests_total`
- `heart_disease_api_request_latency_seconds`
- `heart_disease_api_predictions_total`

## Example Monitoring Flow

1. Call `/health`
2. Call `/predict`
3. Open `/metrics`
4. Verify counters increased
5. Check pod logs

This is enough to demonstrate that:

- requests are being logged
- the application exposes measurable runtime metrics

## Kubernetes Verification

Check the deployment annotations:

```bash
kubectl describe deployment heart-disease-api
```

or inspect the pod:

```bash
kubectl describe pod <pod-name>
```

You should see the Prometheus scrape annotations attached to the pod template.

## If You Want to Mention Prometheus/Grafana in the Report

You can state:

- the API is Prometheus-ready
- metrics are exposed at `/metrics`
- pod annotations are present for scraping
- Docker Desktop / local Kubernetes deployment can be extended with Prometheus and Grafana if needed

This keeps the implementation honest while still aligning with the assignment expectation of simple monitoring.

## Suggested Screenshots for Step 8

Take screenshots of:

1. `/metrics` output in terminal or browser
2. `kubectl logs <pod-name>` showing request logs
3. optional `kubectl describe pod <pod-name>` showing scrape annotations

## Summary

Step 8 is implemented through:

1. request logging in FastAPI
2. Prometheus-style metrics at `/metrics`
3. Kubernetes annotations for scraping

This is a lightweight and assignment-appropriate monitoring setup that builds directly on the current deployment.
