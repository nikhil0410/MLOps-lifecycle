# Step 7 Deployment Guide

This guide explains how to deploy the Dockerized FastAPI application to the local Kubernetes cluster provided by Docker Desktop. It is designed to help complete **Step 7: Production Deployment** of the assignment.

## Objective

Deploy the existing `heart-disease-api` Docker image to Kubernetes and expose the API so that:

- `/health` returns a successful response
- `/predict` accepts JSON input and returns predictions

This satisfies the local Kubernetes deployment path in the assignment.

## Prerequisites

Before starting, make sure the following are already completed:

1. Docker image exists locally.
   Example:

   ```bash
   docker images
   ```

   The image should include a tag such as:

   ```bash
   heart-disease-api:v2
   ```

2. Docker Desktop Kubernetes is enabled.

   In Docker Desktop:

   - Open `Settings`
   - Go to `Kubernetes`
   - Enable Kubernetes

3. `kubectl` is available and pointing to Docker Desktop.

   Verify with:

   ```bash
   kubectl config current-context
   kubectl get nodes
   ```

   The context should usually be:

   ```bash
   docker-desktop
   ```

## Why a New Docker Image Was Needed

A new image was required because the codebase changed significantly:

- updated API behavior
- updated model artifacts
- updated training/export logic
- updated dependencies

Docker images are snapshots. If the image is old, Kubernetes will run stale application code or stale models.

## Recommended Kubernetes Resources

For local deployment, the simplest structure is:

1. `Deployment`
   This manages the API pod lifecycle.

2. `Service`
   This exposes the pod within Kubernetes and makes it reachable.

3. Optional `Ingress`
   This is not required for the first working deployment, but can be added later for a more production-like setup.

## Example Deployment Manifest

Create a file such as `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: heart-disease-api
  labels:
    app: heart-disease-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: heart-disease-api
  template:
    metadata:
      labels:
        app: heart-disease-api
    spec:
      containers:
        - name: heart-disease-api
          image: heart-disease-api:v2
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 20
```

## Example Service Manifest

Create a file such as `k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: heart-disease-api-service
spec:
  selector:
    app: heart-disease-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: NodePort
```

## Why `imagePullPolicy: IfNotPresent` Is Important

Because the Docker image is stored locally in Docker Desktop and is not necessarily pushed to Docker Hub or another registry, Kubernetes should reuse the local image if it already exists.

Using:

```yaml
imagePullPolicy: IfNotPresent
```

helps Docker Desktop Kubernetes use the local image instead of always trying to pull from a remote registry.

## Deployment Commands

From the project root:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Verification Commands

Check deployment status:

```bash
kubectl get deployments
kubectl get pods
kubectl get svc
```

For more detail:

```bash
kubectl describe deployment heart-disease-api
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

## Accessing the API

The easiest local access method is `port-forward`.

Run:

```bash
kubectl port-forward svc/heart-disease-api-service 8000:80
```

Now test the API locally.

Health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
      }
    ]
  }'
```

## Expected Outcome

Successful `/health` response:

```json
{"status":"ok"}
```

Successful `/predict` response should include:

- `predictions`
- `probabilities`

## Common Issues and Fixes

### 1. Pod stuck in `ImagePullBackOff`

Reason:
- Kubernetes cannot find the image locally or is trying to pull from a remote registry.

Fix:

```bash
docker build -t heart-disease-api:v2 .
kubectl rollout restart deployment/heart-disease-api
```

Also verify the manifest uses:

```yaml
image: heart-disease-api:v2
imagePullPolicy: IfNotPresent
```

### 2. Pod starts but API is not reachable

Check:

```bash
kubectl get pods
kubectl logs <pod-name>
kubectl get svc
```

Then use `port-forward` again:

```bash
kubectl port-forward svc/heart-disease-api-service 8000:80
```

### 3. Readiness/Liveness probe failures

Reason:
- The app may not be fully ready when probes start.

Fix:
- Increase `initialDelaySeconds`
- Confirm `/health` works locally in Docker first

## What to Capture for the Assignment

Take screenshots of:

1. `kubectl get pods`
2. `kubectl get svc`
3. successful `/health` response
4. successful `/predict` response

These screenshots can be added to the final report for Step 7.

## Suggested Next Improvement

Once `Deployment + Service` works, the next enhancement is to add:

- `Ingress`

This gives a cleaner, more production-like Kubernetes setup and helps strengthen the Step 7 submission.

## Summary

The deployment flow is:

1. Build the updated Docker image
2. Enable Docker Desktop Kubernetes
3. Apply Deployment and Service manifests
4. Verify pods and services
5. Port-forward the service
6. Test `/health` and `/predict`
7. Capture screenshots for submission

This is the simplest local Kubernetes approach to complete Step 7 successfully.
