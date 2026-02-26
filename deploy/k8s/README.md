# Kubernetes (Optional) / Kubernetes（任意）

These manifests are a minimal starting point for running the reference API in a cluster.

## Apply (kustomize)
```bash
kubectl apply -k deploy/k8s
```

## Notes
- The container expects writable `/tmp` and `/data` (PVC). Root FS can be read-only.
- Set `ZOCR_API_KEY` via Secret for internal auth.
- Expose the Service via your Ingress / Gateway as needed.

