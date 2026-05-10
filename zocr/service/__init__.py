"""Reference service layer for deploying Z-OCR as an API.

This package contains the operational API surface: persistent jobs, tenant
scoping, auth, quotas, Redis worker integration, metrics, audit hooks, and
artifact downloads. Environment-specific controls such as WAF, mTLS, and
central log shipping still belong at the deployment layer.
"""
