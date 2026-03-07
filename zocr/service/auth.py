from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Iterable, Optional
from urllib.request import urlopen

from .metadata import normalize_tenant_id


@dataclass(frozen=True)
class Principal:
    auth_mode: str
    subject: str
    tenant_id: Optional[str]
    is_admin: bool = False
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestAuthContext:
    principal: Principal
    tenant_id: str


@dataclass(frozen=True)
class ApiKeyRecord:
    api_key: str
    subject: str
    tenant_id: Optional[str]
    is_admin: bool = False
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class JwtSettings:
    secret: Optional[str]
    algorithms: tuple[str, ...]
    issuer: Optional[str]
    audience: Optional[str]
    tenant_claim: str
    subject_claim: str
    roles_claim: str
    scope_claim: str
    admin_claim: str
    admin_scope: Optional[str]
    clock_skew_sec: int


@dataclass(frozen=True)
class JwksCache:
    keys: tuple[Dict[str, Any], ...]
    issuer: Optional[str]
    fetched_at: float


_DEFAULT_ROLE_SCOPES: dict[str, tuple[str, ...]] = {
    "viewer": ("zocr.jobs.read", "zocr.metrics.read"),
    "operator": ("zocr.jobs.read", "zocr.metrics.read", "zocr.jobs.write", "zocr.jobs.run"),
    "admin": (
        "zocr.jobs.read",
        "zocr.metrics.read",
        "zocr.audit.read",
        "zocr.jobs.write",
        "zocr.jobs.run",
        "zocr.jobs.delete",
        "zocr.tenants.read",
        "zocr.tenants.write",
        "zocr.tenants.approve",
        "zocr.tenants.override",
    ),
}


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = str(detail)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _split_scopes(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(sorted({item for item in value.replace(",", " ").split() if item}))
    if isinstance(value, Iterable) and not isinstance(value, (bytes, dict, str)):
        scopes = [str(item).strip() for item in value if str(item).strip()]
        return tuple(sorted(set(scopes)))
    return ()


def _split_roles(value: Any) -> tuple[str, ...]:
    roles = _split_scopes(value)
    return tuple(sorted({role for role in roles if role}))


def _normalize_tenant_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw in {"*", "__any__", "any"}:
        return None
    return normalize_tenant_id(raw)


def _extract_bearer_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    token = value.strip()
    if not token.lower().startswith("bearer "):
        return None
    token = token[7:].strip()
    return token or None


def _looks_like_jwt(token: str) -> bool:
    return token.count(".") == 2


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment + padding)
    except (ValueError, binascii.Error) as exc:
        raise AuthError(401, "Invalid JWT encoding") from exc


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _json_from_url(url: str) -> Dict[str, Any]:
    try:
        with urlopen(url, timeout=5) as response:
            raw = response.read()
    except Exception as exc:
        raise AuthError(401, f"Failed to fetch JWKS/OIDC metadata: {exc}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AuthError(401, "Invalid JWKS/OIDC JSON") from exc
    if not isinstance(parsed, dict):
        raise AuthError(401, "Invalid JWKS/OIDC JSON")
    return parsed


def _normalize_api_key_record(api_key: str, payload: Any, *, fallback_subject: str) -> ApiKeyRecord:
    if not api_key:
        raise ValueError("API key must not be empty")
    if isinstance(payload, str):
        return ApiKeyRecord(
            api_key=api_key,
            subject=fallback_subject,
            tenant_id=_normalize_tenant_or_none(payload),
        )
    if not isinstance(payload, dict):
        raise ValueError("API key payload must be an object or tenant string")
    tenant_id = _normalize_tenant_or_none(payload.get("tenant_id"))
    subject = str(payload.get("subject") or fallback_subject).strip() or fallback_subject
    is_admin = bool(payload.get("admin") or payload.get("is_admin"))
    roles = _split_roles(payload.get("roles") or payload.get("role"))
    scopes = _split_scopes(payload.get("scopes") or payload.get("scope"))
    return ApiKeyRecord(
        api_key=api_key,
        subject=subject,
        tenant_id=tenant_id,
        is_admin=is_admin or tenant_id is None or "admin" in roles,
        roles=roles,
        scopes=scopes,
    )


def _load_api_key_records(default_tenant: Optional[str]) -> dict[str, ApiKeyRecord]:
    records: dict[str, ApiKeyRecord] = {}

    legacy_key = (os.environ.get("ZOCR_API_KEY") or "").strip()
    if legacy_key:
        legacy_tenant = os.environ.get("ZOCR_API_KEY_TENANT") or default_tenant or "default"
        legacy_roles = _split_roles(
            os.environ.get("ZOCR_API_KEY_ROLES") or os.environ.get("ZOCR_API_KEY_ROLE") or ""
        )
        legacy_scopes = _split_scopes(os.environ.get("ZOCR_API_KEY_SCOPES") or "")
        record = ApiKeyRecord(
            api_key=legacy_key,
            subject="legacy-api-key",
            tenant_id=_normalize_tenant_or_none(legacy_tenant),
            is_admin="admin" in legacy_roles,
            roles=legacy_roles,
            scopes=legacy_scopes,
        )
        records[record.api_key] = record

    raw_json = (os.environ.get("ZOCR_API_KEYS_JSON") or "").strip()
    if raw_json:
        try:
            loaded = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ZOCR_API_KEYS_JSON must be valid JSON.") from exc
        if isinstance(loaded, dict):
            for api_key, payload in loaded.items():
                record = _normalize_api_key_record(str(api_key), payload, fallback_subject="api-key")
                records[record.api_key] = record
        elif isinstance(loaded, list):
            for index, item in enumerate(loaded):
                if not isinstance(item, dict):
                    raise RuntimeError("ZOCR_API_KEYS_JSON list entries must be objects.")
                api_key = str(item.get("api_key") or item.get("key") or "").strip()
                record = _normalize_api_key_record(api_key, item, fallback_subject=f"api-key-{index}")
                records[record.api_key] = record
        else:
            raise RuntimeError("ZOCR_API_KEYS_JSON must be a JSON object or array.")

    return records


def _build_jwt_settings() -> Optional[JwtSettings]:
    secret = (os.environ.get("ZOCR_API_JWT_SECRET") or "").strip() or None
    has_jwks = any(
        (os.environ.get(name) or "").strip()
        for name in (
            "ZOCR_API_JWT_JWKS_JSON",
            "ZOCR_API_JWT_JWKS_PATH",
            "ZOCR_API_JWT_JWKS_URL",
            "ZOCR_API_OIDC_DISCOVERY_URL",
        )
    )
    if not secret and not has_jwks:
        return None
    algorithms = tuple(
        item.strip().upper()
        for item in (os.environ.get("ZOCR_API_JWT_ALGORITHMS") or "HS256").split(",")
        if item.strip()
    )
    return JwtSettings(
        secret=secret,
        algorithms=algorithms or ("HS256",),
        issuer=(os.environ.get("ZOCR_API_JWT_ISSUER") or "").strip() or None,
        audience=(os.environ.get("ZOCR_API_JWT_AUDIENCE") or "").strip() or None,
        tenant_claim=(os.environ.get("ZOCR_API_JWT_TENANT_CLAIM") or "tenant_id").strip() or "tenant_id",
        subject_claim=(os.environ.get("ZOCR_API_JWT_SUBJECT_CLAIM") or "sub").strip() or "sub",
        roles_claim=(os.environ.get("ZOCR_API_JWT_ROLES_CLAIM") or "roles").strip() or "roles",
        scope_claim=(os.environ.get("ZOCR_API_JWT_SCOPE_CLAIM") or "scope").strip() or "scope",
        admin_claim=(os.environ.get("ZOCR_API_JWT_ADMIN_CLAIM") or "is_admin").strip() or "is_admin",
        admin_scope=(os.environ.get("ZOCR_API_JWT_ADMIN_SCOPE") or "zocr.admin").strip() or None,
        clock_skew_sec=max(0, _env_int("ZOCR_API_JWT_CLOCK_SKEW_SEC", 30)),
    )


class JwksProvider:
    def __init__(self) -> None:
        self.jwks_json = (os.environ.get("ZOCR_API_JWT_JWKS_JSON") or "").strip()
        self.jwks_path = (os.environ.get("ZOCR_API_JWT_JWKS_PATH") or "").strip()
        self.jwks_url = (os.environ.get("ZOCR_API_JWT_JWKS_URL") or "").strip()
        self.oidc_discovery_url = (os.environ.get("ZOCR_API_OIDC_DISCOVERY_URL") or "").strip()
        self.cache_ttl_sec = max(1, _env_int("ZOCR_API_JWT_JWKS_CACHE_TTL_SEC", 300))
        self._cache: Optional[JwksCache] = None
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.jwks_json or self.jwks_path or self.jwks_url or self.oidc_discovery_url)

    def issuer(self) -> Optional[str]:
        cache = self._load_cache()
        return cache.issuer if cache else None

    def resolve_jwk(self, *, kid: Optional[str], alg: str, kty: Optional[str] = None) -> Dict[str, Any]:
        cache = self._load_cache()
        if cache is None:
            raise AuthError(401, "JWKS is not configured")
        candidates = list(cache.keys)
        if kid:
            candidates = [jwk for jwk in candidates if str(jwk.get("kid") or "") == kid]
        if alg:
            alg_candidates = [jwk for jwk in candidates if not jwk.get("alg") or str(jwk.get("alg")).upper() == alg]
            if alg_candidates:
                candidates = alg_candidates
        if kty:
            candidates = [jwk for jwk in candidates if str(jwk.get("kty") or "").upper() == kty.upper()]
        candidates = [
            jwk
            for jwk in candidates
            if not jwk.get("use") or str(jwk.get("use")).lower() == "sig"
        ]
        key_ops_candidates = []
        for jwk in candidates:
            key_ops = jwk.get("key_ops")
            if not key_ops:
                key_ops_candidates.append(jwk)
                continue
            if isinstance(key_ops, list) and any(str(item).lower() == "verify" for item in key_ops):
                key_ops_candidates.append(jwk)
        if key_ops_candidates:
            candidates = key_ops_candidates
        if not candidates:
            raise AuthError(401, "No matching JWKS key")
        return candidates[0]

    def _load_cache(self) -> Optional[JwksCache]:
        if not self.enabled:
            return None
        now = time.time()
        if self._cache is not None and (now - self._cache.fetched_at) < float(self.cache_ttl_sec):
            return self._cache
        with self._lock:
            if self._cache is not None and (now - self._cache.fetched_at) < float(self.cache_ttl_sec):
                return self._cache
            self._cache = self._fetch_cache(now)
            return self._cache

    def _fetch_cache(self, fetched_at: float) -> JwksCache:
        issuer = None
        if self.jwks_json:
            try:
                jwks_payload = json.loads(self.jwks_json)
            except Exception as exc:
                raise AuthError(401, f"Invalid JWKS JSON: {exc}") from exc
        elif self.jwks_path:
            try:
                with open(self.jwks_path, "r", encoding="utf-8") as handle:
                    jwks_payload = json.load(handle)
            except Exception as exc:
                raise AuthError(401, f"Failed to load JWKS file: {exc}") from exc
        else:
            jwks_url = self.jwks_url
            if self.oidc_discovery_url:
                discovery = _json_from_url(self.oidc_discovery_url)
                issuer = str(discovery.get("issuer") or "").strip() or None
                jwks_url = str(discovery.get("jwks_uri") or jwks_url).strip()
            if not jwks_url:
                raise AuthError(401, "JWKS URL is not configured")
            jwks_payload = _json_from_url(jwks_url)

        if not isinstance(jwks_payload, dict):
            raise AuthError(401, "Invalid JWKS payload")
        raw_keys = jwks_payload.get("keys")
        if not isinstance(raw_keys, list):
            raise AuthError(401, "JWKS payload missing keys")
        keys: list[Dict[str, Any]] = []
        for raw in raw_keys:
            if isinstance(raw, dict):
                keys.append(dict(raw))
        if not keys:
            raise AuthError(401, "JWKS payload contains no keys")
        return JwksCache(keys=tuple(keys), issuer=issuer, fetched_at=fetched_at)


class AuthBackend:
    def __init__(self, *, multi_tenant_enabled: bool, default_tenant: str):
        self.multi_tenant_enabled = bool(multi_tenant_enabled)
        self.default_tenant = str(default_tenant or "").strip()
        self.api_keys = _load_api_key_records(self.default_tenant or None)
        self.jwt = _build_jwt_settings()
        self.jwks = JwksProvider()

        mode = (os.environ.get("ZOCR_API_AUTH_MODE") or "auto").strip().lower()
        if mode == "auto":
            if self.api_keys and self.jwt:
                mode = "hybrid"
            elif self.jwt:
                mode = "jwt"
            elif self.api_keys:
                mode = "api_key"
            else:
                mode = "none"
        if mode not in {"none", "api_key", "jwt", "hybrid"}:
            raise RuntimeError(
                "Unsupported ZOCR_API_AUTH_MODE=%r. Supported modes: auto, none, api_key, jwt, hybrid."
                % (mode,)
            )
        if mode in {"jwt", "hybrid"} and self.jwt is None:
            raise RuntimeError(
                "JWT auth requires ZOCR_API_JWT_SECRET or JWKS/OIDC settings."
            )
        if mode in {"api_key", "hybrid"} and not self.api_keys:
            raise RuntimeError(
                "ZOCR_API_KEY or ZOCR_API_KEYS_JSON is required when API key auth is enabled."
            )
        self.mode = mode
        self.role_scopes = dict(_DEFAULT_ROLE_SCOPES)
        self.authz_strict = _env_truthy("ZOCR_API_AUTHZ_STRICT", False)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def api_key_enabled(self) -> bool:
        return self.mode in {"api_key", "hybrid"}

    @property
    def jwt_enabled(self) -> bool:
        return self.mode in {"jwt", "hybrid"}

    def _resolve_tenant(self, principal: Principal, x_tenant_id: Optional[str]) -> str:
        requested_tenant = None
        if x_tenant_id is not None and str(x_tenant_id).strip():
            try:
                requested_tenant = normalize_tenant_id(x_tenant_id)
            except ValueError as exc:
                raise AuthError(400, str(exc)) from exc

        if principal.tenant_id:
            tenant_id = normalize_tenant_id(principal.tenant_id)
            if requested_tenant and requested_tenant != tenant_id:
                raise AuthError(403, "Tenant mismatch")
            return tenant_id

        if self.can_override_tenant(principal):
            if requested_tenant:
                return requested_tenant
            if self.default_tenant:
                return normalize_tenant_id(self.default_tenant)
            raise AuthError(400, "Missing tenant id")

        if not self.enabled:
            if requested_tenant:
                return requested_tenant
            if self.default_tenant:
                return normalize_tenant_id(self.default_tenant)
            raise AuthError(400, "Missing tenant id")

        if self.default_tenant:
            return normalize_tenant_id(self.default_tenant)
        raise AuthError(403, "Authenticated principal has no tenant binding")

    def _principal_from_api_key(self, api_key: str) -> Principal:
        record = self.api_keys.get(api_key)
        if record is None:
            raise AuthError(403, "Invalid API key")
        return Principal(
            auth_mode="api_key",
            subject=record.subject,
            tenant_id=record.tenant_id,
            is_admin=record.is_admin,
            roles=record.roles,
            scopes=record.scopes,
        )

    def _principal_from_jwt(self, token: str) -> Principal:
        assert self.jwt is not None
        try:
            header_segment, payload_segment, signature_segment = token.split(".")
        except ValueError as exc:
            raise AuthError(401, "Malformed JWT") from exc

        header = self._json_segment(header_segment, "JWT header")
        payload = self._json_segment(payload_segment, "JWT payload")
        alg = str(header.get("alg") or "").upper()
        if alg not in self.jwt.algorithms:
            raise AuthError(401, "Unsupported JWT algorithm")
        signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
        signature = _b64url_decode(signature_segment)
        kid = str(header.get("kid") or "").strip() or None
        self._verify_jwt_signature(alg=alg, kid=kid, signing_input=signing_input, signature=signature)

        now = time.time()
        skew = float(self.jwt.clock_skew_sec)
        exp = payload.get("exp")
        if exp is not None and now > float(exp) + skew:
            raise AuthError(401, "JWT expired")
        nbf = payload.get("nbf")
        if nbf is not None and now + skew < float(nbf):
            raise AuthError(401, "JWT not active yet")
        iat = payload.get("iat")
        if iat is not None and now + skew < float(iat):
            raise AuthError(401, "JWT issued in the future")

        expected_issuer = self.jwt.issuer or self.jwks.issuer()
        if expected_issuer and payload.get("iss") != expected_issuer:
            raise AuthError(401, "JWT issuer mismatch")
        if self.jwt.audience:
            aud = payload.get("aud")
            aud_values = {str(item) for item in aud} if isinstance(aud, list) else {str(aud)} if aud else set()
            if self.jwt.audience not in aud_values:
                raise AuthError(401, "JWT audience mismatch")

        roles = _split_roles(payload.get(self.jwt.roles_claim))
        scopes = _split_scopes(payload.get(self.jwt.scope_claim))
        is_admin = bool(payload.get(self.jwt.admin_claim))
        if self.jwt.admin_scope and self.jwt.admin_scope in scopes:
            is_admin = True
        if "admin" in roles:
            is_admin = True

        subject = str(payload.get(self.jwt.subject_claim) or "jwt").strip() or "jwt"
        tenant_id = _normalize_tenant_or_none(payload.get(self.jwt.tenant_claim))
        return Principal(
            auth_mode="jwt",
            subject=subject,
            tenant_id=tenant_id,
            is_admin=is_admin,
            roles=roles,
            scopes=scopes,
        )

    def _verify_jwt_signature(
        self,
        *,
        alg: str,
        kid: Optional[str],
        signing_input: bytes,
        signature: bytes,
    ) -> None:
        if alg.startswith("HS"):
            self._verify_hs_signature(alg=alg, kid=kid, signing_input=signing_input, signature=signature)
            return
        if alg.startswith("RS"):
            self._verify_rs_signature(alg=alg, kid=kid, signing_input=signing_input, signature=signature)
            return
        raise AuthError(401, "Unsupported JWT algorithm")

    def _verify_hs_signature(
        self,
        *,
        alg: str,
        kid: Optional[str],
        signing_input: bytes,
        signature: bytes,
    ) -> None:
        digestmod = {
            "HS256": hashlib.sha256,
            "HS384": hashlib.sha384,
            "HS512": hashlib.sha512,
        }.get(alg)
        if digestmod is None:
            raise AuthError(401, "Unsupported JWT algorithm")
        candidate_keys: list[bytes] = []
        if self.jwt and self.jwt.secret:
            candidate_keys.append(self.jwt.secret.encode("utf-8"))
        if self.jwks.enabled:
            try:
                jwk = self.jwks.resolve_jwk(kid=kid, alg=alg, kty="oct")
            except AuthError:
                jwk = None
            if jwk is not None:
                raw_k = str(jwk.get("k") or "").strip()
                if not raw_k:
                    raise AuthError(401, "Invalid oct JWKS key")
                candidate_keys.append(_b64url_decode(raw_k))
        for key in candidate_keys:
            expected = hmac.new(key, signing_input, digestmod).digest()
            if hmac.compare_digest(expected, signature):
                return
        raise AuthError(401, "Invalid JWT signature")

    def _verify_rs_signature(
        self,
        *,
        alg: str,
        kid: Optional[str],
        signing_input: bytes,
        signature: bytes,
    ) -> None:
        digestmod = {
            "RS256": hashlib.sha256,
            "RS384": hashlib.sha384,
            "RS512": hashlib.sha512,
        }.get(alg)
        digest_prefix = {
            "RS256": bytes.fromhex("3031300d060960864801650304020105000420"),
            "RS384": bytes.fromhex("3041300d060960864801650304020205000430"),
            "RS512": bytes.fromhex("3051300d060960864801650304020305000440"),
        }.get(alg)
        if digestmod is None or digest_prefix is None:
            raise AuthError(401, "Unsupported JWT algorithm")
        jwk = self.jwks.resolve_jwk(kid=kid, alg=alg, kty="RSA")
        modulus_raw = str(jwk.get("n") or "").strip()
        exponent_raw = str(jwk.get("e") or "").strip()
        if not modulus_raw or not exponent_raw:
            raise AuthError(401, "Invalid RSA JWKS key")
        modulus = int.from_bytes(_b64url_decode(modulus_raw), "big")
        exponent = int.from_bytes(_b64url_decode(exponent_raw), "big")
        if modulus <= 0 or exponent <= 0:
            raise AuthError(401, "Invalid RSA JWKS key")
        signature_int = int.from_bytes(signature, "big")
        key_size = (modulus.bit_length() + 7) // 8
        encoded = pow(signature_int, exponent, modulus).to_bytes(key_size, "big")
        digest = digestmod(signing_input).digest()
        digest_info = digest_prefix + digest
        padding_len = key_size - len(digest_info) - 3
        if padding_len < 8:
            raise AuthError(401, "Invalid RSA JWT signature")
        expected = b"\x00\x01" + (b"\xff" * padding_len) + b"\x00" + digest_info
        if not hmac.compare_digest(encoded, expected):
            raise AuthError(401, "Invalid JWT signature")

    def _json_segment(self, segment: str, label: str) -> Dict[str, Any]:
        raw = _b64url_decode(segment)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise AuthError(401, f"Invalid {label}") from exc
        if not isinstance(parsed, dict):
            raise AuthError(401, f"Invalid {label}")
        return parsed

    def authenticate(
        self,
        *,
        x_api_key: Optional[str],
        authorization: Optional[str],
        x_tenant_id: Optional[str],
    ) -> RequestAuthContext:
        principal = self.authenticate_principal(x_api_key=x_api_key, authorization=authorization)
        return self.resolve_request_context(principal=principal, x_tenant_id=x_tenant_id)

    def authenticate_principal(
        self,
        *,
        x_api_key: Optional[str],
        authorization: Optional[str],
    ) -> Principal:
        bearer_token = _extract_bearer_token(authorization)
        api_key_candidate = (x_api_key or "").strip() or None

        principal: Optional[Principal] = None
        if api_key_candidate and self.api_key_enabled:
            principal = self._principal_from_api_key(api_key_candidate)
        elif bearer_token and self.jwt_enabled and _looks_like_jwt(bearer_token):
            principal = self._principal_from_jwt(bearer_token)
        elif bearer_token and self.api_key_enabled:
            principal = self._principal_from_api_key(bearer_token)
        elif self.enabled:
            raise AuthError(401, "Missing credentials")
        else:
            principal = Principal(auth_mode="none", subject="anonymous", tenant_id=None, is_admin=False)
        return principal

    def resolve_request_context(
        self,
        *,
        principal: Principal,
        x_tenant_id: Optional[str],
    ) -> RequestAuthContext:
        tenant_id = self._resolve_tenant(principal, x_tenant_id)
        return RequestAuthContext(principal=principal, tenant_id=tenant_id)

    def expanded_scopes(self, principal: Principal) -> set[str]:
        expanded = set(principal.scopes)
        for role in principal.roles:
            expanded.update(self.role_scopes.get(role, ()))
        return expanded

    def can_override_tenant(self, principal: Principal) -> bool:
        if principal.is_admin:
            return True
        scopes = self.expanded_scopes(principal)
        if not scopes:
            return False
        return "*" in scopes or "zocr.tenants.override" in scopes

    def require_scopes(self, principal: Principal, *required_scopes: str) -> None:
        required = tuple(scope for scope in required_scopes if scope)
        if not required or not self.enabled or principal.is_admin:
            return
        scopes = self.expanded_scopes(principal)
        if not scopes and not self.authz_strict:
            return
        if "*" in scopes:
            return
        if any(scope in scopes for scope in required):
            return
        raise AuthError(403, "Missing required scope: %s" % (required[0],))


def build_auth_backend(*, multi_tenant_enabled: bool, default_tenant: str) -> AuthBackend:
    return AuthBackend(multi_tenant_enabled=multi_tenant_enabled, default_tenant=default_tenant)
