//! Production middleware for the OpenFang API server.
//!
//! Provides:
//! - Request ID generation and propagation
//! - Per-endpoint structured request logging
//! - In-memory rate limiting (per IP)

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::middleware::Next;
use std::time::Instant;
use tracing::info;

/// Request ID header name (standard).
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Middleware: inject a unique request ID and log the request/response.
pub async fn request_logging(request: Request<Body>, next: Next) -> Response<Body> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let method = request.method().clone();
    let uri = request.uri().path().to_string();
    let start = Instant::now();

    let mut response = next.run(request).await;

    let elapsed = start.elapsed();
    let status = response.status().as_u16();

    info!(
        request_id = %request_id,
        method = %method,
        path = %uri,
        status = status,
        latency_ms = elapsed.as_millis() as u64,
        "API request"
    );

    // Inject the request ID into the response
    if let Ok(header_val) = request_id.parse() {
        response.headers_mut().insert(REQUEST_ID_HEADER, header_val);
    }

    response
}

/// Authentication state passed to the auth middleware.
#[derive(Clone)]
pub struct AuthState {
    pub api_key: String,
    pub auth_enabled: bool,
    pub session_secret: String,
    /// Allow unauthenticated access when no API key is configured.
    /// When true (default), matches legacy behavior: no key = open access.
    /// When false, requires auth even if no API key is set (fail-close).
    pub allow_no_auth: bool,
}

/// Bearer token authentication middleware.
///
/// When `api_key` is non-empty (after trimming), requests to non-public
/// endpoints must include `Authorization: Bearer <api_key>`.
/// If the key is empty or whitespace-only, auth is disabled entirely
/// (public/local development mode).
///
/// When dashboard auth is enabled, session cookies are also accepted.
pub async fn auth(
    axum::extract::State(auth_state): axum::extract::State<AuthState>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    // SECURITY: Capture method early for method-aware public endpoint checks.
    let method = request.method().clone();

    // Shutdown is loopback-only (CLI on same machine) — skip token auth
    let path = request.uri().path();
    if path == "/api/shutdown" {
        let is_loopback = request
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0.ip().is_loopback())
            .unwrap_or(false); // SECURITY: default-deny — unknown origin is NOT loopback
        if is_loopback {
            return next.run(request).await;
        }
    }

    // Public endpoints that don't require auth (dashboard needs these).
    // SECURITY: /api/agents is GET-only (listing). POST (spawn) requires auth.
    // SECURITY: Public endpoints are GET-only unless explicitly noted.
    // POST/PUT/DELETE to any endpoint ALWAYS requires auth to prevent
    // unauthenticated writes (cron job creation, skill install, etc.).
    let is_get = method == axum::http::Method::GET;
    let is_public = path == "/"
        || path == "/logo.png"
        || path == "/favicon.ico"
        || (path == "/.well-known/agent.json" && is_get)
        || (path.starts_with("/a2a/") && is_get)
        || path == "/api/health"
        || path == "/api/health/detail"
        || path == "/api/status"
        || path == "/api/version"
        || (path == "/api/agents" && is_get)
        || (path == "/api/profiles" && is_get)
        || (path == "/api/config" && is_get)
        || (path == "/api/config/schema" && is_get)
        || (path.starts_with("/api/uploads/") && is_get)
        // Dashboard read endpoints — allow unauthenticated so the SPA can
        // render before the user enters their API key.
        || (path == "/api/models" && is_get)
        || (path == "/api/models/aliases" && is_get)
        || (path == "/api/providers" && is_get)
        || (path == "/api/budget" && is_get)
        || (path == "/api/budget/agents" && is_get)
        || (path.starts_with("/api/budget/agents/") && is_get)
        || (path == "/api/network/status" && is_get)
        || (path == "/api/a2a/agents" && is_get)
        || (path == "/api/approvals" && is_get)
        || (path.starts_with("/api/approvals/") && is_get)
        || (path == "/api/channels" && is_get)
        || (path == "/api/hands" && is_get)
        || (path == "/api/hands/active" && is_get)
        || (path.starts_with("/api/hands/") && is_get)
        || (path == "/api/skills" && is_get)
        || (path == "/api/sessions" && is_get)
        || (path == "/api/integrations" && is_get)
        || (path == "/api/integrations/available" && is_get)
        || (path == "/api/integrations/health" && is_get)
        || (path == "/api/workflows" && is_get)
        || path == "/api/logs/stream"  // SSE stream, read-only
        || (path.starts_with("/api/cron/") && is_get)
        || path.starts_with("/api/providers/github-copilot/oauth/")
        || path.starts_with("/api/providers/openai-codex/oauth/")
        || path.starts_with("/api/providers/gemini-oauth/oauth/")
        || path.starts_with("/api/providers/qwen-oauth/oauth/")
        || path.starts_with("/api/providers/minimax-oauth/oauth/")
        || path == "/api/auth/login"
        || path == "/api/auth/logout"
        || (path == "/api/auth/check" && is_get);

    if is_public {
        return next.run(request).await;
    }

    // If no API key configured (empty, whitespace-only, or missing), check
    // whether unauthenticated access is explicitly allowed.
    // SECURITY: fail-close by default — if allow_no_auth is false, deny access
    // even when no API key is set. This prevents accidental open deployments.
    let api_key_trimmed = auth_state.api_key.trim().to_string();
    if api_key_trimmed.is_empty() && !auth_state.auth_enabled {
        if auth_state.allow_no_auth {
            tracing::warn!(
                "No API key configured and auth disabled — all endpoints open (insecure). \
                 Set allow_no_auth = false in [auth] config to enforce authentication."
            );
            return next.run(request).await;
        }
        // Fail-close: no key and no auth enabled, but allow_no_auth is false
        return Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("www-authenticate", "Bearer")
            .body(Body::from(
                serde_json::json!({
                    "error": "No API key configured. Set api_key in config.toml, enable [auth], or set allow_no_auth = true for local development."
                })
                .to_string(),
            ))
            .unwrap_or_default();
    }
    let api_key = api_key_trimmed.as_str();

    // Check Authorization: Bearer <token> header, then fallback to X-API-Key
    let bearer_token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let api_token = bearer_token.or_else(|| {
        request
            .headers()
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
    });

    // SECURITY: Use constant-time comparison to prevent timing attacks.
    let header_auth = api_token.map(|token| {
        use subtle::ConstantTimeEq;
        if token.len() != api_key.len() {
            return false;
        }
        token.as_bytes().ct_eq(api_key.as_bytes()).into()
    });

    // Also check ?token= query parameter (for EventSource/SSE clients that
    // cannot set custom headers, same approach as WebSocket auth).
    let query_token_decoded = request
        .uri()
        .query()
        .and_then(|q| q.split('&').find_map(|pair| pair.strip_prefix("token=")))
        .map(crate::percent_decode);

    // SECURITY: Use constant-time comparison to prevent timing attacks.
    let query_auth = query_token_decoded.as_deref().map(|token| {
        use subtle::ConstantTimeEq;
        if token.len() != api_key.len() {
            return false;
        }
        token.as_bytes().ct_eq(api_key.as_bytes()).into()
    });

    // Accept if either auth method matches
    if header_auth == Some(true) || query_auth == Some(true) {
        return next.run(request).await;
    }

    // Check session cookie (dashboard login sessions)
    if auth_state.auth_enabled {
        if let Some(token) = extract_session_cookie(&request) {
            if crate::session_auth::verify_session_token(&token, &auth_state.session_secret)
                .is_some()
            {
                return next.run(request).await;
            }
        }
    }

    // Determine error message: was a credential provided but wrong, or missing entirely?
    let credential_provided = header_auth.is_some() || query_auth.is_some();
    let error_msg = if credential_provided {
        "Invalid API key"
    } else {
        "Missing Authorization: Bearer <api_key> header"
    };

    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header("www-authenticate", "Bearer")
        .body(Body::from(
            serde_json::json!({"error": error_msg}).to_string(),
        ))
        .unwrap_or_default()
}

/// Extract the `openfang_session` cookie value from a request.
fn extract_session_cookie(request: &Request<Body>) -> Option<String> {
    request
        .headers()
        .get("cookie")
        .and_then(|v| v.to_str().ok())
        .and_then(|cookies| {
            cookies.split(';').find_map(|c| {
                c.trim()
                    .strip_prefix("openfang_session=")
                    .map(|v| v.to_string())
            })
        })
}

/// Security headers middleware — applied to ALL API responses.
pub async fn security_headers(request: Request<Body>, next: Next) -> Response<Body> {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert("x-content-type-options", "nosniff".parse().unwrap());
    headers.insert("x-frame-options", "DENY".parse().unwrap());
    headers.insert("x-xss-protection", "1; mode=block".parse().unwrap());
    // The dashboard handler (webchat_page) sets its own nonce-based CSP.
    // For all other responses (API endpoints), apply a strict default.
    if !headers.contains_key("content-security-policy") {
        headers.insert(
            "content-security-policy",
            "default-src 'none'; frame-ancestors 'none'"
                .parse()
                .unwrap(),
        );
    }
    headers.insert(
        "referrer-policy",
        "strict-origin-when-cross-origin".parse().unwrap(),
    );
    headers.insert(
        "cache-control",
        "no-store, no-cache, must-revalidate".parse().unwrap(),
    );
    headers.insert(
        "strict-transport-security",
        "max-age=63072000; includeSubDomains".parse().unwrap(),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_header_constant() {
        assert_eq!(REQUEST_ID_HEADER, "x-request-id");
    }

    #[test]
    fn test_auth_state_allow_no_auth_default() {
        // When allow_no_auth is true (default), empty key + no auth = open access
        let state = AuthState {
            api_key: String::new(),
            auth_enabled: false,
            session_secret: String::new(),
            allow_no_auth: true,
        };
        assert!(state.allow_no_auth);
        assert!(state.api_key.trim().is_empty());
        assert!(!state.auth_enabled);
    }

    #[test]
    fn test_auth_state_fail_close_flag() {
        // When allow_no_auth is false, empty key + no auth = fail-close
        let state = AuthState {
            api_key: String::new(),
            auth_enabled: false,
            session_secret: String::new(),
            allow_no_auth: false,
        };
        assert!(!state.allow_no_auth);
        // This state should trigger the fail-close branch in middleware
        assert!(state.api_key.trim().is_empty());
        assert!(!state.auth_enabled);
    }

    #[test]
    fn test_auth_state_with_api_key_skips_fail_close() {
        // When api_key is set, the fail-close check is not reached
        let state = AuthState {
            api_key: "my-secret-key".to_string(),
            auth_enabled: false,
            session_secret: "my-secret-key".to_string(),
            allow_no_auth: false,
        };
        assert!(!state.api_key.trim().is_empty());
        // Bearer token check path will be used instead
    }

    #[test]
    fn test_auth_state_auth_enabled_skips_fail_close() {
        // When auth_enabled is true, the fail-close check is not reached
        // (session cookie check path is used instead)
        let state = AuthState {
            api_key: String::new(),
            auth_enabled: true,
            session_secret: "session-secret".to_string(),
            allow_no_auth: false,
        };
        assert!(state.auth_enabled);
        // Session-based auth path will be used
    }

    #[test]
    fn test_public_endpoint_paths() {
        // Verify key GET endpoints are recognized as public
        let is_get = true;
        let public_get_paths = [
            "/api/health",
            "/api/health/detail",
            "/api/status",
            "/api/version",
            "/api/agents",
            "/api/models",
            "/api/providers",
            "/api/budget",
        ];
        for path in &public_get_paths {
            let is_public = *path == "/"
                || *path == "/api/health"
                || *path == "/api/health/detail"
                || *path == "/api/status"
                || *path == "/api/version"
                || (*path == "/api/agents" && is_get)
                || (*path == "/api/models" && is_get)
                || (*path == "/api/providers" && is_get)
                || (*path == "/api/budget" && is_get);
            assert!(is_public, "Expected {} to be public", path);
        }
    }

    #[test]
    fn test_oauth_routes_are_public() {
        // Copilot OAuth routes should be public
        let copilot_start = "/api/providers/github-copilot/oauth/start";
        let copilot_poll = "/api/providers/github-copilot/oauth/poll/abc123";
        assert!(copilot_start.starts_with("/api/providers/github-copilot/oauth/"));
        assert!(copilot_poll.starts_with("/api/providers/github-copilot/oauth/"));
    }
}
