# Use Redis for shared coordination and PostgreSQL for appointment writes

Nova Desk serves a single clinic. It requires Redis for distributed rate limiting, short-lived booking idempotency locks, and caching non-PHI content from the public master FAQ before personalization. These are portfolio requirements even at low traffic.

PostgreSQL enforces slot capacity and commits each appointment change together with a record of its outcome. This lets a retry recover the original result if a Redis lock expires or a worker loses the response after committing.

## Initial rate limits

Allow 10 chat messages per minute per visitor and 60 chat messages per minute across all visitors. Allow 3 FAQ uploads per hour per clinic admin. All workers share these limits. When a limit is reached, tell the caller how long to wait and return that delay in `Retry-After` with HTTP 429. These are starting values to tune against observed use.
