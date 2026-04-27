#!/usr/bin/env bash
# First-time TLS certificate issuance via Let's Encrypt.
#
# Usage (explicit):  ./scripts/init-certs.sh <domain> <email>
# Usage (from .env): cp .env.example .env && edit .env, then ./scripts/init-certs.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Load .env if it exists and no args were passed
if [[ $# -eq 0 && -f .env ]]; then
    # shellcheck disable=SC1091
    set -o allexport
    source .env
    set +o allexport
fi

DOMAIN="${1:-${DOMAIN:-}}"
EMAIL="${2:-${EMAIL:-}}"

if [[ -z "${DOMAIN}" || -z "${EMAIL}" ]]; then
    echo "Error: DOMAIN and EMAIL are required."
    echo ""
    echo "Either:"
    echo "  1. Copy .env.example to .env, fill in DOMAIN and EMAIL, then re-run."
    echo "  2. Pass them as arguments: $0 <domain> <email>"
    exit 1
fi

BOOTSTRAP_CONF="deploy/nginx/bootstrap.conf"

echo "==> Swapping in HTTP-only bootstrap config"
docker cp "${BOOTSTRAP_CONF}" openrsvp_nginx:/etc/nginx/conf.d/default.conf

echo "==> Reloading nginx with bootstrap config"
if docker compose ps nginx | grep -q "Up"; then
    docker compose exec nginx nginx -s reload
else
    docker compose up -d nginx
fi

echo "==> Running certbot for initial certificate issuance"
docker compose run --rm certbot certonly \
    --webroot \
    --webroot-path /var/www/certbot \
    --email "${EMAIL}" \
    --agree-tos \
    --no-eff-email \
    -d "${DOMAIN}"

echo "==> Reloading nginx with SSL config (template will render with DOMAIN=${DOMAIN})"
docker compose up -d --force-recreate nginx

echo ""
echo "Certificate issuance complete for ${DOMAIN}."
echo ""
echo "Renewal is automatic — the certbot container runs crond and checks every 12 hours."
echo "Start the full stack with: docker compose up -d"
echo ""
