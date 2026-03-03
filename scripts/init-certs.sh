#!/usr/bin/env bash
# First-time TLS certificate issuance via Let's Encrypt.
# Usage: ./scripts/init-certs.sh <domain> <email>
# Example: ./scripts/init-certs.sh openrsvp.example.com admin@example.com
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DOMAIN="${1:-}"
EMAIL="${2:-}"

if [[ -z "${DOMAIN}" || -z "${EMAIL}" ]]; then
    echo "Usage: $0 <domain> <email>"
    exit 1
fi

NGINX_CONF="deploy/nginx/default.conf"
NGINX_BACKUP="deploy/nginx/default.conf.bak"
BOOTSTRAP_CONF="deploy/nginx/bootstrap.conf"

echo "==> Backing up nginx config to ${NGINX_BACKUP}"
cp "${NGINX_CONF}" "${NGINX_BACKUP}"

echo "==> Swapping in HTTP-only bootstrap config"
cp "${BOOTSTRAP_CONF}" "${NGINX_CONF}"

echo "==> Reloading nginx with bootstrap config"
if docker compose ps nginx | grep -q "Up"; then
    docker compose exec nginx nginx -s reload
else
    docker compose up -d nginx
fi

echo "==> Running certbot for initial certificate issuance"
docker compose run certbot certonly \
    --webroot \
    --webroot-path /var/www/certbot \
    --email "${EMAIL}" \
    --agree-tos \
    --no-eff-email \
    -d "${DOMAIN}"

echo "==> Restoring SSL nginx config from backup"
cp "${NGINX_BACKUP}" "${NGINX_CONF}"
rm "${NGINX_BACKUP}"

echo "==> Reloading nginx with SSL config"
docker compose exec nginx nginx -s reload

echo ""
echo "Certificate issuance complete for ${DOMAIN}."
echo ""
echo "Renewal is automatic — the certbot container runs crond and checks every 12 hours."
echo "Start the full stack with: docker compose up -d"
echo ""
