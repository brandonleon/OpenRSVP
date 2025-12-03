#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Running certbot renew via docker compose..."
docker compose run --rm certbot renew --webroot -w /var/www/certbot

echo "Reloading nginx to pick up renewed certificates..."
docker compose exec nginx nginx -s reload

echo "Certificate renewal complete."
