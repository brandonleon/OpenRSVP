#!/bin/sh
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting renewal check"
certbot renew \
    --webroot \
    -w /var/www/certbot \
    --quiet \
    --deploy-hook "docker exec openrsvp_nginx nginx -s reload"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Renewal check complete"
