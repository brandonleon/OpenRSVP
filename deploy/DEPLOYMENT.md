# OpenRSVP Deployment Guide

## Prerequisites

- EC2 instance (Amazon Linux 2023 / Ubuntu) with Docker and Docker Compose installed
- Domain pointed at the instance's public IP (A record propagated)
- Ports **80** and **443** open in the security group
- Repo cloned to `/home/ec2-user/OpenRSVP` (or similar)

---

## First-Time Setup

### 1. Configure your domain and email

```bash
cp .env.example .env
# Edit .env — set DOMAIN and EMAIL
```

`.env` is read by both Docker Compose (to pass `DOMAIN` into the nginx container)
and by `init-certs.sh` (for certificate issuance).

### 2. Issue the TLS certificate

```bash
./scripts/init-certs.sh
```

This script handles the chicken-and-egg problem: nginx needs certs to start, but
certbot needs nginx for the ACME HTTP challenge. It does this by temporarily loading
the HTTP-only bootstrap config, obtaining the certificate, then restoring the SSL
config (rendered from `deploy/nginx/default.conf.template` with your `DOMAIN`).

You can also pass domain and email as arguments instead of using `.env`:

```bash
./scripts/init-certs.sh yourdomain.com you@example.com
```

### 3. Start the full stack

```bash
docker compose up -d
```

The app will be reachable at `https://<your-domain>`.

---

## How nginx configuration works

`deploy/nginx/default.conf.template` contains `${DOMAIN}` placeholders. The official
nginx Docker image processes this template on startup via `envsubst`, producing the
live config at `/etc/nginx/conf.d/default.conf` inside the container. The `DOMAIN`
variable is passed in from your `.env` file via Docker Compose.

---

## Automated Renewal

Renewal is fully automatic — no host-side cron job required. The certbot container
runs `crond` internally and checks for certificate renewal every 12 hours. When a
certificate is renewed, the deploy hook runs `docker exec openrsvp_nginx nginx -s reload`
to pick up the new certificate without restarting nginx.

> **Note on Docker socket:** The certbot container mounts `/var/run/docker.sock` so
> it can exec into the nginx container on renewal. This grants the certbot container
> root-equivalent access to the Docker daemon, which is the standard trade-off for
> container-to-container reload signaling on a self-hosted single-server deployment.

To trigger a manual renewal check:

```bash
docker compose exec certbot /renew.sh
```

---

## Log Location

Certbot cron logs are written inside the container:

```
/var/log/certbot-cron.log
```

View them with:

```bash
docker compose logs certbot
# or
docker compose exec certbot cat /var/log/certbot-cron.log
```

Certbot's own verbose logs are in:

```
./deploy/certs/logs/letsencrypt.log
```

---

## Starting / Stopping

```bash
# Start all services
docker compose up -d

# Restart
docker compose restart

# Stop
docker compose down
```
