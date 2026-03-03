# OpenRSVP Deployment Guide

## Prerequisites

- EC2 instance (Amazon Linux 2 / Ubuntu) with Docker and Docker Compose installed
- Domain pointed at the instance's public IP (A record propagated)
- Ports **80** and **443** open in the security group
- Repo cloned to `/home/ec2-user/OpenRSVP` (or similar)

---

## Initial TLS Certificate Issuance

The `scripts/init-certs.sh` script handles the chicken-and-egg problem: nginx needs
certs to start, but certbot needs nginx for the ACME HTTP challenge.

It does this by temporarily swapping in `deploy/nginx/bootstrap.conf` (HTTP-only),
obtaining the certificate, then restoring the full SSL config.

```bash
# From the repo root:
./scripts/init-certs.sh yourdomain.com you@example.com
```

After issuance, nginx will be running with TLS and the app will be reachable at
`https://yourdomain.com`.

> **Note:** Edit `deploy/nginx/default.conf` to replace `openrsvp.example.com`
> with your actual domain before running the stack for the first time.

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

## Starting the Stack

```bash
# First time (after init-certs.sh):
docker compose up -d

# Subsequent restarts:
docker compose restart
```

---

## Stopping the Stack

```bash
docker compose down
```
