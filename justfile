# OpenRSVP task runner
# https://github.com/casey/just

# List available recipes
default:
    @just --list

# --- Development ---

# Start the development server
[group('dev')]
dev:
    uv run python main.py --dev

# Run all tests
[group('dev')]
test:
    uv run pytest

# Run a single test (e.g. just test-one test_event_create)
[group('dev')]
test-one name:
    uv run pytest tests/test_api.py::{{ name }}

# Lint and format
[group('dev')]
lint:
    uv run ruff check && uv run ruff format

# Type check
[group('dev')]
typecheck:
    uv run pyright

# Print shell alias instructions for bash, zsh, and nushell
[group('dev')]
aliases:
    @echo ""
    @echo "Add the following alias to your shell config so 'openrsvp' works"
    @echo "from anywhere inside this project directory:"
    @echo ""
    @echo "──────────────────────────────────────────────"
    @echo " bash  →  ~/.bashrc"
    @echo "──────────────────────────────────────────────"
    @echo "  alias openrsvp='uv run openrsvp'"
    @echo ""
    @echo "──────────────────────────────────────────────"
    @echo " zsh   →  ~/.zshrc"
    @echo "──────────────────────────────────────────────"
    @echo "  alias openrsvp='uv run openrsvp'"
    @echo ""
    @echo "──────────────────────────────────────────────"
    @echo " nushell  →  ~/.config/nushell/config.nu"
    @echo "──────────────────────────────────────────────"
    @echo "  alias openrsvp = uv run openrsvp"
    @echo ""
    @echo "After editing, reload your config:"
    @echo "  bash/zsh:  source ~/.bashrc  (or ~/.zshrc)"
    @echo "  nushell:   source ~/.config/nushell/config.nu"
    @echo ""

# --- Docker (dev) ---

# Start the dev Docker stack
[group('docker')]
docker-dev:
    docker compose -f docker-compose.dev.yml up --build

# Start production stack
[group('docker')]
up:
    docker compose up -d

# Stop production stack
[group('docker')]
down:
    docker compose down

# Follow logs (production)
[group('docker')]
logs *args:
    docker compose logs -f {{args}}

# --- Production (run on server) ---

# Default container for exec commands (override: just container=openrsvp_app_dev <recipe>)
container := "openrsvp_app"

# Pull latest code, rebuild containers, and run DB migrations
[group('prod')]
update:
    uv run openrsvp upgrade-container --prod
    uv run openrsvp upgrade-db

# Renew TLS certificates and reload nginx
[group('prod')]
renew-certs *args:
    docker exec -it {{container}} openrsvp renew-certs {{args}}

# Print current root admin token
[group('prod')]
admin-token:
    docker exec -it {{container}} openrsvp admin-token

# Rotate the root admin token
[group('prod')]
rotate-admin-token:
    docker exec -it {{container}} openrsvp rotate-admin-token

# Upgrade the SQLite schema (pass --no-backup to skip backup)
[group('prod')]
upgrade-db *args:
    docker exec -it {{container}} openrsvp upgrade-db {{args}}

# Run a manual decay cycle (pass --vacuum to also vacuum)
[group('prod')]
decay *args:
    docker exec -it {{container}} openrsvp decay {{args}}

# Seed the database with fake data
[group('prod')]
seed *args:
    docker exec -it {{container}} openrsvp seed-data {{args}}

# View or update persistent config (pass --show or --<key> <value>)
[group('prod')]
config *args:
    docker exec -it {{container}} openrsvp config {{args}}

# --- Metrics config ---

# List allowed IP ranges for /metrics
[group('metrics')]
metrics-config-list:
    docker exec -it {{container}} openrsvp metrics-config list

# Set allowed IP ranges (replaces existing): just metrics-config-set 192.168.1.0/24
[group('metrics')]
metrics-config-set *args:
    docker exec -it {{container}} openrsvp metrics-config set {{args}}

# Add a single IP range: just metrics-config-add 10.0.0.1/32
[group('metrics')]
metrics-config-add cidr:
    docker exec -it {{container}} openrsvp metrics-config add {{cidr}}

# Remove a single IP range: just metrics-config-remove 10.0.0.1/32
[group('metrics')]
metrics-config-remove cidr:
    docker exec -it {{container}} openrsvp metrics-config remove {{cidr}}

# Disable metrics (remove all IP ranges)
[group('metrics')]
metrics-config-clear:
    docker exec -it {{container}} openrsvp metrics-config clear --yes

# --- Release ---

# Bump patch version and tag
[group('release')]
release-patch:
    uv run openrsvp release patch

# Bump minor version and tag
[group('release')]
release-minor:
    uv run openrsvp release minor

# Bump major version and tag
[group('release')]
release-major:
    uv run openrsvp release major
