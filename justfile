# OpenRSVP task runner
# https://github.com/casey/just

# List available recipes
default:
    @just --list

# --- Development ---

# Start the development server
dev:
    uv run python main.py --dev

# Run all tests
test:
    uv run pytest

# Run a single test (e.g. just test-one test_event_create)
test-one name:
    uv run pytest tests/test_api.py::{{ name }}

# Lint and format
lint:
    uv run ruff check && uv run ruff format

# Type check
typecheck:
    uv run pyright

# Seed the dev database with fake data
seed:
    uv run openrsvp seed-data

# Run the decay cycle manually
decay:
    uv run openrsvp decay

# --- Docker (dev) ---

# Start the dev Docker stack
docker-dev:
    docker compose -f docker-compose.dev.yml up --build

# --- Production (run on server) ---

# Pull latest code, rebuild containers, and run DB migrations
update:
    uv run openrsvp upgrade-container --prod
    uv run openrsvp upgrade-db

# Print shell alias instructions for bash, zsh, and nushell
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
