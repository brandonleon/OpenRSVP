"""IP filtering and access control for metrics endpoint."""

import ipaddress
from typing import Union

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from .models import Meta


def validate_cidr(cidr: str) -> bool:
    """Validate CIDR notation.

    Args:
        cidr: IP address or network in CIDR notation (e.g., "192.168.1.0/24")

    Returns:
        True if valid, False otherwise
    """
    try:
        # Try parsing as IPv4 or IPv6 network
        ipaddress.ip_network(cidr, strict=False)
        return True
    except ValueError:
        return False


def parse_cidr_list(
    cidr_string: str,
) -> list[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
    """Parse comma-separated list of CIDR ranges.

    Args:
        cidr_string: Comma-separated CIDR ranges (e.g., "192.168.1.0/24,10.0.0.0/8")

    Returns:
        List of IP network objects

    Raises:
        ValueError: If any CIDR is invalid
    """
    if not cidr_string or not cidr_string.strip():
        return []

    networks = []
    for cidr in cidr_string.split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError as e:
            raise ValueError(f"Invalid CIDR '{cidr}': {e}")

    return networks


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request.

    Handles X-Forwarded-For header for proxied requests (nginx).
    Takes the first IP in X-Forwarded-For chain as it represents the original client.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address as string
    """
    # Check X-Forwarded-For header (set by nginx proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take first IP in chain (original client)
        return forwarded_for.split(",")[0].strip()

    # Fall back to direct connection IP
    if request.client:
        return request.client.host

    # Fallback if no client info available
    return "unknown"


def check_ip_allowed(
    client_ip: str,
    allowed_ranges: list[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]],
) -> bool:
    """Check if client IP is in any of the allowed ranges.

    Args:
        client_ip: Client IP address as string
        allowed_ranges: List of allowed IP networks

    Returns:
        True if IP is allowed, False otherwise
    """
    if not allowed_ranges:
        return False

    try:
        client_addr = ipaddress.ip_address(client_ip)
    except ValueError:
        # Invalid IP address
        return False

    # Check if IP is in any allowed range
    for network in allowed_ranges:
        if client_addr in network:
            return True

    return False


def require_metrics_access(request: Request, db: Session) -> None:
    """Require that the request comes from an allowed IP range.

    Raises HTTPException(403) if access is denied.

    Args:
        request: FastAPI request object
        db: Database session

    Raises:
        HTTPException: 403 Forbidden if access denied
    """
    # Get allowed IP ranges from database
    meta_entry = db.query(Meta).filter(Meta.key == "metrics_allowed_ips").first()

    if not meta_entry or not meta_entry.value:
        raise HTTPException(
            status_code=403,
            detail="Metrics access denied. Configure allowed IPs with: openrsvp metrics-config set <cidr>",
        )

    # Parse allowed ranges
    try:
        allowed_ranges = parse_cidr_list(meta_entry.value)
    except ValueError as e:
        # Configuration error - deny access and log
        raise HTTPException(
            status_code=403,
            detail=f"Metrics configuration error: {e}",
        )

    if not allowed_ranges:
        raise HTTPException(
            status_code=403,
            detail="Metrics access denied. No IP ranges configured.",
        )

    # Get client IP and check access
    client_ip = get_client_ip(request)

    if not check_ip_allowed(client_ip, allowed_ranges):
        raise HTTPException(
            status_code=403,
            detail=f"Metrics access denied for IP {client_ip}. Configure allowed IPs with: openrsvp metrics-config set <cidr>",
        )
