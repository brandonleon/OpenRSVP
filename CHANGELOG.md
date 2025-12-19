# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to SemVer.

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

## [0.16.0] - 2025-12-19

### Added

- Deployment guide and updated docs navigation, plus API messaging details.
- Channel discovery page with search, sorting, and pagination.
- "My Events" and "My RSVPs" pages driven by local saved magic links.
- Event share button with native share sheet support and clipboard fallback.
- Location gating for approval-required events in the API and ICS export.
- Dev server reload flag (`openrsvp runserver --dev`).
- Docker upgrade options for host port and data directory overrides.
- Security probe script, Kubernetes diagrams, and scaling notes.
- Expanded API and privacy-focused test coverage.

### Changed

- Hide event location and unapproved RSVP counts from public API/listing views
  unless the requester is authorized.
- API event payloads now omit private channel metadata unless authorized.
- Home page channel selector limited to top-ranked public channels.
- Share and channel UI copy updated across templates and docs.

### Fixed

- Public API privacy leaks for private RSVPs and private channels.
