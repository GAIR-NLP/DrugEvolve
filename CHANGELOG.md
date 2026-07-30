# Changelog

All notable changes to DrugEvolve will be documented here.

## 0.1.0 - Unreleased

### Added

- Dependency-light `drugevolve` core package and CLI.
- YAML run specs with explicit preflight confirmation.
- Atomic local experiment storage and separate cognition storage.
- Random, greedy, UCB1, and island sampling.
- Structured subprocess evaluator with process-group timeouts.
- Resumable state, event logs, per-step artifacts, and best snapshots.
- Minimal deterministic example, tests, packaging metadata, and CI.

### Changed

- Local storage is the compatibility backend; the private remote-storage
  adapter has been removed.
- Agent models and task prompts are deployment configuration.
- Objective/judge scoring channels and score direction are explicit.
- Candidate training now runs directly on the local GPU machine with timeout
  and process-group termination.
- Agent prompt/output logging is redacted by default.

### Fixed

- Algorithm serialization mismatch in duplicate detection.
- Agent file path traversal checks.
- Failure loops that ignored the experiment budget.
- Training monitor cleanup and cancellation behavior.
- Implementer success claims that did not modify candidate code.
- Legacy/core algorithm conversion and sampler metadata consistency.
