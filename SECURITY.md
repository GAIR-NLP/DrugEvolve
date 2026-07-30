# Security Policy

## Reporting a vulnerability

Do not disclose exploitable path traversal, command execution, secret leakage,
or evaluator-isolation vulnerabilities in a public issue. Contact the maintainers
privately through the repository security advisory mechanism.

## Deployment guidance

- Treat evaluator commands and candidate code as untrusted.
- Run evaluators in an isolated container or restricted worker account.
- Mount only approved mutation paths as writable.
- Use short-lived API credentials and never put them in run specs.
- Keep `.env` files, run artifacts, and agent logs out of version control.
- Agent prompt/output bodies are redacted by default. Enable
  `DRUGEVOLVE_LOG_CONTENT=1` only for trusted local debugging.
- Review a run spec before setting `confirmed: true`.
- Keep direct training inside an isolated GPU container or restricted account.

The built-in path checks prevent accidental traversal but do not replace OS or
container isolation for adversarial candidate code.
