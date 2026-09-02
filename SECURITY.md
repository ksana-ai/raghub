# Security policy

## Supported versions

RAGHub is currently an experimental preview. Security fixes are applied only
to the latest commit on `main` until the first stable release.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's
**Report a vulnerability** flow in the Security tab of this repository.
Include the affected revision, reproduction steps, impact, and any suggested
mitigation. Maintainers will acknowledge a complete report within seven days.

Do not include real credentials, personal data, or customer documents in a
report. Test only against systems and data you are authorized to use.

## Security boundaries

The public HTTP API does not authenticate callers. `X-Tenant-ID` and
`X-Principal-ID` are development-only authorization context and are spoofable.
Do not expose RAGHub directly to untrusted networks or use it with sensitive
data without adding a trusted authentication layer and the operational
controls documented in the README.
