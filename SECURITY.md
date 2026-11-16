# Security policy

## Reporting a problem

Please don't open a public issue for a security problem. Write to **sales@4shadow.io** with "Security" in the subject, or use GitHub's private **Report a vulnerability** form on this repository's Security tab where it is enabled.

Tell us what you found, how to reproduce it, and what someone could do with it. We'll confirm we have it within three working days, keep you posted while we fix it, and credit you when the fix is released unless you'd rather we didn't.

This covers the code in this repository. For a problem with the hosted API at api.4shadow.io or the site at 4shadow.io, write to the same address.

## Supported versions

Fixes go into `main`, which is what api.4shadow.io runs.

## What the code is built to protect

- **The server's resources.** Every request is bounded: forecast hours, field size (variables, stride, bounding box), threshold areas (cells and vertices), place-search lengths and MCP table sizes all have limits, so one request can't take a worker's memory or minutes of CPU.
- **The store.** The API only reads it. Run ids, model names, variables and tile coordinates are checked against what exists before any file is opened, so no request can reach a path outside the store.
- **Place search.** It runs on a local, read-only SQLite file with parameterised queries; nothing about a lookup leaves the server.

The API has no authentication of its own. A deployment decides who may call it with a proxy or gateway in front (see [docs/deploy.md](docs/deploy.md)); on the hosted service that is the API-key router.

A report that one of these doesn't hold is exactly what we want to hear.
