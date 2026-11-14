# Security

## Reporting

Please report a vulnerability privately, through GitHub's
[private vulnerability reporting](https://github.com/ix64/sotaigen/security/advisories/new), not in a
public issue. Fixes go into the latest release only.

## What there is to attack

sotaigen runs on your own machine. The parts that take input from elsewhere:

- **The editor's local server** (`pnpm run editor`, `web/serve.mjs`) listens on 127.0.0.1 only and answers
  only a `Host` of `127.0.0.1`, `localhost` or `[::1]`, so a website whose name resolves to your machine
  (DNS rebinding) cannot read from it. A POST must come from one of its own pages or from no page at
  all, so another site's page cannot start or cancel a body. It is meant for one person on one machine:
  don't expose it to a network.
- **Parameter files** are read as JSON data and checked; nothing in them is run.
- **The `.blend` files it writes** hold no Python: the rig uses constraints and simple-expression drivers
  only, so they open correctly with Blender's *Auto Run Python Scripts* off. Blender itself runs with
  `--factory-startup`, which keeps that setting off.
- **`sotaigen swap-head`** opens a `.blend` you give it, in Blender with factory settings. Treat a head file
  from someone else as you would any `.blend`.

Reports about these, or anything else that crosses from data into code, are welcome.
