# The harness box (docs/HARNESS-SANDBOX.md, SELF-IMPROVEMENT-LOG #49):
# OpenCode 1.18.32, Pi 0.87.1 and Codex 0.157.1, the official npm packages at
# the exact versions of the host tests, in one pinned image, with their
# default loadout's tools (docs/HARNESSES.md "Default loadout"): the Playwright
# MCP server (Microsoft; it attaches over CDP to the browser sidecar and
# downloads no browser), typescript-language-server + typescript + pyright
# (language-server diagnostics: OpenCode's built-in lsp, and `tsc` /
# `pyright` through the shell for Pi and Codex), agent-browser (Vercel; Pi's
# browser CLI) and fd. No key, no config and no project is in it:
# harness_box.py mounts a per-run home and the project, and passes the key as
# an environment variable at run time.
#
# Build ONLY through `python bench/sandbox/harness_box.py build`: it refuses
# when the base below is not already in the local image store (a pull is a
# download the operator decides on), and builds with --pull=false.
#
# Base: Node 24 LTS (24.21.0, bookworm), the multi-arch index digest read from
# Docker Hub on 2026-09-26 (`docker buildx imagetools inspect`; amd64
# manifest sha256:5a750d3b..., 8 layers, ~410 MB compressed). The full
# bookworm image, not -slim: it already carries git, curl, ca-certificates
# and python3, which the harnesses and their tools need, so nothing is
# installed from an unpinned apt mirror.
ARG BASE=node:24.21.0-bookworm@sha256:64af3819f9275802414d7cdc38c27e9d82bd564dec4d4da87d008255d36c63b4
FROM ${BASE}

ENV NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false

WORKDIR /opt/harness
COPY package.json package-lock.json ./
# `npm ci`: exactly the lockfile, every tarball checked against its sha512
# (package-lock.json; the five Pi workspace packages its shrinkwrap left
# without one were filled from the registry's dist.integrity). Then
# `npm audit signatures`: every installed package's registry signature (and
# provenance attestation where one is published) is verified, or the build
# fails. Lifecycle scripts OFF, then ONLY opencode-ai's own postinstall,
# which puts the platform binary at bin/opencode.exe (its package.json
# `bin`). agent-browser's postinstall only patches global shims; its linux
# binary ships in the tarball without the exec bit, so it is set here and
# linked directly.
RUN npm ci --omit=dev --ignore-scripts \
 && npm audit signatures --omit=dev \
 && npm rebuild opencode-ai --foreground-scripts \
 && npm cache clean --force \
 && chmod 0755 node_modules/agent-browser/bin/agent-browser-linux-x64 \
 && ln -s /opt/harness/node_modules/.bin/opencode /usr/local/bin/opencode \
 && ln -s /opt/harness/node_modules/.bin/pi /usr/local/bin/pi \
 && ln -s /opt/harness/node_modules/.bin/codex /usr/local/bin/codex \
 && ln -s /opt/harness/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex-path/rg /usr/local/bin/rg \
 && for b in playwright-mcp typescript-language-server tsc tsserver pyright pyright-langserver; do \
      ln -s "/opt/harness/node_modules/.bin/$b" "/usr/local/bin/$b"; done \
 && ln -s /opt/harness/node_modules/agent-browser/bin/agent-browser-linux-x64 /usr/local/bin/agent-browser \
 && opencode --version && pi --version && codex --version && rg --version \
 && playwright-mcp --version && typescript-language-server --version && tsc --version \
 && pyright --version && agent-browser --version

# fd for Pi's `find` tool (the default loadout). The official sharkdp/fd
# release, checked against the sha256 GitHub records for the asset (release
# API `digest`, read 2026-09-26). Pi runs with PI_OFFLINE=1 and never
# downloads it.
ADD --checksum=sha256:761c72dc8e120d85b22292063be8a796e2eeb20eb3e4f38b8fa2343ccf3514a7 \
    https://github.com/sharkdp/fd/releases/download/v10.5.0/fd-v10.5.0-x86_64-unknown-linux-musl.tar.gz /tmp/fd.tgz
RUN tar -xzf /tmp/fd.tgz -C /tmp \
 && install -m 0755 /tmp/fd-v10.5.0-x86_64-unknown-linux-musl/fd /usr/local/bin/fd \
 && rm -rf /tmp/fd.tgz /tmp/fd-v10.5.0-x86_64-unknown-linux-musl \
 && fd --version

# ripgrep on PATH is the one Codex ships (pinned by the lockfile's sha512).
# Pi runs with PI_OFFLINE=1 (as on the host) and never downloads rg/fd, so
# without these its grep/find tools fail; OpenCode uses rg from PATH instead
# of fetching it.
# The unprivileged user of the base image (uid 1000). harness_box.py runs
# the container as this user with every capability dropped.
USER node
WORKDIR /work
LABEL org.opencontainers.image.title="yamadori-harness-box" \
      yamadori.harness.opencode="1.18.32" \
      yamadori.harness.pi="0.87.1" \
      yamadori.harness.codex="0.157.1" \
      yamadori.harness.loadout="tools1"
