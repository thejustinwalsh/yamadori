# PackageLens MCP (npm packagelens-mcp 0.1.11, MIT), the first MCP server the
# proxy hosts for the model (mcp/mcp_host.py; operator, 2026-09-28: "Yes get
# package lens"). The proxy is its MCP client over stdio; the container runs
# long-lived with every capability dropped, a read-only root, no credentials
# (no GITHUB_TOKEN) and its network only through the egress gate
# (bench/sandbox/sandbox_net.py: an --internal network whose one way out is
# HTTPS to global addresses).
#
# Build ONLY through `python mcp/mcp_host.py build packagelens`: --pull=false,
# so the base below must already be in the local image store (it is the
# harness box's base, bench/sandbox/harness/Dockerfile).
#
# Base: Node 24 LTS (24.21.0, bookworm), the multi-arch index digest the
# harness box pins. Node 24's fetch honours HTTP(S)_PROXY when
# NODE_USE_ENV_PROXY=1 (set at run time by mcp_host.py).
ARG BASE=node:24.21.0-bookworm@sha256:64af3819f9275802414d7cdc38c27e9d82bd564dec4d4da87d008255d36c63b4
FROM ${BASE}

ENV NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false

WORKDIR /opt/packagelens
COPY package.json package-lock.json ./
# `npm ci`: exactly the lockfile, every tarball checked against its sha512.
# Lifecycle scripts OFF (none of the 16 packages declares an install
# script; the source audit is recorded in models/manifest.yaml, runtimes:
# mcp-packagelens). `npm audit signatures`: every package's registry
# signature, or the build fails.
RUN npm ci --omit=dev --ignore-scripts \
 && npm audit signatures --omit=dev \
 && npm cache clean --force \
 && node -e "const p=require('/opt/packagelens/node_modules/packagelens-mcp/package.json'); if (p.version !== '0.1.11') process.exit(1); console.log(p.name, p.version)"

# The server's cwd: an empty directory, so smart_search's codebase detection
# (it stats package.json and friends in process.cwd()) finds nothing.
USER node
WORKDIR /home/node
ENTRYPOINT ["node", "/opt/packagelens/node_modules/packagelens-mcp/dist/index.js"]
LABEL org.opencontainers.image.title="yamadori-mcp-packagelens" \
      yamadori.mcp.server="packagelens" \
      yamadori.mcp.packagelens="0.1.11"
