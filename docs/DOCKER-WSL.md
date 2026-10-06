# Docker Engine in WSL Ubuntu (since 2026-10-06)

Yamadori's containers run on **Docker Engine (docker-ce 29.8.2) inside the Ubuntu WSL distro**, not on Docker
Desktop. Operator, 2026-10-06: "docker is down because WSL is misconfigured, has startup errors and uses too much
memory and CPU"; approved "Docker Engine in WSL" and the cleanup of the distro. Docker Desktop stays INSTALLED
(uninstalling it is the operator's call) and is set not to start (`AutoStart` false and its Run key removed in
`%APPDATA%\Docker\settings-store.json`; its WSL integration of Ubuntu off). Its data (the 69 GB
`%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx`) is untouched.

What uses it: `mcp/mcp_host.py` (PackageLens + its gate, `bench/sandbox/sandbox_net.py`), `mcp/npm_resolve.py`,
`mcp/typecheck.py` (the volumes `yamadori-typecheck-tools1`, `yamadori-prove-npm-*`), and `bench/*`.

## How it is wired

- **Engine**: `dockerd` under systemd in Ubuntu. Unix socket (root) plus the API on **127.0.0.1:2376 ONLY, mutual
  TLS** (`/etc/systemd/system/docker.service.d/yamadori.conf`; CA, server and client keys in `/etc/docker/tls`,
  mode 0700). Never 0.0.0.0: this box is on ZeroTier. Checked: the Windows listener is `127.0.0.1:2376` (wslrelay)
  and a connect to 10.242.120.152 / 10.0.0.44 / 172.29.32.1 on 2376 fails; a client with no certificate is refused.
  Why not plain `tcp://127.0.0.1:2375`: unauthenticated root API that any local process or a browser page
  (DNS rebinding) could reach.
- **Windows client**: Docker Desktop's own `docker.exe` (28.1.1, already on PATH) with a **docker context**,
  `wsl-ubuntu` (`tcp://127.0.0.1:2376`, certs in `%USERPROFILE%\.docker\wsl-engine\`), made current
  (`%USERPROFILE%\.docker\config.json` `currentContext`). A context, not `DOCKER_HOST`: every process (the stack's
  Scheduled Tasks, the watchdog's restarts, a shell) reads it with no per-script environment, so
  `start-stack.bat`, `watchdog.ps1` and `bench/deploy_layout_v3.py envs_for` need nothing. (`DOCKER_HOST` or
  `DOCKER_CONTEXT` in a process overrides it.)
- **Storage driver**: classic `overlay2` (`/etc/docker/daemon.json`: `containerd-snapshotter` false). Docker 29's
  default image store is containerd's, whose image id is the manifest digest; `models/manifest.yaml`
  `mcp-packagelens` `image_id` is the config digest, which only the classic store reports.
- **Repo digests**: `docker load` restores tags and image ids but never `RepoDigests`, and the code pins
  `nikolaik/python-nodejs@sha256:1401...` (the gate), `node:24.21.0-bookworm@sha256:64af...` (build base). Those
  references were re-entered in `/var/lib/docker/image/overlay2/repositories.json` (backup
  `/root/repositories.json.bak-20261006`) for the three images whose digests were known. A rebuild of
  PackageLens or the harness box needs no download; `docker pull` of those images would give the same digests.
- **Bind mounts**: the engine reads a mount source as ITS path. `sandbox_net.host_path()` turns `C:\...` into
  `/mnt/c/...` when the docker context is remote (`tcp://`/`ssh://`; `YAMADORI_DOCKER_PATHS=wsl|native` forces it;
  an offline suite is never remote). Applied to the gate's mount and the verify probe's. NOT applied to the
  other `-v` sites (`bench/sandbox/harness_box.py`, `bench/octopus/{grade,toolset_arms}.py`,
  `bench/voxel/check.py`) and `host.docker.internal` (the harness box's `egress:1234 -> host` forward) does not
  exist on this engine: those benchmark runners are not migrated.
- **Keep-alive**: WSL stops a distro seconds after its last `wsl.exe` session; systemd services and running
  containers do not hold it (measured 2026-10-06: Ubuntu Stopped within 25 s, dockerd with it). `scripts\wsl_engine.ps1`
  starts ONE hidden `wsl.exe -d Ubuntu -u root --exec sleep infinity` and waits for the engine. It is called by
  `scripts\start-stack.bat` (before the proxy starts; PackageLens starts at server.py startup) and by every
  `scripts\watchdog.ps1` run. Log: `logs\wsl-engine.log`.

## Start, stop, check

    powershell -File scripts\wsl_engine.ps1 -Status     # keep-alive, engine, docker context
    powershell -File scripts\wsl_engine.ps1             # start the keep-alive, wait for the engine
    powershell -File scripts\wsl_engine.ps1 -Stop       # drop the keep-alive: the VM idles out (vmIdleTimeout 60 s default)
    docker version                                      # a Server section = the engine answers

## .wslconfig (`%USERPROFILE%\.wslconfig`, backup `.wslconfig.bak-20261006`)

`memory=5GB`, `processors=4`, `[experimental] autoMemoryReclaim=dropCache`; `vmIdleTimeout=-1` removed. Before:
only `vmIdleTimeout=-1` (the VM took WSL's defaults: 31,967 MB and 24 CPUs).

Measured 2026-10-06 (1 Hz `/proc/meminfo` and `/proc/stat` in the VM; three runs of: two PackageLens servers with
their gates, npm's resolver on five packages, and a type check of 2 / 3 / 7 held packages with the npm install into a
volume; up to 6 containers; each run 17 s):

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| idle base used (kernel, systemd, dockerd) | 1,312 MB | 1,478 MB | 1,759 MB |
| peak used with the workload | 1,692 MB | 2,071 MB | 2,090 MB |
| CPU peak (24 vCPU) | 4 % | 8 % | 8 % |

Caps = 2 x the measured peak, rounded up: 2 x 2.09 GB -> 5 GB; 2 x 1.9 vCPU -> 4. The factor 2 is a CHOICE
(the operator's "peak plus headroom"), unmeasured above 2.09 GB: bench/octopus's Chromium sidecars and a large tsc
program were not run. If a container is OOM-killed (`dmesg` in Ubuntu), raise `memory`.
`autoMemoryReclaim=dropCache`: measured, a 1.6 GB page cache from reading the image tars was dropped ~115 s after
the reads stopped (used 474 MB, cache 1,601 -> 293 MB); `gradual` keeps a stale cache of every layer read.
`sparseVhd`: not set -- `wsl --manage Ubuntu --set-sparse true` answers "Sparse VHD support is currently disabled
due to potential data corruption" and offers only `--allow-unsafe`, not used.

## Ubuntu clean-up

Ubuntu booted `degraded`: `snap-snapd-19993.mount` and eight `snap-ubuntu\x2ddesktop\x2dinstaller-*.mount` units
failed. `snapd.service`, `snapd.socket`, `snapd.seeded.service` are MASKED; the nine failed mount units are
DISABLED (`systemctl mask` refuses a unit whose file is a regular file in `/etc/systemd/system`); nothing purged or
deleted. `systemctl is-system-running` -> `running`. Records: `/root/failed-units-before-20261006.txt`,
`/root/snap-unit-files-before-20261006.txt`.

## Roll back

1. Windows client: `docker context use desktop-linux`; start Docker Desktop (the data is intact). Re-enable its
   autostart in its Settings (or restore `%APPDATA%\Docker\settings-store.json.bak-20261006` and the Run value in
   `C:\Users\jwals\octo\docker-export\docker-desktop-run-key.txt`).
2. `%USERPROFILE%\.wslconfig.bak-20261006` over `.wslconfig`; `wsl --shutdown`.
3. Ubuntu: `systemctl unmask snapd.service snapd.socket snapd.seeded.service`; `systemctl enable` the nine mounts;
   `systemctl disable --now docker containerd` (or `apt remove docker-ce docker-ce-cli containerd.io
   docker-buildx-plugin`). The `scripts/start-stack.bat` and `scripts/watchdog.ps1` calls of `wsl_engine.ps1` are
   harmless without it (they log a failure).
4. Exports: `C:\Users\jwals\octo\docker-export\` (`images\*.tar`, `volumes\*.tar`, `image-ids.json`; never in git).

Docker Desktop's backend crashed at start on a stale `%LOCALAPPDATA%\Docker\run\dockerInference` socket file
(2026-09-24, "initializing Inference manager"); it was renamed to `dockerInference.stale-20261006` (from Ubuntu: a
Windows socket reparse point cannot be renamed from Windows) so the export could run.
