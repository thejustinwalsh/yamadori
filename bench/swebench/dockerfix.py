#!/usr/bin/env python
"""Run mini-swe-agent and the swebench harness natively on Windows, around a
Docker Desktop whose container-inspect cache is broken.

    python dockerfix.py mini <mini-extra swebench args...>
    python dockerfix.py eval <swebench.harness.run_evaluation args...>
    python dockerfix.py selftest

WHY THIS EXISTS (2026-09-23, after the 23:42 crash). Docker Desktop 4.41.2
came back with its WSL integration for Ubuntu failed. Its dialog offers
"Restart the WSL integration" and waits for a human. It also came back with
its API cache (`com.docker.backend.exe.apicache` in the backend log) serving
stale reads:

- `GET /containers/json` returns `[]` while containers run.
- `GET /containers/<id>/json` returns 404 for a container that was just
  created and is running.

Every write goes through: create, start, exec, put_archive, stop and remove.
So does the exec stream. Only the container READS are stale. The docker CLI's
`docker exec` inspects first, so it fails, and so does the SDK's
`containers.create()`, which calls get(). Nothing here restarts Docker Desktop
(operator constraint: restart nothing).

THE WORKAROUND (it changes plumbing only; every command still runs, with the
same flags, in the same image and working directory):

- mini-swe-agent: `LowLevelDockerEnvironment` replaces the `docker`
  environment class. Same container (`sleep 2h`, `-w /testbed`, auto-remove),
  same `bash -c <command>` per action, the same env vars, and stdout and stderr
  merged, as `docker exec` with `stderr=STDOUT`. The same 60 s timeout gives
  the same `TimeoutExpired` text. It does not use docker inspect.
- the harness: `containers.create()` returns the model built from the create
  response instead of re-reading it, and `containers.get()` falls back to that
  model on 404.
- Both run on Windows, where `Path.write_text` would translate `\\n` to
  `\\r\\n`. The harness writes `patch.diff` and `eval.sh` that way and copies
  them into the Linux container, so CRLF would break both. `write_text` is
  pinned to `newline="\\n"` in this process, which is what Linux does.
  PYTHONUTF8=1 is set by the caller.

`selftest` proves the environment is faithful before anything is scored
(PROTOCOL rule 3): quotes, backslashes, heredocs, unicode, exit codes,
timeouts, and cwd and env handling, against the real image.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import threading
import time
import uuid

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")


def _pin_lf() -> None:
    orig = pathlib.Path.write_text

    def write_text(self, data, encoding=None, errors=None, newline=None):
        return orig(self, data, encoding=encoding or "utf-8", errors=errors,
                    newline="\n" if newline is None else newline)

    pathlib.Path.write_text = write_text


def _patch_sdk() -> None:
    import docker
    from docker.models.containers import ContainerCollection, _create_container_args

    def create(self, image, command=None, **kwargs):
        if isinstance(image, docker.models.images.Image):
            image = image.id
        kwargs["image"] = image
        kwargs["command"] = command
        kwargs["version"] = self.client.api._version
        create_kwargs = _create_container_args(kwargs)
        resp = self.client.api.create_container(**create_kwargs)
        name = kwargs.get("name") or ""
        return self.prepare_model({"Id": resp["Id"], "Name": "/" + name})

    orig_get = ContainerCollection.get

    def get(self, container_id):
        try:
            return orig_get(self, container_id)
        except docker.errors.NotFound:
            return self.prepare_model({"Id": container_id, "Name": ""})

    ContainerCollection.create = create
    ContainerCollection.get = get

    # images.list() lists, then inspects every id; the cache lists ids it
    # then 404s on. Build the models from the list response itself (it
    # carries Id and RepoTags, which is all the harness reads).
    from docker.models.images import ImageCollection

    def list_images(self, name=None, all=False, filters=None):  # noqa: A002
        resp = self.client.api.images(name=name, all=all, filters=filters)
        return [self.prepare_model(r) for r in resp]

    ImageCollection.list = list_images


# ------------------------------------------------------------ mini-swe-agent
def _env_class():
    import docker
    from minisweagent.environments.docker import DockerEnvironment

    class LowLevelDockerEnvironment(DockerEnvironment):
        """DockerEnvironment over the Engine API without inspect calls."""

        def _start_container(self):
            self._api = docker.APIClient(timeout=600)
            name = f"minisweagent-{uuid.uuid4().hex[:8]}"
            img = self.config.image
            try:
                self._api.inspect_image(img)
            except docker.errors.NotFound:
                self._api.pull(img)
            hc = self._api.create_host_config(
                auto_remove="--rm" in self.config.run_args)
            c = self._api.create_container(
                img, ["sleep", self.config.container_timeout], name=name,
                working_dir=self.config.cwd, host_config=hc)
            self._api.start(c["Id"])
            self.container_id = c["Id"]
            self.logger.info(f"Started container {name} with ID {c['Id']}")

        def execute(self, action: dict, cwd: str = "", *,
                    timeout: int | None = None) -> dict:
            command = action.get("command", "")
            cwd = cwd or self.config.cwd
            assert self.container_id, "Container not started"
            env = {}
            for key in self.config.forward_env:
                if (value := os.getenv(key)) is not None:
                    env[key] = value
            env.update(self.config.env)
            argv = [*self.config.interpreter, command]
            limit = timeout or self.config.timeout
            # What the stock class would have run, for the timeout message.
            shown = [self.config.executable, "exec", "-w", cwd]
            for k, v in env.items():
                shown += ["-e", f"{k}={v}"]
            shown += [self.container_id, *argv]
            box: dict = {"out": b""}

            def run():
                try:
                    ex = self._api.exec_create(self.container_id, argv,
                                               workdir=cwd, environment=env)
                    box["id"] = ex["Id"]
                    for chunk in self._api.exec_start(ex["Id"], stream=True):
                        box["out"] += chunk
                    box["rc"] = self._api.exec_inspect(ex["Id"])["ExitCode"]
                except Exception as e:                           # noqa: BLE001
                    box["err"] = e

            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(limit)
            text = box["out"].decode("utf-8", errors="replace")
            if t.is_alive():
                e = subprocess.TimeoutExpired(shown, limit, output=text)
                output = {"output": text, "returncode": -1,
                          "exception_info": f"An error occurred while executing the command: {e}",
                          "extra": {"exception_type": type(e).__name__,
                                    "exception": str(e)}}
            elif "err" in box:
                e = box["err"]
                output = {"output": text, "returncode": -1,
                          "exception_info": f"An error occurred while executing the command: {e}",
                          "extra": {"exception_type": type(e).__name__,
                                    "exception": str(e)}}
            else:
                output = {"output": text, "returncode": box.get("rc", -1),
                          "exception_info": ""}
            prog = os.environ.get("SWEBENCH_PROGRESS")
            if prog:
                # One line per executed action, so a running instance's step
                # count is visible (mini writes its trajectory only at the end).
                try:
                    with open(prog, "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%H:%M:%S')}\t"
                                f"{os.environ.get('SWEBENCH_INSTANCE', '')}\t"
                                f"rc={output['returncode']}\t"
                                f"{command[:100]!r}\n")
                except OSError:
                    pass
            self._check_finished(output)
            return output

        def cleanup(self):
            cid = getattr(self, "container_id", None)
            api = getattr(self, "_api", None)
            if not cid or api is None:
                return
            self.container_id = None

            def stop():
                try:
                    api.stop(cid, timeout=10)
                except Exception:                                # noqa: BLE001
                    pass
                try:
                    api.remove_container(cid, force=True)
                except Exception:                                # noqa: BLE001
                    pass

            threading.Thread(target=stop, daemon=True).start()

    return LowLevelDockerEnvironment


def _install_env() -> None:
    import minisweagent.environments as envs
    cls = _env_class()
    mod = sys.modules[__name__]
    mod.LowLevelDockerEnvironment = cls
    envs._ENVIRONMENT_MAPPING["docker"] = f"{__name__}.LowLevelDockerEnvironment"


def _patch_retry() -> None:
    """A 429 ("all main lanes busy") is a place in line, not a failure: retry
    it after 1-2 s with jitter instead of mini's exponential 4-60 s backoff,
    which lost every race for a lane to the other clients on a shared card
    (2026-09-23, operator decision (b)). Every other error keeps mini's own
    backoff. Retries happen inside one model query, so no step is added."""
    import logging
    import litellm
    from tenacity import (Retrying, before_sleep_log, retry_if_not_exception_type,
                          stop_after_attempt, wait_exponential, wait_random)
    import minisweagent.models.litellm_model as lm

    expo = wait_exponential(multiplier=1, min=4, max=60)
    quick = wait_random(1, 2)

    def wait(rs):
        e = rs.outcome.exception() if rs.outcome else None
        return quick(rs) if isinstance(e, litellm.exceptions.RateLimitError) else expo(rs)

    def retry(*, logger: logging.Logger, abort_exceptions):
        return Retrying(
            reraise=True,
            stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
            wait=wait,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            retry=retry_if_not_exception_type(tuple(abort_exceptions)),
        )

    lm.retry = retry


def run_mini(args: list[str]) -> int:
    _pin_lf()
    _install_env()
    _patch_retry()
    from minisweagent.run.benchmarks.swebench import app
    sys.argv = ["mini-extra-swebench", *args]
    try:
        app()
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def run_eval(args: list[str]) -> int:
    _pin_lf()
    _patch_sdk()
    if os.name == "nt" and "resource" not in sys.modules:
        # swebench.harness/__init__ imports prepare_images, which imports the
        # Unix-only `resource` at module level. run_evaluation itself only
        # calls setrlimit on Linux, so an inert stand-in is enough.
        import types
        stub = types.ModuleType("resource")
        stub.RLIMIT_NOFILE = 7
        stub.setrlimit = lambda *a, **k: None
        stub.getrlimit = lambda *a, **k: (4096, 4096)
        sys.modules["resource"] = stub
    sys.argv = ["run_evaluation", *args]
    import runpy
    try:
        runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def selftest(image: str) -> int:
    _install_env()
    cls = sys.modules[__name__].LowLevelDockerEnvironment
    env = cls(image=image, cwd="/testbed", timeout=5,
              env={"PAGER": "cat", "TQDM_DISABLE": "1"},
              interpreter=["bash", "-c"])
    cases = [
        ("quotes", "echo \"double\" 'single' $PAGER", "double single cat\n", 0),
        ("backslash", "printf '%s\\n' 'a\\b\\\\c'", "a\\b\\\\c\n", 0),
        ("heredoc", "cat <<'EOF'\nline1 \"q\" $x\nline2\\t\nEOF",
         "line1 \"q\" $x\nline2\\t\n", 0),
        ("python_c", "python -c \"import sys; print('hi %s' % sys.argv[1:])\" a 'b c'",
         "hi ['a', 'b c']\n", 0),
        ("pipes", "echo abc | tr a-z A-Z && echo ok; false || echo fallback",
         "ABC\nok\nfallback\n", 0),
        ("unicode", "echo 'héllo ✓'", "héllo ✓\n", 0),
        ("stderr_merged", "echo out; echo err 1>&2", None, 0),
        ("exit_code", "exit 3", "", 3),
        ("cwd", "pwd", "/testbed\n", 0),
        ("no_crlf", "printf 'a\\nb\\n' | od -c | head -1", None, 0),
    ]
    ok = 0
    for name, cmd, want, rc in cases:
        out = env.execute({"command": cmd})
        good = out["returncode"] == rc and (want is None or out["output"] == want)
        if name == "stderr_merged":
            good = good and "out" in out["output"] and "err" in out["output"]
        if name == "no_crlf":
            good = good and "\\r" not in out["output"]
        ok += good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: {out['output'][:80]!r} rc={out['returncode']}")
    t0 = time.time()
    out = env.execute({"command": "sleep 30"})
    good = (out["returncode"] == -1 and "timed out after 5 seconds" in out["exception_info"]
            and time.time() - t0 < 10)
    ok += good
    print(f"  {'ok  ' if good else 'FAIL'} timeout: {out['exception_info'][-60:]!r}")
    try:
        env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo diff"})
        print("  FAIL submit: not raised")
    except Exception as e:                                       # noqa: BLE001
        good = type(e).__name__ == "Submitted"
        ok += good
        print(f"  {'ok  ' if good else 'FAIL'} submit raises {type(e).__name__}")
    env.cleanup()
    n = len(cases) + 2
    print(f"{ok}/{n} checks passed")
    return 0 if ok == n else 1


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "mini":
        sys.exit(run_mini(rest))
    if mode == "eval":
        sys.exit(run_eval(rest))
    if mode == "selftest":
        sys.exit(selftest(rest[0] if rest else
                          "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-17655:latest"))
    sys.exit(f"unknown mode {mode}")
