# Pushing this to GitHub

`gh` is installed (2.32.1) but its token expired, and re-auth is an interactive
browser flow. Two commands:

```powershell
gh auth login -h github.com

# PRIVATE is the right default -- config.yaml contains your ZeroTier IP
# (ai.thejustinwalsh.me) and local filesystem paths.
gh repo create llama-stack --private --source=. --remote=origin --push
```

Already public-safe if you prefer `--public`: no keys, no tokens, no model
weights (`.gitignore` excludes `*.gguf`, `bin/`, `index/`, logs). The only
disclosure is the ZeroTier address and machine paths.

Existing state (re-checked): 55 commits on `main`, 59 files tracked, and the
remote **is** configured — `origin` is `git@github.com:thejustinwalsh/yamadori.git`,
so the `gh repo create` line above is history, not a next step. Note the repo is
named `yamadori`, not `llama-stack`.
