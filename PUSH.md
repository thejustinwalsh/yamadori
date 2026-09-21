# Pushing this to GitHub

`gh` is installed (2.32.1) but its token expired, and re-auth is an interactive
browser flow. Two commands:

```powershell
gh auth login -h github.com

# PRIVATE is the right default -- config.yaml contains your ZeroTier IP
# (10.242.120.152) and local filesystem paths.
gh repo create llama-stack --private --source=. --remote=origin --push
```

Already public-safe if you prefer `--public`: no keys, no tokens, no model
weights (`.gitignore` excludes `*.gguf`, `bin/`, `index/`, logs). The only
disclosure is the ZeroTier address and machine paths.

Existing state: one commit on `main`, 12 files tracked, no remote configured.
