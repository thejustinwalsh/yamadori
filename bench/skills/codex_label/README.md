# Pass C: Codex labels the skill injector's cases

This is a third, independent labelling pass by Codex, a non-Claude model. It uses the same written rubric as passes A and B. You run it on this machine or on a remote one, with Python 3 and the `codex` CLI on PATH.

## Run it

Check one batch first:

```
python label_codex.py --bundle . --model Sol --first-batch
```

Open `answers_C.json` and check it. Then run everything (it resumes where it stopped):

```
python label_codex.py --bundle . --model Sol
```

- `--sample blind` labels only pass B's blind sample.
- `--dry-run` prints the number of calls and runs nothing.
- Leave out `--model` to use Codex's own configured model.

## What each call does

Each batch is one `codex exec` call with these flags: `--sandbox read-only --ephemeral --skip-git-repo-check --cd <empty temp dir> --output-schema <schema> --output-last-message <file>`. The prompt goes on stdin.

- The rubric is included verbatim.
- The prompt tells Codex to answer from the text alone and run nothing.
- No auto-approve or sandbox-bypass flag is used.
- The script never reads or prints a credential or config file.

## Bring back

Bring this whole folder back:

- `answers_C.json` holds the labels in pass B's schema.
- `pass_C.jsonl` holds the same labels, one per line.
- `codex_log.jsonl` has one row per call.

Score it in the repo:

```
python bench/skills/inject_labels.py ingest --out <this folder> --answers <this folder>/answers_C.json --labeller C
python bench/skills/inject_labels.py agree
```

`agree` reports agreement (kappa, with n) for A vs C, B vs C and C vs the hindsight rule.

## What the data is

- `cases.jsonl` holds turns from our own benchmark transcripts (pagoda, Octopus, the lookup probes) and corpus test traffic. The operator allowed this data to go to Codex (2026-09-29).
- `rubric.txt` is the written rubric (bench/skills/inject_labels.py RUBRIC).
- `blind_cases.txt` lists the case ids in pass B's sample.
