# Third-party licences for recipe sources

Each entry names a source whose licence is held verbatim in this directory.

## TypeHero

- **Source:** https://github.com/typehero/typehero
- **Commit:** `7871629e9a77718312e68a367644ce3fc286ef04` (committed 2026-09-03)
- **Licence:** AGPL-3.0 (GNU Affero General Public License v3.0). Repo-root `LICENSE`,
  copied byte-for-byte to `LICENSE-typehero-AGPL-3.0.txt`
  (sha256 `6f1e622c82a380075843bb084a7ec3b1f1d12a4a02526d75e78b0924a860aa75`).
  The GitHub API (`/repos/typehero/typehero`) reports `spdx_id: AGPL-3.0`.
- **Approval:** operator-approved 2026-09-22 (explicit decision in chat): ingest into
  the skills corpus with the licence recorded as AGPL-3.0, and advance past the
  restricted-licence hold on the operator's behalf.
- **Content ingested:** 40 files, all TypeHero-original text, each one dataset with
  a `typehero_` name (rows in `bench/recipes/typehero_*.jsonl`):
  - 14 challenge prompts, `challenges/<slug>/prompt.md` (the TypeScript Foundations
    track plus TypeHero's own `pick`)
  - 25 Advent of TypeScript 2023 prompts, `challenges/aot/2023/<day>/prompt.md`
  - `challenges/challenge-guidelines.md`
- **Not ingested:** the challenges TypeHero ports from type-challenges (MIT). They are
  not in the TypeHero repo; they are copied from type-challenges at database seed time.
  AoT 2024 prompts in the repo are byte-identical to 2023. Solutions are bare code.
  `bench/recipes/TYPEHERO-CANDIDATES.json` lists each file and gives the reasons.
- **Obligation:** anything served from these rows carries AGPL-3.0 with it
  (see `datasets.RESTRICTED`).

## React corpus (react.dev, typescript-cheatsheets, bulletproof-react, interview handbooks)

Recipes extracted from these sources are derived works of them. The candidate
list, one URL per dataset, is `bench/recipes/REACT-CANDIDATES.json`. Licences
were read from the files named below on 2026-09-22 and matched against the
GitHub REST API's `license.spdx_id`. The commit is the default-branch head at
that time; the worker fetches the branch head, so a page can change after this.

| Source | Commit read | Licence | Copy in this directory | Attribution line |
|---|---|---|---|---|
| [reactjs/react.dev](https://github.com/reactjs/react.dev), `src/content/**` (React documentation) | `b011783f` | CC-BY-4.0 (`LICENSE-DOCS.md`; README: "Content submitted to react.dev is CC-BY-4.0 licensed") | `LICENSE-react.dev-docs.md` | React documentation, © Meta Platforms, Inc. and affiliates, https://react.dev, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Recipes are adapted and summarised from the source, not a verbatim copy. |
| [typescript-cheatsheets/react](https://github.com/typescript-cheatsheets/react), `docs/**` | `c550e41b` | MIT | `LICENSE-typescript-cheatsheets-react` | React TypeScript Cheatsheet, Copyright (c) 2018 shawn wang, MIT License. |
| [alan2207/bulletproof-react](https://github.com/alan2207/bulletproof-react), `docs/*.md` | `9506629e` | MIT | `LICENSE-bulletproof-react` | Bulletproof React, Copyright (c) 2024 Alan Alickovic, MIT License. |
| [yangshun/front-end-interview-handbook](https://github.com/yangshun/front-end-interview-handbook), `website/contents/*.md` | `8e867581` | MIT | `LICENSE-front-end-interview-handbook` | Front End Interview Handbook, Copyright (c) 2017-Present Yangshun Tay, MIT License. |
| [sudheerj/reactjs-interview-questions](https://github.com/sudheerj/reactjs-interview-questions), `README.md` | `55b7d491` | MIT | `LICENSE-reactjs-interview-questions` | reactjs-interview-questions, Copyright (c) 2017-Present Sudheer Jonna, MIT License. |

- **CC BY 4.0 obligation (section 3(a)):** keep the creator and copyright
  notice, refer to and link the licence, link the material, and indicate that
  it was modified. Each recipe row carries its own `source_url` and an
  `evidence` quote, which covers the link to the material. The attribution line
  above covers the rest. Anything that redistributes the react.dev-derived rows
  must carry this file and `LICENSE-react.dev-docs.md` with them.
- **Not established: react.dev code samples under MIT.** The repo does not say
  this. The only MIT text in it is the header on the site's own `.tsx` source,
  which refers to a root `LICENSE` file that does not exist. Treat code blocks
  in `src/content/**` as CC-BY-4.0 with the rest of the page.
- **Excluded on licence grounds:** `greatfrontend/top-reactjs-interview-questions`
  (no licence file); the GreatFrontEnd and LeetCode question banks
  (proprietary). The handbook pages link to GreatFrontEnd in places. The
  handbook text in the repo is MIT, and none of the linked pages was fetched.
