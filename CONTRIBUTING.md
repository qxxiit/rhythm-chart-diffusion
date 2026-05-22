# Contributing

Internal team workflow conventions. Keep it simple.

## Branching

- `main` — always working state. Direct pushes discouraged; use PRs.
- `dev` — integration branch for in-progress work.
- `feat/<short-name>` — feature branches off `dev`.
- `fix/<short-name>` — bug fixes.
- `exp/<short-name>` — experimental code (may not merge).

Example: `feat/osu-downloader`, `fix/tokenizer-edge-case`, `exp/diffusion-cosine-schedule`.

## Commit Messages

Loose [Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>: <short description>

[optional body]
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `exp`.

Examples:
- `feat: add osu API downloader for ranked 4K`
- `fix: handle missing TimingPoints in osu parser`
- `exp: try cosine noise schedule with 100 steps`

## Pull Requests

1. Create a branch off `dev`.
2. Make changes, commit frequently.
3. Push and open PR to `dev`.
4. At least one other team member reviews (loosely — light touch is fine).
5. Merge when CI passes and reviewer is OK.

Periodically merge `dev` → `main` when stable (after each phase, say).

## Code Style

- Run `make format` before committing.
- Run `make lint` to check.
- Follow type hints where reasonable.
- Docstrings for public functions (Google style preferred).

## Tests

- New utility functions / data parsers → add unit test in `tests/`.
- Model training code doesn't need full tests, but at minimum a "model can do one forward pass" sanity check.

## Experiment Logging

- All training runs → wandb.
- Use descriptive run names: `<phase>-<model>-<key-hp>` e.g. `p2-diffusion-cosine-100steps`.
- Log results in `docs/EXPERIMENTS.md` after each significant run.

## Meeting Notes

- Weekly meeting notes in Notion.
- Decision-impacting outcomes also captured in `docs/DECISIONS.md` (one-line entries).

## Asking for Help

If stuck > 24h on the same issue:
1. Post in team Slack/KakaoTalk with what you tried.
2. If still stuck → escalate to mentor in next 1:1.
3. Don't suffer in silence.
