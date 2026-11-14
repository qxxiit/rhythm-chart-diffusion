"""Charts chosen by a judge (EXPERIMENTS 2026-11-08 (5)): while a song is drawn, every window
is drawn k times and the draw a judge finds most human is kept (generate_song's select).

A song's judge is the fold judge of a `judge.py --save-models` run that held that song out, so
it never saw the song. It sees each draw as its view shows it (a selection judge is trained on
the rhythm view: the lane passes after the drawing choose the lanes again), with the SR it was
trained with (`--sr human`: the human chart's, which is what a draw is asked for). The judge
that picks is never the one that scores the result (docs/EVALUATION.md): that is a judge
trained on the chosen run with another seed.

    WindowJudge(dir, device)          the fold judges of a judge.py --save-models run
    .scorer(key, s, draws, seed)      score(cands, mel, b) -> logits [k] for the song key, or
                                      None when no fold judge held it out
    combined_scorer(scorers, gaps)    several judges' scorers as one (evaluate.py
                                      --select-judge J1 J2 ...)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.models.diffusion import load_denoiser
from src.models.judge import Judge, stable_hash, view_tokens


class WindowJudge:
    def __init__(self, path: str | Path, device):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "models.json").read_text(encoding="utf-8"))
        self.device = device
        self.view = self.meta["view"]
        self.fold_of = {k: int(f) for k, f in self.meta["fold_of"].items()}
        self._models: dict[int, Judge] = {}

    def _model(self, fold: int) -> Judge:
        if fold not in self._models:
            trunk = load_denoiser(self.meta["ckpt"], self.device)
            judge = Judge(trunk, audio=self.meta["audio"]).to(self.device)
            state = torch.load(self.dir / f"fold{fold}.pt", map_location="cpu")["state"]
            judge.load_state_dict({k: v.float() if v.is_floating_point() else v
                                   for k, v in state.items()})
            self._models[fold] = judge.eval()
        return self._models[fold]

    def scorer(self, key: str, s: float, draws: int = 2, seed: int = 0):
        """score(cands, mel, b): the judge's logit for each of the windows cands ([L, K] each,
        no MASK), mel [L * 4, n_mels] on the device, b the window's beat length; in the rhythm
        and holds views the mean over draws of the random lanes. Higher looks more
        generated."""
        fold = self.fold_of.get(key)
        if fold is None:
            return None
        judge = self._model(fold)
        h, calls = stable_hash(key), [0]
        n_draws = max(1, draws) if self.view in ("rhythm", "holds") else 1

        @torch.no_grad()
        def score(cands: list[np.ndarray], mel: torch.Tensor, b: float) -> np.ndarray:
            calls[0] += 1
            xs = [view_tokens(np.asarray(c), self.view, (seed, h, calls[0], i, d))
                  for i, c in enumerate(cands) for d in range(n_draws)]
            n = len(xs)
            x = torch.as_tensor(np.stack(xs), dtype=torch.long, device=self.device)
            logits = judge(x, mel[None].expand(n, -1, -1),
                           torch.full((n,), float(s), dtype=torch.float32, device=self.device),
                           torch.full((n,), float(b), dtype=torch.float32, device=self.device))
            return logits.float().view(len(cands), n_draws).mean(dim=1).cpu().numpy()

        return score


def combined_scorer(scorers: list, gaps: list[float]):
    """Several judges' scorers as one: each judge's window scores divided by its gap (models.json
    "gap": how far its run's windows sat over the human ones), then summed, so each judge's own
    gap counts one. A window kept is then the one the judges together find most human, every
    judge as loud as its own reach; one scorer is returned as it is (its logits)."""
    if len(scorers) == 1:
        return scorers[0]

    def score(cands, mel, b) -> np.ndarray:
        return sum(np.asarray(sc(cands, mel, b), dtype=np.float64) / float(g)
                   for sc, g in zip(scorers, gaps, strict=True))

    return score
