"""Training entry point.

Uses Hydra to compose model/data/training configs.

Usage:
    # Train baseline
    python scripts/train.py model=transformer_baseline data=osu_mania_4k

    # Train diffusion
    python scripts/train.py model=diffusion data=osu_mania_4k

    # Override hyperparams
    python scripts/train.py model=diffusion training.batch_size=8 training.lr=3e-4

TODO (Phase 1, week of 7/15):
- [ ] Set up Hydra config composition
- [ ] Instantiate model from config
- [ ] Instantiate dataset/loader from config
- [ ] Build trainer (src/training/trainer.py)
- [ ] Hook up wandb logging
- [ ] Checkpoint saving
"""

# import hydra
# from omegaconf import DictConfig
#
#
# @hydra.main(config_path="../configs", config_name="config", version_base=None)
# def main(cfg: DictConfig) -> None:
#     ...


def main() -> None:
    raise NotImplementedError("TODO: implement Hydra-based training entry.")


if __name__ == "__main__":
    main()
