## Repo orientation

This repository implements Single-Path One-Shot NAS (SPOS) focused on CIFAR-10. Key top-level files:
- `train_supernet.py` — trains the SPOS supernet; uses `SinglePath_OneShot` (models/model.py).
- `random_search.py` — evaluates random architectures by sampling `choice` lists and calling `model(inputs, choice)`.
- `retrain_best_choice.py` — retrains a single chosen architecture implemented by `SinglePath_Network`.
- `utils.py` — shared utilities: `data_transforms`, `random_choice`, `set_seed`, `AverageMeter`, `save_checkpoint`.
- `models/model.py` — core: `SinglePath_OneShot` (supernet) and `SinglePath_Network` (fixed choice model).
- `models/block.py` — building blocks: `Choice_Block`, `Choice_Block_x` and `channel_shuffle`.
- `scripts/*.sh` — convenience bash wrappers that run the above Python scripts with logging.

## High-level architecture & conventions

- Supernet vs Choice model: the supernet (`SinglePath_OneShot`) expects to receive a `choice` list when calling forward (signature: `model(x, choice)`); the choice model (`SinglePath_Network`) is constructed with a fixed `choice` list and its forward is `model(x)`.
- `choice` is a Python list of integers length `layers` (default 20). Each int in 0..3 picks one sub-block per layer. Example best path in README: `[1,0,3,1,3,0,...]`.
- Kernel options are `[3,5,7,'x']`, where `'x'` selects the deeper `Choice_Block_x` implementation.
- BatchNorm differences: during supernet training BatchNorm layers are created with `affine=False` for the choices; when building a fixed network, blocks are instantiated with `supernet=False` so BatchNorm has affine parameters — preserve this when modifying blocks.
- Downsampling: `downsample_layers` is inferred from dataset/resize config (see `models/model.py`) — this affects strides and channel shapes.

## Typical developer workflows (how to run)

- Train supernet (CIFAR-10, default):
  - `bash scripts/train_supernet.sh` — creates `logdir/log_spos_c10_train_supernet` and writes checkpoints to `./checkpoints/`.
- Random search using pretrained supernet:
  - `bash scripts/random_search.sh` — loads `./checkpoints/spos_c10_train_supernet_best.pth` and logs to `logdir/log_spos_c10_random_search`.
- Retrain a chosen architecture:
  - Edit the `choice` list in `retrain_best_choice.py` (README points to Line ~116) or pass into `SinglePath_Network` programmatically. Then run `bash scripts/retrain_best_choice.sh`.

Note: scripts use bash + `CUDA_VISIBLE_DEVICES` + `nohup`. On Windows use Git Bash, MSYS2 shell, WSL, or an equivalent POSIX shell to run the provided `.sh` scripts.

## Project-specific patterns & gotchas

- Determinism: the repo fixes seeds via `utils.set_seed(seed)`. If you change randomness, update `set_seed` or the seed args used in scripts.
- Data pipeline: `utils.data_transforms(args)` returns CIFAR/Imagenet transforms and handles `--resize`, `--cutout`, and `--auto_aug`. Tests and training rely on those transforms.
- Checkpoints & logs:
  - Supernet best checkpoint expected at `./checkpoints/spos_c10_train_supernet_best.pth`.
  - Training logs live in `logdir/` and `save_checkpoint` writes snapshots to `./snapshots/`.
- Model shape constants: channel layout and `last_channel` are hard-coded in `models/model.py` — changing them changes many shapes; adjust blocks accordingly.
- Performance measurement: `thop.profile` is used in `retrain_best_choice.py` to compute FLOPS/params when building the fixed `SinglePath_Network`.

## Where to look when modifying behavior

- To change how architectures are sampled: `utils.random_choice` and `random_search.py`.
- To change layer/block implementations: `models/block.py` (Choice_Block, Choice_Block_x) and `models/model.py` (how blocks are composed).
- To change dataset/hyperparameters: modify args at top of `train_supernet.py`, `random_search.py`, or `retrain_best_choice.py`.

## Suggested prompts & safe edits for AI agents

- “Add an option to pass a fixed `choice` to `train_supernet.py` for debugging small runs. Follow existing CLI style and default to current behavior.” — edit parser, use `SinglePath_OneShot` accordingly.
- “When converting a trained supernet path into a retrainable model, ensure all `Choice_Block` instances are created with `supernet=False` so BatchNorm has affine parameters.” — refer to `models/model.py` lines where `supernet` flag is used.
- “If you change kernel lists or channel sizes, run a single forward pass test (random tensor) to detect shape mismatches before long training runs.” — use small batch, CPU mode if needed.

If anything here is unclear or you want more examples (e.g., exact line numbers to edit, or a small unit-test harness to validate forward passes), tell me which part and I’ll update the instructions.
