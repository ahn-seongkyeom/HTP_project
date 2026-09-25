# Near-Duplicate Robustness of Pedestrian Trajectory Predictors

This repository reproduces a controlled experiment on three pedestrian trajectory prediction models
on the ETH/UCY benchmark:

| Model | Social aggregation | Released weights used |
|---|---|---|
| **Social GAN (SGAN-20VP)** | max pooling over neighbours (set operation) | `sgan-p-models` (official Dropbox) |
| **Social-STGCNN** | weighted graph (normalized Laplacian) | `checkpoint/` (included in the official repo) |
| **AgentFormer (DLow)** | attention (weighted) | `agentformer_models.zip` (official Google Drive) |

**Hypothesis.** Max pooling is idempotent on sets, i.e. `max{A, B, C} = max{A, A, B, C}`, so adding
near-duplicates of one pedestrian should leave the other predictions (almost) unchanged. Weighted
aggregation (graph / attention) instead shifts the weights, e.g. from `1/3, 1/3, 1/3` to `1/2, 1/4, 1/4`,
so the predictions of the original pedestrians should change.

---

## Quick start

```bash
bash run_all.sh
```

That is the only command. Starting from an empty machine, the script:

1. installs Miniforge (conda) if conda is not present and creates three Python environments
   (one per model, see `requirements.txt`);
2. clones the three **official** repositories and downloads the **official** data and **released**
   pretrained weights from their original locations (table below);
3. prints the frame interval of every data file (all must be 10);
4. runs the experiment for the three models and collects the results in `~/dup_repro/results_all/`.

The script is idempotent: installed environments, downloads and finished models are skipped.
Use `FORCE=1 bash run_all.sh` to re-run all experiments. The working directory can be changed with
`REPRO_ROOT=/path/to/dir bash run_all.sh` (default `~/dup_repro`).

### Requirements

- Ubuntu (x86_64); WSL2 also works
- NVIDIA GPU with a driver supporting CUDA 11.8 (the SGAN code requires CUDA)
- `git`, `wget`, `unzip`, `curl`, internet access
- about 15 GB of free disk space (three PyTorch environments + data + weights)

### Sources (everything is downloaded from the original locations)

| Item | Source |
|---|---|
| SGAN code | https://github.com/agrimgupta92/sgan |
| SGAN data (incl. `raw/all_data`) | `scripts/download_data.sh` of the SGAN repo (Dropbox) |
| SGAN released weights | `scripts/download_models.sh` of the SGAN repo (Dropbox), `sgan-p-models/<dataset>_12_model.pt` |
| Social-STGCNN code, data, weights | https://github.com/abduallahmohamed/Social-STGCNN (`datasets/`, `checkpoint/`) |
| AgentFormer code, data | https://github.com/Khrylx/AgentFormer |
| AgentFormer released weights | Google Drive link in the AgentFormer README (`agentformer_models.zip`) |

If the Google Drive download is blocked (quota or network), the script still runs SGAN and
Social-STGCNN and prints the link; place `agentformer_models.zip` in `~/dup_repro/AgentFormer/`
and run the script again - only AgentFormer will be executed.

---

## Experimental protocol

**Data.** The standard leave-one-out test splits distributed with SGAN
(`datasets/{eth,hotel,univ,zara1,zara2}/test/*.txt`). All files, including ETH, use a 10-frame
interval, which is the interval the released weights were trained with. Observation 8 steps,
prediction 12 steps.

**Scenes.** Identical to the official `TrajectoryDataset` (sliding window of 20 frames, stride 1,
pedestrians present in all 20 frames, at least 2 pedestrians per scene).

**Duplication.** In every scene one pedestrian `A` is chosen at random (fixed seed). `k` copies of
`A`'s 8 observed points are appended to the scene, each perturbed with independent Gaussian noise
`N(0, sigma^2)` per coordinate. Defaults: `k = 0, 1, 2, 4` and `sigma = 0, 0.02, 0.05` m
(`sigma = 0` = exact duplicates). The original pedestrians' inputs are never modified.

**Evaluation.** ADE/FDE are computed **only for the original pedestrians** (duplicates are excluded).
Best-of-20 follows each model's official evaluation:

| Model | Best-of-20 | Kept fixed before/after duplication |
|---|---|---|
| SGAN | per scene (as `scripts/evaluate_model.py`) | the latent noise `z` selected in the official evaluation run |
| Social-STGCNN | per pedestrian (as `test.py`) | the standard-normal draws used to sample the predicted Gaussian |
| AgentFormer | per pedestrian (as `eval.py`) | the DLow latent codes `z` (duplicates receive `A`'s codes) |

For SGAN the baseline (`k = 0`) is the **output of the official `evaluate_model.py` itself**: the
script runs the official evaluation, records the noise and predictions it used, and derives the
duplication experiment from exactly that run. The log prints both values side by side:

```
[official evaluate check] ... official evaluate_model 0.xxxx / x.xxxx  =  baseline of this experiment 0.xxxx / x.xxxx
```

---

## Outputs

`~/dup_repro/results_all/` contains, for each model (`sgan`, `stgcnn`, `agentformer`):

- `summary_<model>.csv` - averages per dataset, plus the average over the five datasets (`dataset = AVG`)
- `per_pedestrian_<model>.csv` - one row per pedestrian, scene and configuration

### Columns of `summary_<model>.csv`

| Column | Meaning |
|---|---|
| `dataset` | test split (`eth`, `hotel`, `univ`, `zara1`, `zara2`, or `AVG` = mean over the five splits) |
| `k` | number of near-duplicates added |
| `sigma` | noise standard deviation of the duplicates (m) |
| `role` | `all` = all original pedestrians, `source` = the duplicated pedestrian A, `other` = all original pedestrians except A |
| `n` | number of pedestrian-window samples |
| `ADE_base`, `FDE_base` | mean ADE / FDE without duplicates (`k = 0`) |
| `ADE_dup`, `FDE_dup` | mean ADE / FDE after adding the duplicates |
| `dADE`, `dFDE` | mean signed change, `ADE_dup - ADE_base` (resp. FDE) |
| `abs_dADE`, `abs_dFDE` | mean absolute change, `mean(abs(ADE_dup - ADE_base))`; unlike the signed change, improvements and degradations do not cancel out |
| `ratio_ADE(%)`, `ratio_FDE(%)` | relative absolute change in percent: `abs_dADE / ADE_base * 100` (resp. FDE); `0` = unchanged |

### Columns of `per_pedestrian_<model>.csv`

`dataset`, `data_file` (source txt file), `ped_id`, `frame_start`, `frame_obs_end`, `frame_end`
(frames of the 20-step window), `k`, `sigma`, `role`, `ADE_base`, `FDE_base`, `ADE_dup`, `FDE_dup`,
`dADE`, `dFDE`, `ratio_ADE(%)`, `ratio_FDE(%)` (here per pedestrian: `|ADE_dup - ADE_base| / ADE_base * 100`).

For `AVG` rows the error columns are averaged over the five datasets and the ratios are recomputed
from those averages.

**Ratio definition.** For a set `S` of original pedestrians (a dataset and a `role`):

```
ratio_ADE(%) = 100 * mean_{i in S} |ADE_dup_i - ADE_base_i|  /  mean_{i in S} ADE_base_i
ratio_FDE(%) = 100 * mean_{i in S} |FDE_dup_i - FDE_base_i|  /  mean_{i in S} FDE_base_i
```

The absolute value makes improvements and degradations add up instead of cancelling, so the ratio
measures how much the predictions of the original pedestrians move when duplicates are added.
It is a ratio of means (not a mean of per-pedestrian ratios), so pedestrians with a very small
baseline error cannot dominate it.

### Log lines worth checking

- `[official evaluate check]` (SGAN) - baseline equals the official `evaluate_model.py` output.
- `max change of original predictions under exact duplication (sigma=0)` - largest change of any
  original pedestrian's prediction when exact duplicates are added (expected ~0 for SGAN up to
  floating-point error).
- `[graph check]` (Social-STGCNN) - the experiment builds the graph with a vectorised NumPy
  implementation; it is compared once with the official `utils.seq_to_graph` (should be below 1e-5).
- `frame interval` - printed for every data file; all must be 10.

Full logs are written to `~/dup_repro/log_{sgan,stgcnn,agentformer}.txt`.

### Running a subset or other settings

After the first run, each model can be run directly, e.g.

```bash
cd ~/dup_repro/sgan && REPRO_ROOT=~/dup_repro PYTHONPATH=. ../envs/sgan/bin/python sgan_dup.py eth zara1 --k 0,1,2,4,8 --sigma 0,0.02
cd ~/dup_repro/Social-STGCNN && REPRO_ROOT=~/dup_repro ../envs/stgcnn/bin/python stgcnn_dup.py --k 0,1,2
cd ~/dup_repro/AgentFormer && REPRO_ROOT=~/dup_repro ../envs/agentformer/bin/python agentformer_dup.py hotel
```

---

## Compatibility notes

The original repositories target old PyTorch versions (SGAN 0.4, AgentFormer 1.8). To run on current
GPUs the environments use PyTorch 2.0.1 (CUDA 11.8). The official repositories are **not modified**;
the following is handled inside the experiment scripts only and does not change any computed value:

- SGAN: the official loader passes permuted (non-contiguous) tensors, which `Tensor.view` rejects in
  PyTorch >= 1.x; the inputs are made contiguous before the forward pass (memory layout only).
- AgentFormer: `torch.nn.modules.linear._LinearWithBias` (removed after PyTorch 1.8) is aliased to
  `NonDynamicallyQuantizableLinear`, the same bias-enabled linear layer with identical parameter names.
- AgentFormer imports `torchvision` (map encoder, unused for ETH/UCY), so it is installed.
- Social-STGCNN: `networkx==2.8.8` and `scipy==1.11.4` are pinned for the official `seq_to_graph`.
- The Google Drive download forces IPv4, because some environments (e.g. WSL) have no IPv6 route.

Note that the released SGAN weights do not reproduce every entry of Table 1 of the SGAN paper
(e.g. HOTEL); the official `evaluate_model.py` gives the same numbers as this experiment's baseline,
so the difference comes from the released checkpoints, not from the evaluation.

## Credits

Code, data and pretrained weights belong to the authors of
Social GAN (Gupta et al., CVPR 2018), Social-STGCNN (Mohamed et al., CVPR 2020) and
AgentFormer (Yuan et al., ICCV 2021). Please cite their papers and follow their licenses.
