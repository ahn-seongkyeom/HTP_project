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
so predictions of the original pedestrians should change.

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

The script is idempotent: already installed environments, downloads and finished models are skipped.
Use `FORCE=1 bash run_all.sh` to re-run all experiments. The working directory can be changed with
`REPRO_ROOT=/path/to/dir bash run_all.sh` (default `~/dup_repro`).

### Requirements

- Ubuntu (x86_64), also tested under WSL2
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
pedestrians present in all 20 frames, at least 2 pedestrians per scene). For SGAN the script checks
that the scenes it uses are exactly those of the official loader.

**Duplication.** In every scene one pedestrian `A` is chosen at random (fixed seed). `k` copies of
`A`'s observed 8 points are appended to the scene, each perturbed with independent Gaussian noise
`N(0, sigma^2)` per coordinate. Defaults: `k = 0, 1, 2, 4` and `sigma = 0, 0.02, 0.05` m
(`sigma = 0` = exact duplicates). The original pedestrians' inputs are never modified.

**Evaluation.** ADE/FDE are computed **only for the original pedestrians** (duplicates are excluded).
Best-of-20 follows each model's official evaluation:

| Model | Best-of-20 | What is kept fixed before/after duplication |
|---|---|---|
| SGAN | per scene (as `scripts/evaluate_model.py`) | the latent noise `z` chosen in the official evaluation run |
| Social-STGCNN | per pedestrian (as `test.py`) | the standard-normal draws used to sample the predicted Gaussian |
| AgentFormer | per pedestrian (as `eval.py`) | the DLow latent codes `z` (duplicates receive `A`'s codes) |

For SGAN the baseline (`k = 0`) is the **output of the official `evaluate_model.py` itself**: the
script runs the official evaluation, records the noise and predictions it used, and derives the
duplication experiment from exactly that run. The console prints both numbers side by side:

```
[공식 evaluate 대조] ... 공식 evaluate_model 0.xxxx / x.xxxx  =  이 실험의 기존값 0.xxxx / x.xxxx
```
(`official evaluate_model ADE/FDE = baseline ADE/FDE of this experiment`)

---

## Outputs

`~/dup_repro/results_all/` contains, for each model (`sgan`, `stgcnn`, `agentformer`):

- `summary_<model>.csv` - averages per dataset and over the five datasets (`dataset = AVG`)
- `per_pedestrian_<model>.csv` - one row per pedestrian, scene and configuration

Column glossary (headers are in Korean):

| Column | Meaning |
|---|---|
| `dataset`, `k`, `sigma` | test split, number of duplicates, noise std (m) |
| `role` | `all` = all original pedestrians, `source` = the duplicated pedestrian A, `other` = everyone else |
| `건수` | number of pedestrian-scene samples |
| `기존ADE`, `기존FDE` | baseline ADE/FDE (no duplicates, `k = 0`) |
| `만든ADE`, `만든FDE` | ADE/FDE after adding the duplicates |
| `ADE차이`, `FDE차이` | signed change (after - before) |
| `\|ADE차이\|`, `\|FDE차이\|` | mean absolute change |
| `ped_id`, `frame_start`, `frame_obs_end`, `frame_end`, `source` | pedestrian id, frames of the window, data file (per-pedestrian file only) |

Other useful console lines:

- `완전 복제(sigma=0) 시 원래 사람 예측의 최대 변화` - maximum change of the original pedestrians'
  predictions under exact duplication (expected ~0 for SGAN up to floating-point error).
- `[그래프 검사]` (Social-STGCNN) - the script builds the graph with a vectorised NumPy
  implementation and compares it once with the official `utils.seq_to_graph` (difference should be
  below 1e-5).

Logs are written to `~/dup_repro/log_{sgan,stgcnn,agentformer}.txt`.

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
