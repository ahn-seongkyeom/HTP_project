#!/usr/bin/env bash
# =============================================================================
# Near-duplicate robustness: SGAN (20VP, max pooling) / Social-STGCNN / AgentFormer
# Official code + official data + released pretrained weights, all downloaded from their original sources.
# Requirements: Ubuntu (x86_64), NVIDIA GPU + driver, git, wget, unzip, curl
# Usage: bash run_all.sh     (working dir ~/dup_repro, results in ~/dup_repro/results_all)
#        FORCE=1 bash run_all.sh      re-run finished models
#        REPRO_ROOT=/path bash run_all.sh   other working dir
# =============================================================================
set -euo pipefail
export REPRO_ROOT="${REPRO_ROOT:-$HOME/dup_repro}"
mkdir -p "$REPRO_ROOT"; cd "$REPRO_ROOT"
log(){ echo -e "\n\033[1;36m==== $* ====\033[0m"; }

# ---------------------------------------------------------------- 1. conda + environments
if command -v conda >/dev/null 2>&1; then CONDA="$(command -v conda)"; else
  if [ ! -x "$REPRO_ROOT/miniforge/bin/conda" ]; then
    log "Installing Miniforge"
    wget -q -O miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash miniforge.sh -b -p "$REPRO_ROOT/miniforge"; rm -f miniforge.sh
  fi
  CONDA="$REPRO_ROOT/miniforge/bin/conda"
fi
mkenv(){ [ -x "$REPRO_ROOT/envs/$1/bin/python" ] || "$CONDA" create -y -q -p "$REPRO_ROOT/envs/$1" -c conda-forge "python=$2"; }
PY_S="$REPRO_ROOT/envs/sgan/bin/python"
PY_T="$REPRO_ROOT/envs/stgcnn/bin/python"
PY_A="$REPRO_ROOT/envs/agentformer/bin/python"
TORCH="torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118"
log "Python environments (sgan py3.8 / stgcnn py3.10 / agentformer py3.8, PyTorch 2.0.1 + CUDA 11.8)"
mkenv sgan 3.8; mkenv stgcnn 3.10; mkenv agentformer 3.8
"$PY_S" -m pip install -q $TORCH; "$PY_S" -m pip install -q attrdict==2.0.0 "numpy<1.25"
"$PY_T" -m pip install -q $TORCH; "$PY_T" -m pip install -q "numpy<2" networkx==2.8.8 "scipy==1.11.4" tqdm   # networkx 2.8.8 calls scipy.errstate (removed in scipy 1.12)
"$PY_A" -m pip install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118; "$PY_A" -m pip install -q "numpy<1.25" scipy pyyaml easydict glob2 six tensorboard opencv-python-headless gdown
"$PY_S" -c "import torch,sys; print('  GPU:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else sys.exit('CUDA GPU not found. Please check the NVIDIA driver.')"

# ---------------------------------------------------------------- 2. official repositories + data + released weights
log "SGAN: repository + data (Dropbox, incl. raw/all_data) + released weights (Dropbox)"
cd "$REPRO_ROOT"; [ -d sgan/.git ] || git clone -q https://github.com/agrimgupta92/sgan.git
cd sgan
[ -d datasets/eth ] || bash scripts/download_data.sh
[ -d models/sgan-p-models ] || bash scripts/download_models.sh
mkdir -p ckpt_author
for d in eth hotel univ zara1 zara2; do cp models/sgan-p-models/${d}_12_model.pt ckpt_author/${d}_20VP_with_model.pt; done

log "Social-STGCNN: repository (includes data and weights)"
cd "$REPRO_ROOT"; [ -d Social-STGCNN/.git ] || git clone -q https://github.com/abduallahmohamed/Social-STGCNN.git

log "AgentFormer: repository (includes data) + released weights (Google Drive)"
cd "$REPRO_ROOT"; [ -d AgentFormer/.git ] || git clone -q https://github.com/Khrylx/AgentFormer.git
cd AgentFormer
AF_ID=1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB
AF_ZIP="$REPRO_ROOT/AgentFormer/agentformer_models.zip"
af_zip_ok(){ [ -s "$AF_ZIP" ] && unzip -tq "$AF_ZIP" >/dev/null 2>&1; }
if [ ! -d results/eth_agentformer ]; then
  if ! af_zip_ok; then            # 1) gdown with IPv4 forced (some environments such as WSL have no IPv6 route)
    "$PY_A" - << 'GD' || true
import urllib3.util.connection as c; c.HAS_IPV6 = False
import gdown; gdown.download(id="1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB", output="agentformer_models.zip", quiet=False)
GD
  fi
  if ! af_zip_ok; then            # 2) curl, IPv4, large-file confirmation
    curl -4 -L --retry 3 -o "$AF_ZIP" "https://drive.usercontent.google.com/download?id=$AF_ID&export=download&confirm=t" || true
  fi
  if af_zip_ok; then unzip -q -o "$AF_ZIP" && rm -f "$AF_ZIP"; fi
fi
if [ -d results/eth_agentformer ]; then AF_OK=1; else
  AF_OK=0
  echo "  [warning] Could not download the AgentFormer weights automatically (SGAN and Social-STGCNN continue)."
  echo "            Download agentformer_models.zip from https://drive.google.com/file/d/$AF_ID/view,"
  echo "            put it at $AF_ZIP and run this script again (only AgentFormer will run)."
fi

log "Frame interval of every data file (all must be 10)"
cd "$REPRO_ROOT/sgan"
"$PY_S" - << 'CHK'
import numpy as np, glob, os
for f in sorted(glob.glob("datasets/raw/all_data/*.txt")) + sorted(glob.glob("datasets/*/test/*.txt")):
    fr = np.unique(np.loadtxt(f)[:, 0])
    print(f"  {os.path.relpath(f, 'datasets'):<34} frame interval {int(np.median(np.diff(fr)))}")
CHK

# ---------------------------------------------------------------- 3. experiment scripts
cat > "$REPRO_ROOT/sgan/sgan_dup.py" << 'EOF_SGAN_DUP'
# -*- coding: utf-8 -*-
"""
SGAN-20VP (max pooling): near-duplicate experiment.

In every scene one pedestrian A is chosen at random; k copies of A's 8 observed points, each perturbed
with independent N(0, sigma^2) noise per coordinate, are appended to the scene. ADE/FDE are computed for
the ORIGINAL pedestrians only (duplicates are excluded).
The baseline (k = 0) is the output of the official scripts/evaluate_model.py itself: the official
evaluation is run once, and the latent noise z it selected for every scene is reused (fixed) for all
duplicated inputs.
"""
import os, csv, argparse, importlib.util
import numpy as np
import torch

SGAN_ROOT = os.path.join(os.path.expanduser(os.environ.get("REPRO_ROOT", "~")), "sgan")
SEED, K_SAMPLES, OL, PL, MIN_PED = 0, 20, 8, 12, 1
DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]
ROLES = ("all", "source", "other")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--k", default="0,1,2,4", help="numbers of duplicates")
    ap.add_argument("--sigma", default="0,0.02,0.05", help="noise std of the duplicates (m)")
    a = ap.parse_args()
    a.ks = sorted({0} | {int(x) for x in a.k.split(",")})
    a.sigmas = [float(x) for x in a.sigma.split(",")]
    a.configs = [(0, 0.0)] + [(k, s) for k in a.ks if k > 0 for s in a.sigmas]
    a.outdir = os.path.join(SGAN_ROOT, "results_dup", "k" + "-".join(map(str, a.ks)))
    return a


def load_test_files(name, root=SGAN_ROOT):
    d = os.path.join(root, "datasets", name, "test")
    return [(f[:-4], os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".txt")]


def build_scenes(arr):
    """Same windows as the official TrajectoryDataset (20 consecutive frames, >= 2 pedestrians)."""
    d = np.around(arr, 4)
    L = OL + PL
    frames = np.unique(d[:, 0])
    rows = {f: d[d[:, 0] == f] for f in frames}
    out = []
    for i in range(0, max(len(frames) - L + 1, 0)):
        win = frames[i:i + L]
        seg = np.concatenate([rows[f] for f in win], 0)
        ids, tr = [], []
        for pid in np.unique(seg[:, 1]):
            s = seg[seg[:, 1] == pid]
            s = s[np.argsort(s[:, 0])]
            if s[0, 0] != win[0] or s[-1, 0] != win[-1] or len(s) != L:
                continue
            ids.append(int(pid)); tr.append(s[:, 2:4])
        if len(ids) > MIN_PED:
            out.append(dict(frames=win.astype(int), pids=ids, traj=np.stack(tr, 1)))
    return out


def make_scenes(name, a):
    """Scenes + randomly chosen pedestrian A + duplicate noise (used by Social-STGCNN and AgentFormer)."""
    rng, trng = np.random.default_rng(SEED), np.random.default_rng(SEED + 7)
    kmax = max(max(a.ks), 1)
    root = getattr(a, "data_root", SGAN_ROOT)
    scenes = []
    for src, path in load_test_files(name, root):
        sc_ = build_scenes(np.loadtxt(path)[:, :4])
        print(f"  {os.path.relpath(path, root)}: {len(sc_)} scenes, {sum(len(s['pids']) for s in sc_)} pedestrian-windows")
        for sc in sc_:
            sc["src"] = src
            sc["t"] = int(trng.integers(len(sc["pids"])))
            sc["u_dup"] = rng.standard_normal((kmax, OL, 2))
            scenes.append(sc)
    return scenes


def augment(obs, t, k, sg, u_dup):
    """obs (OL, n, 2) -> (OL, n + k, 2); the last k agents are near-duplicates of pedestrian t."""
    if k == 0:
        return obs
    dup = obs[:, t][None] + sg * u_dup[:k]
    return np.concatenate([obs, dup.transpose(1, 0, 2)], axis=1)


def predict(g, A, obs_np, zs):
    from sgan.utils import relative_to_abs
    dev = next(g.parameters()).device
    Kz, n = zs.shape[0], obs_np.shape[1]
    obs = torch.tensor(obs_np, dtype=torch.float32, device=dev).repeat(1, Kz, 1)
    rel = torch.zeros_like(obs); rel[1:] = obs[1:] - obs[:-1]
    se = torch.tensor([[k * n, (k + 1) * n] for k in range(Kz)], device=dev)
    z = zs if A.noise_mix_type == "global" else zs.repeat_interleave(n, 0)
    out = relative_to_abs(g(obs, rel, se, user_noise=z), obs[-1])
    return out.view(out.size(0), Kz, n, 2).permute(1, 0, 2, 3).cpu().numpy()


def _pct(a, b):
    """100 * a / b  (ratio in percent)"""
    return float(100.0 * a / b) if b > 0 else float("nan")


def make_rows(name, sc, t, k, sg, ade0, fde0, ade1, fde1):
    ff = sc["frames"]
    return [[name, sc["src"], pid, int(ff[0]), int(ff[OL - 1]), int(ff[-1]), k, sg,
             "source" if j == t else "other",
             float(ade0[j]), float(fde0[j]), float(ade1[j]), float(fde1[j]),
             float(ade1[j] - ade0[j]), float(fde1[j] - fde0[j]),
             _pct(abs(ade1[j] - ade0[j]), ade0[j]), _pct(abs(fde1[j] - fde0[j]), fde0[j])] for j, pid in enumerate(sc["pids"])]


def run(name, a):
    from attrdict import AttrDict
    import sgan.models as smod
    from sgan.data.loader import data_loader
    from sgan.utils import get_dset_path

    ck = os.path.join(SGAN_ROOT, "ckpt_author", f"{name}_20VP_with_model.pt")
    print(f"\n{'#' * 84}\n[{name}]  SGAN-20VP  {os.path.relpath(ck, SGAN_ROOT)}")
    sp = importlib.util.spec_from_file_location("evaluate_model", os.path.join(SGAN_ROOT, "scripts", "evaluate_model.py"))
    em = importlib.util.module_from_spec(sp); sp.loader.exec_module(em)
    cp = torch.load(ck)
    gen = em.get_generator(cp)
    A = AttrDict(cp["args"])
    print(f"  pooling={A.pooling_type}  best_k={A.best_k}")

    # Scenes built with the same rule, only to recover frame numbers and ids (the official loader drops them)
    ours = []
    for src, path in load_test_files(name):
        for sc in build_scenes(np.loadtxt(path)[:, :4]):
            sc["src"] = src; ours.append(sc)
    by_n = {}
    for i, sc in enumerate(ours):
        by_n.setdefault(len(sc["pids"]), []).append(i)
    full = {n: np.stack([ours[i]["traj"] for i in ix]) for n, ix in by_n.items()}   # (scenes, 20, n, 2)

    # Run the official evaluation and record (1) the batches it reads and (2) the z and predictions it uses
    batches, zrec, rec = [], [], []
    orig_gn = smod.get_noise
    def gn(shape, noise_type):
        z = orig_gn(shape, noise_type); zrec.append(z); return z
    orig_fwd = gen.forward
    def fwd(obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        # The official loader passes permuted (non-contiguous) tensors that Tensor.view rejects in
        # PyTorch >= 1.x; make them contiguous (memory layout only, values unchanged).
        out = orig_fwd(obs_traj.contiguous(), obs_traj_rel.contiguous(), seq_start_end, user_noise)
        rec.append((zrec[-1].detach().cpu().numpy(), out.detach().cpu().numpy().astype(np.float64)))
        return out
    torch.manual_seed(SEED)
    _, loader = data_loader(A, get_dset_path(A.dataset_name, "test"))
    class RecLoader:
        def __iter__(self):
            for b in loader:
                batches.append([x.detach().cpu().numpy() for x in b]); yield b
    smod.get_noise = gn; gen.forward = fwd
    try:
        ade_off, fde_off = em.evaluate(A, RecLoader(), gen, K_SAMPLES)
    finally:
        smod.get_noise = orig_gn
        del gen.forward
    dev = next(gen.parameters()).device
    assert len(rec) == len(batches) * K_SAMPLES

    rng, trng = np.random.default_rng(SEED), np.random.default_rng(SEED + 7)
    kmax = max(max(a.ks), 1)
    rows, exact, tot_a, tot_f, npeds, used, unmatched = [], 0.0, 0.0, 0.0, 0, set(), 0
    with torch.no_grad():
        for bi, b in enumerate(batches):
            obs_b, gt_b, sse = b[0].astype(np.float64), b[1].astype(np.float64), b[6]   # official observations / ground truth
            grp = rec[bi * K_SAMPLES:(bi + 1) * K_SAMPLES]
            P = np.cumsum(np.stack([g_[1] for g_ in grp]), axis=1) + obs_b[-1][None, None]   # relative_to_abs
            Z = np.stack([g_[0] for g_ in grp])
            for j, (s0, e0) in enumerate(sse):
                s0, e0 = int(s0), int(e0); n = e0 - s0
                obs, gt = obs_b[:, s0:e0], gt_b[:, s0:e0]
                # Recover frames/ids by matching all 20 points (first points alone are ambiguous for standing people)
                diff = np.abs(full[n] - np.concatenate([obs, gt])[None]).reshape(len(full[n]), -1).max(1)
                ii = int(diff.argmin())
                if diff[ii] < 1e-3:
                    sc = ours[by_n[n][ii]]; used.add(by_n[n][ii])
                else:   # metrics come from the official batch anyway; only frames/ids are left empty
                    sc = dict(src="?", frames=np.full(OL + PL, -1), pids=list(range(n))); unmatched += 1
                ea = np.linalg.norm(P[:, :, s0:e0] - gt[None], axis=-1)
                k_ade, k_fde = int(ea.sum(1).sum(1).argmin()), int(ea[:, -1].sum(1).argmin())   # scene-level best-of-20
                tot_a += ea[k_ade].sum(); tot_f += ea[k_fde, -1].sum(); npeds += n
                ade0, fde0 = ea[k_ade].mean(0), ea[k_fde, -1]
                idx = sorted({k_ade, k_fde}); ia, jf = idx.index(k_ade), idx.index(k_fde)
                z_fix = torch.tensor(Z[idx, j], dtype=torch.float32, device=dev)
                t = int(trng.integers(n)); u_dup = rng.standard_normal((kmax, OL, 2))
                base = None
                for k, sg in a.configs:
                    if k == 0:
                        ade1, fde1 = ade0, fde0
                    else:
                        if base is None:
                            base = predict(gen, A, obs, z_fix)
                        pb = predict(gen, A, augment(obs, t, k, sg, u_dup), z_fix)[:, :, :n]
                        if sg == 0:
                            exact = max(exact, float(np.abs(pb - base).max()))
                        e = np.linalg.norm(pb - gt[None], axis=-1)
                        ade1, fde1 = e[ia].mean(0), e[jf, -1]
                    rows += make_rows(name, sc, t, k, sg, ade0, fde0, ade1, fde1)

    print(f"  [official evaluate check] {sum(len(b[6]) for b in batches)} official scenes, frames/ids recovered for {len(used)}"
          f"{f' (unmatched {unmatched})' if unmatched else ''};  official evaluate_model "
          f"{float(ade_off):.4f} / {float(fde_off):.4f}  =  baseline of this experiment {tot_a/(npeds*PL):.4f} / {tot_f/npeds:.4f}")
    summ = summarize(name, rows, a)
    print(f"  max change of original predictions under exact duplication (sigma=0): {exact:.2e} m")
    return rows, summ


HDR = ["dataset", "data_file", "ped_id", "frame_start", "frame_obs_end", "frame_end", "k", "sigma", "role",
       "ADE_base", "FDE_base", "ADE_dup", "FDE_dup", "dADE", "dFDE", "ratio_ADE(%)", "ratio_FDE(%)"]
SHDR = ["dataset", "k", "sigma", "role", "n", "ADE_base", "FDE_base", "ADE_dup", "FDE_dup",
        "dADE", "dFDE", "abs_dADE", "abs_dFDE", "ratio_ADE(%)", "ratio_FDE(%)"]


def table(summ, a):
    print(f"\n  {'k':>3} {'sigma':>6} | {'ADE_base':>9} {'ADE_dup':>8} {'dADE':>8} {'ratio%':>7} | "
          f"{'FDE_base':>9} {'FDE_dup':>8} {'dFDE':>8} {'ratio%':>7} | {'other |dADE|':>12}")
    for k, sg in a.configs:
        al = [s for s in summ if s[1] == k and s[2] == sg and s[3] == "all"]
        ot = [s for s in summ if s[1] == k and s[2] == sg and s[3] == "other"]
        if al:
            x = al[0]
            print(f"  {k:>3} {sg:>6} | {x[5]:>9.4f} {x[7]:>8.4f} {x[9]:>+8.4f} {x[13]:>7.2f} | "
                  f"{x[6]:>9.4f} {x[8]:>8.4f} {x[10]:>+8.4f} {x[14]:>7.2f} | {(ot[0][11] if ot else 0):>12.4f}")


def _summary_row(name, k, sg, role, n, M6, absd):
    """M6 = mean of (ADE_base, FDE_base, ADE_dup, FDE_dup, dADE, dFDE), absd = mean |dADE|, |dFDE|.
    ratio_ADE(%) = 100 * mean|ADE_dup - ADE_base| / mean ADE_base  (FDE likewise)."""
    return [name, k, sg, role, n] + list(M6) + list(absd) + [_pct(absd[0], M6[0]), _pct(absd[1], M6[1])]


def summarize(name, rows, a):
    summ = []
    for k, sg in a.configs:
        for role in ROLES:
            sel = [r[9:15] for r in rows if r[6] == k and r[7] == sg and (role == "all" or r[8] == role)]
            if sel:
                M = np.array(sel, float)
                summ.append(_summary_row(name, k, sg, role, len(M), M.mean(0),
                                         [np.abs(M[:, 4]).mean(), np.abs(M[:, 5]).mean()]))
    table(summ, a)
    return summ


def save(all_rows, all_summ, a):
    os.makedirs(a.outdir, exist_ok=True)
    tag = getattr(a, "tag", "sgan")
    avg = []
    for k, sg in a.configs:
        for role in ROLES:
            ms = np.array([s[5:13] for s in all_summ if s[1] == k and s[2] == sg and s[3] == role], float)
            if len(ms):
                n = int(sum(s[4] for s in all_summ if s[1] == k and s[2] == sg and s[3] == role))
                m = ms.mean(0)
                avg.append(_summary_row("AVG", k, sg, role, n, m[:6], m[6:8]))
    with open(f"{a.outdir}/summary_{tag}.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(SHDR)
        for s in all_summ + avg:
            w.writerow(s[:5] + [f"{v:.6f}" for v in s[5:]])
    with open(f"{a.outdir}/per_pedestrian_{tag}.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(HDR)
        for r in all_rows:
            w.writerow(r[:9] + [f"{v:.6f}" for v in r[9:]])
    print(f"\n-> {a.outdir}/summary_{tag}.csv, per_pedestrian_{tag}.csv")
    print(f"\n{'=' * 104}\n{getattr(a, 'label', 'SGAN-20VP')}  near-duplicate experiment - average over datasets, original pedestrians only (m)")
    table(avg, a)


if __name__ == "__main__":
    a = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    R, S = [], []
    for n in a.datasets:
        r = run(n, a)
        R += r[0]; S += r[1]
    save(R, S, a)
EOF_SGAN_DUP
cat > "$REPRO_ROOT/Social-STGCNN/stgcnn_dup.py" << 'EOF_STGCNN_DUP'
# -*- coding: utf-8 -*-
"""Social-STGCNN: near-duplicate experiment with the released weights (checkpoint/ in the official repo).
Best-of-20 per pedestrian as in the official test.py; the standard-normal draws used to sample the
predicted Gaussian are kept fixed before/after duplication."""
import os, sys, pickle, time, importlib.util
import numpy as np
import torch

ROOT = os.path.expanduser(os.environ.get("REPRO_ROOT", "~"))
STG_ROOT = os.path.join(ROOT, "Social-STGCNN")
K_SAMPLES = 20
_sp = importlib.util.spec_from_file_location("sgan_dup", os.path.join(ROOT, "sgan", "sgan_dup.py"))
M = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(M)
OL, PL = M.OL, M.PL
sys.path.insert(0, STG_ROOT)


def load_model(name, dev):
    from model import social_stgcnn
    exp = f"{STG_ROOT}/checkpoint/social-stgcnn-{name}"
    with open(f"{exp}/args.pkl", "rb") as fo:
        args = pickle.load(fo)
    m = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn, output_feat=args.output_size,
                      seq_len=args.obs_seq_len, kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
    m.load_state_dict(torch.load(f"{exp}/val_best.pth", map_location="cpu"))
    return m.eval().to(dev)


def fast_graph(rel):
    """Vectorised equivalent of the official utils.seq_to_graph (V = displacements, A = normalized Laplacian)."""
    r = rel.transpose(2, 0, 1)
    d = np.linalg.norm(r[:, :, None, :] - r[:, None, :, :], axis=-1)
    with np.errstate(divide="ignore"):
        A = np.where(d == 0, 0.0, 1.0 / d)
    i = np.arange(A.shape[1]); A[:, i, i] = 1.0
    deg = A.sum(-1)
    with np.errstate(divide="ignore"):
        dh = 1.0 / np.sqrt(deg)
    dh[~np.isfinite(dh)] = 0.0
    Lp = dh[:, :, None] * (np.eye(A.shape[1])[None] * deg[:, :, None] - A) * dh[:, None, :]
    return torch.from_numpy(r.copy()).float(), torch.from_numpy(Lp).float()


def rel_of(obs):
    seq = obs.transpose(1, 2, 0).astype(np.float64)
    rel = np.zeros_like(seq); rel[:, :, 1:] = seq[:, :, 1:] - seq[:, :, :-1]
    return seq, rel


def params(model, obs, dev):
    _, rel = rel_of(obs)
    V, A = fast_graph(rel)
    with torch.no_grad():
        out, _ = model(V.unsqueeze(0).permute(0, 3, 1, 2).to(dev), A.to(dev))
    out = out.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.float64)
    return out[..., :2], np.exp(out[..., 2]), np.exp(out[..., 3]), np.tanh(out[..., 4])


def sample_abs(mean, sx, sy, corr, eps, last):
    """Same as torch MultivariateNormal.sample() (loc + scale_tril @ eps), then cumulative sum to positions."""
    dx = sx[None] * eps[..., 0]
    dy = (corr * sy)[None] * eps[..., 0] + (sy * np.sqrt(np.clip(1 - corr ** 2, 0, None)))[None] * eps[..., 1]
    return np.cumsum(mean[None] + np.stack([dx, dy], -1), axis=1) + last[None, None]


def check_graph(scenes):
    """Compare the vectorised graph with the official utils.seq_to_graph (informative only)."""
    try:
        from utils import seq_to_graph
        worst = 0.0
        for sc in scenes[:5]:
            seq, rel = rel_of(M.augment(sc["traj"][:OL], sc["t"], 4, 0.02, sc["u_dup"]))
            V1, A1 = seq_to_graph(seq, rel, True)
            V2, A2 = fast_graph(rel)
            worst = max(worst, float((V1 - V2).abs().max()), float((A1 - A2).abs().max()))
        print(f"  [graph check] max difference to official seq_to_graph: {worst:.2e}  (identical if < 1e-5)")
    except Exception as e:
        print(f"  [graph check] skipped ({type(e).__name__}: {e}) - does not affect the experiment")


def run(name, a, dev):
    print(f"\n{'#' * 84}\n[{name}]  Social-STGCNN")
    model = load_model(name, dev)
    scenes = M.make_scenes(name, a)
    check_graph(scenes)
    rng = np.random.default_rng(M.SEED + 1)
    rows, exact, t0 = [], 0.0, time.time()
    for i, sc in enumerate(scenes):
        if i and i % 200 == 0:
            el = time.time() - t0
            print(f"    scene {i}/{len(scenes)}  elapsed {el/60:.1f} min  remaining ~{el/i*(len(scenes)-i)/60:.1f} min", flush=True)
        obs, gt, n, t = sc["traj"][:OL], sc["traj"][OL:], len(sc["pids"]), sc["t"]
        eps = rng.standard_normal((K_SAMPLES, PL, n, 2))
        pa = sample_abs(*params(model, obs, dev), eps, obs[-1])
        e0 = np.linalg.norm(pa - gt[None], axis=-1)
        ka, kf = e0.mean(1).argmin(0), e0[:, -1].argmin(0)      # per-pedestrian best-of-20
        jj = np.arange(n)
        ade0, fde0 = e0.mean(1)[ka, jj], e0[:, -1][kf, jj]
        for k, sg in a.configs:
            if k == 0:
                pb = pa
            else:
                prm = tuple(x[:, :n] for x in params(model, M.augment(obs, t, k, sg, sc["u_dup"]), dev))
                pb = sample_abs(*prm, eps, obs[-1])
                if sg == 0:
                    exact = max(exact, float(np.abs(pb - pa).max()))
            e = np.linalg.norm(pb - gt[None], axis=-1)
            rows += M.make_rows(name, sc, t, k, sg, ade0, fde0, e.mean(1)[ka, jj], e[:, -1][kf, jj])
    summ = M.summarize(name, rows, a)
    print(f"  max change of original predictions under exact duplication (sigma=0): {exact:.2e} m")
    return rows, summ


if __name__ == "__main__":
    a = M.parse_args()
    a.label, a.tag, a.data_root = "Social-STGCNN", "stgcnn", STG_ROOT
    a.outdir = os.path.join(STG_ROOT, "results_dup", "k" + "-".join(map(str, a.ks)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R, S = [], []
    for n in a.datasets:
        r = run(n, a, dev)
        R += r[0]; S += r[1]
    M.save(R, S, a)
EOF_STGCNN_DUP
cat > "$REPRO_ROOT/AgentFormer/agentformer_dup.py" << 'EOF_AGENTFORMER_DUP'
# -*- coding: utf-8 -*-
"""AgentFormer (DLow): near-duplicate experiment with the released weights.
Best-of-20 per pedestrian as in the official eval.py. The official DLow inference (mean=True) computes
20 latent codes z from the observations; they are kept fixed, and each duplicate receives A's codes."""
import os, sys, time, importlib.util
import numpy as np
import torch

ROOT = os.path.expanduser(os.environ.get("REPRO_ROOT", "~"))
AF_ROOT = os.path.join(ROOT, "AgentFormer")
_sp = importlib.util.spec_from_file_location("sgan_dup", os.path.join(ROOT, "sgan", "sgan_dup.py"))
M = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(M)
OL, PL = M.OL, M.PL


def load_model(name, dev):
    os.chdir(AF_ROOT); sys.path.insert(0, AF_ROOT)
    import torch.nn.modules.linear as _lin              # the official code imports _LinearWithBias (PyTorch 1.8 only)
    if not hasattr(_lin, "_LinearWithBias"):            # alias to the same bias-enabled linear layer (same parameter names)
        _lin._LinearWithBias = _lin.NonDynamicallyQuantizableLinear
    from utils.config import Config
    from model.model_lib import model_dict
    cfg = Config(f"{name}_agentformer", tmp=False, create_dirs=False)
    model = model_dict[cfg.get("model_id", "agentformer")](cfg)
    model.load_state_dict(torch.load(cfg.model_path % cfg.get_last_epoch(), map_location="cpu")["model_dict"], strict=False)
    model.set_device(dev); model.eval()
    return model, float(cfg.traj_scale)


def af_predict(model, obs, gt, pids, ts, z=None):
    n = obs.shape[1]
    data = {"pre_motion_3D": [torch.from_numpy(obs[:, j] / ts).float() for j in range(n)],
            "fut_motion_3D": [torch.from_numpy(gt[:, j] / ts).float() for j in range(n)],
            "pre_motion_mask": [torch.ones(OL) for _ in range(n)],
            "fut_motion_mask": [torch.ones(PL) for _ in range(n)],
            "heading": None, "valid_id": list(pids), "traj_scale": ts, "pred_mask": None, "scene_map": None}
    with torch.no_grad():
        model.set_data(data)
        d, pm = model.data, model.pred_model[0]
        pm.context_encoder(d)
        if z is None:                                   # official DLow inference (mean=True)
            z = model.q_b(model.q_mlp(d["agent_context"])).view(-1, model.nz)
        pm.future_decoder(d, mode="infer", sample_num=model.nk, autoregress=True, z=z)
        out = d["infer_dec_motion"][..., :2] * ts
    return out.permute(1, 2, 0, 3).cpu().numpy().astype(np.float64), z


def run(name, a, dev):
    print(f"\n{'#' * 84}\n[{name}]  AgentFormer")
    model, ts = load_model(name, dev)
    K = model.nk
    scenes = M.make_scenes(name, a)
    rows, exact, t0 = [], 0.0, time.time()
    for i, sc in enumerate(scenes):
        if i and i % 200 == 0:
            el = time.time() - t0
            print(f"    scene {i}/{len(scenes)}  elapsed {el/60:.1f} min  remaining ~{el/i*(len(scenes)-i)/60:.1f} min", flush=True)
        obs, gt, n, t, pids = sc["traj"][:OL], sc["traj"][OL:], len(sc["pids"]), sc["t"], list(sc["pids"])
        pa, z0 = af_predict(model, obs, gt, pids, ts)
        e0 = np.linalg.norm(pa - gt[None], axis=-1)
        ka, kf = e0.mean(1).argmin(0), e0[:, -1].argmin(0)      # per-pedestrian best-of-20
        jj = np.arange(n)
        ade0, fde0 = e0.mean(1)[ka, jj], e0[:, -1][kf, jj]
        zt = z0[t * K:(t + 1) * K]                              # the 20 latent codes of pedestrian A
        for k, sg in a.configs:
            if k == 0:
                pb = pa
            else:
                gt_aug = np.concatenate([gt, np.repeat(gt[:, t:t + 1], k, axis=1)], axis=1)
                out, _ = af_predict(model, M.augment(obs, t, k, sg, sc["u_dup"]), gt_aug,
                                    pids + [10 ** 6 + c for c in range(k)], ts, z=torch.cat([z0] + [zt] * k, 0))
                pb = out[:, :, :n]
                if sg == 0:
                    exact = max(exact, float(np.abs(pb - pa).max()))
            e = np.linalg.norm(pb - gt[None], axis=-1)
            rows += M.make_rows(name, sc, t, k, sg, ade0, fde0, e.mean(1)[ka, jj], e[:, -1][kf, jj])
    summ = M.summarize(name, rows, a)
    print(f"  max change of original predictions under exact duplication (sigma=0): {exact:.2e} m")
    return rows, summ


if __name__ == "__main__":
    a = M.parse_args()
    a.label, a.tag, a.data_root = "AgentFormer", "agentformer", M.SGAN_ROOT
    a.outdir = os.path.join(AF_ROOT, "results_dup", "k" + "-".join(map(str, a.ks)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R, S = [], []
    for n in a.datasets:
        r = run(n, a, dev)
        R += r[0]; S += r[1]
    M.save(R, S, a)
EOF_AGENTFORMER_DUP

# ---------------------------------------------------------------- 4. run
finished(){ ls "$REPRO_ROOT/$1"/results_dup/k*/summary_$2.csv >/dev/null 2>&1 && [ "${FORCE:-0}" != 1 ]; }
FAILED=""
if finished sgan sgan; then log "Experiment 1/3  SGAN already finished -> skipped (FORCE=1 bash run_all.sh to re-run)"; else
  log "Experiment 1/3  SGAN (baseline = output of the official evaluate_model.py)"
  cd "$REPRO_ROOT/sgan" && PYTHONPATH=. "$PY_S" -u sgan_dup.py 2>&1 | tee "$REPRO_ROOT/log_sgan.txt" || FAILED="$FAILED SGAN"
fi
if finished Social-STGCNN stgcnn; then log "Experiment 2/3  Social-STGCNN already finished -> skipped"; else
  log "Experiment 2/3  Social-STGCNN"
  cd "$REPRO_ROOT/Social-STGCNN" && "$PY_T" -u stgcnn_dup.py 2>&1 | tee "$REPRO_ROOT/log_stgcnn.txt" || FAILED="$FAILED Social-STGCNN"
fi
if [ "$AF_OK" != 1 ]; then log "Experiment 3/3  AgentFormer skipped (weights missing)"; FAILED="$FAILED AgentFormer(weights-missing)"
elif finished AgentFormer agentformer; then log "Experiment 3/3  AgentFormer already finished -> skipped"; else
  log "Experiment 3/3  AgentFormer"
  cd "$REPRO_ROOT/AgentFormer" && "$PY_A" -u agentformer_dup.py 2>&1 | tee "$REPRO_ROOT/log_agentformer.txt" || FAILED="$FAILED AgentFormer"
fi

log "Results -> $REPRO_ROOT/results_all"
mkdir -p "$REPRO_ROOT/results_all"
for f in "$REPRO_ROOT"/sgan/results_dup/k*/*.csv "$REPRO_ROOT"/Social-STGCNN/results_dup/k*/*.csv \
         "$REPRO_ROOT"/AgentFormer/results_dup/k*/*.csv; do
  if [ -f "$f" ]; then cp "$f" "$REPRO_ROOT/results_all/"; fi
done
ls "$REPRO_ROOT/results_all"
if [ -n "$FAILED" ]; then
  log "Finished, but some models failed:$FAILED  (logs: $REPRO_ROOT/log_*.txt). Fix the cause and run again; only those models will run."
  exit 1
fi
log "Done"
