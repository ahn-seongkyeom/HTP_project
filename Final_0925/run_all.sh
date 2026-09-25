#!/usr/bin/env bash
# =============================================================================
# Near-duplicate robustness: SGAN(20VP, max pooling) / Social-STGCNN / AgentFormer
# 공식 저장소 코드 + 공식 데이터 + 공개 가중치를 모두 원출처에서 받아 실행
# 요구: Ubuntu(x86_64), NVIDIA GPU + 드라이버, git, wget, unzip
# 사용: bash run_all.sh     (작업 폴더 ~/dup_repro, 결과 ~/dup_repro/results_all)
# =============================================================================
set -euo pipefail
export REPRO_ROOT="${REPRO_ROOT:-$HOME/dup_repro}"
mkdir -p "$REPRO_ROOT"; cd "$REPRO_ROOT"
log(){ echo -e "\n\033[1;36m==== $* ====\033[0m"; }

# ---------------------------------------------------------------- 1. conda + 가상환경
if command -v conda >/dev/null 2>&1; then CONDA="$(command -v conda)"; else
  if [ ! -x "$REPRO_ROOT/miniforge/bin/conda" ]; then
    log "Miniforge 설치"
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
log "가상환경 (sgan py3.8 / stgcnn py3.10 / agentformer py3.8, PyTorch 2.0.1 + CUDA 11.8)"
mkenv sgan 3.8; mkenv stgcnn 3.10; mkenv agentformer 3.8
"$PY_S" -m pip install -q $TORCH; "$PY_S" -m pip install -q attrdict==2.0.0 "numpy<1.25"
"$PY_T" -m pip install -q $TORCH; "$PY_T" -m pip install -q "numpy<2" networkx==2.8.8 "scipy==1.11.4" tqdm   # networkx 2.8.8 은 scipy.errstate 사용 (1.12 에서 삭제)
"$PY_A" -m pip install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118; "$PY_A" -m pip install -q "numpy<1.25" scipy pyyaml easydict glob2 six tensorboard opencv-python-headless gdown
"$PY_S" -c "import torch,sys; print('  GPU:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else sys.exit('GPU(CUDA)를 인식하지 못했습니다. NVIDIA 드라이버를 확인하세요')"

# ---------------------------------------------------------------- 2. 공식 저장소 + 데이터 + 공개 가중치
log "SGAN: 저장소 + 데이터(Dropbox, raw 포함) + 공개 가중치(Dropbox)"
cd "$REPRO_ROOT"; [ -d sgan/.git ] || git clone -q https://github.com/agrimgupta92/sgan.git
cd sgan
[ -d datasets/eth ] || bash scripts/download_data.sh
[ -d models/sgan-p-models ] || bash scripts/download_models.sh
mkdir -p ckpt_author
for d in eth hotel univ zara1 zara2; do cp models/sgan-p-models/${d}_12_model.pt ckpt_author/${d}_20VP_with_model.pt; done

log "Social-STGCNN: 저장소 (데이터·가중치 포함)"
cd "$REPRO_ROOT"; [ -d Social-STGCNN/.git ] || git clone -q https://github.com/abduallahmohamed/Social-STGCNN.git

log "AgentFormer: 저장소 (데이터 포함) + 공개 가중치(Google Drive)"
cd "$REPRO_ROOT"; [ -d AgentFormer/.git ] || git clone -q https://github.com/Khrylx/AgentFormer.git
cd AgentFormer
AF_ID=1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB
AF_ZIP="$REPRO_ROOT/AgentFormer/agentformer_models.zip"
af_zip_ok(){ [ -s "$AF_ZIP" ] && unzip -tq "$AF_ZIP" >/dev/null 2>&1; }
if [ ! -d results/eth_agentformer ]; then
  if ! af_zip_ok; then            # 1) gdown, IPv4 강제 (IPv6 경로가 없는 WSL 등 대비)
    "$PY_A" - << 'GD' || true
import urllib3.util.connection as c; c.HAS_IPV6 = False
import gdown; gdown.download(id="1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB", output="agentformer_models.zip", quiet=False)
GD
  fi
  if ! af_zip_ok; then            # 2) curl, IPv4 + 대용량 확인 통과
    curl -4 -L --retry 3 -o "$AF_ZIP" "https://drive.usercontent.google.com/download?id=$AF_ID&export=download&confirm=t" || true
  fi
  if af_zip_ok; then unzip -q -o "$AF_ZIP" && rm -f "$AF_ZIP"; fi
fi
if [ -d results/eth_agentformer ]; then AF_OK=1; else
  AF_OK=0
  echo "  [주의] AgentFormer 가중치를 자동으로 받지 못했습니다 (SGAN·STGCNN 은 계속 진행)."
  echo "         브라우저로 https://drive.google.com/file/d/$AF_ID/view 에서 agentformer_models.zip 을 받아"
  echo "         $AF_ZIP 에 두고 다시 실행하면 AgentFormer 만 이어서 돕니다."
fi

log "데이터 프레임 간격 확인 (전부 10 이어야 함)"
cd "$REPRO_ROOT/sgan"
"$PY_S" - << 'CHK'
import numpy as np, glob, os
for f in sorted(glob.glob("datasets/raw/all_data/*.txt")) + sorted(glob.glob("datasets/*/test/*.txt")):
    fr = np.unique(np.loadtxt(f)[:, 0])
    print(f"  {os.path.relpath(f, 'datasets'):<34} 프레임 간격 {int(np.median(np.diff(fr)))}")
CHK

# ---------------------------------------------------------------- 3. 실험 코드
cat > "$REPRO_ROOT/sgan/sgan_dup.py" << 'EOF_SGAN_DUP'
# -*- coding: utf-8 -*-
"""
SGAN 20VP (max pooling) 복제 실험
장면마다 무작위 1명(A)의 관측 8점에 N(0, sigma^2) 을 더한 복제본 k 개를 추가하고,
원래 보행자들의 ADE/FDE 변화를 본다 (복제본은 평가 제외).
기존값 = 공식 scripts/evaluate_model.py 실행값 그대로. 그 실행에서 장면마다 고른 best z 를 고정해 사용.
"""
import os, csv, time, argparse, importlib.util
import numpy as np
import torch

SGAN_ROOT = os.path.join(os.path.expanduser(os.environ.get("REPRO_ROOT", "~")), "sgan")
SEED, K_SAMPLES, OL, PL, MIN_PED = 0, 20, 8, 12, 1
DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]
ROLES = ("all", "source", "other")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--k", default="0,1,2,4")
    ap.add_argument("--sigma", default="0,0.02,0.05")
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
    """SGAN 공식 TrajectoryDataset 과 같은 창 구성 (20점 연속, 2명 이상)"""
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
    """장면 + 복제할 사람 A + 복제본 잡음 (STGCNN·AgentFormer 가 사용)"""
    rng, trng = np.random.default_rng(SEED), np.random.default_rng(SEED + 7)
    kmax = max(max(a.ks), 1)
    root = getattr(a, "data_root", SGAN_ROOT)
    scenes = []
    for src, path in load_test_files(name, root):
        sc_ = build_scenes(np.loadtxt(path)[:, :4])
        print(f"  {os.path.relpath(path, root)}: 장면 {len(sc_)}, 보행자-창 {sum(len(s['pids']) for s in sc_)}")
        for sc in sc_:
            sc["src"] = src
            sc["t"] = int(trng.integers(len(sc["pids"])))
            sc["u_dup"] = rng.standard_normal((kmax, OL, 2))
            scenes.append(sc)
    return scenes


def augment(obs, t, k, sg, u_dup):
    """obs (OL,n,2) -> (OL,n+k,2). 뒤 k 명이 A 의 복제본"""
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


def make_rows(name, sc, t, k, sg, ade0, fde0, ade1, fde1):
    ff = sc["frames"]
    return [[name, sc["src"], pid, int(ff[0]), int(ff[OL - 1]), int(ff[-1]), k, sg,
             "source" if j == t else "other",
             float(ade0[j]), float(fde0[j]), float(ade1[j]), float(fde1[j]),
             float(ade1[j] - ade0[j]), float(fde1[j] - fde0[j])] for j, pid in enumerate(sc["pids"])]


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

    # 결과 파일의 프레임·ID 용: 같은 규칙으로 만든 장면 (공식 로더는 이 정보를 버림)
    ours = []
    for src, path in load_test_files(name):
        for sc in build_scenes(np.loadtxt(path)[:, :4]):
            sc["src"] = src; ours.append(sc)
    by_n = {}
    for i, sc in enumerate(ours):
        by_n.setdefault(len(sc["pids"]), []).append(i)
    full = {n: np.stack([ours[i]["traj"] for i in ix]) for n, ix in by_n.items()}   # (장면수, 20, n, 2)

    # 공식 evaluate 를 실행하면서 (1) 읽은 배치 (2) 그때의 z·예측 을 그대로 기록
    batches, zrec, rec = [], [], []
    orig_gn = smod.get_noise
    def gn(shape, noise_type):
        z = orig_gn(shape, noise_type); zrec.append(z); return z
    orig_fwd = gen.forward
    def fwd(obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        # 공식 로더는 permute 된(비연속) 텐서를 넘김. PyTorch 1.x 이후 .view() 가 거부하므로 값은 그대로 두고 메모리만 정렬
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
            obs_b, gt_b, sse = b[0].astype(np.float64), b[1].astype(np.float64), b[6]   # 공식 배치의 관측·정답
            grp = rec[bi * K_SAMPLES:(bi + 1) * K_SAMPLES]
            P = np.cumsum(np.stack([g_[1] for g_ in grp]), axis=1) + obs_b[-1][None, None]   # relative_to_abs
            Z = np.stack([g_[0] for g_ in grp])
            for j, (s0, e0) in enumerate(sse):
                s0, e0 = int(s0), int(e0); n = e0 - s0
                obs, gt = obs_b[:, s0:e0], gt_b[:, s0:e0]
                # 프레임·ID 찾기: 20점 전체로 비교 (첫 위치만 보면 서 있는 사람 때문에 헷갈림)
                diff = np.abs(full[n] - np.concatenate([obs, gt])[None]).reshape(len(full[n]), -1).max(1)
                ii = int(diff.argmin())
                if diff[ii] < 1e-3:
                    sc = ours[by_n[n][ii]]; used.add(by_n[n][ii])
                else:                       # 못 찾아도 수치는 공식 배치로 계산하므로 영향 없음. 프레임·ID 만 비워 둠
                    sc = dict(src="?", frames=np.full(OL + PL, -1), pids=list(range(n))); unmatched += 1
                ea = np.linalg.norm(P[:, :, s0:e0] - gt[None], axis=-1)
                k_ade, k_fde = int(ea.sum(1).sum(1).argmin()), int(ea[:, -1].sum(1).argmin())
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

    print(f"  [공식 evaluate 대조] 공식 장면 {sum(len(b[6]) for b in batches)}개 중 프레임·ID 대응 {len(used)}"
          f"{f' (대응 못 함 {unmatched})' if unmatched else ''},  공식 evaluate_model "
          f"{float(ade_off):.4f} / {float(fde_off):.4f}  =  이 실험의 기존값 {tot_a/(npeds*PL):.4f} / {tot_f/npeds:.4f}")
    summ = summarize(name, rows, a)
    print(f"  완전 복제(sigma=0) 시 원래 사람 예측의 최대 변화: {exact:.2e} m")
    return rows, summ


HDR = ["dataset", "source", "ped_id", "frame_start", "frame_obs_end", "frame_end", "k", "sigma", "role",
       "기존ADE", "기존FDE", "만든ADE", "만든FDE", "ADE차이", "FDE차이"]
SHDR = ["dataset", "k", "sigma", "role", "건수", "기존ADE", "기존FDE", "만든ADE", "만든FDE",
        "ADE차이", "FDE차이", "|ADE차이|", "|FDE차이|"]


def table(summ, a):
    print(f"\n  {'k':>3} {'sigma':>6} | {'전체 기존ADE':>11} {'만든ADE':>8} {'차이':>8} | "
          f"{'기존FDE':>8} {'만든FDE':>8} {'차이':>8} | {'나머지|ΔADE|':>12}")
    for k, sg in a.configs:
        al = [s for s in summ if s[1] == k and s[2] == sg and s[3] == "all"]
        ot = [s for s in summ if s[1] == k and s[2] == sg and s[3] == "other"]
        if al:
            x = al[0]
            print(f"  {k:>3} {sg:>6} | {x[5]:>11.4f} {x[7]:>8.4f} {x[9]:>+8.4f} | "
                  f"{x[6]:>8.4f} {x[8]:>8.4f} {x[10]:>+8.4f} | {(ot[0][11] if ot else 0):>12.4f}")


def summarize(name, rows, a):
    summ = []
    for k, sg in a.configs:
        for role in ROLES:
            sel = [r[9:] for r in rows if r[6] == k and r[7] == sg and (role == "all" or r[8] == role)]
            if sel:
                M = np.array(sel, float)
                summ.append([name, k, sg, role, len(M)] + list(M.mean(0)) +
                            [float(np.abs(M[:, 4]).mean()), float(np.abs(M[:, 5]).mean())])
    table(summ, a)
    return summ


def save(all_rows, all_summ, a):
    os.makedirs(a.outdir, exist_ok=True)
    tag = getattr(a, "tag", "sgan")
    avg = []
    for k, sg in a.configs:
        for role in ROLES:
            ms = np.array([s[5:] for s in all_summ if s[1] == k and s[2] == sg and s[3] == role], float)
            if len(ms):
                n = int(sum(s[4] for s in all_summ if s[1] == k and s[2] == sg and s[3] == role))
                avg.append(["AVG", k, sg, role, n] + list(ms.mean(0)))
    with open(f"{a.outdir}/summary_{tag}.csv", "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo); w.writerow(SHDR)
        for s in all_summ + avg:
            w.writerow(s[:5] + [f"{v:.6f}" for v in s[5:]])
    with open(f"{a.outdir}/per_pedestrian_{tag}.csv", "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo); w.writerow(HDR)
        for r in all_rows:
            w.writerow(r[:9] + [f"{v:.6f}" for v in r[9:]])
    print(f"\n-> {a.outdir}/summary_{tag}.csv, per_pedestrian_{tag}.csv")
    print(f"\n{'=' * 96}\n{getattr(a, 'label', 'SGAN-20VP')}  복제 실험 — 데이터셋 평균, 원래 보행자만 평가 (m)")
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
"""Social-STGCNN 복제 실험 — 저장소 포함 공개 weight, 공식 test.py 와 같은 보행자별 best-of-20, 샘플링 난수 고정"""
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
    """utils.seq_to_graph 와 같은 결과를 numpy 로 (V = 이동량, A = 정규화 라플라시안)"""
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
    """torch MultivariateNormal.sample() 과 같은 식 (loc + scale_tril @ eps), 누적합으로 위치"""
    dx = sx[None] * eps[..., 0]
    dy = (corr * sy)[None] * eps[..., 0] + (sy * np.sqrt(np.clip(1 - corr ** 2, 0, None)))[None] * eps[..., 1]
    return np.cumsum(mean[None] + np.stack([dx, dy], -1), axis=1) + last[None, None]


def check_graph(scenes):
    """빠른 그래프가 공식 utils.seq_to_graph 와 같은지 대조 (참고용: 실패해도 실험은 계속)"""
    try:
        from utils import seq_to_graph
        worst = 0.0
        for sc in scenes[:5]:
            seq, rel = rel_of(M.augment(sc["traj"][:OL], sc["t"], 4, 0.02, sc["u_dup"]))
            V1, A1 = seq_to_graph(seq, rel, True)
            V2, A2 = fast_graph(rel)
            worst = max(worst, float((V1 - V2).abs().max()), float((A1 - A2).abs().max()))
        print(f"  [그래프 검사] 공식 seq_to_graph 와 최대 차이 {worst:.2e}  (1e-5 이하면 동일)")
    except Exception as e:
        print(f"  [그래프 검사] 건너뜀 ({type(e).__name__}: {e}) - 실험 계산에는 영향 없음")


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
            print(f"    장면 {i}/{len(scenes)}  경과 {el/60:.1f}분  남은 약 {el/i*(len(scenes)-i)/60:.1f}분", flush=True)
        obs, gt, n, t = sc["traj"][:OL], sc["traj"][OL:], len(sc["pids"]), sc["t"]
        eps = rng.standard_normal((K_SAMPLES, PL, n, 2))
        pa = sample_abs(*params(model, obs, dev), eps, obs[-1])
        e0 = np.linalg.norm(pa - gt[None], axis=-1)
        ka, kf = e0.mean(1).argmin(0), e0[:, -1].argmin(0)
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
    print(f"  완전 복제(sigma=0) 시 원래 사람 예측의 최대 변화: {exact:.2e} m")
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
"""AgentFormer (DLow) 복제 실험 — 공개 weight, 공식 eval.py 와 같은 보행자별 best-of-20, 잠재 z 고정 (복제본은 A 의 z 복사)"""
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
    import torch.nn.modules.linear as _lin              # 공식 코드는 PyTorch 1.8 전용 _LinearWithBias 를 import 함
    if not hasattr(_lin, "_LinearWithBias"):            # 이후 버전의 같은 클래스(편향 있는 Linear)로 연결, 가중치 이름 동일
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
        if z is None:                                   # 공식 DLow 추론 (mean=True)
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
            print(f"    장면 {i}/{len(scenes)}  경과 {el/60:.1f}분  남은 약 {el/i*(len(scenes)-i)/60:.1f}분", flush=True)
        obs, gt, n, t, pids = sc["traj"][:OL], sc["traj"][OL:], len(sc["pids"]), sc["t"], list(sc["pids"])
        pa, z0 = af_predict(model, obs, gt, pids, ts)
        e0 = np.linalg.norm(pa - gt[None], axis=-1)
        ka, kf = e0.mean(1).argmin(0), e0[:, -1].argmin(0)
        jj = np.arange(n)
        ade0, fde0 = e0.mean(1)[ka, jj], e0[:, -1][kf, jj]
        zt = z0[t * K:(t + 1) * K]
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
    print(f"  완전 복제(sigma=0) 시 원래 사람 예측의 최대 변화: {exact:.2e} m")
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

# ---------------------------------------------------------------- 4. 실행
finished(){ ls "$REPRO_ROOT/$1"/results_dup/k*/summary_$2.csv >/dev/null 2>&1 && [ "${FORCE:-0}" != 1 ]; }
FAILED=""
if finished sgan sgan; then log "실험 1/3  SGAN 이미 완료 -> 건너뜀 (다시 하려면 FORCE=1 bash run_all.sh)"; else
  log "실험 1/3  SGAN (기존값 = 공식 evaluate_model 실행값)"
  cd "$REPRO_ROOT/sgan" && PYTHONPATH=. "$PY_S" -u sgan_dup.py 2>&1 | tee "$REPRO_ROOT/log_sgan.txt" || FAILED="$FAILED SGAN"
fi
if finished Social-STGCNN stgcnn; then log "실험 2/3  Social-STGCNN 이미 완료 -> 건너뜀"; else
  log "실험 2/3  Social-STGCNN"
  cd "$REPRO_ROOT/Social-STGCNN" && "$PY_T" -u stgcnn_dup.py 2>&1 | tee "$REPRO_ROOT/log_stgcnn.txt" || FAILED="$FAILED Social-STGCNN"
fi
if [ "$AF_OK" != 1 ]; then log "실험 3/3  AgentFormer 건너뜀 (가중치 없음)"; FAILED="$FAILED AgentFormer(가중치없음)"
elif finished AgentFormer agentformer; then log "실험 3/3  AgentFormer 이미 완료 -> 건너뜀"; else
  log "실험 3/3  AgentFormer"
  cd "$REPRO_ROOT/AgentFormer" && "$PY_A" -u agentformer_dup.py 2>&1 | tee "$REPRO_ROOT/log_agentformer.txt" || FAILED="$FAILED AgentFormer"
fi

log "결과 -> $REPRO_ROOT/results_all"
mkdir -p "$REPRO_ROOT/results_all"
for f in "$REPRO_ROOT"/sgan/results_dup/k*/*.csv "$REPRO_ROOT"/Social-STGCNN/results_dup/k*/*.csv \
         "$REPRO_ROOT"/AgentFormer/results_dup/k*/*.csv; do
  if [ -f "$f" ]; then cp "$f" "$REPRO_ROOT/results_all/"; fi
done
ls "$REPRO_ROOT/results_all"
if [ -n "$FAILED" ]; then
  log "끝났지만 실패한 모델이 있습니다:$FAILED  (로그: $REPRO_ROOT/log_*.txt) -> 원인 해결 후 다시 실행하면 그 모델만 이어서 돕니다"
  exit 1
fi
log "완료"
