import os, re, io, gc, time, math, hashlib, shutil, zipfile, tempfile
import numpy as np, pandas as pd, cv2, joblib
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.morphology import h_maxima, remove_small_objects, binary_closing, disk
from skimage.measure import label as sk_label, regionprops_table
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import local_binary_pattern
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
TILE_PX = 128
TARGET_UM_PX = 0.024267578125
UM_PX_DEFAULT = 0.024267578125
CROP_MODE = 'auto'
CROP_FIXED_FRACTION = 0.0625
CROP_MARGIN_PX = 3
SAT_FRAC_MAX = 0.5
LAP_VAR_MIN = 2.0
NET_INPUT = 224
TTA_LIST = [(0, False), (0, True), (2, False), (2, True)]
FEAT_BATCH = 256
FEAT_DIM = 2048
NORM_ON = False
TAU = -5.219522953033447
HIDDEN = 128
DROPOUT = 0.5
DIMPLE_WORK_UM_PX = 0.05
DIMPLE_BG_UM = 2.0
DIMPLE_RIDGE_UM = (0.08, 0.15, 0.25)
DIMPLE_RIM_PCT = 80
DIMPLE_H_UM = 0.15
DIMPLE_MIN_UM = 0.3
DIMPLE_MAX_UM = 8.0
DIMPLE_MIN_SOLIDITY = 0.65
DIMPLE_REGION = 'ductile'
LBP_WORK_UM_PX = 0.05
LBP_SMOOTH_UM = 1.5
KM_K = 4
KM_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ffbf00']
CLUSTER_NAMES = {0: 'Shear Dimples', 1: 'Shear Walls', 2: 'Conical Dimples', 3: 'Brittle Features'}
CV = {'auc': 0.8703703703703705, 'bacc': 0.8478009259259259, 'n': 59}

def read_gray8(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None: raise IOError(path)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2GRAY)
    if a.dtype == np.uint16:
        a = np.round(a / 257.0).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return a

def info_bar_top(g):
    # Row where the black SEM info bar starts, or None.
    H = g.shape[0]; start = int(H * 0.80)
    idx = np.where(np.median(g, axis=1)[start:] <= 3)[0]
    if len(idx) == 0: return None
    y = start + int(idx[0])
    return y if 0.01 * H <= H - y <= 0.15 * H else None

def crop_bottom(g):
    H = g.shape[0]
    y = info_bar_top(g) if CROP_MODE == "auto" else None
    method = "auto"
    if y is None: y = int(round(H * (1 - CROP_FIXED_FRACTION))); method = "fixed"
    return g[:max(1, y - (CROP_MARGIN_PX if method == "auto" else 0))], method

def load_frame(path, hfw_um):
    # Cropped grayscale frame resampled to TARGET_UM_PX. Returns (frame, scale_factor_applied).
    raw = read_gray8(path); c, _ = crop_bottom(raw)
    um_px = hfw_um / raw.shape[1] if pd.notna(hfw_um) else UM_PX_DEFAULT
    s = um_px / TARGET_UM_PX   # >1: image is coarser than target (upsampled), <1: finer (downsampled)
    if abs(s - 1) > 0.02:
        c = cv2.resize(c, (int(round(c.shape[1] * s)), int(round(c.shape[0] * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return c, s

def tile_frame(frame):
    # Non-overlapping TILE_PX tiles. Returns tiles (N,T,T) uint8 and origins (N,2) = (row, col) in frame pixels.
    H, W = frame.shape; T = TILE_PX
    org = [(r, c) for r in range(0, H - T + 1, T) for c in range(0, W - T + 1, T)]
    return np.stack([frame[r:r + T, c:c + T] for r, c in org]), np.array(org)

def tile_quality(tiles):
    sat = ((tiles == 0) | (tiles == 255)).mean(axis=(1, 2))
    lap = np.array([cv2.Laplacian(t, cv2.CV_64F).var() for t in tiles])
    return (sat <= SAT_FRAC_MAX) & (lap >= LAP_VAR_MIN), sat, lap

def norm_tile(t):
    x = t.astype(np.float32); x = (x - x.mean()) / (x.std() + 1e-6)
    return np.clip(128 + 48 * x, 0, 255).astype(np.uint8)

@torch.no_grad()
def embed_tiles(tiles_u8):
    out = []
    for i in range(0, len(tiles_u8), FEAT_BATCH):
        x = torch.from_numpy(tiles_u8[i:i + FEAT_BATCH]).to(device).float().div(255).unsqueeze(1)
        x = F.interpolate(x, size=(NET_INPUT, NET_INPUT), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        x = (x - MEAN) / STD
        acc = 0
        for k, flip in TTA_LIST:
            v = torch.rot90(x, k, (2, 3)); v = torch.flip(v, (3,)) if flip else v
            acc = acc + resnet(v)
        out.append((acc / len(TTA_LIST)).float().cpu().numpy())
    return np.concatenate(out)

def tiles_to_frame(vals, coords, shape):
    # paint one value per tile into a frame-sized array (NaN = no tile / excluded)
    out = np.full(shape, np.nan, np.float32)
    for v, (r, c) in zip(vals, coords): out[r:r + TILE_PX, c:c + TILE_PX] = v
    return out

def add_scale_bar(ax, um_px, w, h, length_um=None):
    if length_um is None:
        target = w * um_px / 5; length_um = min((1, 2, 5, 10, 20, 50, 100), key=lambda v: abs(v - target))
    L = length_um / um_px; x0, y0 = w * 0.04, h * 0.94
    ax.add_patch(Rectangle((x0 - 10, y0 - h * 0.06), L + 20, h * 0.085, color="black", alpha=0.6))
    ax.add_patch(Rectangle((x0, y0), L, h * 0.012, color="white"))
    ax.text(x0 + L / 2, y0 - h * 0.012, f"{length_um:g} µm", color="white", ha="center", va="bottom", fontsize=10)

def measure_dimples(frame, um_px, region=None):
    # work at DIMPLE_WORK_UM_PX
    f = DIMPLE_WORK_UM_PX / um_px
    if f > 1.0:
        w = cv2.resize(frame, (int(frame.shape[1] / f), int(frame.shape[0] / f)), interpolation=cv2.INTER_AREA); wpx = um_px * f
        reg = None if region is None else cv2.resize(region.astype(np.uint8), (w.shape[1], w.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        w, wpx, reg = frame, um_px, region
    wf = w.astype(np.float32) / 255.0
    d = wf - cv2.GaussianBlur(wf, (0, 0), DIMPLE_BG_UM / wpx)
    rid = sato(d, sigmas=[max(s / wpx, 1.0) for s in DIMPLE_RIDGE_UM], black_ridges=False)
    rim = binary_closing(rid > np.percentile(rid, DIMPLE_RIM_PCT), disk(1))
    inter = remove_small_objects(~rim, max(4, int(math.pi * (DIMPLE_MIN_UM / 2) ** 2 / wpx ** 2)))
    dist = cv2.GaussianBlur(ndi.distance_transform_edt(inter).astype(np.float32), (0, 0), 1.0)
    seeds = sk_label(h_maxima(dist, max(DIMPLE_H_UM / wpx, 0.5)))
    lab = watershed(-dist, seeds, mask=inter)
    pr = pd.DataFrame(regionprops_table(lab, properties=("label", "area", "centroid", "bbox", "solidity")))
    if len(pr) == 0: return np.zeros(frame.shape, np.int32), pd.DataFrame(), 0.0
    H, W = lab.shape
    diam = 2 * np.sqrt(pr.area * wpx ** 2 / math.pi)
    ok = (diam >= DIMPLE_MIN_UM) & (diam <= DIMPLE_MAX_UM) & (pr.solidity >= DIMPLE_MIN_SOLIDITY)
    ok &= (pr["bbox-0"] > 0) & (pr["bbox-1"] > 0) & (pr["bbox-2"] < H) & (pr["bbox-3"] < W)
    if reg is not None and DIMPLE_REGION == "ductile":
        cy = pr["centroid-0"].round().astype(int).clip(0, H - 1); cx = pr["centroid-1"].round().astype(int).clip(0, W - 1)
        ok &= reg[cy.values, cx.values]
    keep_ids = pr.label[ok].values
    out = np.where(np.isin(lab, keep_ids), lab, 0).astype(np.int32)
    tab = pd.DataFrame(dict(dimple_id=keep_ids, area_um2=pr.area[ok].values * wpx ** 2, diam_um=diam[ok].values,
                            cy_px=pr["centroid-0"][ok].values * f if f > 1 else pr["centroid-0"][ok].values,
                            cx_px=pr["centroid-1"][ok].values * f if f > 1 else pr["centroid-1"][ok].values))
    out_full = cv2.resize(out, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) if f > 1 else out
    area_um2 = (reg.sum() if reg is not None and DIMPLE_REGION == "ductile" else reg.size if reg is not None else w.size) * wpx ** 2
    return out_full, tab, float(area_um2)

def ductile_region(bag, c_kept, shape):
    # True where the tile is kept and its score is <= tau
    r = tiles_to_frame((c_kept <= TAU).astype(np.float32), bag["coords"][bag["keep"]], shape)
    return np.nan_to_num(r, nan=0.0) > 0.5

def dimple_summary(tab, area_um2):
    return dict(n_dimples=len(tab), ductile_area_um2=round(area_um2, 1),
                density_per_100um2=round(100 * len(tab) / area_um2, 3) if area_um2 > 0 else np.nan,
                diam_mean_um=round(tab.diam_um.mean(), 3) if len(tab) else np.nan, diam_median_um=round(tab.diam_um.median(), 3) if len(tab) else np.nan,
                area_mean_um2=round(tab.area_um2.mean(), 3) if len(tab) else np.nan)

def lbp_edge_map(frame, um_px):
    f = LBP_WORK_UM_PX / um_px
    w = cv2.resize(frame, (int(frame.shape[1] / f), int(frame.shape[0] / f)), interpolation=cv2.INTER_AREA) if f > 1 else frame
    wpx = um_px * max(f, 1.0)
    code = local_binary_pattern(w, 8, 1, method="uniform")      # 0 = spot, 8 = flat, 4 = edge, 9 = non-uniform, others = corner
    E, Fl, N = [cv2.GaussianBlur((code == v).astype(np.float32), (0, 0), LBP_SMOOTH_UM / wpx) for v in (4, 8, 9)]
    m = E / (E + Fl + N + 1e-6)
    return cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)


class AdditiveMIL(nn.Module):
    def __init__(self, mu, sd, d=FEAT_DIM, h=HIDDEN, p=DROPOUT):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32)); self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        self.drop = nn.Dropout(p); self.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p))
        self.att, self.cls = nn.Linear(h, 1), nn.Linear(h, 1); self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x, mask):
        z = self.enc(self.drop((x - self.mu) / self.sd))
        a = self.att(z).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(1)
        c = self.cls(z).squeeze(-1)
        return self.bias + (a * c).sum(1), c, a

import streamlit as st
st.set_page_config(page_title="SEM fractography", layout="wide")
D = os.path.dirname(os.path.abspath(__file__))
FEAT_BATCH = min(int(FEAT_BATCH), 32)          # 256 tiles at once can exceed the free cloud's ~2.7 GB; same results, less memory
UP = tempfile.gettempdir()          # uploads go to a temp folder (the app folder may be read-only in the cloud)

@st.cache_resource
def load_resnet():
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity(); m = m.to(device).eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m

@st.cache_resource
def load_heads():
    ck = torch.load(f"{D}/additive_mil_head.pt", map_location=device, weights_only=False)
    ms = []
    for s in ck["states"]:
        m = AdditiveMIL(ck["mu"], ck["sd"]).to(device); m.load_state_dict(s); ms.append(m.eval())
    return ms, joblib.load(f"{D}/morphology.joblib")

resnet = load_resnet(); heads, MK = load_heads()

def score_image(path, hfw):
    fr, s = load_frame(path, hfw); tl, org = tile_frame(fr); keep, _, _ = tile_quality(tl)
    if keep.sum() == 0: return None
    t = np.stack([norm_tile(x) for x in tl[keep]]) if NORM_ON else tl[keep]
    feats = embed_tiles(t).astype(np.float16).astype(np.float32)             # same float16 rounding as training
    X = torch.from_numpy(feats)[None].to(device); M = torch.ones(1, X.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad(): out = [m(X, M) for m in heads]
    return dict(frame=fr, scale=s, coords=org, keep=keep, feats=feats,
                logit=float(np.mean([o[0].item() for o in out])), c=np.mean([o[1][0].cpu().numpy() for o in out], 0))
run_model = st.cache_data(show_spinner="Scoring tiles ...")(score_image)

def fig_panel(img, um_px, cmap="gray", overlay=None, kw=None, title=None, cbar=None, legend=None):
    h, w = img.shape[:2]; fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(img, cmap=cmap)
    if overlay is not None:
        im = ax.imshow(overlay, **kw)
        if cbar: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cbar)
    if legend: ax.legend(handles=legend, loc="lower right", fontsize=7)
    add_scale_bar(ax, um_px, w, h); ax.axis("off")
    if title: ax.set_title(title, fontsize=9)
    fig.tight_layout(); return fig

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Pixel size", ["Same as training (49.7 µm across 2048 px)", "Set image width in µm"])
    hfw = float(st.number_input("Full image width (µm)", value=49.7, min_value=0.1, step=0.1)) if mode.startswith("Set") else np.nan
    st.subheader("Dimple detection")
    DIMPLE_RIM_PCT = st.slider("Rim sensitivity (percentile)", 50, 95, int(DIMPLE_RIM_PCT), help="Higher = only the strongest ridges count as rims, so cups merge.")
    DIMPLE_H_UM    = st.slider("Seed height (µm)", 0.02, 0.6, float(DIMPLE_H_UM), 0.01, help="Larger merges neighbouring cups, smaller splits them.")
    DIMPLE_MIN_UM, DIMPLE_MAX_UM = st.slider("Dimple diameter range (µm)", 0.1, 20.0, (float(DIMPLE_MIN_UM), float(DIMPLE_MAX_UM)), 0.1)
    DIMPLE_REGION  = st.radio("Count dimples in", ["ductile", "all"], index=0 if DIMPLE_REGION == "ductile" else 1,
                              format_func=lambda v: "ductile-evidence tiles only" if v == "ductile" else "whole image")
    st.subheader("About the model")
    st.caption("Trained on 6000× SEM images only (see the notice at the top of the page).")
    st.caption("Frozen ResNet50 tile features + Additive MIL head, trained on all labelled images. Scores for images that were in training are optimistic.")
    if CV: st.caption(f"Held-out (cross-validated, {CV['n']} images): AUC {CV['auc']:.2f}, balanced accuracy {CV['bacc']:.2f}. Wide error bars. Brittle area has not been checked against hand-drawn outlines.")

def single_section():
    files = st.file_uploader("SEM micrograph(s)", type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"], accept_multiple_files=True, key="single_upload")

    for f in files or []:
        raw_bytes = f.getvalue(); tag = hashlib.md5(raw_bytes[:200000] + str(len(raw_bytes)).encode()).hexdigest()[:8]
        path = os.path.join(UP, f"upload_{tag}_{os.path.basename(f.name)}")
        with open(path, "wb") as fh: fh.write(raw_bytes)
        try: R = run_model(path, hfw)
        except Exception as e: st.error(f"{f.name}: could not analyse ({e})"); continue
        if R is None: st.error(f"{f.name}: no usable tiles (all blurred or saturated)."); continue
        fr, org, keep, c = R["frame"], R["coords"], R["keep"], R["c"]; h, w = fr.shape; um_px = TARGET_UM_PX
        p_b = 1 / (1 + np.exp(-R["logit"])); brittle = c > TAU
        area_pct = 100 * float(brittle.mean()); used_um2 = int(keep.sum()) * TILE_PX ** 2 * um_px ** 2
        if keep.sum() < 8: st.warning("Fewer than 8 usable tiles: treat results with caution.")
        region = ductile_region(dict(coords=org, keep=keep), c, fr.shape)
        lab, tab, dim_area = measure_dimples(fr, um_px, region); ds = dimple_summary(tab, dim_area)
        cl = MK["km"].predict(MK["pca"].transform((R["feats"] - MK["mu"]) / MK["sd"])); share = np.bincount(cl, minlength=KM_K) / len(cl)

        st.header(f.name)
        m = st.columns(5)
        m[0].metric("Decision", "Brittle" if R["logit"] > 0 else "Ductile"); m[1].metric("P(Brittle)", f"{p_b:.2f}")
        m[2].metric("Brittle-evidence area", f"{area_pct:.1f} %"); m[3].metric("Ductile-evidence area", f"{100 - area_pct:.1f} %")
        m[4].metric("Dimples counted", f"{ds['n_dimples']}")
        m = st.columns(4)
        m[0].metric("Brittle area (µm²)", f"{area_pct / 100 * used_um2:.0f} of {used_um2:.0f}")
        m[1].metric("Dimple density (/100 µm² ductile)", f"{ds['density_per_100um2']}")
        m[2].metric("Median dimple diameter (µm)", f"{ds['diam_median_um']}")
        m[3].metric("Mean dimple area (µm²)", f"{ds['area_mean_um2']}")

        t1, t2, t3, t4 = st.tabs(["Evidence map", "Dimples", "Morphology and texture", "Data and downloads"])
        with t1:
            cmap_ = tiles_to_frame(c - TAU, org[keep], fr.shape); lim = max(1.0, np.nanpercentile(np.abs(cmap_), 99))
            c1, c2 = st.columns(2)
            c1.pyplot(fig_panel(fr, um_px, title="SEM frame (info bar cropped)")); c1.caption("Grayscale frame resampled to the training scale.")
            c2.pyplot(fig_panel(fr, um_px, overlay=np.ma.masked_invalid(cmap_), kw=dict(cmap="coolwarm", vmin=-lim, vmax=lim, alpha=0.5),
                                cbar="tile score - tau", title=f"Signed evidence map | brittle area {area_pct:.0f}%"))
            c2.caption(f"Each {TILE_PX * um_px:.1f} µm tile is scored. Red pushes towards Brittle, blue towards Ductile; the threshold tau = {TAU:.2f} was set so about 5% of held-out ductile tiles come out red. Blurred or saturated tiles are excluded.")
            plt.close("all")
        with t2:
            c1, c2 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 6 * h / w)); ax.imshow(fr, cmap="gray")
            bnd = find_boundaries(lab, mode="inner"); ov = np.zeros((*bnd.shape, 4)); ov[bnd] = (1, 0.1, 0.1, 0.9); ax.imshow(ov)
            add_scale_bar(ax, um_px, w, h); ax.axis("off"); ax.set_title(f"{ds['n_dimples']} dimples outlined", fontsize=9); fig.tight_layout(); c1.pyplot(fig)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            if len(tab): ax.hist(tab.diam_um, bins=25, color="#1f77b4")
            ax.set_xlabel("equivalent diameter (µm)"); ax.set_ylabel("count"); ax.set_title("Dimple size distribution", fontsize=9); fig.tight_layout(); c2.pyplot(fig)
            c2.table(pd.DataFrame({"Measure": ["Count", "Ductile-region area (µm²)", "Density (per 100 µm²)", "Mean diameter (µm)", "Median diameter (µm)", "Mean area (µm²)"],
                                   "Value": [ds["n_dimples"], ds["ductile_area_um2"], ds["density_per_100um2"], ds["diam_mean_um"], ds["diam_median_um"], ds["area_mean_um2"]]}))
            st.caption("Watershed on the rim network, counted inside ductile-evidence tiles. Outlines are automatic: tune the sliders in the sidebar until they follow the dimples you would count by eye. Depth is not measured: a single top-down SEM image has no height information (that needs a stereo pair at two tilt angles).")
            plt.close("all")
        with t3:
            c1, c2 = st.columns(2)
            cm = tiles_to_frame(cl.astype(np.float32), org[keep], fr.shape)
            leg = [Patch(color=KM_COLORS[k], label=f"C{k}: {CLUSTER_NAMES.get(k, 'unnamed')}") for k in range(KM_K)]
            c1.pyplot(fig_panel(fr, um_px, overlay=np.ma.masked_invalid(cm), kw=dict(cmap=ListedColormap(KM_COLORS), vmin=-0.5, vmax=KM_K - 0.5, alpha=0.5),
                                legend=leg, title="Unsupervised morphology map (ResNet50 features + k-means)"))
            lm = lbp_edge_map(fr, um_px); lo, hi = np.percentile(lm, [2, 98])
            c2.pyplot(fig_panel(fr, um_px, overlay=lm, kw=dict(cmap="RdYlBu_r", vmin=lo, vmax=hi, alpha=0.5), cbar="edge / (edge + flat + non-uniform)",
                                title="LBP texture map"))
            st.table(pd.DataFrame({"Cluster": [f"C{k}: {CLUSTER_NAMES.get(k, 'unnamed')}" for k in range(KM_K)], "Share of tiles (%)": (100 * share).round(1)}))
            st.caption("Clusters are unsupervised and were named by visual inspection of exemplar tiles. They describe appearance; they are not a validated fractographic classification.")
            plt.close("all")
        with t4:
            row = dict(file=f.name, decision="Brittle" if R["logit"] > 0 else "Ductile", p_brittle=round(p_b, 4), brittle_area_pct=round(area_pct, 2),
                       brittle_area_um2=round(area_pct / 100 * used_um2, 1), analysed_area_um2=round(used_um2, 1), tiles_used=int(keep.sum()),
                       tiles_excluded=int((~keep).sum()), um_per_px=round(um_px, 5), tau=round(TAU, 3), **ds,
                       **{f"cluster_{k}_pct": round(100 * share[k], 1) for k in range(KM_K)})
            st.dataframe(pd.DataFrame([row]).T.rename(columns={0: "value"}).astype(str))
            st.download_button("Download summary (CSV)", pd.DataFrame([row]).to_csv(index=False), f"{os.path.splitext(f.name)[0]}_summary.csv", "text/csv", key=f"s{tag}")
            st.download_button("Download per-dimple table (CSV)", tab.to_csv(index=False), f"{os.path.splitext(f.name)[0]}_dimples.csv", "text/csv", key=f"d{tag}")
            if st.button("Prepare the map images (ZIP of PNGs: MIL evidence, dimples, clustering)", key=f"pm{tag}"):
                _td = tempfile.mkdtemp(prefix="m1_"); _stem = safe_name(os.path.splitext(f.name)[0])
                save_maps((_td, [], _stem), fr, org, keep, c, lab, cl, um_px, "Brittle" if R["logit"] > 0 else "Ductile", area_pct, ds["n_dimples"])
                _b = io.BytesIO()
                with zipfile.ZipFile(_b, "w", zipfile.ZIP_STORED) as zo:
                    for dp, _, fns in os.walk(_td):
                        for fn_ in fns: zo.write(os.path.join(dp, fn_), os.path.relpath(os.path.join(dp, fn_), _td))
                shutil.rmtree(_td, ignore_errors=True); st.session_state[f"mz{tag}"] = _b.getvalue()
            if f"mz{tag}" in st.session_state:
                st.download_button("Download the map images (ZIP)", st.session_state[f"mz{tag}"], f"{os.path.splitext(f.name)[0]}_maps.zip", "application/zip", key=f"z{tag}")
            st.dataframe(tab)
        with st.expander("Crop check"):
            raw = read_gray8(path); fig, ax = plt.subplots(figsize=(6, 6 * raw.shape[0] / raw.shape[1]))
            ax.imshow(raw, cmap="gray"); ax.axhline(crop_bottom(raw)[0].shape[0], color="red", lw=2); ax.axis("off")
            st.pyplot(fig); plt.close(fig); st.caption("Raw image; everything below the red line is removed before analysis.")

# =====================================================================================================
#  SECTION 2: compare charging tests from a zip (test folder / region folder / images)
# =====================================================================================================
IMG_EXT_APP = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
REGION_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]

def natkey(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def list_zip(zf):
    # list the images INSIDE the zip without extracting anything. group = the folder an image sits in (region) and that folder's
    # parent (test). Folder NAMES are never interpreted.
    rows = []
    for info in zf.infolist():
        parts = [p for p in info.filename.replace("\\", "/").split("/") if p]
        if info.is_dir() or not parts or parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts): continue
        if not parts[-1].lower().endswith(IMG_EXT_APP): continue
        dirs = parts[:-1]
        test = dirs[-2] if len(dirs) >= 2 else (dirs[0] if len(dirs) == 1 else "(top level)")
        region = dirs[-1] if len(dirs) >= 2 else "(no region folder)"
        rows.append(dict(path=info.filename, file=parts[-1], test=test, region=region, rel="/".join(parts)))
    return pd.DataFrame(rows, columns=["path", "file", "test", "region", "rel"])

def peak_mb():
    import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024      # peak memory of the app so far (Linux: KB)

def safe_name(x): return re.sub(r"[^\w\-. ]", "_", str(x)).strip() or "_"

def save_maps(maps, fr, org, keep, c, lab, cl, um_px, decision, brittle_pct, n_dimples):
    # three PNGs per image, written to disk under <kind>/<test>/<region>/<image>.png (zipped later on request)
    outdir, sub, name = maps; h, w = fr.shape
    def save(kind, fig):
        d = os.path.join(outdir, kind, *sub); os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name + ".png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    # 1) MIL signed evidence map
    cmap_ = tiles_to_frame(c - TAU, org[keep], fr.shape); lim = max(1.0, np.nanpercentile(np.abs(cmap_), 99))
    save("1_MIL_evidence_maps", fig_panel(fr, um_px, overlay=np.ma.masked_invalid(cmap_), kw=dict(cmap="coolwarm", vmin=-lim, vmax=lim, alpha=0.5),
                                          cbar="tile score - tau", title=f"{name} | {decision} | brittle-evidence area {brittle_pct:.0f}%"))
    # 2) dimple outlines
    fig, ax = plt.subplots(figsize=(6, 6 * h / w)); ax.imshow(fr, cmap="gray")
    bnd = find_boundaries(lab, mode="inner"); ov = np.zeros((*bnd.shape, 4)); ov[bnd] = (1, 0.1, 0.1, 0.9); ax.imshow(ov)
    add_scale_bar(ax, um_px, w, h); ax.axis("off"); ax.set_title(f"{name} | {n_dimples} dimples outlined", fontsize=9); fig.tight_layout()
    save("2_dimple_maps", fig)
    # 3) clustering: morphology map (k-means) next to the LBP texture map
    cm = tiles_to_frame(cl.astype(np.float32), org[keep], fr.shape)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6 * h / w))
    ax[0].imshow(fr, cmap="gray"); ax[0].imshow(np.ma.masked_invalid(cm), cmap=ListedColormap(KM_COLORS), vmin=-0.5, vmax=KM_K - 0.5, alpha=0.5)
    ax[0].legend(handles=[Patch(color=KM_COLORS[k], label=f"C{k}: {CLUSTER_NAMES.get(k, 'unnamed')}") for k in range(KM_K)], loc="lower right", fontsize=7)
    ax[0].set_title(f"{name}: morphology map (k-means clusters)", fontsize=9)
    lm = lbp_edge_map(fr, um_px); lo, hi = np.percentile(lm, [2, 98])
    ax[1].imshow(fr, cmap="gray"); im = ax[1].imshow(lm, cmap="RdYlBu_r", vmin=lo, vmax=hi, alpha=0.5)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02); ax[1].set_title("LBP texture map", fontsize=9)
    for a_ in ax: a_.axis("off")
    add_scale_bar(ax[0], um_px, w, h); fig.tight_layout(); save("3_clustering_maps", fig)

def classify_one(path, hfw, fast=False):
    # STEP 1 (the slow part): tiles -> frozen ResNet50 -> Additive MIL head. Only small arrays are kept (about 1 MB per image),
    # so the optional steps later can reuse them without running ResNet again.
    global TTA_LIST
    old = TTA_LIST
    if fast: TTA_LIST = TTA_LIST[:1]
    try: R = score_image(path, hfw)
    finally: TTA_LIST = old
    if R is None: return None
    keep, c = R["keep"], R["c"]; brittle = 100 * float((c > TAU).mean())
    row = dict(decision="Brittle" if R["logit"] > 0 else "Ductile", p_brittle=float(1 / (1 + np.exp(-R["logit"]))),
               brittle_area_pct=brittle, ductile_area_pct=100 - brittle, tiles_used=int(keep.sum()),
               analysed_area_um2=int(keep.sum()) * TILE_PX ** 2 * TARGET_UM_PX ** 2, mode="fast" if fast else "full")
    core = dict(coords=R["coords"], keep=keep, c=c, feats=R["feats"].astype(np.float16), hfw=hfw)
    del R; gc.collect()
    return row, core

def detail_one(fr, core):
    # STEP 2/3: dimples (watershed) and the four morphology features, from the stored tile scores and features
    um_px = TARGET_UM_PX; org, keep, c = core["coords"], core["keep"], core["c"]
    region = ductile_region(dict(coords=org, keep=keep), c, fr.shape)
    lab, tab, dim_area = measure_dimples(fr, um_px, region); ds = dimple_summary(tab, dim_area)
    cl = MK["km"].predict(MK["pca"].transform((core["feats"].astype(np.float32) - MK["mu"]) / MK["sd"]))
    cnt = np.bincount(cl, minlength=KM_K)
    out = dict(n_dimples=ds["n_dimples"], dimple_density_per_100um2=ds["density_per_100um2"], dimple_median_diam_um=ds["diam_median_um"])
    for k in range(KM_K):
        out[f"feat{k}_pct"] = 100 * cnt[k] / cnt.sum(); out[f"feat{k}_um2"] = cnt[k] * TILE_PX ** 2 * um_px ** 2
    return out, lab, cl

def with_frame(zf, p, hfw, fn):
    # pull ONE image out of the zip into a temp file, load it, run fn(frame), delete the temp file
    tmp = None
    try:
        with zf.open(p) as a, tempfile.NamedTemporaryFile(suffix=os.path.splitext(p)[1], delete=False) as t:
            shutil.copyfileobj(a, t); tmp = t.name
        fr, _ = load_frame(tmp, hfw)
        return fn(fr)
    finally:
        if tmp and os.path.exists(tmp): os.remove(tmp)

def fig_png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=200, bbox_inches="tight"); return b.getvalue()

def ductility_scatter(df, tests, regions):
    # one dot per image. y = ductility loss = brittle-evidence area (% of analysed surface). Ring = image classified Brittle.
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(tests) + 2.5), 4.8)); rng = np.random.RandomState(0)
    off = lambda j: (j - (len(regions) - 1) / 2) * 0.24
    ax.axhline(5, color="#999", lw=1, ls=":", zorder=0)
    ax.text(len(tests) - 0.5, 5.6, "≈5% is expected even on a fully ductile image", ha="right", va="bottom", fontsize=7, color="#777")
    for j, r in enumerate(regions):
        sub = df[df.region == r]
        if sub.empty: continue
        col = REGION_COLORS[j % 6]
        xs = np.array([tests.index(t) for t in sub.test]) + off(j) + rng.uniform(-0.06, 0.06, len(sub)); ys = sub.brittle_area_pct.values
        isb = (sub.decision == "Brittle").values
        ax.scatter(xs[~isb], ys[~isb], s=48, color=col, alpha=0.9, edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter(xs[isb], ys[isb], s=78, color=col, edgecolor="black", linewidth=1.8, zorder=4)
        mean = sub.groupby("test").brittle_area_pct.mean(); mean = mean.iloc[sorted(range(len(mean)), key=lambda i: tests.index(mean.index[i]))]
        mx = [tests.index(t) + off(j) for t in mean.index]
        ax.plot(mx, mean.values, color=col, lw=1.3, ls="--", alpha=0.8, zorder=2); ax.scatter(mx, mean.values, marker="_", s=520, color=col, linewidth=3, zorder=5)
    ax.set_xticks(range(len(tests))); ax.set_xticklabels(tests, rotation=20 if len(tests) > 4 else 0, ha="right" if len(tests) > 4 else "center")
    ax.set_xlim(-0.6, len(tests) - 0.4); ax.set_ylim(-3, max(20, float(df.brittle_area_pct.max()) * 1.15))
    ax.set_ylabel("Ductility loss (% brittle-evidence area)"); ax.set_xlabel("charging test")
    ax.set_title("Ductility loss per image (dash = folder mean)", fontsize=10)
    hs = [Line2D([], [], marker="o", ls="", color=REGION_COLORS[j % 6], label=r) for j, r in enumerate(regions)]
    hs.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.8, label="classified Brittle"))
    ax.legend(handles=hs, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); return fig

def dimple_scatter(df, regions):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for j, r in enumerate(regions):
        sub = df[df.region == r]
        if sub.empty: continue
        isb = (sub.decision == "Brittle").values; col = REGION_COLORS[j % 6]
        ax.scatter(sub.dimple_density_per_100um2[~isb], sub.brittle_area_pct[~isb], s=48, color=col, edgecolor="white", linewidth=0.7)
        ax.scatter(sub.dimple_density_per_100um2[isb], sub.brittle_area_pct[isb], s=78, color=col, edgecolor="black", linewidth=1.8)
    ax.set_xlabel("dimple density (per 100 µm² of ductile area)"); ax.set_ylabel("Ductility loss (% brittle-evidence area)")
    ax.set_title("Ductility loss against dimple density, one dot per image", fontsize=10)
    hs = [Line2D([], [], marker="o", ls="", color=REGION_COLORS[j % 6], label=r) for j, r in enumerate(regions)]
    hs.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.8, label="classified Brittle"))
    ax.legend(handles=hs, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); return fig

def batch_section():
    S = st.session_state
    st.subheader("Compare charging tests")
    st.caption("Upload a .zip that holds one folder per charging test, each with an inner and an outer region folder of micrographs. "
               "Folder names do not matter: an image's region is the folder it sits in and its test is that folder's parent. Every image is analysed.")
    z = st.file_uploader("Zip of charging tests", type=["zip"], key="zip_upload")
    if z is None: return
    try: zf = zipfile.ZipFile(z)                       # read straight from the upload: no extra copies of the zip in memory
    except zipfile.BadZipFile:
        st.error("This file is not a valid zip."); return
    z.seek(0); tag = hashlib.md5(z.read(200000) + f"{z.name}{z.size}".encode()).hexdigest(); z.seek(0)
    def reset_all():
        shutil.rmtree(S.get("map_dir") or "", ignore_errors=True)
        S.update(zip_tag=tag, batch_res={}, batch_core={}, batch_fail={}, batch_detail={}, map_done=set(), zip_cache=None, map_dir=tempfile.mkdtemp(prefix="maps_"))
    if S.get("zip_tag") != tag or "batch_core" not in S: reset_all()
    items = list_zip(zf)
    if items.empty: st.error("No images (.tif, .png, .jpg, .bmp) found in the zip."); return

    summ = items.groupby(["test", "region"]).size().rename("images").reset_index()
    summ = summ.iloc[sorted(range(len(summ)), key=lambda i: (natkey(summ.test[i]), natkey(summ.region[i])))]
    st.markdown(f"**Detected {len(items)} images in {summ.test.nunique()} test folder(s) and {len(summ)} region folder(s).** Check this before running:")
    st.dataframe(summ.rename(columns={"test": "Test folder", "region": "Region folder", "images": "Images"}), hide_index=True)

    res, core, fail, detail = S["batch_res"], S["batch_core"], S["batch_fail"], S["batch_detail"]
    for p_ in [p_ for p_ in res if p_ not in core or "ductile_area_pct" not in res[p_]]: res.pop(p_)           # results kept from an older app version
    todo = [p for p in items.path if p not in res and p not in fail]
    fast = st.checkbox("Fast mode: about 4x quicker, but scores differ slightly from the trained model (fine for a first look)", value=False, key="fast")
    st.caption("Step 1 is the slow one: roughly 20-65 s per image on the free cloud CPU (about a quarter of that in fast mode). Keep this tab open while it runs. "
               "Finished images are kept, so pressing the button again continues where it stopped.")
    c_a, c_b = st.columns([3, 1])
    run1 = c_a.button(f"Step 1: classify {len(todo)} image(s) as brittle or ductile" if todo else "Step 1 done: all images classified", disabled=not todo, type="primary", key="run1")
    if c_b.button("Clear results and start over", key="reset_batch"):
        reset_all(); st.rerun()
    if run1:
        bar, status, t0 = st.progress(0.0), st.empty(), time.time()
        for i, p in enumerate(todo):
            eta = (time.time() - t0) / i * (len(todo) - i) if i else None
            status.write(f"Image {i + 1} of {len(todo)}: {items.rel[items.path == p].iloc[0]}" + (f"  | about {eta / 60:.1f} min left" if eta else "") + f"  | peak memory {peak_mb():.0f} MB")
            tmp = None
            try:
                with zf.open(p) as a, tempfile.NamedTemporaryFile(suffix=os.path.splitext(p)[1], delete=False) as t:
                    shutil.copyfileobj(a, t); tmp = t.name           # one image on disk at a time, deleted right after
                r = classify_one(tmp, hfw, fast)
                if r is None: fail[p] = "no usable tiles (blurred or saturated)"
                else: res[p], core[p] = r
            except Exception as e: fail[p] = str(e)
            finally:
                if tmp and os.path.exists(tmp): os.remove(tmp)
            bar.progress((i + 1) / len(todo))
        status.write(f"Done: {len(todo)} image(s) in {(time.time() - t0) / 60:.1f} min.")
    if fail:
        with st.expander(f"{len(fail)} image(s) could not be analysed"):
            st.dataframe(pd.DataFrame([dict(image=items.rel[items.path == p].iloc[0], reason=r) for p, r in fail.items()]), hide_index=True)
    if not res: return

    df = items[items.path.isin(res)].reset_index(drop=True)
    df = pd.concat([df, pd.DataFrame([res[p] for p in df.path])], axis=1)
    tests = sorted(df.test.unique(), key=natkey); regions = sorted(df.region.unique(), key=natkey)
    if len(res) < len(items): st.info(f"Results below cover the {len(res)} image(s) classified so far.")
    nat = lambda d: d.iloc[sorted(range(len(d)), key=lambda i: (natkey(d.test.iloc[i]), natkey(d.region.iloc[i]), natkey(d.file.iloc[i]) if "file" in d else 0))].reset_index(drop=True)

    # =================== RESULTS OF STEP 1 ===================
    st.markdown("## Brittle or ductile?")
    m = st.columns(4)
    m[0].metric("Images classified", len(df)); m[1].metric("Classified Brittle", int((df.decision == "Brittle").sum()))
    m[2].metric("Mean brittle area", f"{df.brittle_area_pct.mean():.1f} %"); m[3].metric("Mean ductile area", f"{df.ductile_area_pct.mean():.1f} %")
    cls = nat(df)[["test", "region", "file", "decision", "p_brittle", "brittle_area_pct", "ductile_area_pct"]]
    cls = cls.rename(columns={"test": "Test", "region": "Region", "file": "Image", "decision": "Decision", "p_brittle": "P(Brittle)",
                              "brittle_area_pct": "Brittle area (%)", "ductile_area_pct": "Ductile area (%)"})
    cls[["P(Brittle)"]] = cls[["P(Brittle)"]].round(3); cls[["Brittle area (%)", "Ductile area (%)"]] = cls[["Brittle area (%)", "Ductile area (%)"]].round(1)
    st.dataframe(cls, hide_index=True)
    fold = df.groupby(["test", "region"]).agg(Images=("file", "count"), Brittle_images=("decision", lambda x: int((x == "Brittle").sum())),
                                              Ductile_images=("decision", lambda x: int((x == "Ductile").sum())), brittle_mean=("brittle_area_pct", "mean"),
                                              brittle_sem=("brittle_area_pct", "sem"), ductile_mean=("ductile_area_pct", "mean")).reset_index()
    fold = nat(fold).rename(columns={"test": "Test", "region": "Region", "Brittle_images": "Images classified Brittle", "Ductile_images": "Images classified Ductile",
                                     "brittle_mean": "Mean brittle area (%)", "brittle_sem": "s.e.m.", "ductile_mean": "Mean ductile area (%)"}).round(1)
    st.markdown("**Per test and region:**"); st.dataframe(fold, hide_index=True)
    st.caption("Decision = the Additive MIL image score (P(Brittle) above 0.5). Brittle area = share of usable 3.1 µm tiles whose evidence score is above the threshold "
               "(set so about 5% of tiles on a truly ductile image count as brittle). Held-out accuracy on the training images was about 85% (AUC 0.87), so treat single images as a first pass.")

    st.markdown("### Ductility loss")
    fig = ductility_scatter(df, tests, regions); st.pyplot(fig); png_loss = fig_png(fig); plt.close(fig)
    st.caption("Ductility loss here means the share of the fracture surface the model reads as brittle (100% minus the ductile area). It is an image-based proxy for hydrogen "
               "embrittlement, not a mechanical ductility measurement. Tests are ordered by name, so the dashed lines only make sense if the names sort in charging order.")
    d1, d2_, d3 = st.columns(3)
    d1.download_button("Per-image table (CSV)", cls.to_csv(index=False), "brittle_ductile_per_image.csv", "text/csv", key="dl_cls")
    d2_.download_button("Per-folder table (CSV)", fold.to_csv(index=False), "brittle_ductile_per_folder.csv", "text/csv", key="dl_fold")
    d3.download_button("Ductility-loss plot (PNG)", png_loss, "ductility_loss.png", "image/png", key="dl_loss")

    # =================== OPTIONAL STEP 2: dimples + features ===================
    st.markdown("---"); st.markdown("## Optional: dimples and feature areas")
    miss = [p for p in res if p not in detail]
    st.write(f"Counts the dimples and measures the area of the four features for each image ({len(res) - len(miss)} of {len(res)} done). "
             "This reuses the Step 1 results, so it only takes a few seconds per image.")
    b1, b2 = st.columns(2)
    go2 = b1.button(f"Measure dimples and feature areas ({len(miss)} image(s))", disabled=not miss, key="run2")
    redo2 = b2.button("Redo all with the current sidebar dimple settings", disabled=len(detail) == 0, key="redo2")
    if go2 or redo2:
        if redo2: detail.clear()
        todo2 = [p for p in res if p not in detail]; bar, t0 = st.progress(0.0), time.time()
        for i, p in enumerate(todo2):
            try: detail[p] = with_frame(zf, p, core[p]["hfw"], lambda fr, p=p: detail_one(fr, core[p])[0])
            except Exception as e: fail[p] = f"dimple step: {e}"
            bar.progress((i + 1) / len(todo2))
        st.success(f"Done in {(time.time() - t0) / 60:.1f} min.")
    if detail:
        d2 = df[df.path.isin(detail)].reset_index(drop=True)
        d2 = pd.concat([d2, pd.DataFrame([detail[p] for p in d2.path])], axis=1)
        # ---- graph 1: dimples per image, per test and region
        st.markdown("### Dimples per image")
        what = st.radio("Show", ["Average number of dimples per image", "Dimple density (per 100 µm² of ductile area)"], horizontal=True, key="dimple_what")
        col = "n_dimples" if what.startswith("Average") else "dimple_density_per_100um2"
        g = d2.groupby(["test", "region"])[col].agg(["mean", "sem", "count"]).fillna({"sem": 0})
        fig, ax = plt.subplots(figsize=(max(6.5, 1.4 * len(tests) * max(1, len(regions) * 0.7)), 4.4)); wd = 0.8 / len(regions)
        for j, r in enumerate(regions):
            xs = [i + (j - (len(regions) - 1) / 2) * wd for i, t in enumerate(tests) if (t, r) in g.index]
            rr = [g.loc[(t, r)] for t in tests if (t, r) in g.index]
            if not rr: continue
            ax.bar(xs, [v["mean"] for v in rr], wd * 0.92, yerr=[v["sem"] for v in rr], capsize=3, color=REGION_COLORS[j % 6], label=r, ecolor="#333")
            for x, v in zip(xs, rr): ax.text(x, v["mean"] + v["sem"], f"n={int(v['count'])}", ha="center", va="bottom", fontsize=7, color="#444")
        ax.set_xticks(range(len(tests))); ax.set_xticklabels(tests, rotation=20 if len(tests) > 4 else 0, ha="right" if len(tests) > 4 else "center")
        ax.set_ylabel("dimples per image" if col == "n_dimples" else "dimples per 100 µm²"); ax.set_xlabel("charging test")
        ax.set_title("Mean ± s.e.m. over the images in each region folder", fontsize=10); ax.legend(title="region", frameon=False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); png1 = fig_png(fig); plt.close(fig)
        st.caption("Dimples are counted inside ductile-evidence tiles only (cleavage facets are never counted), so a region with more brittle area has fewer countable dimples. "
                   "The density option divides by the ductile area to remove that effect.")


        # ---- graph 2: area of the four features per folder
        st.markdown("### Area occupied by the four features")
        unit = st.radio("Show", ["% of analysed area", "µm² per image"], horizontal=True, key="area_unit")
        sfx = "pct" if unit.startswith("%") else "um2"
        folders = [(t, r) for t in tests for r in regions if ((d2.test == t) & (d2.region == r)).any()]
        M = pd.DataFrame({k: [d2[(d2.test == t) & (d2.region == r)][f"feat{k}_{sfx}"].mean() for t, r in folders] for k in range(KM_K)})
        fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(folders) + 2), 4.8)); bottom = np.zeros(len(folders))
        for k in range(KM_K):
            ax.bar(range(len(folders)), M[k], bottom=bottom, color=KM_COLORS[k], label=CLUSTER_NAMES.get(k, f"C{k}"), width=0.75)
            for i, v in enumerate(M[k]):
                if v >= (5 if sfx == "pct" else 0.06 * M.sum(axis=1).max()): ax.text(i, bottom[i] + v / 2, f"{v:.0f}" if sfx == "pct" else f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
            bottom += M[k].values
        ax.set_xticks(range(len(folders))); ax.set_xticklabels([f"{t}\n{r}" for t, r in folders], rotation=35 if len(folders) > 6 else 0, ha="right" if len(folders) > 6 else "center", fontsize=8)
        ax.set_ylabel("% of analysed area" if sfx == "pct" else "µm² per image"); ax.set_title("Mean over the images in each region folder", fontsize=10)
        ax.legend(title="feature", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); png2 = fig_png(fig); plt.close(fig)
        st.caption("Area = share of usable 3.1 µm tiles assigned to each unsupervised morphology cluster, so it is approximate (tile resolution) and the names describe appearance, "
                   "not a validated fractographic classification.")

        st.markdown("### Ductility loss against dimple density")
        fig = dimple_scatter(d2, regions); st.pyplot(fig); png3 = fig_png(fig); plt.close(fig)
        agg = {"images": ("file", "count"), "dimples_mean": ("n_dimples", "mean"), "dimples_sem": ("n_dimples", "sem"),
               "density_per_100um2_mean": ("dimple_density_per_100um2", "mean"), "ductility_loss_pct_mean": ("brittle_area_pct", "mean")}
        for k in range(KM_K):
            nm = CLUSTER_NAMES.get(k, f"C{k}")
            agg[f"{nm}_pct"] = (f"feat{k}_pct", "mean"); agg[f"{nm}_um2_per_image"] = (f"feat{k}_um2", "mean")
        folder_tab = nat(d2.groupby(["test", "region"]).agg(**agg).round(3).reset_index())
        st.markdown("### Tables"); st.dataframe(folder_tab, hide_index=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.download_button("Per-folder summary (CSV)", folder_tab.to_csv(index=False), "folder_summary.csv", "text/csv", key="dl_f2")
        c2.download_button("Per-image results (CSV)", d2.drop(columns=["path"]).round(4).to_csv(index=False), "per_image_results.csv", "text/csv", key="dl_i2")
        c3.download_button("Dimple graph (PNG)", png1, "dimples_comparison.png", "image/png", key="dl_p1")
        c4.download_button("Feature-area graph (PNG)", png2, "feature_area_comparison.png", "image/png", key="dl_p2")
        c5.download_button("Loss vs dimples (PNG)", png3, "ductility_loss_vs_dimples.png", "image/png", key="dl_p3")

    # =================== OPTIONAL STEP 3: map images ===================
    st.markdown("---"); st.markdown("## Optional: map images")
    mdone = S["map_done"]; mmiss = [p for p in res if p not in mdone]
    st.write("**Would you like the map images?** This makes three PNGs per image (MIL evidence map, dimple outlines, clustering maps) and packs them into one ZIP per map type, "
             f"with a folder per test and region ({len(mdone)} of {len(res)} images done). A few seconds per image.")
    if st.button(f"Create map images for {len(mmiss)} image(s)", disabled=not mmiss, key="run3"):
        bar, t0 = st.progress(0.0), time.time(); where = items.set_index("path")
        for i, p in enumerate(mmiss):
            rw = where.loc[p]
            def make(fr, p=p, rw=rw):
                out, lab, cl = detail_one(fr, core[p])
                if p not in detail: detail[p] = out
                save_maps((S["map_dir"], [safe_name(rw.test), safe_name(rw.region)], safe_name(os.path.splitext(rw.file)[0])), fr, core[p]["coords"], core[p]["keep"], core[p]["c"],
                          lab, cl, TARGET_UM_PX, res[p]["decision"], res[p]["brittle_area_pct"], out["n_dimples"])
            try: with_frame(zf, p, core[p]["hfw"], make); mdone.add(p)
            except Exception as e: fail[p] = f"map step: {e}"
            bar.progress((i + 1) / len(mmiss))
        st.success(f"Done in {(time.time() - t0) / 60:.1f} min."); S["zip_cache"] = None
    kinds = [("1_MIL_evidence_maps", "MIL evidence maps"), ("2_dimple_maps", "Dimple maps"), ("3_clustering_maps", "Clustering maps")]
    key = (len(mdone), S["map_dir"])
    if mdone and (S.get("zip_cache") or (None,))[0] != key:
        out_z = {}
        for k, _ in kinds:
            kd = os.path.join(S["map_dir"], k)
            if not os.path.isdir(kd): continue
            b = io.BytesIO()
            with zipfile.ZipFile(b, "w", zipfile.ZIP_STORED) as zo:          # PNGs are already compressed
                for dp, _, fns in os.walk(kd):
                    for f in fns: zo.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), kd))
            out_z[k] = b.getvalue()
        S["zip_cache"] = (key, out_z)
    zc = (S.get("zip_cache") or (None, {}))[1]
    if zc:
        cols = st.columns(3)
        for col_, (k, label) in zip(cols, kinds):
            if k in zc: col_.download_button(f"{label} (ZIP, {len(zc[k]) / 1e6:.0f} MB)", zc[k], f"{k}.zip", "application/zip", key=f"dl_{k}")
        st.caption("Each ZIP holds one folder per test, with a sub-folder per region and one PNG per image. Download them before leaving the page.")

st.title("SEM fracture surface analysis")
st.warning("**Please note:** the model was trained only on SEM micrographs taken at **6000× magnification** (49.7 µm field width, secondary-electron images). "
           "Results for other magnifications, detectors or instruments are not validated and may be unreliable. Use it as a quick first pass alongside the examiner's judgement.")
tab_single, tab_batch = st.tabs(["Single image", "Compare charging tests (zip)"])
with tab_single: single_section()
with tab_batch: batch_section()
