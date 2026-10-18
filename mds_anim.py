import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation, Slerp
import subprocess, os, sys

NAVY, TEAL, BURG, GREY, BG = "#0b1d39", "#3f827a", "#722428", "#8a96a8", "#ffffff"
plt.rcParams["font.family"] = "Inter"
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Inter", "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold"})
FPS, T = 30, 30
NF = FPS * T
PREVIEW = len(sys.argv) > 1

# ---------------- data: 3 standardised predictors, defaults from a logistic model
rng = np.random.default_rng(11)
n = 36
X = rng.normal(size=(n, 3))
beta_true = np.array([1.6, -1.1, 0.9])
p = 1 / (1 + np.exp(-(X @ beta_true - 0.9)))
y = (rng.uniform(size=n) < p).astype(float)
mu = np.array([1.2, -0.6, 0.8])           # raw (uncentred) location of predictors
Xraw = X + mu

# ---------------- classical MDS from distances only
D = np.sqrt(((Xraw[:, None] - Xraw[None]) ** 2).sum(-1))
Delta = D ** 2
J = np.eye(n) - np.ones((n, n)) / n
G = -0.5 * J @ Delta @ J
w, V = np.linalg.eigh(G)
idx = np.argsort(w)[::-1][:3]
Xt = V[:, idx] * np.sqrt(w[idx])          # principal coordinates

# Procrustes: Xt R = Xraw - mean
Xc = Xraw - Xraw.mean(0)
U, s, Wt = np.linalg.svd(Xt.T @ Xc)
R = U @ Wt
if np.linalg.det(R) < 0:                   # eigenvector sign is arbitrary: flip one axis
    Xt[:, 2] *= -1
    U, s, Wt = np.linalg.svd(Xt.T @ Xc)
    R = U @ Wt
err_proc = np.abs(Xt @ R - Xc).max()

# ---------------- logistic (IWLS) on raw X and DB-GLM-style on principal coords
def irls(Z, y, it=50):
    A = np.c_[np.ones(len(Z)), Z]
    b = np.zeros(A.shape[1])
    for _ in range(it):
        eta = A @ b; m = 1 / (1 + np.exp(-eta)); W = m * (1 - m)
        z = eta + (y - m) / W
        b = np.linalg.solve(A.T @ (W[:, None] * A), A.T @ (W * z))
    return b, 1 / (1 + np.exp(-(A @ b)))
b_log, p_log = irls(Xraw, y)
b_db, p_db = irls(Xt, y)
err_p = np.abs(p_log - p_db).max()
print("procrustes err", err_proc, "max |p_log - p_db|", err_p)

# ---------------- edges for the graph (k nearest)
edges = set()
for i in range(n):
    for j in np.argsort(D[i])[1:4]:
        edges.add(tuple(sorted((i, int(j)))))
edges = sorted(edges)
order = np.argsort(y, kind="stable")       # healthy first, defaults last (matrix ordering)

cols = np.where(y == 1, BURG, TEAL)
cm_d = LinearSegmentedColormap.from_list("d", ["#ffffff", "#c9d2de", NAVY])
cm_g = LinearSegmentedColormap.from_list("g", [BURG, "#ffffff", TEAL])

def ease(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

def seg(f, a, b):
    return ease((f - a) / (b - a))


# timeline (frames, 30 fps, 30 s)
A0, A1 = 0, 120          # euclidean graph
B0, B1 = 130, 240        # delta flies in, matrix fills
C0, C1 = 285, 345        # card flip  Delta -> G
D0, D1 = 400, 500        # rows of G fly out -> principal coordinates
E0, E1 = 545, 690        # rotate + shift onto X
F0 = 715                 # logistic plane

captions = [
    (0, 130, "Euclidean distances between firms", "δᵢⱼ = ‖xᵢ − xⱼ‖ is all we keep"),
    (130, 262, "Distance matrix Δ", "squared distances, one cell per pair"),
    (262, 392, "Double centring", "G = −½ J Δ J: distances become inner products"),
    (392, 530, "Principal coordinates", "eigen-decomposition of G: one point per firm"),
    (530, 702, "Back to the original predictors", "only rotated and shifted"),
    (702, NF + 1, "Same model as logistic regression", "exact because the distance is Euclidean"),
]

FLY = {}
NOUT = NF
lim = np.abs(np.r_[Xraw - Xraw.mean(0), Xt]).max() * 0.92
Z0 = -lim * 0.98

nh = b_log[1:] / np.linalg.norm(b_log[1:])
best = None
for az in np.arange(-180, 180, 1.0):
    e = np.radians(16); a = np.radians(az)
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    sc = abs(abs(v @ nh) - 0.3)
    if best is None or sc < best[0]: best = (sc, az)
AZ_END = best[1]
AZ_START = AZ_END - 110
key = Rotation.from_matrix(np.stack([np.eye(3), R.T]))
sl = Slerp([0, 1], key)

fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
ax = fig.add_axes([0.36, -0.04, 0.64, 1.04], projection="3d")
MPOS = [0.045, 0.22, 0.30, 0.533]
axm = fig.add_axes(MPOS)
axo = fig.add_axes([0, 0, 1, 1], zorder=10); axo.set_axis_off()
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba
rank = np.empty(n, int); rank[order] = np.arange(n)   # matrix row of each firm
nd = int((y == 0).sum())
Mo = (Delta[np.ix_(order, order)] / Delta.max()) ** 0.6
Go = G[np.ix_(order, order)]; Go = Go / np.abs(Go).max()
IMG_D = cm_d(Mo); IMG_G = cm_g((Go + 1) / 2)
I_HL, J_HL = 3, int(np.argmax(D[3]))

def plane_box(b, L):
    pts = []
    c = np.array([-L, L])
    for ax_ in range(3):
        o = [k for k in range(3) if k != ax_]
        for u in c:
            for v in c:
                x = np.zeros(3); x[o[0]] = u; x[o[1]] = v
                t = -(b[0] + b[1:][o[0]] * u + b[1:][o[1]] * v) / b[1:][ax_]
                if -L <= t <= L:
                    x[ax_] = t; pts.append(x)
    if len(pts) < 3: return None
    P = np.array(pts); ctr = P.mean(0)
    nrm = b[1:] / np.linalg.norm(b[1:])
    e1 = np.cross(nrm, [1, 0, 0])
    if np.linalg.norm(e1) < 1e-6: e1 = np.cross(nrm, [0, 1, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
    ang = np.arctan2((P - ctr) @ e2, (P - ctr) @ e1)
    return P[np.argsort(ang)]

def to_fig(pts):
    x2, y2, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())
    return fig.transFigure.inverted().transform(ax.transData.transform(np.c_[x2, y2]))

def floor(P, alpha):
    if alpha <= 0.01: return
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), Z0), s=46, c=GREY, alpha=0.20 * alpha, depthshade=False, edgecolors="none")
    ax.add_collection3d(Line3DCollection([[p, [p[0], p[1], Z0]] for p in P], colors=GREY, linewidths=0.7, alpha=0.13 * alpha))

def pts_rgba(alpha):
    c = np.array([to_rgba(cc) for cc in cols]); c[:, 3] = np.clip(alpha, 0, 1); return c

def draw(f):
    ax.cla(); axm.cla(); axo.cla(); axo.set_axis_off(); axo.set_xlim(0, 1); axo.set_ylim(0, 1)
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0)); a.line.set_color((0, 0, 0, 0))
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=14 + 12 * np.sin(f / NF * np.pi), azim=AZ_START + (AZ_END - AZ_START) * ease(f / NF))
    try: ax.set_box_aspect((1, 1, 1), zoom=1.3)
    except TypeError: ax.set_box_aspect((1, 1, 1))

    # floor grid (depth cue), always present
    fl = seg(f, 0, 30)
    g = np.linspace(-lim * 0.85, lim * 0.85, 7)
    gl = [[[v, g[0], Z0], [v, g[-1], Z0]] for v in g] + [[[g[0], v, Z0], [g[-1], v, Z0]] for v in g]
    ax.add_collection3d(Line3DCollection(gl, colors=GREY, linewidths=0.8, alpha=0.16 * fl))

    # --- stage A/B: euclidean graph
    fade_graph = 1 - seg(f, B0 + 25, B1)
    P = Xraw - Xraw.mean(0)
    if fade_graph > 0.01:
        a_edge = 0.35 * seg(f, A0, A0 + 30) * fade_graph
        ax.add_collection3d(Line3DCollection([[P[i], P[j]] for i, j in edges], colors=NAVY, linewidths=1.2, alpha=a_edge))
        floor(P, fade_graph * seg(f, 0, 30))
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=80, c=cols, alpha=0.95 * fade_graph, depthshade=True, edgecolors="white", linewidths=0.8)
        if f < B0 + 50:
            hl = seg(f, 40, 80) * fade_graph
            ax.plot(*np.c_[P[I_HL], P[J_HL]], color=BURG, lw=2.6, alpha=hl, ls=(0, (4, 3)))
            mid = (P[I_HL] + P[J_HL]) / 2
            ax.text(*mid, "  δᵢⱼ", color=BURG, fontsize=26, style="italic", alpha=hl * (1 - seg(f, B0 - 15, B0 - 5)))
            FLY["src"] = to_fig(mid[None])[0]

    # --- stage D/E/F: recovered coordinates (rows of G fly in)
    if f >= D0:
        tr = seg(f, E0, E1)
        rot = sl([tr]).as_matrix()[0]
        Q = Xt @ rot.T
        arrive = np.array([seg(f, D0 + 1.4 * rank[i] + 30, D0 + 1.4 * rank[i] + 42) for i in range(n)])
        ghost = seg(f, E0 - 10, E0 + 15)
        if ghost > 0:
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=280, facecolors="none", edgecolors=GREY, linewidths=1.4, alpha=0.6 * ghost, depthshade=False)
        floor(Q, arrive.mean())
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=80, c=pts_rgba(arrive), depthshade=False, edgecolors="white", linewidths=0.8)
        if f >= F0:
            pa = seg(f, F0, F0 + 45)
            bc = b_log.copy(); bc[0] = b_log[0] + b_log[1:] @ Xraw.mean(0)
            poly = plane_box(bc, lim * 0.62)
            if poly is not None:
                ax.add_collection3d(Poly3DCollection([poly], facecolors=TEAL, edgecolors=TEAL, linewidths=1.2, alpha=0.16 * pa))
        # flying particles: row of G  ->  point in 3D
        if f < D0 + 1.4 * n + 45:
            dst = to_fig(Q)
            for i in range(n):
                t = seg(f, D0 + 1.4 * rank[i], D0 + 1.4 * rank[i] + 40)
                if 0 < t < 1:
                    r = rank[i]
                    src = np.array([MPOS[0] + MPOS[2], MPOS[1] + MPOS[3] * (1 - (r + 0.5) / n)])
                    p = (1 - t) * src + t * dst[i] + np.array([0, 0.07 * np.sin(np.pi * t)])
                    axo.scatter([p[0]], [p[1]], s=40 + 40 * np.sin(np.pi * t), c=cols[i], alpha=0.9, edgecolors="white", linewidths=0.6)

    # --- matrix panel (with a card-flip from Delta to G)
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values(): sp.set_visible(False)
    m_in = seg(f, B0, B1 - 20)
    m_out = 1 - seg(f, D0 + 1.4 * n + 10, D0 + 1.4 * n + 40)
    k = seg(f, C0, C1)
    sc = abs(np.cos(np.pi * k))
    axm.set_position([MPOS[0] + MPOS[2] * (1 - sc) / 2, MPOS[1] - 0.02 * np.sin(np.pi * k), MPOS[2] * max(sc, 0.01), MPOS[3] * (1 + 0.04 * np.sin(np.pi * k))])
    if m_in > 0 and m_out > 0:
        img = (IMG_D if k < 0.5 else IMG_G).copy()
        rows = int(np.ceil(m_in * n))
        img[rows:, :, 3] = 0
        img[rank[I_HL], rank[J_HL], 3] *= seg(f, B0 + 34, B0 + 42)
        if f >= D0:   # rows dim as they leave
            for i in range(n):
                img[rank[i], :, 3] *= 1 - 0.75 * seg(f, D0 + 1.4 * rank[i], D0 + 1.4 * rank[i] + 12)
        img[..., 3] *= m_out
        axm.imshow(img, interpolation="nearest", aspect="auto")
        axm.add_patch(plt.Rectangle((-0.5, -0.5), n, n, fill=False, ec=NAVY, lw=1.2, alpha=m_in * m_out))
        axm.axhline(nd - 0.5, color=BURG, lw=1.6, alpha=0.8 * m_in * m_out)
        axm.axvline(nd - 0.5, color=BURG, lw=1.6, alpha=0.8 * m_in * m_out)
        axm.set_title("Δ" if k < 0.5 else "G", fontsize=30, color=NAVY, pad=14, alpha=m_in * m_out * (0.3 + 0.7 * sc), fontweight="medium")

    # --- overlay texts
    for t in list(fig.texts): t.remove()
    fig.patches.clear()
    tgt = np.array([MPOS[0] + MPOS[2] * (rank[J_HL] + 0.5) / n, MPOS[1] + MPOS[3] * (1 - (rank[I_HL] + 0.5) / n)])
    if B0 - 15 <= f < B0 + 42 and "src" in FLY:
        pop = seg(f, B0 - 15, B0 + 8); fly = seg(f, B0 + 12, B0 + 40)
        pos = (1 - fly) * FLY["src"] + fly * tgt + np.array([0, 0.05 * np.sin(np.pi * fly)])
        fs = (26 + 34 * pop) * (1 - fly) + 10 * fly
        sq = seg(f, B0 + 2, B0 + 12)
        a0 = min(1, pop * 1.5) * (1 - seg(f, B0 + 34, B0 + 42))
        off = 0.006 * pop * (1 - fly)
        for lab, al in ((r"$\delta_{ij}$", 1 - sq), (r"$\delta_{ij}^{\,2}$", sq)):
            if al < 0.01: continue
            fig.text(pos[0] + off, pos[1] - off, lab, fontsize=fs, color=GREY, ha="center", va="center", alpha=0.35 * a0 * al * (1 - fly))
            fig.text(*pos, lab, fontsize=fs, color=BURG, ha="center", va="center", alpha=a0 * al,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BURG, lw=1.5 * (1 - fly) + 0.5, alpha=a0 * al))
    if B0 + 8 <= f < C0:
        al = seg(f, B0 + 8, B0 + 18) * (1 - seg(f, C0 - 25, C0 - 5))
        w, h = MPOS[2] / n, MPOS[3] / n
        fig.patches.append(plt.Rectangle((tgt[0] - w * 0.9, tgt[1] - h * 0.9), w * 1.8, h * 1.8, transform=fig.transFigure,
                                         fill=False, ec=BURG, lw=2.5, alpha=al, figure=fig))
    for a, b, head, sub in captions:
        if a <= f < b:
            al = min(seg(f, a, a + 15), 1 - seg(f, b - 12, b)) if b < NF else seg(f, a, a + 15)
            fig.text(0.045, 0.89, head, fontsize=50, color=NAVY, fontweight="bold", alpha=al)
            if sub: fig.text(0.045, 0.835, sub, fontsize=30, color=TEAL, alpha=al)
    fig.text(0.045, 0.045, "●", color=TEAL, fontsize=20); fig.text(0.06, 0.045, "healthy firm", color=NAVY, fontsize=18)
    fig.text(0.16, 0.045, "●", color=BURG, fontsize=20); fig.text(0.175, 0.045, "default", color=NAVY, fontsize=18)
    if f >= E0 - 10:
        fig.text(0.25, 0.045, "○", color=GREY, fontsize=20, alpha=seg(f, E0 - 10, E0 + 15))
        fig.text(0.265, 0.045, "original predictors X", color=NAVY, fontsize=18, alpha=seg(f, E0 - 10, E0 + 15))

out = os.path.dirname(os.path.abspath(__file__))
if PREVIEW:
    for f in [int(x) for x in sys.argv[1:]]:
        g = f * NF / NOUT; draw(g - 1); draw(g); fig.savefig(f"{out}/prev_{f:03d}.png", facecolor=BG)
    sys.exit()

os.makedirs(f"{out}/frames", exist_ok=True)
for f in range(NOUT):
    draw(f * NF / NOUT); fig.savefig(f"{out}/frames/f{f:03d}.png", facecolor=BG)
subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", f"{out}/frames/f%03d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                f"{out}/distances_to_logistic.mp4"], check=True, capture_output=True)
print("done")
