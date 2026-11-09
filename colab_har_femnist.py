# ===== Colab: PSO-FMTL vs FedAvg (single task) vs Online FMTL on HAR and FEMNIST =====
# Runtime: CPU is fine. Paste into one cell (or save as .py and run). ~10-30 min for 10 seeds.
# %pip install -q datasets pillow
import os, io, zipfile, urllib.request, subprocess, sys
try:
    import datasets  # noqa
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "pillow"])

"""(core) Real-data check: FedAvg (single task), Online FMTL (full sharing + Omega), PSO-FMTL (partial sharing + Omega).
Clients = subjects (HAR) or writers (FEMNIST). Multi-output RFF least squares, online minibatch LMS.

Usage:
  python real_data_fmtl.py har    /path/to/"UCI HAR Dataset"
  python real_data_fmtl.py femnist /path/to/leaf/data/femnist/data/train   (LEAF json files)
  python real_data_fmtl.py demo   (synthetic stand-in, only to test the code)
Env: SEEDS=10 ROUNDS=300 D=300 MU=0.5 BATCH=8 FRAC=0.175 KMAX=50
"""
import sys, os, json, glob, numpy as np

def load_har(root):
    X, y, s = [], [], []
    for part in ("train", "test"):
        p = os.path.join(root, part)
        X.append(np.loadtxt(os.path.join(p, f"X_{part}.txt")))
        y.append(np.loadtxt(os.path.join(p, f"y_{part}.txt")).astype(int) - 1)
        s.append(np.loadtxt(os.path.join(p, f"subject_{part}.txt")).astype(int))
    X, y, s = np.vstack(X), np.concatenate(y), np.concatenate(s)
    return [(X[s == u], y[s == u]) for u in np.unique(s)], 6

def load_femnist(folder, kmax=50, min_n=150):
    cl = []
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(f))
        for u in d["users"]:
            x = np.array(d["user_data"][u]["x"], float); yy = np.array(d["user_data"][u]["y"], int)
            if len(yy) >= min_n: cl.append((x, yy))
    return cl[:kmax], 62

def load_demo(K=20, C=5, d=40, n=240, seed=0):
    r = np.random.default_rng(seed); cen = r.normal(size=(C, d)); cl = []
    for k in range(K):
        shift = r.normal(scale=0.7, size=d); perm_noise = r.normal(scale=0.3, size=(C, d))
        y = r.integers(0, C, n); cl.append(((cen + perm_noise * (k % 3))[y] + shift + r.normal(size=(n, d)), y))
    return cl, C

def omega(Wf, theta=1.0):
    A = Wf.T @ Wf; A = A + theta * np.mean(np.diag(A)) * np.eye(len(A))
    ev, U = np.linalg.eigh(A); S = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T
    S = S / np.trace(S); np.fill_diagonal(S, 0); S = np.clip(S, 0, None)
    return S / np.maximum(S.sum(1, keepdims=True), 1e-12)

def run(clients, C, method, M, seed, rounds, D, mu, B, lam=0.05, TO=20, sigma=None):
    r = np.random.default_rng(seed); K = len(clients)
    d = clients[0][0].shape[1]
    allx = np.vstack([c[0] for c in clients]); sub = allx[r.choice(len(allx), min(500, len(allx)), replace=False)]
    mu_, sd_ = allx.mean(0), allx.std(0) + 1e-8
    dist = np.sqrt(((sub[:200, None] - sub[None, :200]) ** 2).sum(-1)); g = 1.0 / np.median(dist[dist > 0]) ** 2
    Wr = r.normal(scale=np.sqrt(2 * g), size=(d, D)); br = r.uniform(0, 2 * np.pi, D)
    phi = lambda X: np.sqrt(2.0 / D) * np.cos(X @ Wr + br)  # raw features (median-heuristic scale)
    data = []
    for X, y in clients:
        p = r.permutation(len(y)); cut = int(0.7 * len(y)); tr, te = p[:cut], p[cut:]
        data.append((phi(X[tr]), np.eye(C)[y[tr]], phi(X[te]), y[te]))
    Wg = np.zeros((D, C)); Wl = [np.zeros((D, C)) for _ in range(K)]
    Om = np.ones((K, K)) / max(K - 1, 1); np.fill_diagonal(Om, 0)
    sig = sigma or M
    for m in range(rounds):
        csum = np.zeros((D, C)); ccnt = np.zeros(D); new = []
        for k in range(K):
            Z, Y = data[k][0], data[k][1]
            idx = (np.arange(B) + m * B) % len(Y); z, y = Z[idx], Y[idx]
            if method == "fedavg": view = Wg
            else:
                if M >= D: mask = np.ones(D)
                else:
                    pos = (m * sig + k * (D // K)) % D; mask = np.zeros(D); mask[(pos + np.arange(M)) % D] = 1
                view = mask[:, None] * Wg + (1 - mask[:, None]) * Wl[k]
            grad = -z.T @ (y - z @ view) / B
            if method != "fedavg":
                nb = sum(Om[k, j] * Wl[j] for j in range(K) if Om[k, j] > 0)
                grad = grad + lam * (view - nb)
            wn = view - mu * grad; new.append(wn)
            if method == "fedavg": csum += wn; ccnt += 1
            else: csum += mask[:, None] * wn; ccnt += mask
        Wl = new
        avg = np.where(ccnt[:, None] > 0, csum / np.maximum(ccnt, 1)[:, None], Wg); Wg = avg
        if method != "fedavg" and m % TO == 0:
            Om = omega(np.stack([w.ravel() for w in Wl], 1))
    acc = []
    for k in range(K):
        W = Wg if method == "fedavg" else (Wl[k] if M >= D else (mask_for(k, rounds, M, D, K, sig) [:, None] * Wg + (1 - mask_for(k, rounds, M, D, K, sig)[:, None]) * Wl[k]))
        acc.append((np.argmax(data[k][2] @ W, 1) == data[k][3]).mean())
    return float(np.mean(acc))

def mask_for(k, m, M, D, K, sig):
    pos = (m * sig + k * (D // K)) % D; mk = np.zeros(D); mk[(pos + np.arange(M)) % D] = 1; return mk



# ---------------------------- data download -----------------------------
def get_har():
    root = "UCI HAR Dataset"
    if not os.path.isdir(root):
        url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
        urllib.request.urlretrieve(url, "har_outer.zip")
        with zipfile.ZipFile("har_outer.zip") as z: z.extractall(".")
        if os.path.exists("UCI HAR Dataset.zip"):
            with zipfile.ZipFile("UCI HAR Dataset.zip") as z: z.extractall(".")
    return load_har(root)

def get_femnist(kmax=50, min_n=150):
    from datasets import load_dataset
    import numpy as np
    ds = load_dataset("flwrlabs/femnist", split="train")        # columns: image, writer_id, character, ...
    wid = np.array(ds["writer_id"]); u, cnt = np.unique(wid, return_counts=True)
    keep = u[cnt >= min_n][:kmax]
    chars = sorted(set(ds["character"])); cmap = {c: i for i, c in enumerate(chars)}
    clients = []
    for w in keep:
        sub = ds.select(np.where(wid == w)[0].tolist())
        X = np.stack([np.asarray(im.convert("L"), float).ravel() / 255.0 for im in sub["image"]])
        clients.append((X, np.array([cmap[c] for c in sub["character"]])))
    return clients, len(chars)

# ------------------------------- experiment -----------------------------
def experiment(name, clients, C, seeds=10, rounds=300, D=300, B=8, frac=0.175, mus=(0.25, 0.5, 1.0)):
    M = int(round(frac * D)); rows = []
    cfg = {"FedAvg (single task)": ("fedavg", D), "Online FMTL": ("fmtl", D), "PSO-FMTL": ("fmtl", M)}
    print(f"\n== {name}: K={len(clients)} clients, C={C} classes, D={D}, M={M}, {seeds} seeds, {rounds} rounds ==")
    for mu in mus:
        for n, (a, mm) in cfg.items():
            v = np.array([run(clients, C, a, mm, s, rounds, D, mu, B) for s in range(seeds)])
            params = 2 * C * (D if a == "fedavg" or mm >= D else mm)
            rows.append((mu, n, 100 * v.mean(), 100 * v.std(), params))
            print(f"mu={mu:<5} {n:22s} acc {100*v.mean():6.2f} +- {100*v.std():4.2f} %   params/client/round {params}")
    return rows

if __name__ == "__main__":      # in Colab just run the lines below in cells
    har, C1 = get_har()
    experiment("HAR (subject = task)", har, C1)
    fem, C2 = get_femnist()
    experiment("FEMNIST (writer = task)", fem, C2)
