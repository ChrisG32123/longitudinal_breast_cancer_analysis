#!/usr/bin/env python3
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import torch.optim as optim
import matplotlib.pyplot as plt


# ─── 1. UTILS ────────────────────────────────────────────────────────────────────

def load_delta_times(data_dir: Path, patient_ids: list[int]) -> dict[int, np.ndarray]:
    """
    For each pid, look for a folder ending with that pid,
    parse subfolder names as MM-DD-YYYY, compute days since T0.
    Returns pid -> 1D int array of length T.
    """
    missing = []
    delta = {}
    for pid in patient_ids:
        # find ANY subfolder whose name ends with str(pid)
        for sub in data_dir.iterdir():
            if sub.is_dir() and sub.name.endswith(str(pid)):
                path = sub
                break
        else:
            missing.append(pid)
            continue

        dates = sorted(d.name for d in path.iterdir() if d.is_dir())
        dates = np.array([datetime.strptime(ds, "%m-%d-%Y") for ds in dates])
        diffs = (dates - dates[0]).astype("timedelta64[D]").astype(int)
        delta[pid] = diffs

    if missing:
        print(f"Warning: no data for {len(missing)} patients → dropped: {missing}")
    return delta


# ─── 2. DATASET & COLLATE ────────────────────────────────────────────────────────

class ISPY2Dataset(Dataset):
    def __init__(
        self,
        rads_list: list[pd.DataFrame],
        clin_df: pd.DataFrame,
        delta_times: dict[int, np.ndarray],
        rad_cols: list[str],
        clin_cols: list[str],
        target_col: str = "pCR",
    ):
        self.rads_list = rads_list
        self.clin_df   = clin_df.set_index("Patient_ID")
        self.delta     = delta_times
        self.rad_cols  = rad_cols
        self.clin_cols = clin_cols
        self.target    = target_col

        # only keep pids present in radiomics, clinical, AND have correct T
        rad_ids  = {int(x) for x in rads_list[0]["Patient_ID"].unique()}
        clin_ids = set(self.clin_df.index)
        time_ids = {pid for pid, arr in delta_times.items() if len(arr) == len(rads_list)}

        self.patients = sorted(rad_ids & clin_ids & time_ids)
        if len(self.patients) < len(rad_ids & clin_ids):
            dropped = sorted((rad_ids & clin_ids) - set(self.patients))
            print(f"Dropped {len(dropped)} with mismatched timepoints: {dropped}")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx:int):
        pid = self.patients[idx]

        # Radiomics: stack rows per timepoint → [T, F]
        rad_tensors = []
        for df in self.rads_list:
            row = df[df["Patient_ID"] == pid]
            vals = row[self.rad_cols].values.astype(np.float32)
            assert vals.shape[0] == 1, f"Expected one row for pid {pid}"
            rad_tensors.append(torch.from_numpy(vals[0]))
        rad = torch.stack(rad_tensors, dim=0)

        # Clinical → [C]
        clin_vals = self.clin_df.loc[pid, self.clin_cols].values.astype(np.float32)
        clin = torch.from_numpy(clin_vals)

        # Delta times → [T]
        delta = torch.from_numpy(self.delta[pid].astype(np.float32))

        # Label scalar
        lbl = float(self.clin_df.loc[pid, self.target])
        label = torch.tensor(lbl, dtype=torch.float32)

        return rad, clin, delta, label


def collate_fn(batch):
    rad, clin, delta, label = zip(*batch)
    return (
        torch.stack(rad, dim=0),      # [B, T, F]
        torch.stack(clin, dim=0),     # [B, C]
        torch.stack(delta, dim=0),    # [B, T]
        torch.stack(label).unsqueeze(1)   # [B,1]
    )


# ─── 3. MODEL ───────────────────────────────────────────────────────────────────

class PickTimeModel(nn.Module):
    def __init__(self, T:int, F:int, C:int, H:int, velocity_layer:int=-1):
        """
        T = # timepoints, F = # radiomics features,
        C = # clinical features, H = hidden size.
        velocity_layer = -1 (none) or 0 (at input).
        """
        super().__init__()
        self.velocity_layer = velocity_layer
        # Radiomics path
        self.fc1 = nn.Linear(T*F, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        # Clinical path
        self.c1  = nn.Linear(C, H)
        self.c2  = nn.Linear(H, H)
        # Combine → 1 logit
        self.out = nn.Linear(2*H, 1)

    def calculate_velocity(self, x:torch.Tensor, delta:torch.Tensor):
        # x:[B,T,F], delta:[B,T] → output same shape
        diffs = x[:,1:,:] - x[:,:-1,:]
        denom = delta[:,1:].unsqueeze(-1) + 1e-6
        vel = diffs / denom
        return torch.cat([x[:,:1,:], vel], dim=1)

    def forward(self, rad, clin, delta):
        # optionally inject velocity
        if self.velocity_layer == 0:
            rad = self.calculate_velocity(rad, delta)

        B,T,F = rad.shape
        x = rad.view(B, T*F)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        c = torch.relu(self.c1(clin))
        c = torch.relu(self.c2(c))

        h = torch.cat([x,c], dim=1)
        return self.out(h)  # raw logit


# ─── 4. TRAIN & EVAL ───────────────────────────────────────────────────────────

def train_epoch(model : PickTimeModel, loader : DataLoader[ISPY2Dataset], crit, opt, device):
    model.train()
    running = 0.0
    for rad, clin, delta, labels in loader:
        rad, clin, delta, labels = [t.to(device) for t in (rad, clin, delta, labels)]
        opt.zero_grad()
        logits = model(rad, clin, delta)
        loss   = crit(logits, labels)
        loss.backward()
        opt.step()
        running += loss.item() * rad.size(0)
    return running / len(loader.dataset)

def eval_model(model : PickTimeModel, loader : DataLoader[ISPY2Dataset], device):
    model.eval()
    P, A, Y = [], [], []
    with torch.no_grad():
        for rad, clin, delta, labels in loader:
            rad, clin, delta, labels = [t.to(device) for t in (rad, clin, delta, labels)]
            logits = model(rad, clin, delta)
            probs  = torch.sigmoid(logits).squeeze(1)
            preds  = (probs > 0.5).float()
            P.append(preds.cpu())
            A.append(probs.cpu())
            Y.append(labels.squeeze(1).cpu())
    P = torch.cat(P); A = torch.cat(A); Y = torch.cat(Y)
    return {
        "accuracy": accuracy_score(Y, P),
        "auc": roc_auc_score(Y, A),
        "report": classification_report(Y, P, digits=4),
        "cm": confusion_matrix(Y, P),
    }


# ─── 5. MAIN ────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42); np.random.seed(42)

    # Paths
    RAD_PATH  = Path("/mnt/home/gerlac37/ISPY2/data/Multi-feature-MRI-NACT-Data.xlsx")
    CLIN_PATH = Path("/mnt/home/gerlac37/ISPY2/data/clinical_processed.csv")
    DATA_DIR  = Path("/mnt/scratch/gerlac37/ISPY2/data")

    # Load clinical
    clin = pd.read_csv(CLIN_PATH)
    clin_cols = [c for c in clin.columns if c not in ("Patient_ID", "pCR")]

    # Load radiomics (if you truly have per-timepoint data, parse by a Time column)
    rad_df = pd.read_excel(RAD_PATH)
    rad_df.rename(columns={"CLINICAL-TRIAL-SUBJECT-ID":"Patient_ID"}, inplace=True)
    rad_cols = [c for c in rad_df.columns if c.startswith(("VOLUME_","SPHERICITY_","LD_","BPE_"))]
    # here we *fake* 4 timepoints by repeating the same DF; replace with real splits
    rads_list = [rad_df[["Patient_ID"]+rad_cols] for _ in range(4)]

    # Build list of patients *with* correct delta times
    all_pids = sorted(set(clin["Patient_ID"]).intersection(rad_df["Patient_ID"]))
    deltas   = load_delta_times(DATA_DIR, all_pids)
    # keep only those with exactly 4 timepoints
    valid_pids = [pid for pid, arr in deltas.items() if len(arr)==len(rads_list)]
    print(f"Using {len(valid_pids)} patients (dropped {len(all_pids)-len(valid_pids)})")

    # Train/test split
    y = clin.set_index("Patient_ID").loc[valid_pids,"pCR"]
    train_pids, test_pids = train_test_split(valid_pids, test_size=0.2,
                                             stratify=y, random_state=42)

    # Filter DataFrames
    def flt(df, p): return df[df["Patient_ID"].isin(p)].copy()
    train_r, test_r = [ [flt(df, train_pids) for df in rads_list],
                        [flt(df, test_pids ) for df in rads_list] ]
    train_c, test_c = clin[clin["Patient_ID"].isin(train_pids)].copy(), clin[clin["Patient_ID"].isin(test_pids)].copy()

    # Scale radiomics
    scaler = StandardScaler()
    train_stack = np.vstack([df[rad_cols].values for df in train_r])
    scaler.fit(train_stack)
    for lst in (train_r, test_r):
        for df in lst:
            df[rad_cols] = scaler.transform(df[rad_cols].values)

    # Datasets & loaders
    train_ds = ISPY2Dataset(train_r, train_c, {pid:deltas[pid] for pid in train_pids}, rad_cols, clin_cols)
    test_ds  = ISPY2Dataset(test_r,  test_c,  {pid:deltas[pid] for pid in test_pids},  rad_cols, clin_cols)
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True,  collate_fn=collate_fn)
    test_ld  = DataLoader(test_ds,  batch_size=16, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate models
    models = {
        "NoVelocity":  PickTimeModel(4, len(rad_cols), len(clin_cols), 64, velocity_layer=-1),
        "Vel@Input":   PickTimeModel(4, len(rad_cols), len(clin_cols), 64, velocity_layer=0),
        "Vel@Layer1":   PickTimeModel(4, len(rad_cols), len(clin_cols), 64, velocity_layer=0),
        "Vel@Layer2":   PickTimeModel(4, len(rad_cols), len(clin_cols), 64, velocity_layer=0),
        "Vel@Layer3":   PickTimeModel(4, len(rad_cols), len(clin_cols), 64, velocity_layer=0),
        "Vel@OutputLayer":   PickTimeModel(4, len(rad_cols), len(clin_cols), 64, velocity_layer=0),
    }

    criterion = nn.BCEWithLogitsLoss()
    results = {}

    # Train & evaluate
    for name, model in models.items():
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(20):
            loss = train_epoch(model, train_ld, criterion, opt, device)
        metrics = eval_model(model, test_ld, device)
        print(f"\n{name}: Acc={metrics['accuracy']:.4f}  AUC={metrics['auc']:.4f}")
        print(metrics["report"])
        results[name] = metrics

    # Plot comparison
    plt.figure(figsize=(5,3))
    plt.bar(results.keys(), [r["auc"] for r in results.values()])
    plt.ylabel("ROC AUC")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
