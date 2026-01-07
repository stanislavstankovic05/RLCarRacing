import os
import csv
from typing import List, Dict
import matplotlib.pyplot as plt

def read_csv(path):
    cols = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, [])
                try:
                    cols[k].append(float(v))
                except Exception:
                    pass
    return cols

def moving_average(x, window = 50):
    out = []
    s = 0.0
    q: List[float] = []
    for v in x:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def plot_training_curves(csv_paths, labels, outdir):
    os.makedirs(outdir, exist_ok=True)

    plt.figure()
    for p, lab in zip(csv_paths, labels):
        d = read_csv(p)
        y = d.get("reward", [])
        plt.plot(moving_average(y, 50), label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Reward (MA50)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reward_vs_episodes.png"))
    plt.close()

    plt.figure()
    for p, lab in zip(csv_paths, labels):
        d = read_csv(p)
        fin = d.get("finished", [])
        cum = []
        s = 0.0
        for i, v in enumerate(fin, start=1):
            s += 1.0 if v > 0.5 else 0.0
            cum.append(s / i)
        plt.plot(cum, label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Completion rate (cumulative)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "completion_rate.png"))
    plt.close()
