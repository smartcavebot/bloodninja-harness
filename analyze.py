"""Paired, task-cluster bootstrap analysis of role-queue experiment scores.
Usage: python analyze.py results.csv --reps 4000
Columns required: task_id, arm, score. Optional: trial, cost_usd, total_tokens, latency_seconds.
"""
import argparse, csv, random, statistics
from collections import defaultdict

ARMS = ("single", "multi", "queue")

def read(path):
    rows = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            arm = row["arm"].strip().lower()
            if arm not in ARMS:
                continue
            tid = row["task_id"].strip()
            value = float(row["score"])
            if not (0 <= value <= 1):
                raise ValueError(f"score outside [0,1]: {value}")
            rows[tid][arm].append(value)
    return rows

def summarize(rows, reps=4000, seed=20261019):
    paired = {tid: {a: statistics.mean(arms[a]) for a in ARMS}
              for tid, arms in rows.items() if all(arms.get(a) for a in ARMS)}
    ids = sorted(paired)
    if not ids:
        raise ValueError("No task_id with all three arms; cannot do paired analysis")
    def calc(sample):
        mean = {a: statistics.mean(paired[tid][a] for tid in sample) for a in ARMS}
        numerator = mean["queue"] - mean["single"]
        denominator = mean["multi"] - mean["single"]
        return {**mean,
                "queue_minus_multi": mean["queue"] - mean["multi"],
                "captured_gain": numerator/denominator if denominator > 1e-9 else float("nan")}
    point = calc(ids)
    rng = random.Random(seed)
    draws = [calc([rng.choice(ids) for _ in ids]) for _ in range(reps)]
    result = {}
    for metric, estimate in point.items():
        vals = sorted(d[metric] for d in draws if d[metric] == d[metric])
        interval = (vals[int(.025 * len(vals))], vals[min(len(vals)-1,int(.975 * len(vals)))]) if vals else (float('nan'), float('nan'))
        result[metric] = {"estimate": estimate, "ci95": interval}
    return len(ids), result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_file")
    p.add_argument("--reps", type=int, default=4000)
    args = p.parse_args()
    n, metrics = summarize(read(args.csv_file), args.reps)
    print(f"Paired task clusters: {n}")
    for name, values in metrics.items():
        v, (lo, hi) = values["estimate"], values["ci95"]
        print(f"{name:20s} {v:8.4f} 95% task-bootstrap CI [{lo:8.4f}, {hi:8.4f}]")
    print("Warning: captured_gain unstable when original multi-minus-single effect is small or crosses zero.")
