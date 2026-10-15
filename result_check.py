import re
import csv
import statistics
from pathlib import Path
from collections import defaultdict


# =========================
# CONFIG
# =========================

INPUT_FILE = r"C:\Users\hongo\Desktop\result.txt"
OUTPUT_DIR = r"C:\Users\hongo"

SEEDS = [42, 123, 999]

STRATEGIES = ["Baseline", "LA", "SGA"]
SKEWS = [10, 5, 2]
EVAL_ROUNDS = [0, 10, 25, 50, 75, 99]


# =========================
# READ TEXT FILE
# =========================

input_path = Path(INPUT_FILE)

if not input_path.exists():
    raise FileNotFoundError(f"Could not find input file: {input_path}")

text = input_path.read_text(encoding="utf-8", errors="replace")


# =========================
# PARSE RESULT TABLE
# =========================

row_pattern = re.compile(
    r"^(Baseline|LA|SGA)\s+"
    r"(\d+)\s+"
    r"(\d+)\s+"
    r"([0-9]*\.[0-9]+)\s+"
    r"([0-9]*\.[0-9]+)\s+"
    r"([0-9]*\.[0-9]+)\s+"
    r"([0-9]*\.[0-9]+)\s*$",
    re.MULTILINE
)

rows = []

for match in row_pattern.finditer(text):
    strategy, skew, round_, global_acc, global_loss, avg_client_acc, avg_client_loss = match.groups()

    rows.append({
        "strategy": strategy,
        "skew": int(skew),
        "classes_per_client": int(skew),
        "round": int(round_),
        "global_acc": float(global_acc),
        "global_loss": float(global_loss),
        "avg_client_acc": float(avg_client_acc),
        "avg_client_loss": float(avg_client_loss),
    })


# =========================
# VALIDATE ROW COUNT
# =========================

rows_per_seed = len(STRATEGIES) * len(SKEWS) * len(EVAL_ROUNDS)
expected_rows = rows_per_seed * len(SEEDS)

if len(rows) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} result rows, but found {len(rows)}.\n"
        f"Make sure {INPUT_FILE} contains the full result table for all seeds."
    )


# =========================
# ADD SEED + DERIVED COLUMNS
# =========================

for i, row in enumerate(rows):
    seed_index = i // rows_per_seed
    row["seed"] = SEEDS[seed_index]

    row["client_global_acc_gap"] = row["avg_client_acc"] - row["global_acc"]
    row["client_global_loss_gap"] = row["avg_client_loss"] - row["global_loss"]


raw_columns = [
    "seed",
    "strategy",
    "skew",
    "classes_per_client",
    "round",
    "global_acc",
    "global_loss",
    "avg_client_acc",
    "avg_client_loss",
    "client_global_acc_gap",
    "client_global_loss_gap",
]


# =========================
# GROUP RESULTS
# =========================

groups = defaultdict(list)

for row in rows:
    key = (row["seed"], row["strategy"], row["skew"])
    groups[key].append(row)


# =========================
# COMPUTE STABILITY PER SEED
# =========================

stability_rows = []
enhanced_rows = []

for (seed, strategy, skew), group in groups.items():
    group = sorted(group, key=lambda r: r["round"])

    losses = [r["global_loss"] for r in group]
    accs = [r["global_acc"] for r in group]
    client_accs = [r["avg_client_acc"] for r in group]
    client_losses = [r["avg_client_loss"] for r in group]
    acc_gaps = [r["client_global_acc_gap"] for r in group]

    loss_deltas = [
        group[i]["global_loss"] - group[i - 1]["global_loss"]
        for i in range(1, len(group))
    ]

    acc_deltas = [
        group[i]["global_acc"] - group[i - 1]["global_acc"]
        for i in range(1, len(group))
    ]

    initial = group[0]
    final = group[-1]

    loss_variance = statistics.pvariance(losses)
    loss_std = statistics.pstdev(losses)
    loss_range = max(losses) - min(losses)
    mean_abs_loss_delta = statistics.mean(abs(x) for x in loss_deltas)

    acc_variance = statistics.pvariance(accs)
    acc_std = statistics.pstdev(accs)
    acc_range = max(accs) - min(accs)
    mean_abs_acc_delta = statistics.mean(abs(x) for x in acc_deltas)

    stability_row = {
        "seed": seed,
        "strategy": strategy,
        "skew": skew,
        "classes_per_client": skew,
        "rounds_count": len(group),

        "loss_variance": loss_variance,
        "loss_std": loss_std,
        "loss_range": loss_range,
        "mean_abs_loss_delta_between_eval_rounds": mean_abs_loss_delta,

        "acc_variance": acc_variance,
        "acc_std": acc_std,
        "acc_range": acc_range,
        "mean_abs_acc_delta_between_eval_rounds": mean_abs_acc_delta,

        "initial_global_acc": initial["global_acc"],
        "final_global_acc": final["global_acc"],
        "global_acc_change": final["global_acc"] - initial["global_acc"],

        "initial_global_loss": initial["global_loss"],
        "final_global_loss": final["global_loss"],
        "global_loss_change": final["global_loss"] - initial["global_loss"],

        "final_avg_client_acc": final["avg_client_acc"],
        "final_avg_client_loss": final["avg_client_loss"],

        "final_client_global_acc_gap": final["client_global_acc_gap"],
        "mean_client_global_acc_gap": statistics.mean(acc_gaps),

        "mean_global_acc": statistics.mean(accs),
        "mean_global_loss": statistics.mean(losses),
        "mean_avg_client_acc": statistics.mean(client_accs),
        "mean_avg_client_loss": statistics.mean(client_losses),
    }

    stability_rows.append(stability_row)

    previous = None

    for row in group:
        enhanced = dict(row)

        if previous is None:
            enhanced["global_acc_delta_from_prev_eval"] = ""
            enhanced["global_loss_delta_from_prev_eval"] = ""
        else:
            enhanced["global_acc_delta_from_prev_eval"] = row["global_acc"] - previous["global_acc"]
            enhanced["global_loss_delta_from_prev_eval"] = row["global_loss"] - previous["global_loss"]

        enhanced["run_loss_std"] = loss_std
        enhanced["run_loss_variance"] = loss_variance
        enhanced["run_loss_range"] = loss_range
        enhanced["run_mean_abs_loss_delta_between_eval_rounds"] = mean_abs_loss_delta

        enhanced["run_acc_std"] = acc_std
        enhanced["run_acc_variance"] = acc_variance
        enhanced["run_acc_range"] = acc_range
        enhanced["run_mean_abs_acc_delta_between_eval_rounds"] = mean_abs_acc_delta

        enhanced["run_final_global_acc"] = final["global_acc"]
        enhanced["run_final_global_loss"] = final["global_loss"]
        enhanced["run_final_client_global_acc_gap"] = final["client_global_acc_gap"]

        enhanced_rows.append(enhanced)
        previous = row


# =========================
# SUMMARY ACROSS SEEDS
# =========================

summary_groups = defaultdict(list)

for row in stability_rows:
    key = (row["strategy"], row["skew"])
    summary_groups[key].append(row)


summary_metrics = [
    "loss_std",
    "loss_variance",
    "loss_range",
    "mean_abs_loss_delta_between_eval_rounds",

    "acc_std",
    "acc_variance",
    "acc_range",
    "mean_abs_acc_delta_between_eval_rounds",

    "final_global_acc",
    "final_global_loss",
    "final_avg_client_acc",
    "final_avg_client_loss",

    "final_client_global_acc_gap",
    "mean_client_global_acc_gap",

    "mean_global_acc",
    "mean_global_loss",
    "mean_avg_client_acc",
    "mean_avg_client_loss",
]

summary_rows = []

for (strategy, skew), group in sorted(summary_groups.items(), key=lambda x: (x[0][0], x[0][1])):
    summary = {
        "strategy": strategy,
        "skew": skew,
        "classes_per_client": skew,
        "seed_count": len(group),
    }

    for metric in summary_metrics:
        values = [row[metric] for row in group]

        summary[f"{metric}_mean"] = statistics.mean(values)

        if len(values) > 1:
            summary[f"{metric}_std_across_seeds"] = statistics.stdev(values)
        else:
            summary[f"{metric}_std_across_seeds"] = 0.0

    summary_rows.append(summary)


# =========================
# WRITE CSV FILES
# =========================

def write_csv(filename, data, columns):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for row in data:
            writer.writerow({col: row.get(col, "") for col in columns})

    print(f"Saved: {path.resolve()}")


stability_columns = [
    "seed",
    "strategy",
    "skew",
    "classes_per_client",
    "rounds_count",

    "loss_variance",
    "loss_std",
    "loss_range",
    "mean_abs_loss_delta_between_eval_rounds",

    "acc_variance",
    "acc_std",
    "acc_range",
    "mean_abs_acc_delta_between_eval_rounds",

    "initial_global_acc",
    "final_global_acc",
    "global_acc_change",

    "initial_global_loss",
    "final_global_loss",
    "global_loss_change",

    "final_avg_client_acc",
    "final_avg_client_loss",

    "final_client_global_acc_gap",
    "mean_client_global_acc_gap",

    "mean_global_acc",
    "mean_global_loss",
    "mean_avg_client_acc",
    "mean_avg_client_loss",
]


enhanced_columns = [
    "seed",
    "strategy",
    "skew",
    "classes_per_client",
    "round",

    "global_acc",
    "global_loss",
    "avg_client_acc",
    "avg_client_loss",

    "client_global_acc_gap",
    "client_global_loss_gap",

    "global_acc_delta_from_prev_eval",
    "global_loss_delta_from_prev_eval",

    "run_loss_std",
    "run_loss_variance",
    "run_loss_range",
    "run_mean_abs_loss_delta_between_eval_rounds",

    "run_acc_std",
    "run_acc_variance",
    "run_acc_range",
    "run_mean_abs_acc_delta_between_eval_rounds",

    "run_final_global_acc",
    "run_final_global_loss",
    "run_final_client_global_acc_gap",
]


summary_columns = list(summary_rows[0].keys())


write_csv("results_reconstructed.csv", rows, raw_columns)

write_csv(
    "stability_per_seed.csv",
    sorted(stability_rows, key=lambda r: (r["seed"], r["strategy"], r["skew"])),
    stability_columns
)

write_csv(
    "stability_summary_across_seeds.csv",
    summary_rows,
    summary_columns
)

write_csv(
    "results_with_stability_features.csv",
    sorted(enhanced_rows, key=lambda r: (r["seed"], r["strategy"], r["skew"], r["round"])),
    enhanced_columns
)


# =========================
# FINAL CHECK
# =========================

print()
print("Done.")
print(f"Input file: {input_path.resolve()}")
print(f"Output folder: {Path(OUTPUT_DIR).resolve()}")
print(f"Parsed result rows: {len(rows)}")
print(f"Expected result rows: {expected_rows}")
print(f"Stability rows: {len(stability_rows)}")
print(f"Summary rows: {len(summary_rows)}")

print()
print("Created files:")
print("1. results_reconstructed.csv")
print("2. stability_per_seed.csv")
print("3. stability_summary_across_seeds.csv")
print("4. results_with_stability_features.csv")