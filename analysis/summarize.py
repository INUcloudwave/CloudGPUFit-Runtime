#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


TARGETS = (
    "g4dn.xlarge",
    "g5.xlarge",
    "g6.xlarge",
    "g6e.xlarge",
    "g7.2xlarge",
    "g7e.2xlarge",
)

CSV_FIELDS = (
    "case_id",
    "model_key",
    "model_id",
    "workload",
    "family",
    "batch_size",
    "input_length",
    "output_length",
    "resolution",
    "target",
    "engine_mode",
    "actual_status",
    "prediction_status",
    "comparison_status",
    "actual_median_ms",
    "actual_mean_ms",
    "actual_std_ms",
    "actual_cv_percent",
    "predicted_ms",
    "ape_percent",
    "bias_percent",
    "prediction_unsupported_count",
    "prediction_unsupported_operations",
    "actual_error",
    "prediction_error",
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_row(record, mode):
    for row in record.get("rows", []):
        if row.get("engine_mode") == mode:
            return row
    return None


def final_predictions(target_dir):
    eager_dir = target_dir / "eager"
    records = {
        row["case_id"]: row
        for row in read_jsonl(eager_dir / "prediction_results_ridge.jsonl")
    }
    vision_path = eager_dir / "prediction_results_vision_fixed.jsonl"
    if vision_path.exists():
        records.update({row["case_id"]: row for row in read_jsonl(vision_path)})
    rows = [records[key] for key in sorted(records)]
    write_jsonl(eager_dir / "prediction_results_final.jsonl", rows)
    return records


def compare_target(target_dir):
    actual_records = {
        row["case_id"]: row
        for row in read_jsonl(target_dir / "eager" / "actual_results.jsonl")
    }
    prediction_records = final_predictions(target_dir)
    output = []

    for case_id, actual in actual_records.items():
        prediction = prediction_records.get(case_id)
        predicted = prediction_row(prediction, "eager") if prediction else None
        actual_status = actual.get("status", "failed")
        prediction_status = "ok" if predicted and predicted.get("predicted_ms") is not None else "failed"

        if actual_status == "ok" and prediction_status == "ok":
            comparison_status = "ok"
        elif actual_status in {"oom", "skipped_oom"}:
            comparison_status = "actual_oom"
        elif actual_status != "ok":
            comparison_status = "actual_failed"
        else:
            comparison_status = "prediction_failed"

        actual_ms = actual.get("actual_median_ms")
        predicted_ms = predicted.get("predicted_ms") if predicted else None
        ape = None
        bias = None
        if comparison_status == "ok" and actual_ms:
            bias = (predicted_ms - actual_ms) / actual_ms * 100.0
            ape = abs(bias)

        unsupported = predicted.get("unsupported_operations", []) if predicted else []
        output.append(
            {
                "case_id": case_id,
                "model_key": actual.get("model_key"),
                "model_id": actual.get("model_id"),
                "workload": actual.get("workload"),
                "family": actual.get("family"),
                "batch_size": actual.get("batch_size"),
                "input_length": actual.get("input_length"),
                "output_length": actual.get("output_length"),
                "resolution": actual.get("resolution"),
                "target": actual.get("target"),
                "engine_mode": "eager",
                "actual_status": actual_status,
                "prediction_status": prediction_status,
                "comparison_status": comparison_status,
                "actual_median_ms": actual_ms,
                "actual_mean_ms": actual.get("actual_mean_ms"),
                "actual_std_ms": actual.get("actual_std_ms"),
                "actual_cv_percent": actual.get("actual_cv_percent"),
                "predicted_ms": predicted_ms,
                "ape_percent": ape,
                "bias_percent": bias,
                "prediction_unsupported_count": len(unsupported),
                "prediction_unsupported_operations": json.dumps(unsupported, ensure_ascii=False),
                "actual_error": actual.get("error", ""),
                "prediction_error": prediction.get("error", "") if prediction else "missing prediction",
            }
        )

    output.sort(key=lambda row: row["case_id"])
    path = target_dir / "eager" / "all_results.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    return output


def metric(rows, target, mode, scope, group):
    ok = [row for row in rows if row["comparison_status"] == "ok"]
    oom = [row for row in rows if row["comparison_status"] == "actual_oom"]
    failed = len(rows) - len(ok) - len(oom)
    apes = [float(row["ape_percent"]) for row in ok]
    biases = [float(row["bias_percent"]) for row in ok]
    return {
        "target": target,
        "mode": mode,
        "scope": scope,
        "group": group,
        "conditions": len(rows),
        "success": len(ok),
        "oom": len(oom),
        "failed": failed,
        "mape_percent": statistics.fmean(apes) if apes else None,
        "median_ape_percent": statistics.median(apes) if apes else None,
        "bias_percent": statistics.fmean(biases) if biases else None,
    }


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def grouped_metrics(rows, key, target, mode, scope):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [metric(group_rows, target, mode, scope, name) for name, group_rows in sorted(groups.items())]


def round_metrics(rows):
    for row in rows:
        for key in ("mape_percent", "median_ape_percent", "bias_percent"):
            if row[key] is not None:
                row[key] = round(row[key], 4)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    target_root = args.results / "targets"
    summary_root = args.results / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    eager_metrics = []
    eager_workloads = []
    eager_models = []
    compiled_metrics = []
    all_eager_rows = []
    all_compiled_rows = []

    for target in TARGETS:
        target_dir = target_root / target
        eager_rows = compare_target(target_dir)
        compiled_rows = read_csv(target_dir / "compiled_smoke" / "all_results.csv")
        all_eager_rows.extend(eager_rows)
        all_compiled_rows.extend(compiled_rows)
        eager_metrics.append(metric(eager_rows, target, "eager", "full", "all"))
        eager_workloads.extend(grouped_metrics(eager_rows, "workload", target, "eager", "full"))
        eager_models.extend(grouped_metrics(eager_rows, "model_key", target, "eager", "full"))
        compiled_metrics.append(metric(compiled_rows, target, "compiled", "smoke", "all"))

    eager_metrics.append(metric(all_eager_rows, "all", "eager", "full", "all"))
    compiled_metrics.append(metric(all_compiled_rows, "all", "compiled", "smoke", "all"))

    write_csv(summary_root / "eager_by_target.csv", round_metrics(eager_metrics))
    write_csv(summary_root / "eager_by_target_workload.csv", round_metrics(eager_workloads))
    write_csv(summary_root / "eager_by_target_model.csv", round_metrics(eager_models))
    write_csv(summary_root / "compiled_smoke_by_target.csv", round_metrics(compiled_metrics))

    manifest = {
        "eager": eager_metrics[-1],
        "compiled_smoke": compiled_metrics[-1],
        "targets": list(TARGETS),
        "metric_definition": {
            "APE_percent": "abs(predicted-actual)/actual*100",
            "MAPE_percent": "mean APE over comparison_status=ok rows",
            "Bias_percent": "mean((predicted-actual)/actual*100) over comparison_status=ok rows",
        },
    }
    (summary_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
