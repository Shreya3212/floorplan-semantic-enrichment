import json
import csv
import argparse

def summarize_rule_results(json_path, csv_out):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Prepare CSV
    with open(csv_out, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "id", "type", "passed_checks", "failed_checks"])

        for item in data:
            writer.writerow([
                item.get("image", ""),
                item.get("id", ""),
                item.get("type", "unknown"),
                ", ".join(item.get("passed_checks", [])),
                ", ".join(item.get("failed_checks", [])),
            ])

    print(f"✅ Summary CSV written to: {csv_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to rule_based_results.json")
    parser.add_argument("--out", required=True, help="Output CSV path")

    args = parser.parse_args()
    summarize_rule_results(args.json, args.out)
