import json
import argparse

def load_graph(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def run_checks(graph_data):
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    results = []

    for node in nodes:
        result = {
            "id": node.get("id", ""),
            "image": node.get("image", ""),
            "type": node.get("ifc_type", "unknown"),
            "passed_checks": [],
            "failed_checks": []
        }

        # === Rule 1: Fire extinguisher must have confidence >= 0.5
        if node.get("ifc_type") == "FIREEXTINGUISHER":
            if node.get("confidence", 0) >= 0.5:
                result["passed_checks"].append("FireExtinguisher Confidence OK")
            else:
                result["failed_checks"].append("FireExtinguisher Confidence TOO LOW")

        # === Rule 2: Exit must have OCR text like "EXIT" or "EXIT ROUTE"
        if node.get("ifc_type") == "EXIT":
            ocr = node.get("ocr_text", "").lower()
            if "exit" in ocr:
                result["passed_checks"].append("Exit OCR OK")
            else:
                result["failed_checks"].append("Exit OCR MISSING")

        results.append(result)

    return results

def save_results(results, out_path):
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Rule-based results saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help="Path to input graph_output.json")
    parser.add_argument('--out', required=True, help="Path to output JSON file")
    args = parser.parse_args()

    graph_data = load_graph(args.json)
    rule_results = run_checks(graph_data)
    save_results(rule_results, args.out)

if __name__ == "__main__":
    main()

