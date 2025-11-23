import json
import argparse
from collections import defaultdict

def build_graph(data):
    graph = {"nodes": [], "edges": []}
    image_nodes = {}
    element_counter = defaultdict(int)

    for item in data:
        image_name = item.get("image")
        element_type = item.get("ifc_type")
        element_class = item.get("ifc_class")
        ocr_text = item.get("ocr_text", "")
        confidence = item.get("confidence", 0)

        node_id = f"{image_name}_{element_type}_{element_counter[element_type]}"
        element_counter[element_type] += 1

        # Create element node
        node = {
            "id": node_id,
            "image": image_name,
            "ifc_class": element_class,
            "ifc_type": element_type,
            "ocr_text": ocr_text,
            "confidence": confidence
        }
        graph["nodes"].append(node)

        # Add image node (if not already added)
        if image_name not in image_nodes:
            image_node = {
                "id": image_name,
                "type": "Image"
            }
            graph["nodes"].append(image_node)
            image_nodes[image_name] = image_node

        # Create edge: image -> element
        graph["edges"].append({
            "source": image_name,
            "target": node_id,
            "relation": "contains"
        })

    return graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to enrichment.json")
    parser.add_argument("--out", required=True, help="Path to output graph_output.json")
    args = parser.parse_args()

    # Load enriched data
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten if loaded as dict
    if isinstance(data, dict):
        data = list(data.values())

    # Build graph
    graph = build_graph(data)

    # Save graph
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print(f"✅ Graph exported with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")

if __name__ == "__main__":
    main()
