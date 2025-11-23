import json
import os
import matplotlib.pyplot as plt

# === Path to input Graph-BIM JSON ===
json_path = "ocr_out/evac_v12_full_fused_v4/graph_output.json"

# === Output folder for PNG visualizations ===
output_dir = "ocr_out/evac_v12_full_fused_v4/visualized"
os.makedirs(output_dir, exist_ok=True)

# === Load Graph JSON ===
with open(json_path, "r") as f:
    graph_data = json.load(f)

# === Visualize graph ===
nodes = graph_data.get("nodes", [])
edges = graph_data.get("edges", [])

plt.figure(figsize=(10, 8))

# Plot nodes
for node in nodes:
    label = node.get("ifc_type", "unknown")
    plt.scatter(len(label), node.get("confidence", 0.5) * 100, s=100, label=label)

# Plot edges
for edge in edges:
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)

plt.title("Graph-BIM Visualization")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()

# Save visualization
output_path = os.path.join(output_dir, "graph_visualization.png")
plt.savefig(output_path)
plt.close()

print(f"✅ Graph visualization saved to: {output_path}")
