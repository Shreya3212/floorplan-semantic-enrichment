# file: plot_ifc_type_counts.py

import pandas as pd
import matplotlib.pyplot as plt

# Load your exported semantic enrichment CSV
csv_path = 'ocr_out/evac_v12_full_fused_v4/ifc_export.csv'
df = pd.read_csv(csv_path)

# Drop rows with missing or unknown IFC type
df = df[df['ifc_type'].notna() & (df['ifc_type'] != '')]

# Count how many times each IFC type appears
type_counts = df['ifc_type'].value_counts()

# Plot as a bar chart
type_counts.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Detected Object Counts by IFC Type')
plt.xlabel('IFC Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

# Save to image file
plt.savefig('fig_detection_count.png', dpi=300)
plt.show()
