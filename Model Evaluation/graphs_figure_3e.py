import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
labels = ['dBR', 'ADM', 'AMM', 'MBO']
variants_300 = [0.11, 0.67, 0.63, 0.61]  # Replace with actual values
variants_11 = [0.65, -0.18, 0.48, 0.61]  # Replace with actual values
val_20_percent = [0.45, 0.48, 0.51]  # Replace with actual values

# Define the width of the bars
bar_width = 0.2

# Set up positions for the bars
r1 = np.arange(len(labels))
r2 = [x + bar_width + 0.02 for x in r1]  # Adjusted constant offset for smaller spacing
r3 = [x + 2 * bar_width + 0.04 for x in r1]  # Adjusted constant offset for smaller spacing

plt.figure(figsize=(12, 6))

# Create the bar plot
plt.bar(r1, variants_300, color='#0c70bc', width=bar_width, edgecolor='grey', label='300 validation variants')
plt.bar(r2, variants_11, color="#da5118", width=bar_width, edgecolor='grey', label='11 validation sURS')

# Add 'val_20_percent' to all models except 'DBR'
plt.bar(r3[1], val_20_percent[0], color='#edb11f', width=bar_width, edgecolor='grey', label='20% test set (5CV)')
plt.bar(r3[2], val_20_percent[1], color='#edb11f', width=bar_width, edgecolor='grey')
plt.bar(r3[3], val_20_percent[2], color='#edb11f', width=bar_width, edgecolor='grey')

# Add labels, title, and legend
plt.ylabel('Pearson correlation', fontsize=13)

# Set legend at the top
plt.legend(loc='upper left', fontsize=11)

# Add a gap between bar pairs
plt.xticks([r + 1.5 * bar_width for r in r1], labels, fontsize=11)
plt.yticks([0.1 * i for i in range(-2, 11)])

# Add a horizontal line at y=0
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Adjust left and right padding
plt.subplots_adjust(left=0.18, right=0.9, top=0.90)

plt.savefig("figure_3e_revised.png", dpi=300)
# Show the plot
plt.show()
