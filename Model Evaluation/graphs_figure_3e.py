import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
variants_300 = [0.11, 0.67, 0.64, 0.61]  # Replace with actual values
variants_11 = [0.65, -0.18, 0.45, 0.60]  # Replace with actual values
error_11 = [0.066, 0.144, 0.087, 0.075]  # Replace with actual error values

# Define the width of the bars
bar_width = 0.2

# Set up positions for the bars
r1 = [0.65 + 0.23 * i for i in range(len(variants_11))]
r2 = [2 + 0.23 * i for i in range(len(variants_300))]

plt.figure(figsize=(10, 6))

# Create the bar plot for variants_11 with error bars
for i in range(len(variants_11)):
    plt.bar(r1[i], variants_11[i], color=['#0c70bc', '#da5118', '#edb11f', '#6bd47a'][i], width=bar_width, edgecolor='grey', yerr=error_11[i], capsize=5)

# Create the bar plot for variants_300 one by one
for i in range(len(variants_300)):
    plt.bar(r2[i], variants_300[i], color=['#0c70bc', '#da5118', '#edb11f', '#6bd47a'][i], width=bar_width, edgecolor='grey')

# Add labels, title, and manually add legend
plt.ylabel('Pearson correlation', fontsize=13)

# Add a gap between bar pairs
plt.xticks([0.9, 2.4], ["11 validation sURS", "300 validation variants"], fontsize=12)
plt.yticks([0.1 * i for i in range(-4, 11)], fontsize=12)

# Add a horizontal line at y=0
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Manually add legend
legend_labels = ['dBR', 'ADM', 'AMM', 'MBO']
legend_colors = ['#0c70bc', '#da5118', '#edb11f', '#6bd47a']
legend_handles = [plt.Rectangle((0, 0), 0.75, 0.75, color=color) for color in legend_colors]
plt.legend(legend_handles, legend_labels, loc='upper left', fontsize=12, ncol=len(legend_labels))

# Adjust left and right padding
plt.subplots_adjust(left=0.18, right=0.9, top=0.90)

plt.savefig("figure_3e_revised.png", dpi=300)
# Show the plot
plt.show()
