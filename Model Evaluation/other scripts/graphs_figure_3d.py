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

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7), gridspec_kw={'width_ratios': [4, 1]}, sharey=True)

# Create the bar plot for variants_11 with error bars
for i in range(len(variants_11)):
    axes[0].bar(r1[i], variants_11[i], color=['#0c70bc', '#da5118', '#edb11f', '#6bd47a'][i], width=bar_width, edgecolor='grey', yerr=error_11[i], capsize=5)

# Create the bar plot for variants_300 one by one
for i in range(len(variants_300)):
    axes[0].bar(r2[i], variants_300[i], color=['#0c70bc', '#da5118', '#edb11f', '#6bd47a'][i], width=bar_width, edgecolor='grey')

# Add labels, title, and manually add legend for the first subplot
axes[0].set_ylabel('Pearson correlation', fontsize=16)
axes[0].set_xticks([0.9, 2.4])
axes[0].set_xticklabels(["11 validation sURS", "300 validation variants"], fontsize=16)
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
legend_labels = ['dBR', 'ADM', 'AMM', 'MBO']
legend_colors = ['#0c70bc', '#da5118', '#edb11f', '#6bd47a']
legend_handles = [plt.Rectangle((0, 0), 0.75, 0.75, color=color) for color in legend_colors]
axes[0].legend(legend_handles, legend_labels, loc='upper left', fontsize=13, ncol=len(legend_labels))

# Add a second subplot
ax2 = axes[1]
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Sample data for the second subplot
correlations = [0.6, 0.53]
labels = ["Corr\n(dBR,MBO)", "Corr\n(dBR,AMM)"]
colors = ['#9580c5']

# Create the bar plot for the second subplot
ax2.bar(labels, correlations, color=colors, edgecolor='grey',width=0.44)
ax2.tick_params(axis='x',labelsize=16)

ax2.tick_params(axis='x', rotation=90)
ax2.set_yticks([])  # Set y-ticks to an empty list to hide them

axes[0].set_yticks([0.1 * i for i in range(-3, 11)])
axes[0].set_yticklabels([f"{0.1 * i:.1f}" for i in range(-3, 11)], fontsize=13)

# Adjust left and right padding for the second subplot
fig.subplots_adjust(right=0.95, top=0.95, bottom=0.19)

# Save the figure
plt.savefig("figure_3d.png", dpi=300)

# Show the plot
plt.show()
