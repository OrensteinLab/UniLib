import matplotlib.pyplot as plt

# Data for the first graph
pearson_scores1 = [0.56139, 0.602278]
x_labels1 = ['11 validation sURS', '300 validation variants']

# Data for the second graph
pearson_scores2 = [0.68, -0.02]
x_labels2 = ['dBR Model', 'ADM Model']

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 7))  # 1 row, 2 columns

# Create the first bar graph
bars1 = ax2.bar(x_labels1, pearson_scores1, color=['#edb11f', '#da5118'], align='center', width=0.4)
ax2.set_xlabel('Validation set', fontsize=12)
ax2.set_ylabel('Pearson correlation', fontsize=12)
ax2.set_xticklabels([])
ax2.set_ylim(-0.1, 1)
ax2.set_yticks([0.1 * i for i in range(-1,11)])
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax2.legend(bars1, x_labels1, loc='upper left', fontsize=12)

# Create the second bar graph
bars2 = ax1.bar(x_labels2, pearson_scores2, color=['#0c70bc', '#6bd47a'], align='center', width=0.4)
ax1.set_xlabel('Yeast measurements', fontsize=12)
ax1.set_ylabel('Pearson correlation on validation variants', fontsize=12)
ax1.set_ylim(-0.1, 1)
ax1.set_xticklabels([])
ax1.set_yticks([0.1 * i for i in range(-1, 11)])
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax1.legend(bars2, x_labels2, loc='upper left', fontsize=12)

# Adjust spacing between subplots
plt.tight_layout()

# Show the combined graph
plt.savefig("figure_3e.png", dpi=300)

plt.show()
