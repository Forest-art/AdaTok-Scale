import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from the new table provided
data = {
    "Token Length": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "FlexRep": [303.11, 276.44, 139.47, 58.30, 21.39, 9.84, 4.18, 1.89, 1.16],
    "OneDPiece": [346.48, 328.73, 100.41, 34.27, 15.25, 7.37, 3.88, 1.95, 1.48],
    "FlexTok": [7.74, 6.68, 5.11, 4.83, 3.59, 3.19, 2.84, 2.86, 3.19],
    "TiTok-S-128": [np.nan, np.nan, np.nan, np.nan, np.nan,  np.nan,  np.nan, 1.70,  np.nan],
    "TiTok-B-64": [np.nan, np.nan, np.nan, np.nan, np.nan,  np.nan, 1.71,  np.nan,  np.nan],
    "TiTok-L-32": [np.nan, np.nan, np.nan, np.nan, np.nan, 2.21, np.nan, np.nan, np.nan]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(8,6))

# Plot each method's data
plt.plot(df['Token Length'], df['FlexRep'], label='AdaTok', marker='o', linestyle='-', color='orange', linewidth=4, markersize=12)
# plt.plot(df['Token Length'], df['OneDPiece'], label='OneDPiece', marker='s', linestyle='--', color='red')
plt.plot(df['Token Length'], df['FlexTok'], label='FlexTok', marker='^', linestyle='-.', color='purple', linewidth=4, markersize=12)
plt.plot(df['Token Length'], df['TiTok-S-128'], label='TiTok-S-128', marker='D', linestyle=':', color='blue', linewidth=4, markersize=12)
plt.plot(df['Token Length'], df['TiTok-B-64'], label='TiTok-B-64', marker='v', linestyle='-', color='green', linewidth=4, markersize=12)
plt.plot(df['Token Length'], df['TiTok-L-32'], label='TiTok-L-32', marker='x', linestyle='-', color='brown', linewidth=4, markersize=12)


# Set the scale for both axes with different log bases
plt.xscale('log', base=2)  # Log base 2 for the x-axis
plt.yscale('log', base=10)  # Log base 10 for the y-axis

# Customize tick marks for the x and y axes
plt.xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], ['1', '2', '4', '8', '16', '32', '64', '128', '256'])  # x-axis custom ticks
plt.yticks([1, 10, 100], ['1', '10', '100'])  # y-axis custom ticks

# Labels and title
plt.xlabel('Token Length', fontsize=15)
plt.ylabel('rFID', fontsize=15)
# plt.title('rFID vs Token Length for Different Methods', fontsize=18)

# Add legend
plt.legend(fontsize=12)

# Remove top and right borders
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')

# Remove gridlines
plt.grid(False)



# Show plot
plt.savefig("token-rFID.pdf", dpi=200)