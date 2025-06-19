import torch
import matplotlib.pyplot as plt

# Load the two saved probability tensors
probs_1 = torch.load('probs_tensor.pt')
probs_2 = torch.load('probs_tensor_init.pt')

# If probs_1 and probs_2 are batch-wise probabilities, select the first element of each batch
# Assuming probs_1 and probs_2 are both numpy arrays (detached, CPU)
probs_1 = probs_1[0]  # If you want to visualize the first sample
probs_2 = probs_2[0]  # Same for the second tensor

# Create a bar plot with two sets of probabilities, using different colors
plt.figure(figsize=(8, 6))

# Plot the first probability distribution
plt.bar(range(len(probs_1)), probs_1, alpha=0.6, label='w ATA', color='blue', linewidth=4)

# Plot the second probability distribution
plt.bar(range(len(probs_2)), probs_2, alpha=0.6, label='w/o ATA', color='red', linewidth=4)

# Adding labels and title
plt.xlabel('Action', fontsize=15)
plt.ylabel('Probability', fontsize=15)
# plt.title('Comparison of Two Categorical Distributions', fontsize=16)
# Remove top and right borders
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')

# Add a legend to distinguish between the two distributions
plt.legend()

# Show the plot
plt.savefig("probs.pdf", dpi=200)
