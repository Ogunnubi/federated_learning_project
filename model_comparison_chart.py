import matplotlib.pyplot as plt

# Validation accuracies for each model
models = ['Central', 'Federated', 'Hospital A', 'Hospital B', 'Hospital C']
accuracies = [94.00, 81.33, 89.90, 87.88, 87.88]

plt.figure()
plt.bar(models, accuracies, color=['blue','green','red','orange','purple'])
plt.ylabel('Validation Accuracy (%)')
plt.title('Model Comparison')
plt.ylim(0, 100)

# Save the bar chart as an image file
plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')

# Show the chart on screen (optional)
plt.show()