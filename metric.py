import numpy as np

real_data = np.array(["سالم", "سالم", "سالم", "مشکوک", "مشکوک", "بیمار", "بیمار", "بیمار", "مشکوک", "مشکوک", "بیمار", "بیمار"])
predicted_data = np.array(["سالم", "سالم", "سالم", "مشکوک", "مشکوک", "مشکوک", "بیمار", "بیمار", "مشکوک", "مشکوک", "بیمار", "بیمار"])

labels = np.unique(real_data)

def calculate_metrics(label):
    tp = np.sum((real_data == label) & (predicted_data == label))
    fp = np.sum((real_data != label) & (predicted_data == label))
    fn = np.sum((real_data == label) & (predicted_data != label))
    return tp, fp, fn

def compute_metrics(label):
    tp, fp, fn = calculate_metrics(label)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return label, {"precision": precision, "recall": recall, "f1_score": f1_score}

metrics = dict(map(compute_metrics, labels))

accuracy = np.sum(real_data == predicted_data) / len(real_data)

print("Metrics for each label:")
for label, values in metrics.items():
    print(f"{label}: Precision: {values['precision']:.2f}, Recall: {values['recall']:.2f}, F1-Score: {values['f1_score']:.2f}")
print(f"\nOverall Accuracy: {accuracy:.2f}")
