'''
Metrics for use in training keras models
'''

from tensorflow.keras import metrics


METRICS = [
    metrics.CategoricalAccuracy(name='accuracy'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall'),
    metrics.TopKCategoricalAccuracy(name='top_3_categorical_accuracy', k=3),
    metrics.TopKCategoricalAccuracy(name='top_5_categorical_accuracy', k=5)]

# METRICS = [
#           metrics.TruePositives(name='tp'),
#           metrics.FalsePositives(name='fp'),
#           metrics.TrueNegatives(name='tn'),
#           metrics.FalseNegatives(name='fn'),
#           metrics.CategoricalAccuracy(name='accuracy'),
#           metrics.Precision(name='precision'),
#           metrics.Recall(name='recall'),
#           metrics.TopKCategoricalAccuracy(name='top_k_categorical_accuracy', k=5),
#           metrics.AUC(name='auc')
# ]
