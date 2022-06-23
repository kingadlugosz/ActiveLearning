import numpy as np
from scipy.stats import ttest_rel
from main import labeling_methods

scores = np.load('results.npy')

alpha = .05
t_statistic = np.zeros((len(labeling_methods), len(labeling_methods)))
p_value = np.zeros((len(labeling_methods), len(labeling_methods)))

# for i in range(t_statistic.shape[0]):
#     for j in range(t_statistic.shape[1]):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
#
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
