import numpy as np
from scipy.stats import binom_test
import numpy as np
from scipy.stats import ttest_ind, ttest_ind_from_stats, mannwhitneyu, wilcoxon
from scipy.special import stdtr
import random

# Define the number of samples and the observed accuracy
n = 2721
accuracy = 0.503

a = np.zeros(n) # 327,27
b = np.zeros(n) #299,14

for i in range(int(accuracy* n)):
    a[i] = 1.0

for i in range(int(0.5* n)):
    b[i] = 1.0

print(np.mean(a))
print(np.mean(b))

random.shuffle(b)

result = wilcoxon(a, b)

pass



