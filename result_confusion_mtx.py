import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dt_val_con = np.array([[9564, 23, 12, 16, 47, ],
                       [5, 379, 1, 1, 0],
                       [198, 21, 39027, 1, 2],
                       [3, 0, 0, 1, 0],
                       [4, 0, 0, 8, 89]])

dt_test_con = np.array([[6301, 34, 17, 9, 7],
                        [4, 800, 4, 0, 1],
                        [187, 17, 41515, 1, 0],
                        [0, 0, 0, 3, 0],
                        [1075, 4, 0, 19, 2]])

mlp_val_con = np.array([[9678, 1, 2, 1, 6],
                        [8, 400, 2, 0, 0],
                        [2, 2, 39187, 0, 0],
                        [5, 0, 0, 0, 1],
                        [6, 0, 2, 0, 99]])

mlp_test_con = np.array([[6320, 39, 7, 0, 2],
                         [9, 795, 5, 0, 0],
                         [156, 1, 41563, 0, 0],
                         [3, 0, 0, 0, 0],
                         [1095, 1, 1, 0, 3]])

plt.figure(figsize=(12, 9))
# sns.heatmap(dt_val_con, annot=True, fmt='g')
# sns.heatmap(dt_test_con, annot=True, fmt='g')
# sns.heatmap(mlp_val_con, annot=True, fmt='g')
sns.heatmap(mlp_test_con, annot=True, fmt='g')
plt.show()
