
import pickle
import sys

# from type3_eval import *
from mat_pred import *

test_model = '/data/yisupeng/workspace/cafa4/prediction/mfo/M037.mat'

pred = MatPred.read_mat_pred(test_model)

for i, (pred_t, row) in enumerate(pred.items()):
    if i >= 10:
        break
    print(i, pred_t, len(row))

print(pred.score.shape)
for i, (pred_t, row) in enumerate(pred.items()):
    pass


