import scipy
import scipy.io
import scipy.sparse
import numpy as np
import h5py
from utils import unpack_ref_object
import traceback

class MatPred:
    def __init__(self, matpred):
        self.mat = matpred

        scores = matpred['pred']['score']

        jc = np.asarray(scores["jc"]).astype(np.int32)
        if 'data' not in scores:
            print(scores.keys())
            shape = (0, jc.shape[0])
            self.score = scipy.sparse.csc_matrix(shape)
            # self.target_ind = {}
            # print(len(self.target_ind))
            return
        data = np.asarray(scores["data"])
        ir = np.asarray(scores["ir"]).astype(np.int32)

        self.score = scipy.sparse.csc_matrix((data, ir, jc)).tocsr()

        terms = self.mat['pred']['ontology']['term']['id']
        try:
            self.terms = unpack_ref_object(terms, self.mat)
        except Exception as e:
            traceback.print_exc()
            return

        pred_targets = self.mat['pred']['object']
        try:
            self.pred_targets = unpack_ref_object(pred_targets, self.mat)
        except Exception as e:
            traceback.print_exc()
            return

        self.target_ind = {k: i for i, k in enumerate(self.pred_targets)}
        print(len(self.target_ind))
        # print(self.target_ind)

    def __contains__(self, item):
        return item in self.target_ind

    def __getitem__(self, item):
        i = self.target_ind[item]
        row = self.score[i, :]
        scores = {}
        for j in row.indices:
            s = row[:, j].todense()[0, 0]
            term = self.terms[j]
            # newpred[pred_t].add(term)
            if term in scores:
                print(f'duplicate prediction for {pred_t}, taking max score')
                scores[term] = max(scores[term], s)
            else:
                scores[term] = s
        return scores

    def items(self):
        for i, row in enumerate(self.score):
            pred_t = self.pred_targets[i]
            scores = {}
            for j in row.indices:
                s = row[:, j].todense()[0, 0]
                term = self.terms[j]
                # newpred[pred_t].add(term)
                if term in scores:
                    print(f'duplicate prediction for {pred_t}, taking max score')
                    scores[term] = max(scores[term], s)
                else:
                    scores[term] = s

            yield pred_t, scores

    def __len__(self):
        return self.score.shape[0]

    @classmethod
    def read_mat_pred(cls, filename):
        mat = h5py.File(filename)
        return cls(mat)



