import os
import scipy
import scipy.io
import scipy.sparse
import numpy as np

from annotation import *
from ontology import *
from gene_ontology import *


class MatBenchmark:
    def __init__(self, bm_matfile):
        self.mat = scipy.io.loadmat(bm_matfile, struct_as_record=False, simplify_cells=True)
        self.oa = self.mat['oa']
        self.terms = list(self._get_terms_from_oa())

        self.ann = self.oa['annotation']
        self.targets = self.oa['object']

    def _get_terms_from_oa(self):
        oa = self.oa
        ont = oa['ontology']
        termlist = ont['term']
        for i, node in enumerate(ont['DAG']):
            #print(node)
            term = Term(termlist[i]['id'])
            term.setName(termlist[i]['name'])
            term.setAspect('F')
            #print(term)
            for item in node:
                if len(item.indices) == 0:
                    continue
                #print(item.indices, item.data)
                #print(ont['term'][item.indices[0]])
                #print('item', item)
                term.addParent(ont['term'][item.indices[0]]['id'])

            yield term

    def get_terms_from_oa(self):
        return self.terms

    def get_term(self, i):
        return self.terms[i]

