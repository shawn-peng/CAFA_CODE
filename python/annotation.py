from collections import deque, defaultdict
from copy import copy, deepcopy
from ontology import Ontology

class LeafAnnotation:
    def __init__(self, termids = None):
        if not termids:
            self.terms = set()
        else:
            self.terms = set(termids)
        
    def addTerm(self, termid):
        self.terms.add(termid)
    
class FullAnnotation(Ontology):
    def __init__(self, ont, leaf_terms = None):
        self.ont = ont
        super().__init__(self._propagate(leaf_terms))
    
    def _propagate(self, leaf_terms = None):
        if not leaf_terms:
            # raise StopIteration
            return
        queue = deque(leaf_terms)
        # queue.append(tid)
        visited = set()
        while(queue):
            tid = queue.pop()
            if tid not in self.ont.terms:
                print('term %s not found in ontology, perhaps obsolete. Skipped' % tid)
                continue
            if tid in visited:
                continue
            visited.add(tid)
            term = self.ont.terms[tid]
            for pid in term.parents:
                queue.append(pid)
            term = deepcopy(term)
            term.children.clear()
            yield term

    
    
