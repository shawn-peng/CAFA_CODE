from collections import deque, defaultdict
from gene_ontology import *
from copy import copy, deepcopy

class Ontology:
    def __init__(self, terms = None):
        # parent_map = {}
        self.terms = {}
        self.leaves = set()
        self.roots = set()
        self.inner_terms = set()
        if not terms:
            return
        
        self.terms = {term.id: term for term in terms}
        # print('finding leaves')
        self.findLeaves()
        # print('finding roots')
        self.findRoots()
        # print('updating children')
        self.updateChildren()

    def __contains__(self, tid):
        return tid in self.terms

    def __getitem__(self, tid):
        return self.terms[tid]

    def keys(self):
        return self.terms.keys()

    def values(self):
        return self.terms.values()

    def items(self):
        return self.terms.items()

    def topological_order(self, roots=None):
        if not roots:
            roots = self.roots

        queue = deque()
        for rootid in roots:
            queue.append(rootid)
        visited = defaultdict(int)
        while(queue):
            tid = queue.pop()
            term = self.terms[tid]
            # print('visiting', tid)
            for chid in term.children:
                visited[chid] += 1
                child = self.terms[chid]
                if visited[chid] == len(child.parents):
                    # print('enque', chid)
                    queue.append(chid)
                    continue
            yield tid

    def reversed_topological_order(self):
        order = list(self.topological_order())
        return reversed(order)

    def updateChildren(self):
        for term in self.terms.values():
            for pid in term.parents:
                self.terms[pid].addChild(term.id)

    def findRoots(self):
        for term in self.terms.values():
            if not term.parents:
                self.roots.add(term.id)

    
    def findLeaves(self):
        for term in self.terms.values():
            self.leaves.add(term.id)
            for pid in term.parents:
                self.inner_terms.add(pid)
        
        for term in self.inner_terms:
            self.leaves.remove(term)
        
    
    def addTerm(self, term):
        self.terms[term.id] = term

    def removeTerm(self, termid):
        term = self.terms[termid]
        for chid in term.children:
            child = self.terms[chid]
            child.parents.remove(termid)
        for pid in term.parents:
            parent = self.terms[pid]
            parent.children.remove(termid)
            for chid in term.children:
                child = self.terms[chid]

                parent.addChild(child.id)
                child.addParent(parent.id)
        if termid in self.terms:
            del self.terms[termid]
                
    
    def __and__(self, other):
        # new_termids = set.intersection(set(self.terms.keys()), set(other.terms.keys()))
        # new_ont = Ontology()
        # for tid in new_termids:
        #     new_term = Term(tid)
        #     term = self.terms[tid]
        #     new_term.setName(term.name)
        #     new_term.setAspect(term.aspect)
        #     for pid in self.terms[tid].parents:
        #         if pid in new_termids:
        #             new_term.addParent(pid)
        #     new_ont.addTerm(new_term)
        # print('making deepcopy')
        # new_ont = deepcopy(self)
        new_ont = Ontology()
        new_ont.terms = deepcopy(self.terms)
        new_ont.leaves = copy(self.leaves)
        new_ont.roots = copy(self.roots)
        # print('deepcopy finished')
        for tid in self.reversed_topological_order():
            # print('checking term', tid)
            # print('parents', new_ont.terms[tid].parents)
            if tid not in other.terms:
                # print('removing', tid)
                new_ont.removeTerm(tid)
        new_ont.leaves.clear()
        # print('finding leaves')
        new_ont.findLeaves()
        new_ont.roots.clear()
        # print('finding roots')
        new_ont.findRoots()

        return new_ont

    def __or__(self, other):
        new_termids = set.union(set(self.terms.keys()), set(other.terms.keys()))
        new_ont = Ontology()
        for tid in new_termids:
            new_term = Term(tid)
            term = self.terms[tid]
            new_term.setName(term.name)
            new_term.setAspect(term.aspect)
            for pid in set.union(self.terms[tid].parents, other.terms[tid].parents): # probably we don't need a union because ontologies are consistent subgraphs
                new_term.addParent(pid)
            new_ont.addTerm(new_term)
        new_ont.findLeaves()
        new_ont.findRoots()
        new_ont.updateChildren()
        return new_ont

    def __sub__(self, other):
        new_ont = Ontology()
        for tid, term in self.terms.items():
            if tid not in other.terms:
                newterm = deepcopy(term)
                new_ont.terms[tid] = newterm
                for pid in term.parents:
                    if pid in other.terms:
                        newterm.parents.remove(pid)
                for cid in term.children:
                    if cid in other.terms:
                        newterm.children.remove(cid)
        new_ont.findLeaves()
        new_ont.findRoots()
        new_ont.updateChildren()
        return new_ont


    
    def getAncestors(self, tid):
        if tid not in self.terms:
            return
        queue = deque()
        queue.append(tid)
        visited = set()
        while(queue):
            tid = queue.pop()
            if tid in visited:
                continue
            visited.add(tid)
            for pid in self.terms[tid].parents:
                queue.append(pid)
            yield tid


    
    def getDescendentsSubgraph(self, tid):
        newg = Ontology()
        if tid not in self.terms:
            return None
        queue = deque()
        queue.append(tid)
        visited = set()
        while(queue):
            tid = queue.pop()
            for cid in self.terms[tid].children:
                if cid in visited:
                    continue
                visited.add(cid)
                queue.append(cid)
            # yield tid
            newt = deepcopy(self.terms[tid])
            for pid in self.terms[tid].parents:
                if pid not in newg.terms:
                    newt.parents.remove(pid)
            
            newg.terms[tid] = newt

        newg.findLeaves()
        newg.findRoots()
        newg.updateChildren()
        return newg

    def getSubgraph(self, terms):
        newg = Ontology()
        for tid in terms:
            newt = deepcopy(self.terms[tid])
            for pid in self.terms[tid].parents:
                if pid not in terms:
                    newt.parents.remove(pid)
            
            newg.terms[tid] = newt

        newg.findLeaves()
        newg.findRoots()
        newg.updateChildren()
        return newg

    def getTermList(self):
        return self.terms.keys()



