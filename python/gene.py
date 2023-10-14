# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:48:49 2019

@author: Shawn
"""

import re
from collections import defaultdict

from utils import dict_diff

__all__ = ['Gene']

class Gene:
    def __init__(self):
        self.annotations = {}
        
    def add_annotation(self, gaf_rec):
        ann = Annotation(gaf_rec["GO_ID"])
        ann.add_evidence(gaf_rec["Evidence"])
        for extstr in re.split(",|\|", gaf_rec["Annotation_Extension"]):
            if extstr == "":
                continue
            ext = Extension(extstr)
            ann.add_extension(ext)
        
        self.annotations[ann.term] = ann
       
    def __sub__(self, other):
        res = Gene()
        res.annotations = dict_diff(self.annotations, other.annotations)
        if not res.annotations:
            return False
        return res
    
    def __repr__(self):
        return "Annotations: %s" % (self.annotations)
    
    __str__ = __repr__
    
    

class Annotation:
    def __init__(self, term):
        self.term = term
        self.evidences = set()
        self.extensions = defaultdict(set)
    
    def add_evidence(self, e):
        self.evidences.add(e)
    
    def add_extension(self, ext):
        self.extensions[ext.rel].add(ext.target)
    
    def __repr__(self):
        return "Term: '%s', Evidence: %s, Extensions: %s" % (self.term, self.evidences, dict(self.extensions))
    
    __str__ = __repr__
       
    def __sub__(self, other):
        return not (self.term == other.term)

ext_re = re.compile("(\w+)\((.+)\)")

class Extension(object):
    def __init__(self, s):
        self.rel = ext_re.match(s).group(1)
        self.target = ext_re.match(s).group(2)
    
    def __repr__(self):
        return "%s : %s" % (self.rel, self.target)
    
    __str__ = __repr__
