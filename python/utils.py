#!/usr/bin/env python
# coding: utf-8


import os
import sys
import glob
import re
import itertools
import csv
import pandas as pd
import numpy as np
from collections import defaultdict

def findOne(seq, f):
    return next(x for x in seq if f(x))
# findOne(cafa4_ont_asp['mfo'].values(), lambda x: x.name == 'protein binding')

def findOneByValue(d, f):
    return next((k, v) for k, v in d.items() if f(v))
# findOneByValue(cafa4_ont_asp['mfo'], lambda x: x.name == 'protein binding')


def unpack_ref_object(hdf5_object, root):
    def _():
        for ref in hdf5_object[0]:
            x = root[ref]
            yield bytes(x[:]).decode('utf-16')
    return list(_())


subroot = "last_submissions"


filename_regex = re.compile(r'([\w\d -]*)_([123])_([\d]*|hpo|HPO).txt')
author_regex = re.compile('^AUTHOR\s*(.*)$')
modelnum_regex = re.compile('^MODEL\s*([123])$')
keywords_regex = re.compile('^KEYWORDS\s*(.*)$')
accuracy_regex = re.compile('^ACCURACY\s*(.*)$')
# pred_regex = re.compile('^((?:T|EFI)[0-9]+)\s+((?:GO|HP):[0-9]{7})\s+(1\.0{2}|0\.[0-9]{2})\s*$')
pred_regex = re.compile('^((?:T|EFI)[0-9]+)\s+((?:GO|HP|DO):[0-9]+)\s+(1\.0{2}|0\.[0-9]{2})\s*$')
class PredFile:
    def __init__(self, filename):
        self.filename = filename
        m = filename_regex.match(os.path.basename(filename))
        assert(m)
        self.onto = m.group(3)
        self.file = open(filename)
        self._nextline()
        self._getAuthor()
        self._getModelNum()
        self._getKeywords()
        self._getAccuracy()
    
    def _nextline(self):
        line = self.file.readline()
        if not line:
            self.line = None
            return
        self.line = line.strip()

    def _nextrow(self):
        line = self.file.readline()
        if not line:
            self.line = None
            self.row = None
            return
        self.line = line.rstrip()
        self.row = self.line.split('\t')

    def _getAuthor(self):
        line = self.line
        m = author_regex.match(line)
        if not m:
            raise ValueError(self.filename, 'wrong format, expecting AUTHOR line while getting:', line)
        self.author = m.group(1)
        self._nextline()
        return

    def _getModelNum(self):
        line = self.line
        m = modelnum_regex.match(line)
        if not m:
            raise ValueError(self.filename, 'wrong format, expecting MODEL line while getting:', line)
        self.modelnum = (m.group(1))
        self._nextline()
        return

    def _getKeywords(self):
        line = self.line
        m = keywords_regex.match(line)
        if not m:
            return
            raise ValueError(self.filename, 'wrong format, expecting KEYWORDS line while getting:', line)
        self.keywords = m.group(1)
        self._nextline()
        return

    def _getAccuracy(self):
        line = self.line
        m = accuracy_regex.match(line)
        if not m:
            return
            #raise ValueError('wrong format, expecting ACCURACY line while getting:', line)
        self.accuracy = m.group(1)
        self._nextline()
        return
        
    def __iter__(self):
        return self

    def __next__(self):
        
        while not self.line:
            if self.line is None:
                raise StopIteration
            
            self._nextline()

        if self.line == 'END':
            raise StopIteration
            

        line = self.line
        m = pred_regex.match(line)
        if not m:
            raise ValueError('wrong format, expecting PREDICTION line while getting:', line, self.filename)
        
        row = [m.group(1), m.group(2), float(m.group(3))]
        self._nextline()
        return row



def dict_diff(dict1, dict2):
    res = {}
    for k, v1 in dict1.items():
        if k not in dict2:
            res[k] = v1
            continue
        v2 = dict2[k]
        diff = v1 - v2
        if diff:
            res[k] = diff
    return res


def write_fact(output, fact_format, obj_tuple):
    s = (fact_format % obj_tuple) + '\n'
    s = s.replace('-', '_').replace(':', '').lower()
    output.write(s)



