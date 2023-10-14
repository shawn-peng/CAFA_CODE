#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:53:49 2019

@author: yisupeng
"""

import sys
from collections import defaultdict
import urllib,urllib3
import urllib.request

idmap_url = 'https://www.uniprot.org/uploadlists/'


def mapping_id(idlist, f, t):

    query_params = {
        'from': f,
        'to': t,
        'format':'tab',
        'query':''
    }
    
    query_results = {}
    
    mappings = {}
    
    idlist = set(idlist)
    query_params['query'] = ' '.join(idlist)
    
    data = urllib.parse.urlencode(query_params).encode("utf-8")
    request = urllib.request.Request(idmap_url, data)
    contact = "peng.yis@husky.neu.edu" # Please set a contact email address here to help us debug in case of problems (see https://www.uniprot.org/help/privacy).
    request.add_header('User-Agent', 'Python %s' % contact)
    response = urllib.request.urlopen(request)
    page = response.read().decode()
    print(page, file=sys.stderr)
    print(len(page), file=sys.stderr)
    query_results = page
    
    for line in page.split("\n"):
        if len(line) == 0:
            continue
        row = line.split("\t")
#        print(row)
        for f in row[0].split(","):
            if f in mappings:
#                raise(KeyError("%s has mutiple mappings" % f))
                print("%s has mutiple mappings" % f, file=sys.stderr)
                continue
            t = row[1]
            if f == 'From' and t == 'To':
                continue
            mappings[f] = t
    
    return mappings


def multi_mapping_id(idlist, f, t):

    query_params = {
        'from': f,
        'to': t,
        'format':'tab',
        'query':''
    }
    
    query_results = {}
    
    mappings = defaultdict(set)
    
    query_params['query'] = ' '.join(idlist)
    
    data = urllib.parse.urlencode(query_params).encode("utf-8")
    request = urllib.request.Request(idmap_url, data)
    contact = "peng.yis@husky.neu.edu" # Please set a contact email address here to help us debug in case of problems (see https://www.uniprot.org/help/privacy).
    request.add_header('User-Agent', 'Python %s' % contact)
    response = urllib.request.urlopen(request)
    page = response.read().decode()
    print(page, file=sys.stderr)
    print(len(page), file=sys.stderr)
    query_results = page
    
    for line in page.split("\n"):
        if len(line) == 0:
            continue
        row = line.split("\t")
#        print(row)
        for f in row[0].split(","):
            t = row[1]
            if f == 'From' and t == 'To':
                continue
            mappings[f].add(t)
    
    return mappings






