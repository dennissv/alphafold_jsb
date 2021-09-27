#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:27:25 2021

@author: dennis
"""

import pickle
import os
import time
import colabfold as cf
from os.path import isfile, join

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

proteins = load_obj('enccn_proteins.pkl')
proteins = [(name, seq) for name, seq in proteins.items()]
proteins.sort(key=lambda p: len(p[1]))
proteins = proteins[1000:1800]
result_dir = 'ready'
os.makedirs(result_dir, exist_ok=True) 
ready = set([f for f in os.listdir(result_dir) if isfile(join(result_dir, f))])

for name, seq in proteins[:5]:
    start = time.time()
    print("Running: " + name)
    try:  
        a3m_lines = cf.run_mmseqs2(seq, name, True)
    except:
        print(name + " could not be processed")
        continue
    os.mknod(join(result_dir, name))
    end = time.time()
    time.sleep(max(0, 300 - (end-start)))
