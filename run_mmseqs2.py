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
os.makedirs('msas', exist_ok=True)
ready = set([f for f in os.listdir(result_dir) if isfile(join(result_dir, f))])

for name, seq in proteins:
    if name in ready:
        continue
    start = time.time()
    print("Running: " + name)
    try:
        a3m_lines = cf.run_mmseqs2(seq, join('msas', name), True)
    except:
        print(name + " could not be processed")
        continue
    with open(f'{join(result_dir, name)}.a3m', "w") as text_file:
        text_file.write(a3m_lines)
    end = time.time()
    if end - start < 10:
        time.sleep(30)
    time.sleep(max(0, 300 - (end-start)))
