# -*- coding: utf-8 -*-
"""
Project: neurohacking
File: frommitchell.py
Authors: wffirilat & kindler
"""

from random import randint
import numpy

def genData(seed):
    setData = []
    setData.append(numpy.random.normal(seed * 100, 30))
    setData.append(numpy.random.normal(seed * 100, 50))
    return setData

def tuplechi(spart, tpart):
    return (
        (tpart[0] - spart[0]) ** 2 +
        (tpart[1] - spart[1]) ** 2
    )

def test(t, sett, stor):
    ss = [[] for _ in range(4)]
    for i in stor:
        ss[i[0]].append(i)
    svars = []
    for j, s in enumerate(ss):
        for i in s:
            svars[j] += tuplechi(i[1], sett)
    svars = [var / len(ss[i]) for i, var in enumerate(svars)]
    choice = min(svars)
    if (svars.index(choice) + 1 == t):
        return True
    else:
        # print(var.index(choice)+1)
        # print(t)
        return False

def main(train, tests):
    storage = []
    for x in range(train):
        g = randint(1, 4)
        storage.append((g, genData(g)))
    hit = miss = 0
    for x in range(0, tests):
        t = randint(1, 4)
        if (test(t, genData(t), storage) == 1):
            hit += 1
        else:
            miss += 1
    # print(str(hit)+" " +str(miss))
    return hit, miss

def score(train, tests, sets):
    hits = misses = 0
    r = sets
    for x in range(0, r):
        s = main(train, tests)
        hits += s[0]
        misses += s[1]
    hits = hits / float(r)
    misses = misses / float(r)
    print(hits)
    print(misses)

score(100, 100, 100)
score(50, 100, 100)
score(500, 100, 100)
