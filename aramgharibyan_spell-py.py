#!/usr/bin/env python
# coding: utf-8



# Assumes big.txt is in the same dir.

import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('../input/big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / len(word)
def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return known([word])or  known(edits0(word))or known(edits4(word))or known(edits3(word))            or known(edits1(word))or known(edits2(word))               or known(edits5(word))           or [word]

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
def edits0(word):
    "Duplicates all letters"
    splits = [(word[:i], word[i:])for i in range(len(word) + 1)]
    newList = [L + R[0] + R[0] + R[1:] for L, R in splits if R]
    
    return set(newList)
def edits4(word):
    "Change some letters"
    splits = [(word[:i], word[i:])for i in range(len(word) + 1)]
    newList2 = []
    for L, R in splits:
        if (R):
            if(R[0] == 'c'):
                newList2.append(L + "s" + R[1:])
            elif(R[0] == 's'):
                newList2.append(L + "c" + R[1:])
            elif(R[0] == 'a'):
                newList2.append(L + "e" + R[1:])
            elif(R[0] == 'e'):
                newList2.append(L + "a" + R[1:])
            elif(R[0] == 't'):
                newList2.append(L + "d" + R[1:])
            elif(R[0] == 'd'):
                newList2.append(L + "t" + R[1:])
            elif(R[0] == 'b'):
                newList2.append(L + "p" + R[1:])
            elif(R[0] == 'p'):
                newList2.append(L + "b" + R[1:])
            elif(R[0] == 'n'):
                newList2.append(L + "m" + R[1:])
            elif(R[0] == 'm'):
                newList2.append(L + "n" + R[1:])
    return set(newList2)
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def edits3(word): 
    "Apply edits1 in edits0"
    return (e2 for e1 in edits0(word) for e2 in edits1(e1))
def edits5(word): 
    "Apply edits1 in edits4"
    return (e2 for e1 in edits0(word) for e2 in edits4(e1))

def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = correction(wrong)
        good += (w == right)
        if w != right:
            #print(right,wrong,w)
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
    dt = time.clock() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))
    
def Testset(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

def test_corpus(filename):
    print("Testing " + filename)
    spelltest(Testset(open('../input/' + filename)))     

test_corpus('spell-testset1.txt') # Development set
test_corpus('spell-testset2.txt') # Final test set

# Supplementary sets
test_corpus('wikipedia.txt')
test_corpus('aspell.txt')

