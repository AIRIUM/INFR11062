import numpy as np
import pickle

PATH = './data/'


w2i = pickle.load(open(PATH+'w2i.dict', "rb"))
i2w = pickle.load(open(PATH+'i2w.dict', "rb"))
vocab = pickle.load(open(PATH+'vocab.dict', "rb"))
vocab_size_en = min(len(i2w["en"]), 10000)
vocab_size_fr = min(len(i2w["fr"]), 10000)
print("vocab size, en={0:d}, fr={1:d}".format(vocab_size_en, vocab_size_fr))
print("{0:s}".format("-"*50))


with open(PATH+'text.fr', 'rb') as f:
    jp = f.readlines()
with open(PATH+'text.en', 'rb') as f:
    en = f.readlines()
def count_word(txt, name=None):
    w = ''
    wtokn = []
    wtype = []
    for line in txt:
        for c in line.split():
            wtokn.append(c)
            if c not in wtype:
                wtype.append(c)
    if name == 'jp':
        dname = 'fr'
    else:
        dname = 'en'
    num_tk2unk = 0
    for w in wtokn:
        if w not in list(vocab[dname].keys()):
            num_tk2unk = num_tk2unk + 1
    print(name)
    print("NUM OF LINE  : {0:d}".format(len(txt)))
    print("NUM OF TYPE  : {0:d}".format(len(wtype)))
    print("NUM OF TOKN  : {0:d}".format(len(wtokn)))
    print("NUM OF TK2UNK: {0:d}".format(num_tk2unk))

    return wtokn, wtype

jp_tk, jp_tp = count_word(jp, name="jp")
en_tk, en_tp = count_word(en, name="en")

corr =  {}

for (jpl, enl) in zip(jp,en):
    num_tk_jpl = 0
    num_tk_jpl = len(jpl.split())
    '''
    for char in jpl.split():
        #if char == ' ' or char == '\n':
        if char == ' ':            
            num_tk_jpl = num_tk_jpl + 1
    '''
    num_tk_enl = 0       
    num_tk_enl = len(enl.split())
    '''
    for char in enl.split():
        #if char == ' ' or char == '\n':
        if char == ' ':
            num_tk_enl = num_tk_enl + 1
    '''
    key = str(num_tk_jpl)+'-'+str(num_tk_enl)
    if key in list(corr.keys()):
        corr[key] = corr[key] + 1
    else:
        corr[key] = 1

X = []
Y = []
W = []
for key in sorted(corr):
    X.append(int(key.split('-')[0]))
    Y.append(int(key.split('-')[1]))
    W.append(corr[key])

import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

plt.figure()
plt.xlim(0,max(max(X),max(Y)))
plt.ylim(0,max(max(X),max(Y)))
plt.gca().set_aspect(aspect='equal', adjustable='box')
plt.grid(color='black', linestyle='-.')
plt.plot([0, max(max(X),max(Y))],[0, max(max(X),max(Y))],'-.', color='black', linewidth=1)
plt.title('Distribution of token number in JPN/ENG sentences')
plt.xlabel("Token number of Japanese sentences")
plt.ylabel("Token number of English sentences")
plt.legend(handles=[
    mpatches.Patch(color='#0000FF', linestyle='-.', linewidth=1, label='Average Token Number'),
    mpatches.Patch(color='#00FF00', linewidth=1, label='Fewer Occurrences'),
    mpatches.Patch(color='#FFFF00', linewidth=1, label='Moderate Occurrences'),
    mpatches.Patch(color='#FF0000', linewidth=1, label='More Occurrences'),
    ])

W_max = max(W)
avg = {}
for (x,y,w) in zip(X,Y,W):

    if x in list(avg.keys()):
        c = c + w
        avg[x] = avg[x]*(c-w)/c + y*w/c
    else:
        avg[x] = y
        c = w

    w = round(511 / W_max * w - 256)
    if w >= 240:
        w = '#FF0' + hex(255-w)[2:].upper() + '00'
    elif w >= 0:
        w = '#FF' + hex(255-w)[2:].upper() + '00'
    elif w >= -240:
        w = '#' + hex(256+w)[2:].upper() + 'FF00'
    else:
        w = '#0' + hex(256+w)[2:].upper() + 'FF00'
    plt.plot(x, y, '.', color=w)

x4avg = []
y4avg = []
for k in sorted(avg):
    x4avg.append(k)
    y4avg.append(avg[k])
plt.plot(x4avg,y4avg, '-', color='blue')

plt.show()