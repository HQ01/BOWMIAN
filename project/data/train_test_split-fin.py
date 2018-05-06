import random

lines = open('eng-fin_cleaned.txt', encoding='utf-8').\
    read().strip().split('\n')

random.shuffle(lines)
size = len(lines)
train_set = lines[:int(size*0.8)]
test_set = lines[int(size*0.8):]

with open("train_fin.txt", 'w') as f:
    for s in train_set:
        f.write(s + '\n')
with open("test_fin.txt", 'w') as f:
    for s in test_set:
        f.write(s + '\n')
