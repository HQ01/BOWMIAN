import random


# For eng-fra.txt

lines = open('eng-fra.txt', encoding='utf-8').\
    read().strip().split('\n')

random.shuffle(lines)
size = len(lines)
train_set = lines[:int(size*0.8)]
test_set = lines[int(size*0.2):]

with open("train.txt", 'w') as f:
    for s in train_set:
        f.write(s + '\n')
with open("test.txt", 'w') as f:
    for s in test_set:
        f.write(s + '\n')


# For eng-fra-MT.txt (for Machine Learning Task)

lines_MT = open('eng-fra-MT.txt', encoding='utf-8').\
    read().strip().split('\n')

random.shuffle(lines_MT)
size_MT = len(lines_MT)
train_set_MT = lines_MT[:int(size_MT*0.8)]
test_set_MT = lines_MT[int(size_MT*0.2):]

with open("train-MT.txt", 'w') as f:
    for s in train_set_MT:
        f.write(s + '\n')
with open("test-MT.txt", 'w') as f:
    for s in test_set_MT:
        f.write(s + '\n')
