import pickle
import numpy as np
import os

path = os.getcwd()
path = path + '/data/wordvecs/'
glove_files = os.listdir(path)

embeddings_dict = {}

for i in glove_files:
    temp_name = i.replace('.txt', '').replace('.', '-') + '.pkl'
    with open(path + i, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    f = open(temp_name, "wb")
    pickle.dump(embeddings_dict, f)
    f.close()