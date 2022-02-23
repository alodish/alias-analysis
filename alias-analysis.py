# Imports
import os
from scipy import spatial
import pickle
import pandas as pd

names = []

with open("names.txt", "r") as txt:
    for line in txt:
        if line not in names:  # Avoiding duplicate occurences of names
            names.append(line.replace("\n", ""))

names = [i.lower() for i in names]

print(names[:5])

# Get path to our pickle dictionaries and store their names in list form
path = os.getcwd()
dict_path = os.path.join(os.getcwd(), "pickle-dicts/")
dicts = os.listdir(dict_path)
results = [i.replace('.pkl', ".csv") for i in dicts]


def find_closest_embeddings(embedding, embeddings_dict=None):
    return sorted(embeddings_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))[1:6]


# Results loop
for d in range(len(dicts)):
    file = open(dict_path + '/' + dicts[d], "rb")
    temp_d = pickle.load(file)
    csv_dict = {}
    for name in names:
        try:
            csv_dict[name] = find_closest_embeddings(temp_d[name], embeddings_dict=temp_d)
        except:
            pass

    df = pd.DataFrame(csv_dict)
    df.to_csv(results[d])

# Path to our results folder
results_path = path + '/results/'
csv_results = os.listdir(results_path)
names = ['beth', 'stephen', 'andrew', 'john', 'jennifer']

# Result viewing loop
for csv in csv_results:
    df = pd.DataFrame(pd.read_csv(results_path + csv))
    print(csv, '\n')
    for n in names:
        print(n, df[n].values)
    print('\n\n')


def find_all_embeddings(embedding, embeddings_dict=None):
    return sorted(embeddings_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


# These dictionaries found "Steven" for "Stephen"
dicts = ['glove-42b-300d.pkl', 'glove-6B-300d.pkl', 'glove-6b-100d.pkl']

# Initialize a dictionary of names that stores the relative position of expected names as values
name_dict = {'beth': [], 'andrew': [], 'jennifer': []}

# List of dict keys
names = [i for i in name_dict.keys()]

# The expected values
nicknames = ['elizabeth', 'drew', 'jenn']

# Results loop
for d in range(len(dicts)):
    file = open(dict_path + '/' + dicts[d], "rb")
    temp_d = pickle.load(file)
    for n in range(len(names)):
        temp_array = find_all_embeddings(temp_d[names[n]], embeddings_dict=temp_d)
        try:
            temp_idx = temp_array.index(nicknames[n])
            temp_len = len(temp_array)
            string = [nicknames[n] + ': found at index ' + str(temp_idx) + '/' + str(temp_len)]
            name_dict[names[n]].extend(string)
        except:
            string = [nicknames[n] + ': not found in ' + str(len(temp_array)) + ' indices.']
            name_dict[names[n]].extend(string)

f = open('follow-up-results.pkl', "wb")
pickle.dump(name_dict, f)
f.close()

# Print key:values from our results dictionary
file = open(results_path + '/' + 'follow-up-results.pkl', "rb")
temp_d = pickle.load(file)
d_keys = [i for i in temp_d.keys()]

for k in range(3):
    print("Results for ", dicts[k].replace('.pkl', ': '), '\n')
    for i in range(3):
        print('Key: ', d_keys[i])
        print(temp_d[d_keys[i]][k], '\n')
    print('\n')

# List of common nouns
nouns = ['frog', 'tree', 'star', 'pencil', 'car']

# Re-define our results loop to simply print results rather than save them to CSVs

for d in range(len(dicts)):
    file = open(dict_path + '/' + dicts[d], "rb")
    temp_d = pickle.load(file)
    print("Results for: ", dicts[d], '\n')
    for noun in nouns:
        r = find_closest_embeddings(temp_d[noun], embeddings_dict=temp_d)
        string = noun + ' - 5 closest embeddings: '
        print(string)
        print(*r, '\n')
    print('\n')
