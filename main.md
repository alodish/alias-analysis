# Alias Analysis with GloVe
---


```python
# Imports
import os
from scipy import spatial
import pickle
import pandas as pd
```

### Attempt at utilizing [GloVe](https://nlp.stanford.edu/projects/glove/)  to associate aliases with their proper names
Create list of names from text file filled with randomly generated names


```python
names = []

with open("names.txt", "r") as txt:
    for line in txt:
        if line not in names:   # Avoiding duplicate occurences of names
            names.append(line.replace("\n", ""))
            
# Now we have a list of names, but we need to make them lowercase to be usable with Glove's word vectors
names = [i.lower() for i in names]

print(names[:5])
```

    ['delmer', 'laurel', 'jody', 'lavonne', 'beth']
    

### Preparations

In order to use Glove's multi-dimensional word vectors, we need to turn the vectors into dictionaries.

I've already done this using the Pickle Module (.pkl files are hosted in this repository)

We're going to send the list of names through each of 5 word vector dictionaries all varying in dimensionality


```python
# Get path to our pickle dictionaries and store their names in list form
path = os.getcwd()
dict_path = os.path.join(os.getcwd(), "pickle-dicts")
dicts = os.listdir(path)
results = [i.replace('.pkl', ".csv") for i in dicts]
```


```python
# Define a function for determining closely related words

def find_closest_embeddings(embedding, embeddings_dict=None):
    return sorted(embeddings_dict.keys(),
                           key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))[1:6]
```

### Now for the results loop
We iterate through our dictionaries finding the top 5 matches for each name in our list

These will be treated as key : value pairs and sent to a pandas DataFrame

Each cycle of the nested loop results in the DataFrame being saved to a csv with the name of the dictionary in use


```python
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
```

### Viewing the results
With our top 5 closest matches for each dictionary now stored in CSVs


```python
# Path to our results folder
results_path = path + '/results/'
csv_results = os.listdir(results_path)

# Some of the embeddings are very lengthy so we will just take a sample of our names as key and print their values
# This sample contains names that are either common nicknames or names with common nicknames (i.e. Andrew = Drew)
names = ['beth', 'stephen', 'andrew', 'john', 'jennifer'] 

# Result viewing loop
for csv in csv_results:
    df = pd.DataFrame(pd.read_csv(results_path + csv))
    print(csv, '\n')
    for n in names:
        print(n, df[n].values)
    print('\n\n')
```

    glove-42b-300d-results.csv 
    
    beth ['becky' 'sara' 'lori' 'brenda' 'julie']
    stephen ['andrew' 'steven' 'alan' 'jonathan' 'david']
    andrew ['stephen' 'brian' 'alan' 'nicholas' 'andy']
    john ['james' 'george' 'william' 'richard' 'paul']
    jennifer ['nicole' 'jessica' 'christina' 'amanda' 'julie']
    
    
    
    glove-6b-100d-results.csv 
    
    beth ['donna' 'phyllis' 'jane' 'pamela' 'joanne']
    stephen ['marshall' 'peter' 'steven' 'moore' 'andrew']
    andrew ['james' 'matthew' 'harris' 'stephen' 'stuart']
    john ['james' 'george' 'thomas' 'paul' 'william']
    jennifer ['amy' 'laura' 'michelle' 'julie' 'cynthia']
    
    
    
    glove-6B-200d-results.csv 
    
    beth ['garmaï' 'hamedrash' 'ostrosky' 'mccarthy-miller' 'hensperger']
    stephen ['murphy' 'evans' 'miller' 'cooper' 'matthew']
    andrew ['smith' 'matthew' 'james' 'harris' 'thahl']
    john ['william' 'james' 'george' 'smith' 'thompson']
    jennifer ['connelly' 'amy' 'lisa' 'samantha' 'aniston']
    
    
    
    glove-6B-300d-results.csv 
    
    beth ['marykane2000' 'svahng' 'muhlt' 'prihn' 'sihp']
    stephen ['murphy' 'bruhth' 'rohch' 'rohsh' 'steven']
    andrew ['matthew' 'thahl' 'stuart' 'bruhth' 'stephen']
    john ['james' 'rohch' 'thomas' 'hyoon' 'rohsh']
    jennifer ['amy' 'lisa' 'michelle' 'connelly' 'oxeant']
    
    
    
    glove-6B-50d-results.csv 
    
    beth ['amy' 'melissa' 'tracey' 'joanne' 'walters']
    stephen ['andrew' 'matthew' 'clarke' 'peter' 'stuart']
    andrew ['stephen' 'clarke' 'stuart' 'nathan' 'howard']
    john ['james' 'william' 'thomas' 'henry' 'george']
    jennifer ['lisa' 'michelle' 'amy' 'jessica' 'lindsay']





### Discussion of Results
While there were some cases where nicknames were found as close embeddings, this analysis missed the mark.

Only a handful of times were the expected embeddings found within the top 5 results.

We did find "Steven" for "Stephen" in a few cases, but never "Elizabeth" for "Beth" or "Jenn" for "Jennifer".



###  Follow-up
Let's check if these aliases are found anywhere within the vector of embeddings

This will utilize a lot of memory and is only for demonstration purposes

Results are saved as .pkl file in results folder


```python
# We'll remove the limit from our previously defined function and search for expected nicknames/ names

def find_all_embeddings(embedding, embeddings_dict=None):
    return sorted(embeddings_dict.keys(),
                           key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

# These dictionaries found "Steven" for "Stephen"
dicts = ['glove-42b-300d.pkl', 'glove-6B-300d.pkl', 'glove-6b-100d.pkl']

# Initialize a dictionary of names that stores the relative position of expected names as values
name_dict = {'beth' : [], 'andrew' : [], 'jennifer' : []}

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
```


```python
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
```

    Results for  glove-42b-300d:  
    
    Key:  beth
    elizabeth: found at index 632/1917495 
    
    Key:  andrew
    drew: found at index 310/1917495 
    
    Key:  jennifer
    jenn: found at index 6206/1917495 
    
    
    
    Results for  glove-6B-300d:  
    
    Key:  beth
    elizabeth: found at index 272621/1958333 
    
    Key:  andrew
    drew: found at index 1035763/1958333 
    
    Key:  jennifer
    jenn: found at index 1040676/1958333 
    
    
    
    Results for  glove-6b-100d:  
    
    Key:  beth
    elizabeth: found at index 1401/400001 
    
    Key:  andrew
    drew: found at index 1327/400001 
    
    Key:  jennifer
    jenn: found at index 62073/400001 
    
    
    
    

### Follow up discussion
As expected, the arrays resulting from removing the limit are huge.

Despite all of them containing our expected output somewhere in the array, 

this just isn't a feasible method for determining name-nickname associations.

### Verifying the results
Based on this outcome, we are without indication that this is either an effective or efficient method of finding nicknames

In a simple attempt at verifying our methods, we will take some simple nouns and check their closest embeddigs.


```python
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
```

    Results for:  glove-42b-300d.pkl 
    
    frog - 5 closest embeddings: 
    frogs turtle monkey toad snake 
    
    tree - 5 closest embeddings: 
    trees branches leaf willow pine 
    
    star - 5 closest embeddings: 
    stars superstar winplayed musicdw moviesdw 
    
    pencil - 5 closest embeddings: 
    pencils pen crayon chalk drawing 
    
    car - 5 closest embeddings: 
    cars vehicle automobile truck auto 
    
    
    
    Results for:  glove-6B-300d.pkl 
    
    frog - 5 closest embeddings: 
    toad frogs monkey chemicals-wholesale squirrel 
    
    tree - 5 closest embeddings: 
    trees pine instance yehv shade 
    
    star - 5 closest embeddings: 
    stars superstar http://www.nwguild.org __________________________________ once 
    
    pencil - 5 closest embeddings: 
    pencils crayon ink male/philippines teen.gay.ten-inch 
    
    car - 5 closest embeddings: 
    cars vehicle truck driver driving 
    
    
    
    Results for:  glove-6b-100d.pkl 
    
    frog - 5 closest embeddings: 
    toad snake ape monkey frogs 
    
    tree - 5 closest embeddings: 
    trees grass pine bushes leaf 
    
    star - 5 closest embeddings: 
    stars superstar legend hero newcomer 
    
    pencil - 5 closest embeddings: 
    pencils crayon ink pens erasers 
    
    car - 5 closest embeddings: 
    vehicle truck cars driver driving 
    
    
    
    

### Expectations Met
These results are much closer to what you would anticipate, despite some very strange associations

### Final Thoughts
If you made it this far, thank you for your time.

Feedback for this project is welcome!
