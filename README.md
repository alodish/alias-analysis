# Alias Analysis

Examined GloVe (Global Vectors for Word Representation) vectors in an attempt to find nicknames for proper names

In collaboration with [Doug Horner](https://github.com/horner)

### Notes

Check out the [notebook](https://github.com/alodish/alias-analysis/blob/main/alias-analysis.ipynb) file in this directory to see results.

I had hoped to share all of my .pkl files, but they are far too large. The text files are available [here](https://github.com/stanfordnlp/GloVe).

After downloading the text file(s) of your chosing, feel free to use the pickle-dicts.py file to convert them all to Pickle files.

Thank you to Stanford's NLP GloVe group for the countless hours they put into this project!


### Findings

**Long story short**: This is not a great way to find the associated nicknames for a given proper name.

It became evident that the results derived from crawling the web for millions of words rarely results in a close relationship between names and nicknames.

The possibilities are great when it comes to utilizing these word vectors, but I think it's safe to say "Alias Analysis" is off the table (without further tuning).

---

#### If you are familiar with how these GloVe's can be used, you may have thought, "Why not add *name* + *nickname*?"

That was tried as well, but the results weren't worth publishing. 

Essentailly, our output was the same but with a few related embeddings to the word *nickname* (name, nicknames, etc.)
