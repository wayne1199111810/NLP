# Sentence Boundary Detection
Using the subset of Treebnak dataset, I train a decision tree classifier to classify that if a period is the end of a sentence or not.

## Assumption
* It only Focus on occurance of the period. Thus, assuming that no other puctuation marks can end a sentence.
* It only handle periods that end a word. For example, this project ignore periods that embedded in a word, 27.4
* It only make use of EOS (End of Sentence) and NEOS (Not End of Sentence), and ignore all the TOK labels.

## Description of feature
* Word to the left of "."
* Word to the Right of "."
* Length of left word < 3
* Is left word capitalized
* Is right word capitalized
* If there are more periods on the right word
* If the right word is any punctuation
* Number of upper letter in the left word

## Accuracy
|                  | Accuracy  |
| ---------------- |:---------:|
| 8 features       | 98.13%    |
| First 5 features | 98.12%    |
| Last 3 features  | 93.28%    |

## Implementation
First of all, I take words in SBD.train whose label is not 'TOK'. If the label is 'EOS', I remove the period in the word and save it into my dictionary. As well as the right word. Moreover, I use LabelEncoder and OneHotEncoder to represent the words in my dictionary. Secondly, if the word ends with period, I check whether the word with or without period is in the dictionary. Once it is not in the dictionary, I represent it as unkonwn token. Finally, using the selected features to train the decision tree.