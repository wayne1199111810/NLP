Author: Chien-Wei Lin 
## Data Source
Penn Treebank tag

## Part 1 Viterbi Part-of-speech Tagger
### The tag-level accuracy of Viterbi
87.72% with add one (Laplace) smoothing
74.78% without add one
### The tag-level accuracy of baseline
87.24%
### Error Example

1. 'Intelogic' should be NP, but is JJ. Even though 'Intelogic' is NP in the training dataset, there are very less NP happen before NN. This might be the reason that Viterbi predict it as the JJ.
Line 56 
An/DT Intelogic/JJ spokesman/NN said/VBD the/DT company/NN does/VBZ n't/RB have/VB any/DT comment/NN beyond/IN the/DT filings/NNS ./. 

2. 'ended' should be VBN, but is VBD. In this case, there are multiple different cases that ended is after NN and before NP. However, VBD and VBN can be used in this case. 
Line 238
Gross/DT lending/NN at/IN the/DT IBRD/NN in/IN the/DT year/NN ended/VBD June/NP 30/CD fell/VBD slightly/RB to/TO $/$ 11.3/CD billion/CD from/IN $/$ 11.6/CD billion/CD ./. 

3. There are multiple error in Line 286. First of all 'Spirit' should be NP, but is NN. 'Spirit' only appears once in the training data, and the tag is NN. And it is also reasonable to predict NN after JJ.

4. Secondly, witchy should be JJ, but is NN. 'witchy' is not in the training data. It is much more easier to wrongly predict a unknown as NN.
Line 286
In/IN part/NN because/IN of/IN such/JJ concerns/NNS ,/, ``/`` Unsolved/PP Mysteries/. ''/'' has/VBZ decided/VBN to/TO give/VB added/VBN attention/IN this/DT season/NN to/TO the/DT wiggier/JJ stories/NNS --/: reality/DT TV/NN 's/POS answer/NN to/TO Fox/NP 's/POS ``/`` Alien/PP Nation/. ''/'' and/CC ABC/NP 's/POS witchy/NN ``/`` Free/JJ Spirit/NN ,/, ''/'' both/DT new/JJ this/DT year/NN ./. 

5. 'cut' should be VBD, but is VB. In the training data the 'cut' are VB after MD and before PP$. Although there are also multiple cases that cut is VBD before a PP$, Viterbi will intuitively consider the consecutive tags. This is the reason that this error occur
Line 364
Mr./NP Kenyon/NP said/VBD Congress/NP substantially/MD cut/VB its/PP$ credit/NN lines/NNS to/TO Federated/VB and/CC Allied/NN in/IN mid-August/NN because/IN of/IN concern/NN about/IN Campeau/NP 's/POS financial/JJ situation/NN ./. 

* To sum up, my errors seem to be the ambiguity caused by next tag and previous tag. One way to solve this type of error is to increase the bigram to multi-gram such as 3-gram or 4-gram. Considering longer sequence should increase the accuracy. On the other hands, I am using Laplace smoothing (add one smoothing) which may highly influence those words which appear only once. Using other smoothing method may be helpful for this kind of error.


## Part 2 Training on large data
### The tag-level accuracy of Viterbi
93.53%
### The tag-level accuracy of baseline
91.98%