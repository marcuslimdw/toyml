# toyml

I believe in the power of artificial intelligence. I also like writing code. The natural conclusion, then, is to write my own machine learning library.

What am I aiming to be able to do, eventually?

* Simple preprocessing
    * Scaling
    * Imputation
    * One-hot encoding
    
* Regression and classification
    * K-nearest neighbours
    * Linear regression
    * Logistic regression
    * Random forests
    * Markov chains

* Unsupervised learning
    * Some simple clustering algorithms - TBD

* Basic deep learning
    * Densely connected layers
    * Feedforward networks with backpropagation
    
The deep learning part will probably be the most interesting, but overall I think developing something like this requires competency in both software engineering and mathematics. I don't just want to write code - I want to write good, clean, maintainable code. This means a minimum of repetition, unit tests, consistency, and other things I might not have the vocabulary for right now.

# How to run

I'm not making this a real package right now since it's so incomplete, but if you want to clone this and test it out, you will need to run the following code so that you can do ``import toyml``:

```
import os
import sys

path = <enter toyml's path here>

sys.path.append(path)
```

Using the command prompt/terminal, in ``toyml``'s directory, you can also run the ``Makefile`` with make. Right now, there are only the ``clean`` and ``test`` options, so the most you can do is ``make clean test``.

# Version history

## 0.1.0

 * First version!
 * Contains a Markov Chain generator and unit tests. Not much else, actually.
