# Word2vec
A simple word2vec implementation with numpy

## Features
* Skip-gram model
* Negative sampling
* Subsampling of frequent words
* Evaluation on WordSim353 and analogies dataset
* Comparison with Gensim implementation

## Results
I trained the model on the text8 corpus and the model was able to learn the semantic and syntactic relationships between words. The performance is comparable to the [Gensim](https://radimrehurek.com/gensim/) implementation of word2vec.
See the [eval notebook](./eval.ipynb) for the full results.

### Closest words
```
october    -> [november, june, december, july, april]
oxygen	   -> [dioxide, nitrogen, hydrogen, monoxide, hydrogenase]
washington -> [illinois, ohio, brazos, maryland, baltimore]
```

### Analogies
```
man : king :: woman : ?       -> queen
fast : faster :: slow : ?     -> slower
quick : quickly :: slow : ?   -> slowly
```


## Used Datasets:
- text8: [http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip)
- wordsim 353: [https://gabrilovich.com/resources/data/wordsim353/wordsim353.html](https://gabrilovich.com/resources/data/wordsim353/wordsim353.html)
- analogies [https://www.fit.vut.cz/person/imikolov/public/rnnlm/word-test.v1.txt](https://www.fit.vut.cz/person/imikolov/public/rnnlm/word-test.v1.txt)

## References
- Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean: “Efficient Estimation of Word Representations in Vector Space”, 2013; [arXiv:1301.3781](https://arxiv.org/abs/1301.3781).
- Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean: “Distributed Representations of Words and Phrases and their Compositionality”, 2013; [arXiv:1310.4546](https://arxiv.org/abs/1310.4546).
- Yoav Goldberg, Omer Levy: “word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method”, 2014; [arXiv:1402.3722](https://arxiv.org/abs/1402.3722).
- https://www.tensorflow.org/text/tutorials/word2vec
