# Word2vec
An implementation of Word2vec from-scratch using the skip-gram formulation with negative sampling.

## Results
I trained the model on the text8 corpus and the model was able to learn similar words and some analogies. The performance is comparable to the [Gensim](https://radimrehurek.com/gensim/) implementation of word2vec. See the eval notebook.

## Used Datasets:
- text8: [http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip)
- wordsim 353: [https://gabrilovich.com/resources/data/wordsim353/wordsim353.html](https://gabrilovich.com/resources/data/wordsim353/wordsim353.html)
- analogies [https://www.fit.vut.cz/person/imikolov/public/rnnlm/word-test.v1.txt](https://www.fit.vut.cz/person/imikolov/public/rnnlm/word-test.v1.txt)

## References
- Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean: “Efficient Estimation of Word Representations in Vector Space”, 2013; [arXiv:1301.3781](https://arxiv.org/abs/1301.3781).
- Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean: “Distributed Representations of Words and Phrases and their Compositionality”, 2013; [arXiv:1310.4546](https://arxiv.org/abs/1310.4546).
- Yoav Goldberg, Omer Levy: “word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method”, 2014; [arXiv:1402.3722](https://arxiv.org/abs/1402.3722).
- https://www.tensorflow.org/text/tutorials/word2vec
