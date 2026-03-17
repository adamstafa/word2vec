import argparse
import numpy as np

class Model:
    def __init__(self, vocabulary_size, embedding_dim):
        self.embeddings = self._random_embedding(vocabulary_size, embedding_dim)
        self.context_embeddings = self._random_embedding(vocabulary_size, embedding_dim)

    def _random_embedding(self, vocabulary_size, embedding_dim):
        init_range = 0.5 / embedding_dim
        return np.random.uniform(-init_range, init_range, (vocabulary_size, embedding_dim))
    
    def update(self, center_word, context_word, label):
        pass


class Vocabulary:
    def __init__(self, words):
        self.word_to_index = dict()
        self.index_to_word = dict()

        i = 0
        for w in words:
            if w not in self.word_to_index:
                self.word_to_index[w] = i
                self.index_to_word[i] = w
                i += 1

        self.size = len(self.word_to_index)


class SkipGramLoader:
    def __init__(self, dataset, context_size, negative_samples):
        self.dataset = dataset
        self.context_size = context_size
        self.negative_samples = negative_samples

    def __iter__(self):
        c = self.context_size
        for i in range(0, len(self.dataset) - 2 * c - 1):
            window = self.dataset[i : i + 2*c + 1]
            center_word = window[c]
            context = window[:c] + window[c+1:]

            for skip_gram in self.generate_skip_grams(center_word, context):
                yield skip_gram
            

    def generate_skip_grams(self, center_word, context):
        # generates (center, context, label) tuples
        for context_word in context:
            # positive sample
            yield center_word, context_word, True

            # negative samples
            for _ in range(self.negative_samples):
                # TODO: sampling distribution
                negative_word = np.random.choice(self.dataset)
                yield center_word, negative_word, False


class Trainer:
    def __init__(self, dataset_path, embedding_dim, epochs):
        with open(dataset_path, 'r') as f:
            words = f.read().split()

        self.vocabulary = Vocabulary(words)
        self.data = [ self.vocabulary.word_to_index[w] for w in words ]
        self.data_loader = SkipGramLoader(self.data, context_size=10, negative_samples=5)
        self.model = Model(self.vocabulary.size, embedding_dim)

    def train(self):
        for center, context, label in self.data_loader:
            print(center, context, label)
            self.model.update(center, context, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to the dataset')
    parser.add_argument('--embedding_dim', type=int, help='dimension of the word embeddings')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    args = parser.parse_args()

    trainer = Trainer(args.dataset, args.embedding_dim, args.epochs)
    print("Loaded dataset")
    print("Training ...")
    trainer.train()
