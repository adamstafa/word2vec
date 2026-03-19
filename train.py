import argparse
import numpy as np
import pickle
from collections import Counter

class Model:
    def __init__(self, vocabulary_size, embedding_dim):
        self.embedding = self._random_embedding(vocabulary_size, embedding_dim)
        self.context_embedding = self._random_embedding(vocabulary_size, embedding_dim)

    def _random_embedding(self, vocabulary_size, embedding_dim):
        return np.random.normal(0.0, np.sqrt(1.0 / embedding_dim), (vocabulary_size, embedding_dim)).astype(np.float32)
    
    def update(self, center_word, context_word, label, learning_rate):
        # forward pass
        u = self.embedding[center_word].copy()
        v = self.context_embedding[context_word].copy()
        z = np.dot(u, v)
        p = 1 / (1 + np.exp(-z))

        clip_p = np.clip(p, 1e-8, 1 - 1e-8)
        loss = - (label * np.log(clip_p) + (1 - label) * np.log(1-clip_p))

        # The backward pass is:
        # dL/dz = sigmoid(z) - label
        # dz/du = v
        # dz/dv = u

        diff = p - label
        self.embedding[center_word, :] -= learning_rate * diff * v
        self.context_embedding[context_word, :] -= learning_rate * diff * u

        return loss


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
    def __init__(self, dataset, context_size, negative_samples, negative_sample_exponent=0.75, skip_threshold=1e-5):
        self.dataset = dataset
        self.context_size = context_size
        self.negative_samples = negative_samples
        self.skip_threshold = skip_threshold

        # negative sampling distribution
        counts = Counter(dataset)
        self.word_freq = np.array([ counts[i] for i in range(len(counts)) ], dtype=np.float32)
        self.word_freq /= np.sum(self.word_freq)
        self.negative_sample_dist = self.word_freq ** negative_sample_exponent
        self.negative_sample_dist /= np.sum(self.negative_sample_dist)
        

    def __iter__(self):
        c = self.context_size
        for i in range(0, len(self.dataset) - 2 * c - 1):
            window = self.dataset[i : i + 2*c + 1]
            center_word = window[c]
            context = window[:c] + window[c+1:]

            skip_prob = max(0, 1 - np.sqrt(self.skip_threshold / self.word_freq[center_word]))
            if np.random.rand() < skip_prob:
                continue

            for skip_gram in self.generate_skip_grams(center_word, context):
                yield skip_gram
            

    def generate_skip_grams(self, center_word, context):
        # generates (center, context, label) tuples
        for context_word in context:
            # positive sample
            yield center_word, context_word, 1

            # negative samples
            for negative_word in np.random.choice(self.negative_sample_dist.size, p=self.negative_sample_dist, size=self.negative_samples):
                yield center_word, negative_word, 0


class Trainer:
    def __init__(self, dataset_path, embedding_dim, context_size, negative_samples, epochs, learning_rate):
        with open(dataset_path, 'r') as f:
            words = f.read().split()

        self.vocabulary = Vocabulary(words)
        self.data = [ self.vocabulary.word_to_index[w] for w in words ]
        self.data_loader = SkipGramLoader(self.data, context_size=context_size, negative_samples=negative_samples)
        self.model = Model(self.vocabulary.size, embedding_dim)

        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch} started...")
            loss = self.train_epoch()
            print(f"Epoch {epoch} loss: {loss}")

    def train_epoch(self):
        loss = 0.0
        for center, context, label in self.data_loader:
            loss += self.model.update(center, context, label, self.learning_rate)

        return loss


class Embedder:
    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary
        self.norm_embedding = self.model.embedding / np.linalg.norm(self.model.embedding, axis=1, keepdims=True)

    def __getitem__(self, word):
        index = self.vocabulary.word_to_index.get(word)
        if index is not None:
            return self.norm_embedding[index]
        else:
            raise ValueError(f"Word '{word}' not found in vocabulary")
        
    def closest_words(self, embedding, top_k=5):
        similarities = np.dot(self.norm_embedding, embedding)
        best_indices = np.argsort(similarities)[-top_k-1:-1][::-1]
        return [self.vocabulary.index_to_word[i] for i in best_indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to the dataset')
    parser.add_argument('--embedding_dim', type=int, help='dimension of the word embeddings', default=100)
    parser.add_argument('--context_size', type=int, help='maximum distance between center word and context word', default=3)
    parser.add_argument('--negative_samples', type=int, help='number of negative samples per positive sample', default=5)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=5)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.025)
    parser.add_argument('--output_path', type=str, help='path to save the embedder', default='embedder.pkl')
    args = parser.parse_args()

    trainer = Trainer(args.dataset, args.embedding_dim, args.context_size, args.negative_samples, args.epochs, args.learning_rate)
    print("Loaded dataset")
    print("Training ...")
    trainer.train()
    print("Training Completed")

    embedder = Embedder(trainer.model, trainer.vocabulary)
    with open(args.output_path, 'wb') as f:
        pickle.dump(embedder, f)
    print(f"Saved emebedder to {args.output_path}")
