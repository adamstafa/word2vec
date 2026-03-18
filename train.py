import argparse
import numpy as np
import pickle

class Model:
    def __init__(self, vocabulary_size, embedding_dim):
        self.embedding = self._random_embedding(vocabulary_size, embedding_dim)
        self.context_embedding = self._random_embedding(vocabulary_size, embedding_dim)

    def _random_embedding(self, vocabulary_size, embedding_dim):
        init_range = 0.5 / embedding_dim
        return np.random.uniform(-init_range, init_range, (vocabulary_size, embedding_dim)).astype(np.float32)
    
    def update(self, center_word, context_word, label, learning_rate):
        # forward pass
        u = self.embedding[center_word]
        v = self.context_embedding[context_word]
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
            yield center_word, context_word, 1

            # negative samples
            for _ in range(self.negative_samples):
                # TODO: sampling distribution
                negative_word = self.dataset[np.random.randint(0, len(self.dataset))]
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
        n = 0
        for center, context, label in self.data_loader:
            loss += self.model.update(center, context, label, self.learning_rate)
            n += 1

            if (n + 1) % 1000 == 0:
                max_norm = np.max(np.linalg.norm(self.model.embedding, axis=1))
                max_norm_idx = np.argmax(np.linalg.norm(self.model.embedding, axis=1))
                max_norm_word = self.vocabulary.index_to_word[max_norm_idx]
                avg_norm = np.mean(np.linalg.norm(self.model.embedding, axis=1))
                print(f"Step {n + 1}: Max embedding norm = {max_norm:.4f} (word: '{max_norm_word}'), Avg norm = {avg_norm:.4f}")

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
        
    def closest_words(self, word, top_k=5):
        embedding = self[word]
        similarities = np.dot(self.norm_embedding, embedding)
        best_indices = np.argsort(similarities)[-top_k-1:-1][::-1]
        return [self.vocabulary.index_to_word[i] for i in best_indices]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to the dataset')
    parser.add_argument('--embedding_dim', type=int, help='dimension of the word embeddings')
    parser.add_argument('--context_size', type=int, help='size of the context window', default=3)
    parser.add_argument('--negative_samples', type=int, help='number of negative samples per positive sample', default=10)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=5)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.01)
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
