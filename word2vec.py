import numpy as np
from collections import Counter


class Word2Vec:
    def __init__(self, sentences, embed_size=50, window_size=3, neg_samples=5, lr=0.05):
        self.embed_size = embed_size
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.lr = lr

        # Создаем словарь
        self.word_counts = Counter([word for sentence in sentences for word in sentence])
        self.vocab = sorted(self.word_counts.keys())
        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # Инициализация весов
        self.W_in = np.random.uniform(-0.5 / embed_size, 0.5 / embed_size, (self.vocab_size, embed_size))
        self.W_out = np.zeros((self.vocab_size, embed_size))

        # Распределение для Negative Sampling
        counts = np.array([self.word_counts[self.idx2word[i]] for i in range(self.vocab_size)])
        self.neg_dist = (counts ** 0.75) / np.sum(counts ** 0.75)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def train_step(self, center_idx, context_idx):
        # 1. Forward
        v_c = self.W_in[center_idx]
        u_o = self.W_out[context_idx]

        neg_indices = np.random.choice(self.vocab_size, size=self.neg_samples, p=self.neg_dist)
        u_neg = self.W_out[neg_indices]

        # 2. Ошибки
        score_pos = self._sigmoid(np.dot(u_o, v_c))
        err_pos = score_pos - 1

        scores_neg = self._sigmoid(np.dot(u_neg, v_c))
        err_neg = scores_neg

        # 3. Градиенты
        grad_v_c = err_pos * u_o + np.dot(err_neg, u_neg)
        grad_u_o = err_pos * v_c
        grad_u_neg = np.outer(err_neg, v_c)

        # 4. Обновление весов
        self.W_in[center_idx] -= self.lr * grad_v_c
        self.W_out[context_idx] -= self.lr * grad_u_o
        self.W_out[neg_indices] -= self.lr * grad_u_neg

        return -np.log(score_pos + 1e-10) - np.sum(np.log(1 - scores_neg + 1e-10))

    def train(self, sentences, epochs=5):
        for epoch in range(epochs):
            total_loss = 0
            for sentence in sentences:
                indices = [self.word2idx[w] for w in sentence if w in self.word2idx]
                for i, center_idx in enumerate(indices):
                    start = max(0, i - self.window_size)
                    end = min(len(indices), i + self.window_size + 1)
                    for j in range(start, end):
                        if i == j: continue
                        total_loss += self.train_step(center_idx, indices[j])
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.2f}")

    def find_similar(self, word, top_n=5):
        if word not in self.word2idx: return "Word not found"
        v = self.W_in[self.word2idx[word]]
        v = v / np.linalg.norm(v)
        sims = []
        for i in range(self.vocab_size):
            v_i = self.W_in[i] / (np.linalg.norm(self.W_in[i]) + 1e-10)
            sims.append((self.idx2word[i], np.dot(v, v_i)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[1:top_n + 1]


# --- ЗАПУСК ---

# Читаем данные из сохраненного файла
print("Чтение dataset.txt...")
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Превращаем текст в предложения (sentences)
sentences = [s.split() for s in text.split('\n') if len(s.split()) > 1]

# Берем первые 3000 предложений, чтобы не ждать долго
data_subset = sentences[:3000]

print(f"Всего предложений: {len(data_subset)}")
model = Word2Vec(data_subset, embed_size=50)
model.train(data_subset, epochs=5)

# Проверка
print("\nРезультат для слова 'alice':")
print(model.find_similar("alice"))