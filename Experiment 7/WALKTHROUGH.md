# Walkthrough: Experiment 7 - Sentiment Analysis on IMDB Movie Reviews Using Logistic Regression, LSTM, Vanilla Transformer, and Pretrained BERT

This walkthrough provides an exhaustive, line-by-line explanation of every code statement, parameter, algorithm, mathematical formulation, and linguistic concept implemented in [Experiment_7.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%207/Experiment_7.ipynb). The experiment implements sentiment classification across four distinct architectural paradigms:
1. **Model 1**: Logistic Regression with Term Frequency-Inverse Document Frequency (TF-IDF) feature engineering.
2. **Model 2**: Bidirectional Long Short-Term Memory (BiLSTM) recurrent neural network implemented in PyTorch.
3. **Model 3**: Vanilla Transformer Encoder with Multi-Head Self-Attention implemented in PyTorch.
4. **Model 4**: Pretrained BERT-family Transformer model (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`) via HuggingFace Transformers.
5. **Comparative Analysis**: Systematic evaluation using Precision, Recall, F1-Score, Accuracy, and Confusion Matrices across all models.

---

## 1. Setup and Environment Imports

### Purpose
Import standard system utilities, numerical libraries (`numpy`), tabular formatting tools (`pandas`), PyTorch deep learning modules (`torch`, `torch.nn`, `DataLoader`, `TensorDataset`), scikit-learn evaluation metrics and baseline models (`TfidfVectorizer`, `LogisticRegression`), HuggingFace dataset utilities (`load_dataset`), and sequence classification pipelines (`pipeline`).

### Code Executed
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from datasets import load_dataset
from transformers import pipeline
```

### Line-by-Line & Syntax Explanation
- `import os`: Operating system interface for environment configuration.
- `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`: Silences OpenMP duplicate library runtime conflicts on Windows systems where multiple numerical packages (PyTorch, NumPy, MKL) link separate OpenMP instances.
- `import re`: Regular expression engine used for cleaning HTML artifacts, punctuation, and non-alphabetic noise.
- `import numpy as np`: Core library for array indexing, label vectorization, and matrix manipulation.
- `import pandas as pd`: Creates and styles comparative summary tables.
- `from collections import Counter`: Computes corpus-wide token frequencies to build a fixed-size vocabulary.
- `import torch` and `import torch.nn as nn`: PyTorch core library and neural network module containing layers (`Embedding`, `LSTM`, `TransformerEncoder`, `Linear`).
- `from torch.utils.data import DataLoader, TensorDataset`: Wraps PyTorch tensors into iterable mini-batches with automatic shuffling and collation.
- `from sklearn.feature_extraction.text import TfidfVectorizer`: Computes statistical n-gram weighting matrices for linear baselines.
- `from sklearn.linear_model import LogisticRegression`: Fits maximum-likelihood linear decision boundaries with L2 weight regularization.
- `from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix`: Generates standard classification benchmarks.
- `from datasets import load_dataset`: HuggingFace library for accessing benchmark NLP datasets.
- `from transformers import pipeline`: High-level HuggingFace inference wrapper for Transformer models.

---

## 2. Dataset Loading and Preprocessing

### Purpose
Load the benchmark Stanford IMDB Movie Reviews dataset, extract balanced training and test subsets, clean raw review strings, construct an indexed vocabulary with special tokens (`<pad>` and `<unk>`), and convert variable-length text sequences into uniform fixed-length integer tensors.

### Code Executed
```python
dataset = load_dataset('stanfordnlp/imdb')
train_subset = dataset['train'].shuffle(seed=42).select(range(600))
test_subset = dataset['test'].shuffle(seed=42).select(range(200))

train_texts = [item['text'] for item in train_subset]
train_labels = np.array(train_subset['label'])
test_texts = [item['text'] for item in test_subset]
test_labels = np.array(test_subset['label'])

def clean_and_tokenize(text):
    return re.sub(r'[^a-zA-Z ]', ' ', text.lower()).split()

all_train_tokens = [tok for text in train_texts for tok in clean_and_tokenize(text)]
vocab_counts = Counter(all_train_tokens).most_common(5000)
word_to_id = {w: i + 2 for i, (w, c) in enumerate(vocab_counts)}
word_to_id['<pad>'] = 0
word_to_id['<unk>'] = 1

max_seq_length = 100

def text_to_padded_ids(text, max_len=max_seq_length):
    tokens = clean_and_tokenize(text)[:max_len]
    token_ids = [word_to_id.get(tok, 1) for tok in tokens]
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    return token_ids

train_tensor_ids = torch.tensor([text_to_padded_ids(t) for t in train_texts], dtype=torch.long)
train_tensor_labels = torch.tensor(train_labels, dtype=torch.long)
test_tensor_ids = torch.tensor([text_to_padded_ids(t) for t in test_texts], dtype=torch.long)
test_tensor_labels = torch.tensor(test_labels, dtype=torch.long)

train_dataloader = DataLoader(TensorDataset(train_tensor_ids, train_tensor_labels), batch_size=32, shuffle=True)
test_dataloader = DataLoader(TensorDataset(test_tensor_ids, test_tensor_labels), batch_size=32, shuffle=False)
```

### Line-by-Line & Syntax Explanation
1. `load_dataset('stanfordnlp/imdb')`: Loads the authentic Stanford Large Movie Review Dataset (25,000 train, 25,000 test reviews).
2. `shuffle(seed=42).select(range(...))`: Samples a reproducible balanced subset (600 train, 200 test reviews) to ensure efficient and stable CPU execution.
3. `clean_and_tokenize(text)`: Lowercases text and removes punctuation, special characters, and numeric digits.
4. `vocab_counts = Counter(...).most_common(5000)`: Restricts vocabulary to the 5,000 most informative lexical tokens, reducing parameter count and memory footprint.
5. `word_to_id`: Maps vocabulary words to positive integers starting at index 2, reserving index 0 for `<pad>` and index 1 for `<unk>`.
6. `text_to_padded_ids(text, max_len=100)`: Truncates sequences exceeding 100 tokens and post-pads shorter sequences with zero IDs (`<pad>`).
7. `TensorDataset` and `DataLoader`: Bundles input matrices and label vectors into mini-batches of size 32.

---

## 3. Model 1: Logistic Regression with TF-IDF Representation

### Theoretical Background & Mathematical Foundations
TF-IDF calculates the relative statistical importance of an n-gram $t$ within a document $d$ relative to corpus $D$:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
$$\text{IDF}(t, D) = \ln \left( \frac{1 + |D|}{1 + \text{DF}(t, D)} \right) + 1$$
Logistic Regression predicts class probability using the sigmoid activation function:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$
The model minimizes binary cross-entropy with L2 weight decay:
$$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln (1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

### Code Executed
```python
tfidf_vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)

lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train_tfidf, train_labels)

lr_predictions = lr_model.predict(X_test_tfidf)
lr_precision, lr_recall, lr_f1, _ = precision_recall_fscore_support(test_labels, lr_predictions, average='binary')
lr_accuracy = accuracy_score(test_labels, lr_predictions)
lr_cm = confusion_matrix(test_labels, lr_predictions)
```

### Empirical Results
- **Accuracy**: 0.7400 (74.0%)
- **Precision**: 0.7500 (75.0%)
- **Recall**: 0.6875 (68.75%)
- **F1-Score**: 0.7174 (71.74%)
- **Confusion Matrix**:
  $$\begin{bmatrix} 82 & 22 \\ 30 & 66 \end{bmatrix}$$
  - True Negatives: 82, False Positives: 22
  - False Negatives: 30, True Positives: 66

---

## 4. Model 2: Bidirectional LSTM Neural Network

### Theoretical Background & Mathematical Foundations
Recurrent Neural Networks suffer from vanishing and exploding gradients when processing long sequences. Long Short-Term Memory (LSTM) cells solve this by maintaining an internal cell state $C_t$ governed by three multiplicative gates:
1. **Forget Gate**: $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$
2. **Input Gate**: $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$
3. **Candidate Memory**: $\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$
4. **Updated Cell State**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
5. **Output Gate**: $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$
6. **Hidden State**: $h_t = o_t \odot \tanh(C_t)$

A **Bidirectional LSTM** runs two separate hidden recurrence passes:
- Forward pass: $\overrightarrow{h}_t = \text{LSTM}_{fwd}(x_t, \overrightarrow{h}_{t-1})$
- Backward pass: $\overleftarrow{h}_t = \text{LSTM}_{bwd}(x_t, \overleftarrow{h}_{t+1})$
The combined hidden representation is concatenated: $h_t = [\overrightarrow{h}_t \,\|\, \overleftarrow{h}_t] \in \mathbb{R}^{2 \cdot d_{hidden}}$. Global mean pooling aggregates temporal hidden states across the sequence before classification.

### Code Executed
```python
class BidirectionalLSTMClassifier(nn.Module):
    def __init__(self, vocab_dim, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 2)
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        mean_pooled = torch.mean(lstm_out, dim=1)
        return self.classifier(mean_pooled)

torch.manual_seed(42)
lstm_network = BidirectionalLSTMClassifier(len(word_to_id) + 2)
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = torch.optim.Adam(lstm_network.parameters(), lr=0.003)

for epoch in range(4):
    lstm_network.train()
    for batch_x, batch_y in train_dataloader:
        lstm_optimizer.zero_grad()
        logits = lstm_network(batch_x)
        loss = lstm_criterion(logits, batch_y)
        loss.backward()
        lstm_optimizer.step()
```

### Empirical Results
- **Accuracy**: 0.6300 (63.0%)
- **Precision**: 0.6038 (60.38%)
- **Recall**: 0.6667 (66.67%)
- **F1-Score**: 0.6337 (63.37%)
- **Confusion Matrix**:
  $$\begin{bmatrix} 62 & 42 \\ 32 & 64 \end{bmatrix}$$

---

## 5. Model 3: Vanilla Transformer Encoder

### Theoretical Background & Mathematical Foundations
The Transformer architecture (Vaswani et al., 2017) completely eliminates recurrent loops in favor of multi-head self-attention mechanisms.

#### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V$$
where $Q = X W_Q$, $K = X W_K$, and $V = X W_V$. The scaling factor $1/\sqrt{d_k}$ prevents the dot products from growing excessively large for large dimensions, which would push the softmax into regions with vanishing gradients.

#### 2. Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

#### 3. Positional Encoding
Because self-attention is permutation-invariant, positional encodings are added to token embeddings:
$$\mathbf{Z}_0 = \mathbf{E}_{token} + \mathbf{E}_{position}$$

#### 4. Residual Connection and LayerNorm
Each sub-layer incorporates a residual skip connection followed by Layer Normalization:
$$\mathbf{Z}' = \text{LayerNorm}(\mathbf{Z} + \text{SubLayer}(\mathbf{Z}))$$

### Code Executed
```python
class VanillaTransformerClassifier(nn.Module):
    def __init__(self, vocab_dim, max_len=100, embedding_dim=64, num_heads=4, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(embedding_dim, 2)
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        pos_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        tokens_plus_positions = self.embedding(input_ids) + self.positional_embedding(pos_indices)
        encoded_representations = self.transformer_encoder(tokens_plus_positions)
        mean_pooled = torch.mean(encoded_representations, dim=1)
        return self.classifier(mean_pooled)

torch.manual_seed(42)
transformer_network = VanillaTransformerClassifier(len(word_to_id) + 2)
transformer_criterion = nn.CrossEntropyLoss()
transformer_optimizer = torch.optim.Adam(transformer_network.parameters(), lr=0.001)

for epoch in range(4):
    transformer_network.train()
    for batch_x, batch_y in train_dataloader:
        transformer_optimizer.zero_grad()
        logits = transformer_network(batch_x)
        loss = transformer_criterion(logits, batch_y)
        loss.backward()
        transformer_optimizer.step()
```

### Empirical Results
- **Accuracy**: 0.5500 (55.0%)
- **Precision**: 0.5294 (52.94%)
- **Recall**: 0.5625 (56.25%)
- **F1-Score**: 0.5455 (54.55%)
- **Confusion Matrix**:
  $$\begin{bmatrix} 56 & 48 \\ 42 & 54 \end{bmatrix}$$

#### Why does the Vanilla Transformer train more slowly from scratch on small data?
Transformers have minimal inductive bias compared to LSTMs and CNNs (which assume sequential locality). Without massive pre-training corpora (millions of examples), self-attention weights cannot easily discover syntactic structure from a small sample of 600 reviews.

---

## 6. Model 4: Pretrained BERT Transformer Model

### Theoretical Background & Pre-training Paradigms
BERT (Bidirectional Encoder Representations from Transformers; Devlin et al., 2019) pre-trains deep bidirectional representations over massive corpora (BooksCorpus and English Wikipedia, 3.3 billion words) using two self-supervised objectives:
1. **Masked Language Modeling (MLM)**: Randomly masks 15% of input tokens and predicts them from context, forcing the network to build deep bidirectional representations.
2. **Next Sentence Prediction (NSP)**: Binary classification determining if sentence B logically follows sentence A.

In our implementation, we use `distilbert/distilbert-base-uncased-finetuned-sst-2-english`. DistilBERT compresses the 12-layer BERT base architecture into 6 layers via knowledge distillation, retaining 97% of BERT's language understanding capabilities while executing 60% faster with 40% fewer parameters.

### Code Executed
```python
sentiment_pipeline = pipeline('text-classification', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', device=-1)

bert_predictions_list = []
batch_size_bert = 32
for i in range(0, len(test_texts), batch_size_bert):
    batch_texts = [text[:512] for text in test_texts[i:i + batch_size_bert]]
    batch_outputs = sentiment_pipeline(batch_texts)
    for output in batch_outputs:
        label_int = 1 if output['label'] == 'POSITIVE' else 0
        bert_predictions_list.append(label_int)

bert_predictions = np.array(bert_predictions_list)
bert_precision, bert_recall, bert_f1, _ = precision_recall_fscore_support(test_labels, bert_predictions, average='binary')
bert_accuracy = accuracy_score(test_labels, bert_predictions)
bert_cm = confusion_matrix(test_labels, bert_predictions)
```

### Empirical Results
- **Accuracy**: 0.7900 (79.0%)
- **Precision**: 0.7700 (77.0%)
- **Recall**: 0.8021 (80.21%)
- **F1-Score**: 0.7857 (78.57%)
- **Confusion Matrix**:
  $$\begin{bmatrix} 81 & 23 \\ 19 & 77 \end{bmatrix}$$

---

## 7. Comprehensive Comparative Performance Analysis

### Performance Comparison Table
| Model Architecture | Accuracy | Precision | Recall | F1-Score | Parameter Complexity | Inductive Bias |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Logistic Regression (TF-IDF)** | **0.7400** | 0.7500 | 0.6875 | 0.7174 | Low (~2,500 linear weights) | Strong lexical prior (bag-of-words linear separator) |
| **Bidirectional LSTM (PyTorch)** | **0.6300** | 0.6038 | 0.6667 | 0.6337 | Medium (~350k parameters) | Strong sequential recurrence prior |
| **Vanilla Transformer (PyTorch)** | **0.5500** | 0.5294 | 0.5625 | 0.5455 | High (~420k parameters) | Weak inductive bias; requires large pre-training |
| **Pretrained BERT (Transformers)** | **0.7900** | **0.7700** | **0.8021** | **0.7857** | Very High (66M parameters) | Pretrained on billions of words; transfer learning |

### Key Observations & Insights
1. **Pretrained BERT Outperforms All Models**: Demonstrates the dominance of transfer learning. Pre-training on massive corpora allows the model to understand nuanced sentiment, sarcasm, negation, and complex syntax that cannot be learned from small task-specific datasets.
2. **Logistic Regression Beats Small Deep Models**: On small training datasets (600 samples), linear models with TF-IDF features frequently outperform deep neural networks trained from scratch. Deep networks have high variance and easily overfit small training samples, while TF-IDF + L2 regularization provides a strong, low-variance linear baseline.
3. **Vanilla Transformer vs. LSTM on Small Data**: The BiLSTM outperforms the scratch Vanilla Transformer (0.63 vs 0.55). Recurrent networks possess a built-in sequential bias ($h_t$ strictly depends on $h_{t-1}$), which helps when training on small corpora. Transformers must learn pairwise token interactions from scratch without any inherent sequential bias.

---

## 8. Comprehensive Viva Voce Preparation

### Q1: What is the difference between Precision, Recall, and F1-Score?
- **Answer**:
  - **Precision**: Proportion of predicted positive instances that are actually positive:
    $$\text{Precision} = \frac{TP}{TP + FP}$$
    High precision means low false alarm rate.
  - **Recall (Sensitivity)**: Proportion of actual positive instances that were correctly identified:
    $$\text{Recall} = \frac{TP}{TP + FN}$$
    High recall means low miss rate.
  - **F1-Score**: Harmonic mean of Precision and Recall:
    $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$
    It balances both metrics, especially when classes are unevenly distributed.

### Q2: Why is the harmonic mean used for F1-score rather than the arithmetic mean?
- **Answer**: The arithmetic mean evenly weights extremes. If a model predicts positive for every single review, its recall is 1.0, but precision might be 0.1; the arithmetic mean would be 0.55, which is misleading. The harmonic mean punishes extreme discrepancies; in this scenario, the harmonic mean yields $2(0.1 \times 1.0)/(0.1 + 1.0) \approx 0.18$, accurately reflecting poor model balance.

### Q3: Why does Logistic Regression with TF-IDF perform surprisingly well compared to deep models trained from scratch?
- **Answer**: This is governed by the **bias-variance trade-off**. Deep neural networks (LSTMs, Transformers) have tens of thousands of free parameters and weak inductive priors. When given only 600 training samples, they suffer from high variance and quickly overfit. In contrast, Logistic Regression is a highly constrained, linear model (high bias, low variance). Combined with TF-IDF n-grams (which capture strong sentiment keywords like "horrible", "masterpiece", "boring"), it generalizes reliably on small sample sizes.

### Q4: Why is Bidirectional LSTM preferred over Unidirectional LSTM for sentiment classification?
- **Answer**: In sentiment analysis, downstream tokens often qualify or invert earlier tokens (e.g. "The acting was not at all convincing" or "Although the plot was weak, the visual effects were breathtaking"). A unidirectional LSTM only sees past context. A bidirectional LSTM processes the sequence forward ($0 \to T$) and backward ($T \to 0$), allowing every hidden state $h_t = [\overrightarrow{h}_t \,\|\, \overleftarrow{h}_t]$ to encode both preceding and succeeding linguistic context.

### Q5: What is the purpose of Positional Encoding in Transformers?
- **Answer**: Self-attention computes dot-product similarities between all token pairs simultaneously:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
This operation is completely permutation-invariant (order-independent). "Dog bites man" and "Man bites dog" would produce identical representations without positional encodings. Positional encodings inject positional indices into the embeddings, enabling the attention mechanism to distinguish relative and absolute word order.

### Q6: What is the difference between Self-Attention and Cross-Attention?
- **Answer**:
  - **Self-Attention**: Queries ($Q$), Keys ($K$), and Values ($V$) all originate from the same input sequence (e.g. Encoder self-attention, where every word attends to every other word in the review).
  - **Cross-Attention**: Queries ($Q$) come from the target sequence (e.g. Decoder), while Keys ($K$) and Values ($V$) come from the source sequence (e.g. Encoder output representations in sequence-to-sequence translation or summarization).

### Q7: Why is BERT bidirectional while GPT is unidirectional?
- **Answer**:
  - **BERT** is designed for natural language understanding (classification, NER, extractive QA). It uses bidirectional self-attention where every token attends to all tokens to the left and right, enabling deep contextual comprehension.
  - **GPT** is an autoregressive generative language model. It uses causal masked self-attention (tokens can only attend to past and present tokens, not future tokens) to ensure valid probabilistic next-token generation during text synthesis.

### Q8: What is Knowledge Distillation in DistilBERT?
- **Answer**: Knowledge distillation is a compression technique where a smaller "student" network (DistilBERT with 6 layers) is trained to reproduce the behavior of a larger "teacher" network (BERT-base with 12 layers). The loss function combines:
1. Distillation loss: Kullback-Leibler (KL) divergence between student and teacher output probability distributions (soft targets).
2. Supervised cross-entropy loss against ground truth labels.
3. Cosine embedding loss aligning the student and teacher hidden vectors.

### Q9: What are the axes of a Confusion Matrix and how do they map to classification outcomes?
- **Answer**:
  - **Rows** represent Ground Truth (True Classes): Row 0 = Actual Negatives, Row 1 = Actual Positives.
  - **Columns** represent Model Predictions: Column 0 = Predicted Negatives, Column 1 = Predicted Positives.
  - Elements:
    - Position $(0, 0)$: True Negatives (TN)
    - Position $(0, 1)$: False Positives (FP, Type I Error)
    - Position $(1, 0)$: False Negatives (FN, Type II Error)
    - Position $(1, 1)$: True Positives (TP)

### Q10: What is the computational complexity of Self-Attention vs. LSTM recurrence?
- **Answer**:
  - **Self-Attention**: $O(n^2 \cdot d)$ per layer, where $n$ is sequence length and $d$ is representation dimension. It is quadratic in sequence length but parallelizable across all tokens in $O(1)$ sequential operations.
  - **LSTM Recurrence**: $O(n \cdot d^2)$ per layer. It is linear in sequence length, but sequential operations require $O(n)$ steps because step $t$ strictly requires hidden state $h_{t-1}$, preventing parallelization across time.
