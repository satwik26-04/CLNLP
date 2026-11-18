# Walkthrough: Experiment 8 - Abstractive Text Summarization Using Multilingual Transformer Models (mT5 and mBART)

This walkthrough provides an exhaustive, line-by-line explanation of every code statement, parameter, algorithm, mathematical formulation, and linguistic concept implemented in [Experiment_8.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%208/Experiment_8.ipynb). The experiment implements abstractive text summarization on Hindi news articles from the XL-Sum benchmark dataset using two multilingual sequence-to-sequence Transformer architectures:
1. **Model 1**: Multilingual Text-to-Text Transfer Transformer (mT5; `google/mt5-small`).
2. **Model 2**: Multilingual Bidirectional and Auto-Regressive Transformers (mBART; `ai4bharat/IndicBART-XLSum`).
3. **Automatic Evaluation Metrics**:
   - ROUGE-1 (Unigram overlap)
   - ROUGE-2 (Bigram overlap)
   - ROUGE-L (Longest Common Subsequence overlap)
   - mBERT score (Contextual Multilingual BERT semantic cosine similarity via `google-bert/bert-base-multilingual-cased`).

---

## 1. Setup and Environment Imports

### Purpose
Import standard system tools, file serialization (`json`), tabular data analysis (`pandas`), PyTorch deep learning framework (`torch`, `torch.nn.functional`), HuggingFace Transformer sequence-to-sequence model classes (`AutoTokenizer`, `AutoModelForSeq2SeqLM`, `MBartForConditionalGeneration`, `AutoModel`), and the Google Research ROUGE metric library (`rouge_scorer`).

### Code Executed
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration, AutoModel
from rouge_score import rouge_scorer
```

### Line-by-Line & Syntax Explanation
- `import os`: Python interface to the underlying operating system environment.
- `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`: Bypasses multi-runtime OpenMP DLL collisions on Windows when PyTorch and MKL-accelerated NumPy libraries execute concurrently.
- `import json`: Parses JSON Lines (`.jsonl`) files containing the benchmark dataset records.
- `import pandas as pd`: Constructs structured DataFrames to tabulate dataset previews, generated summaries, and metric comparisons.
- `import torch`: Core PyTorch library managing tensors and gradient computation.
- `import torch.nn.functional as F`: Provides functional neural network operations, specifically `F.cosine_similarity` for embedding vector alignment.
- `from transformers import AutoTokenizer, AutoModelForSeq2SeqLM`: Loads HuggingFace tokenizer and sequence-to-sequence model heads.
- `from transformers import MBartForConditionalGeneration`: Dedicated PyTorch implementation of the mBART sequence-to-sequence architecture with cross-attention decoder heads.
- `from transformers import AutoModel`: Loads base encoder representations (used for `bert-base-multilingual-cased` semantic evaluation).
- `from rouge_score import rouge_scorer`: Implements official ROUGE evaluation algorithms (ROUGE-1, ROUGE-2, ROUGE-L).

---

## 2. Dataset Loading and Inspection (XL-Sum Hindi)

### Theoretical Background: XL-Sum Benchmark
XL-Sum (Hasan et al., Findings of ACL 2021) is a comprehensive cross-lingual summarization benchmark comprising 1.35 million professionally written news articles and human summaries sourced from BBC News across 44 languages (including Hindi, Bengali, Nepali, Telugu, Tamil, Urdu, and English). Unlike heuristic extractive datasets (such as CNN/DailyMail), XL-Sum summaries are professionally written abstractive summaries characterized by high compression ratios and substantial novel n-gram density.

### Code Executed
```python
samples = []
with open('xlsum_hindi_sample.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line.strip()))

dataset_preview = []
for i, item in enumerate(samples[:3]):
    dataset_preview.append({
        'Sample Index': i + 1,
        'Article Title': item['title'],
        'Text Length (chars)': len(item['text']),
        'Summary Length (chars)': len(item['summary'])
    })

df_preview = pd.DataFrame(dataset_preview)
print('=== XL-Sum Hindi Benchmark Dataset Preview ===')
print(df_preview)

test_sample = samples[0]
source_article = test_sample['text'][:600]
reference_summary = test_sample['summary']

print('--- Target Evaluation Sample ---')
print('Article Snippet:')
print(source_article[:250])
print('Human Reference Summary:')
print(reference_summary)
```

### Line-by-Line & Syntax Explanation
1. `with open('xlsum_hindi_sample.jsonl', ...)`: Loads genuine BBC Hindi news articles stored in UTF-8 JSON Lines format.
2. `dataset_preview`: Tabulates sample indices, article headlines, character lengths of original articles, and character lengths of human reference summaries.
3. `source_article`: Extracts the leading 600 characters of the target evaluation article (discussing the court order regarding the Narco and brain mapping tests of Sadhvi Pragya Singh Thakur in Nashik).
4. `reference_summary`: Extracts the ground truth summary written by professional BBC Hindi journalists:
   *"मालेगाँव धमाकों के सिलसिले में गिरफ़्तार साध्वी प्रज्ञा सिंह ठाकुर को अब पॉलीग्राफ़, ब्रेन मैपिंग और नॉर्को टेस्ट से गुज़रना होगा. ये सारे टेस्ट झूठ पकड़ने के लिए किए जाते हैं."*

---

## 3. Model 1: Multilingual Text-to-Text Transfer Transformer (mT5)

### Theoretical Background & Architecture
mT5 (Xue et al., NAACL 2021) is the multilingual variant of Google's T5 (Text-to-Text Transfer Transformer).

#### 1. Architecture Details
- **Parameter Scale**: `google/mt5-small` contains approximately 300 million parameters.
- **Unified Text-to-Text Framework**: Every NLP task (translation, classification, summarization) is framed as consuming a string prefix and outputting a target string.
- **Self-Attention with Relative Position Bias**: Instead of fixed sinusoidal or learned absolute position vectors, T5 learns a scalar bias $b_{i, j}$ added directly to attention logits:
$$\text{Attention Logit}(i, j) = \frac{q_i k_j^\top}{\sqrt{d_k}} + b_{i, j}$$
- **Pre-training Corpus**: Pretrained on mC4 (Multilingual Colossal Clean Crawled Corpus) spanning 101 languages and over 6.3 trillion tokens.
- **Span Corruption Objective**: During pre-training, random spans of tokens (average length 3) are masked with unique sentinel tokens (`<extra_id_0>`, `<extra_id_1>`), and the decoder predicts the masked spans.

### Code Executed
```python
mt5_model_id = 'google/mt5-small'
mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_model_id)
mt5_model = AutoModelForSeq2SeqLM.from_pretrained(mt5_model_id)

mt5_input_text = 'summarize: ' + source_article
mt5_inputs = mt5_tokenizer(mt5_input_text, return_tensors='pt', truncation=True, max_length=512)

with torch.no_grad():
    mt5_generated_tokens = mt5_model.generate(
        **mt5_inputs,
        max_length=80,
        num_beams=2,
        length_penalty=1.0,
        early_stopping=True
    )

mt5_summary = mt5_tokenizer.decode(mt5_generated_tokens[0], skip_special_tokens=True).strip()

print('=== Model 1: mT5 Generated Summary ===')
print(mt5_summary)
```

### Line-by-Line & Syntax Explanation
1. `AutoTokenizer.from_pretrained('google/mt5-small')`: Loads the multilingual SentencePiece tokenizer operating on raw character streams.
2. `mt5_input_text = 'summarize: ' + source_article`: Prepends the canonical task prefix instructing the text-to-text model to perform summarization.
3. `mt5_model.generate(...)`: Executes beam search decoding with beam width 2, maximum sequence length 80, and early stopping upon emitting end-of-sequence (`</s>`).
4. `mt5_tokenizer.decode(..., skip_special_tokens=True)`: Reconstructs readable Unicode text from generated subword token IDs.

---

## 4. Model 2: Multilingual Bidirectional and Auto-Regressive Transformers (mBART)

### Theoretical Background & Architecture
mBART (Liu et al., TACL 2020) is a multilingual sequence-to-sequence denoising autoencoder. Here we deploy `ai4bharat/IndicBART-XLSum` (Kakwani et al., EMNLP 2020 / Dabre et al., Findings of EMNLP 2022).

#### 1. Architecture Details
- **Backbone**: Standard sequence-to-sequence Transformer with a bidirectional encoder (like BERT) and an autoregressive left-to-right causal decoder (like GPT).
- **Denoising Pre-training Tasks**:
  1. *Token Masking*: Random spans of tokens are replaced with a single `<mask>` token (forcing bidirectional contextualization).
  2. *Sentence Permutation*: Sentences within a document are randomly shuffled (forcing the model to learn discourse coherence).
- **Indic Pre-training & XL-Sum Fine-Tuning**: `IndicBART-XLSum` is specifically pre-trained on 12 major Indian languages (including Hindi) using ALBERT-style shared parameter layers across depth to remain compact (244M parameters), followed by supervised fine-tuning directly on XL-Sum summarization datasets.

### Code Executed
```python
mbart_model_id = 'ai4bharat/IndicBART-XLSum'
mbart_tokenizer = AutoTokenizer.from_pretrained(mbart_model_id)
mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_id)

mbart_inputs = mbart_tokenizer(source_article, return_tensors='pt', truncation=True, max_length=512)

with torch.no_grad():
    mbart_generated_tokens = mbart_model.generate(
        **mbart_inputs,
        max_length=80,
        num_beams=2,
        length_penalty=1.0,
        early_stopping=True
    )

mbart_summary = mbart_tokenizer.decode(mbart_generated_tokens[0], skip_special_tokens=True).strip()

print('=== Model 2: mBART (IndicBART-XLSum) Generated Summary ===')
print(mbart_summary)
```

### Empirical Summary Generated
```
साध्वी प्रज्ञा सिंह ठाकुर को तीन नवंबर तक पुलिस हिरासत में रखने का फ़ैसला
```
*(Translation: Decision to keep Sadhvi Pragya Singh Thakur in police custody until November 3)*
- Notice that the model produced a concise, grammatically correct, highly fluent Hindi headline summary capturing the exact subject ("साध्वी प्रज्ञा सिंह ठाकुर") and the core judicial action ("पुलिस हिरासत में रखने का फ़ैसला").

---

## 5. Automatic Evaluation Metrics (ROUGE-1, ROUGE-2, ROUGE-L)

### Theoretical Background & Mathematical Formulations
ROUGE (Recall-Oriented Understudy for Gisting Evaluation; Lin, 2004) measures n-gram and sequence overlap between a candidate summary $C$ and reference summary $R$.

#### 1. ROUGE-N Formulation
$$\text{ROUGE-N Recall} = \frac{\sum_{S \in \{R\}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \{R\}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$
$$\text{ROUGE-N Precision} = \frac{\sum_{S \in \{C\}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \{C\}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$
$$\text{ROUGE-N } F_1 = \frac{2 \cdot \text{Recall} \cdot \text{Precision}}{\text{Recall} + \text{Precision}}$$
- **ROUGE-1**: Evaluates unigram (single-word) overlap, measuring information coverage.
- **ROUGE-2**: Evaluates bigram (word-pair) overlap, measuring local fluency and phrasal coherence.

#### 2. ROUGE-L (Longest Common Subsequence)
ROUGE-L uses the Longest Common Subsequence (LCS) between candidate string $C$ of length $m$ and reference string $R$ of length $n$:
$$\text{Recall}_L = \frac{\text{LCS}(C, R)}{n}, \quad \text{Precision}_L = \frac{\text{LCS}(C, R)}{m}$$
$$F_L = \frac{(1 + \beta^2) \text{Recall}_L \text{Precision}_L}{\text{Recall}_L + \beta^2 \text{Precision}_L}$$
ROUGE-L does not require consecutive matches; it rewards in-sequence words matching sentence-level structure.

#### 3. Why Standard English ROUGE Tokenizers Fail on Multilingual Data
Standard implementations of ROUGE (including Google Research `rouge_score`) default to an ASCII alphanumeric regular expression:
`re.sub(r'[^a-z0-9]+', ' ', text.lower())`
Because Hindi (Devanagari Unicode block `\u0900-\u097F`) contains non-ASCII characters, the default English regex strips every single Hindi character, resulting in empty strings `""` and returning false 0.0 scores! Providing a `MultilingualWhitespaceTokenizer` preserves Devanagari words intact.

### Code Executed
```python
class MultilingualWhitespaceTokenizer:
    def tokenize(self, text):
        return text.strip().split()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=MultilingualWhitespaceTokenizer())

rouge_scores_mt5 = scorer.score(reference_summary, mt5_summary)
rouge_scores_mbart = scorer.score(reference_summary, mbart_summary)

rouge_comparison = [
    {
        'Model Architecture': 'mT5 (google/mt5-small)',
        'ROUGE-1 F1': round(float(rouge_scores_mt5['rouge1'].fmeasure), 4),
        'ROUGE-2 F1': round(float(rouge_scores_mt5['rouge2'].fmeasure), 4),
        'ROUGE-L F1': round(float(rouge_scores_mt5['rougeL'].fmeasure), 4)
    },
    {
        'Model Architecture': 'mBART (IndicBART-XLSum)',
        'ROUGE-1 F1': round(float(rouge_scores_mbart['rouge1'].fmeasure), 4),
        'ROUGE-2 F1': round(float(rouge_scores_mbart['rouge2'].fmeasure), 4),
        'ROUGE-L F1': round(float(rouge_scores_mbart['rougeL'].fmeasure), 4)
    }
]

df_rouge = pd.DataFrame(rouge_comparison)
print('=== ROUGE Evaluation Results ===')
print(df_rouge)
```

### ROUGE Benchmark Output
| Model Architecture | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
| :--- | :---: | :---: | :---: |
| **mT5 (`google/mt5-small`)** | 0.0000 | 0.0000 | 0.0000 |
| **mBART (`ai4bharat/IndicBART-XLSum`)** | **0.2609** | **0.1818** | **0.2174** |

---

## 6. Semantic Similarity Evaluation via Multilingual BERT (mBERT)

### Theoretical Background & Mathematical Foundations
Surface-level n-gram overlap metrics (like ROUGE) have well-known limitations: they penalize valid synonyms, paraphrases, and alternate sentence structures. To capture semantic fidelity, we evaluate contextual representations using Multilingual BERT (`bert-base-multilingual-cased`), which is pretrained on Wikipedia across 104 languages.

#### Mathematical Formulation
Given candidate summary $S_{cand}$ and reference summary $S_{ref}$:
1. Pass tokens through the 12-layer mBERT encoder to extract hidden state sequences:
   $$\mathbf{H}_{cand} \in \mathbb{R}^{L_c \times 768}, \quad \mathbf{H}_{ref} \in \mathbb{R}^{L_r \times 768}$$
2. Compute global mean pooling over tokens to derive sentence representations $\mathbf{u}$ and $\mathbf{v}$:
   $$\mathbf{u} = \frac{1}{L_c} \sum_{i=1}^{L_c} \mathbf{h}_{cand}^{(i)}, \quad \mathbf{v} = \frac{1}{L_r} \sum_{j=1}^{L_r} \mathbf{h}_{ref}^{(j)}$$
3. Compute cosine similarity between the semantic vectors:
   $$\text{mBERT\_Score}(S_{cand}, S_{ref}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
Cosine similarity ranges from $-1.0$ to $+1.0$, where values approaching $1.0$ indicate close semantic alignment in high-dimensional multilingual embedding space.

### Code Executed
```python
mbert_model_id = 'google-bert/bert-base-multilingual-cased'
mbert_tokenizer = AutoTokenizer.from_pretrained(mbert_model_id)
mbert_model = AutoModel.from_pretrained(mbert_model_id)

def compute_mbert_similarity(hypothesis_text, reference_text):
    hypo_inputs = mbert_tokenizer(hypothesis_text, return_tensors='pt', truncation=True, max_length=128)
    ref_inputs = mbert_tokenizer(reference_text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        hypo_emb = mbert_model(**hypo_inputs).last_hidden_state.mean(dim=1)
        ref_emb = mbert_model(**ref_inputs).last_hidden_state.mean(dim=1)
    cosine_sim = F.cosine_similarity(hypo_emb, ref_emb).item()
    return round(float(cosine_sim), 4)

mbert_sim_mt5 = compute_mbert_similarity(mt5_summary, reference_summary)
mbert_sim_mbart = compute_mbert_similarity(mbart_summary, reference_summary)

print('=== mBERT Semantic Cosine Similarity ===')
print('mT5 vs Reference Summary  :', mbert_sim_mt5)
print('mBART vs Reference Summary:', mbert_sim_mbart)
```

### Empirical Semantic Scores
- **mT5 vs Reference Summary**: 0.3134
- **mBART vs Reference Summary**: **0.7926** (79.26% semantic similarity)

---

## 7. Comprehensive Comparative Performance Analysis

### Consolidated Performance Table
| Model Architecture | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | mBERT Score | Architectural Paradigm | Pre-training Objective |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **mT5 (`google/mt5-small`)** | 0.0000 | 0.0000 | 0.0000 | 0.3134 | Encoder-Decoder (T5) | Multilingual Span Corruption (mC4) |
| **mBART (`ai4bharat/IndicBART-XLSum`)** | **0.2609** | **0.1818** | **0.2174** | **0.7926** | Sequence-to-Sequence (BART) | Denoising Autoencoder + XL-Sum Fine-tuning |

### In-Depth Analysis of Results
1. **mBART Dominance**: The fine-tuned mBART model achieved **ROUGE-1 of 0.2609**, **ROUGE-2 of 0.1818**, and **mBERT similarity of 0.7926**. It produced an informative, grammatically cohesive abstractive summary capturing the essential news event.
2. **Pre-training vs. Task Fine-tuning**: Raw `google/mt5-small` was pretrained on unsupervised span masking across 101 languages, but was never fine-tuned on the supervised summarization task. Consequently, when prompted with `summarize: ...`, the small base checkpoint produced partial sentinel tokens (`<extra_id_0> :`) rather than generating a complete paragraph summary. This highlights the indispensable necessity of supervised downstream fine-tuning for seq2seq generation.
3. **mBERT vs. ROUGE Sensitivity**: While ROUGE-1 scored 0.2609 (reflecting that the model paraphrased using different vocabulary than the journalist), mBERT score reached **0.7926**. This confirms that mBERT semantic embeddings capture thematic and topical congruence even when exact lexical n-gram surface overlap is modest.

---

## 8. Comprehensive Viva Voce Preparation

### Q1: What is the fundamental difference between Extractive and Abstractive Summarization?
- **Answer**:
  - **Extractive Summarization**: Identifies, scores, and extracts verbatim sentences or phrases directly from the original source text without altering words (e.g. TextRank, Lead-3, LexRank).
  - **Abstractive Summarization**: Understands the semantic content and generates entirely new sentences, synonyms, and rephrasings that may not appear verbatim in the original document (acting like a human editor). It requires a sequence-to-sequence encoder-decoder model.

### Q2: What is the core architectural difference between mT5 and mBART?
- **Answer**:
  - **mT5**: Uses the T5 architecture with relative position biases in self-attention rather than fixed position embeddings. Pre-training uses span corruption (predicting random dropped spans of tokens marked by sentinel tokens `<extra_id_k>`).
  - **mBART**: Uses the BART architecture with standard sinusoidal/learned positional embeddings and a BERT-style bidirectional encoder paired with a GPT-style causal autoregressive decoder. Pre-training uses full-text denoising autoencoding (token masking + sentence permutation).

### Q3: Why did raw `google/mt5-small` fail to generate fluent summaries out-of-the-box?
- **Answer**: Base T5 models are pretrained strictly on masked span filling (infill prediction), where the model learns to fill small missing spans rather than generate long open-ended paragraphs. Without downstream fine-tuning on sequence-to-sequence summarization pairs $(X_{doc}, Y_{summary})$, the raw base model defaults to predicting sentinel tokens. In contrast, `IndicBART-XLSum` underwent supervised instruction fine-tuning on XL-Sum pairs, enabling fluent target generation.

### Q4: Why is cross-attention essential in Seq2Seq Transformer decoders?
- **Answer**: Self-attention in the decoder allows generated tokens to attend to previous tokens in the summary. However, to ground the generated summary in the source article, the decoder requires **Cross-Attention** layers where:
  - Queries ($Q$) come from the decoder's current hidden representation.
  - Keys ($K$) and Values ($V$) come from the encoder's output representations of the source document.
Without cross-attention, the decoder cannot access information from the original article.

### Q5: What is Teacher Forcing during sequence-to-sequence training?
- **Answer**: During training, the decoder generates tokens one step at a time. In **Teacher Forcing**, instead of feeding the model's own predicted token from step $t-1$ into step $t$, the ground truth target token from the reference summary is fed as input. This stabilizes training, prevents compounding errors, and accelerates convergence.

### Q6: What is Exposure Bias in autoregressive text generation?
- **Answer**: Exposure bias arises from the discrepancy between training and inference:
  - At training time, the model is fed ground-truth tokens at every step (Teacher Forcing).
  - At inference time, the model must feed its own generated output from step $t-1$ back into step $t$.
If the model makes a slight mistake at step $t$, the error propagates and compounds across subsequent tokens, potentially leading to degenerative repetitions or hallucinations.

### Q7: How does Beam Search improve over Greedy Decoding?
- **Answer**:
  - **Greedy Decoding**: Selects the single highest probability token at each time step: $w_t = \arg\max P(w \mid w_{<t})$. It is computationally fast ($O(1)$ branches) but easily falls into locally optimal traps that lead to poor long-term sequence likelihood.
  - **Beam Search**: Maintains a beam of the top-$k$ most probable hypotheses (e.g. $k=2$ or $k=4$) at each time step, pruning lower-probability paths. It explores a wider search tree and finds sequences with higher global probability.

### Q8: What does Length Penalty do in beam search decoding?
- **Answer**: Beam search evaluates sequences by accumulating log-probabilities: $\sum_{t=1}^T \ln P(w_t \mid w_{<t})$. Because log-probabilities are negative, longer sequences naturally accumulate lower total sums, creating an artificial bias toward overly short summaries. The length penalty normalizes the score by $\left(\frac{5 + T}{6}\right)^\alpha$, where $\alpha \in [0.6, 1.0]$ penalizes overly short completions.

### Q9: Why is mBERT similarity a valuable complement to ROUGE?
- **Answer**: ROUGE measures hard n-gram string matching. If a reference summary says "The court ordered the testing" and the model writes "The judicial magistrate mandated the examination", ROUGE scores them as 0 overlap despite identical semantics. mBERT maps both sentences into high-dimensional multilingual contextual vector space; cosine similarity captures semantic equivalence regardless of lexical variation.

### Q10: How does mBART handle multiple languages during generation?
- **Answer**: mBART appends a special language indicator token (e.g., `hi_IN` for Hindi, `en_XX` for English, `bn_IN` for Bengali) at the start and end of sequences. This signal conditions the decoder's internal cross-attention layers to project output embeddings into the phonology, vocabulary, and grammar of the target language.
