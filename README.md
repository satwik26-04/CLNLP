# Computational Linguistics and Natural Language Processing (CLNLP) Lab

## Student and Course Details

- **Name**: Satwik Shukla
- **SAP ID**: 500123679
- **Batch**: 1 AIML
- **Program**: B.Tech CSE
- **Faculty**: Prof. Laskar
- **Course**: Computational Linguistics and Natural Language Processing

---

## Lab Experiments Overview

### Experiment 1: Employee Data Analysis and Visualization
- **Experiment 1.1**: Calculate and display the average salary of each department using a bar chart.
- **Experiment 1.2**: Compute and visualize employee headcount per department.
- **Experiment 1.3**: Calculate the percentage of male and female employees and display using a pie chart.
- **Experiment 1.4**: Visualize salary distribution and frequency using a histogram with Kernel Density Estimation.
- **Experiment 1.5**: Analyze the relationship between years of experience and salary using a scatter plot.
- **Experiment 1.6**: Identify and display the top 10 highest-earning employees in the organization.
- **Experiment 1.7**: Determine the maximum salary achieved within each department.
- **Experiment 1.8**: Filter and list employees earning above the overall company average salary.
- **Experiment 1.9**: Compute the average years of work experience across each department.
- **Experiment 1.10**: Visualize workforce demographic spread and age distribution using a histogram.

### Experiment 2: Basic Text Preprocessing and Tokenization
- **Experiment 2.1**: Text cleaning pipeline performing lowercase conversion with uppercase count, punctuation removal with punctuation type metrics, number removal with digit counts, extra whitespace normalization, and final cleaned text generation.
- **Experiment 2.2**: Sentence and word tokenization implemented and compared across three methodologies:
  - Approach A: Pure Python with Regular Expressions
  - Approach B: NLTK Tokenizers (sent_tokenize, word_tokenize)
  - Approach C: spaCy Language Pipeline (en_core_web_sm)
- **Experiment 2.3**: Stop words detection, isolation, and removal using the NLTK English stopwords corpus to produce filtered alphanumeric content tokens.

### Experiment 3: Stemming, Lemmatization, and Regular Expressions
- **Experiment 3.1**: Rule-based suffix stripping using the Porter Stemmer algorithm on target inflectional and derivational words.
- **Experiment 3.2**: Morphological reduction using WordNet Lemmatizer comparing default noun lemmatization against POS-aware lemmatization for nouns, verbs, and adjectives.
- **Experiment 3.3**: Information extraction using Regular Expressions (Regex) to extract email addresses, URLs, mobile numbers, hashtags, and social media mentions from announcement text.

### Experiment 4: Term Frequency Analysis, Named Entity Recognition, and TF-IDF
- **Experiment 4.1**: Term Frequency analysis and Named Entity Recognition (NER) using spaCy, saving word frequencies to CSV and identifying the top 10 most frequent terms.
- **Experiment 4.2**: Term Frequency analysis using pure Python without external toolkits, exporting word frequencies to CSV and identifying top 10 terms.
- **Experiment 4.3**: End-to-end multi-document TF-IDF pipeline implemented completely from scratch without external NLP libraries, calculating TF, DF, IDF, and TF-IDF scores across three documents.

### Experiment 5: Subword Tokenization, Part-of-Speech Tagging, and Word2Vec Embeddings
- **Experiment 5.1**: Subword-level tokenization using Byte Pair Encoding (BPE), implemented both with a pretrained GPT-2 model and from scratch on `input_sub_word_data.txt`.
- **Experiment 5.2**: Subword-level tokenization using SentencePiece, implemented both with a pretrained T5 model and by training a custom BPE model from scratch on `input_sub_word_data.txt`.
- **Experiment 5.3**: Sentence-level Part-of-Speech (POS) tagging on a target sentence with human-readable tag descriptions using spaCy and NLTK.
- **Experiment 5.4**: Corpus-level Part-of-Speech tagging and frequency profiling on `input_sub_word_data.txt` using spaCy.
- **Experiment 5.5**: Training a Word2Vec Continuous Bag-of-Words (CBOW) model on `corpus.txt`, displaying dense vector representations, and finding the three most similar words.
- **Experiment 5.6**: Training a Word2Vec Skip-Gram model on `corpus.txt`, retrieving dense vectors for "language", and displaying the five most similar words.
- **Experiment 5.7**: Calculating cosine similarity between specific word pairs using the trained Word2Vec model.

### Experiment 6: GloVe Word Embeddings and Statistical N-gram Language Modeling
- **Experiment 6.1**: GloVe word representations trained from scratch on custom corpus using global co-occurrence matrix construction, harmonic context weighting, and AdaGrad optimization.
- **Experiment 6.2**: Semantic similarity analysis and analogy evaluation using pretrained GloVe vectors (`glove-wiki-gigaword-50`).
- **Experiment 6.3**: Statistical N-gram language modeling from scratch implementing Unigram, Bigram, and Trigram Maximum Likelihood Estimation (MLE) with conditional probabilities and greedy next-word prediction.

### Experiment 7: Text Classification with Classical ML, Recurrent Neural Networks, and Transformers
- **Experiment 7.1**: Baseline text sentiment classification using TF-IDF feature extraction with Logistic Regression on the Stanford IMDB movie review dataset.
- **Experiment 7.2**: Deep sequential modeling for sentiment classification using a Bidirectional LSTM (BiLSTM) network implemented in PyTorch.
- **Experiment 7.3**: Deep contextual sequence modeling using a custom Transformer Encoder architecture with multi-head self-attention and positional encodings in PyTorch.
- **Experiment 7.4**: Transfer learning and sequence classification using a pretrained BERT model (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`).
- **Experiment 7.5**: Comprehensive comparative performance evaluation across all four architectures reporting Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

### Experiment 8: Multilingual Abstractive Text Summarization and Evaluation
- **Experiment 8.1**: Abstractive text summarization for low-resource Indic languages (Hindi) using sequence-to-sequence transformer architectures: mT5 (`google/mt5-small`) and mBART (`ai4bharat/IndicBART-XLSum`).
- **Experiment 8.2**: Quantitative evaluation of generated abstractive summaries using n-gram overlap metrics (ROUGE-1, ROUGE-2, ROUGE-L) with a Devanagari-preserving multilingual tokenizer.
- **Experiment 8.3**: Semantic content preservation assessment using multilingual BERT (`google-bert/bert-base-multilingual-cased`) mean-pooled contextual embeddings and cosine similarity.

### Experiment 9: Multiclass Cyberbullying Detection in Code-Mixed Hinglish (`MC-Hinglish1.0`)
- **Experiment 9.1**: Exploratory data analysis and code-mixing profiling on `sahinurlaskar/MC-Hinglish1.0` (7,392 deduplicated training tweets and 1,000 test tweets across 7 classes), diagnosing class imbalance, elongation, and copy-paste outlier skew.
- **Experiment 9.2**: Hinglish orthographic normalization, elongation compression, spam token deduplication, and 5-fold Stratified CV ablation comparing Porter Stemming vs. WordNet Lemmatization alongside Chi-Square discriminative n-gram analysis.
- **Experiment 9.3**: Sparse hybrid Word (1-2 grams) + Character (`char_wb` 3-5 grams) TF-IDF classification using Complement Naive Bayes, Balanced Logistic Regression, and Balanced Linear SVM (`0.7019` Abusive-F1).
- **Experiment 9.4**: Grounded dense embeddings concatenating custom 100-d Skip-Gram FastText with pretrained 100-d GloVe vectors, comparing TF-IDF-weighted averaging against a Multi-Scale 1D-CNN with Global Max-Pooling (`68.80%` accuracy) and a Hybrid Sparse + Dense Margin Ensemble (`76.00%` accuracy, `0.7453` Macro-F1).

### Experiment 10: Low-Resource Neural Machine Translation (`WMT 2026 Indic MT: English to Bodo`)
- **Experiment 10.1**: Parallel corpus collection, Unicode NFC normalization, deduplication, and length-ratio filtering on the WMT 2026 Indic MT (`INDICNE-CORP 2.0`, Category II) English-to-Bodo dataset (`15,214` raw pairs -> `12,762` cleaned pairs: `11,762` train, `500` val, `500` test).
- **Experiment 10.2**: From-scratch Seq2Seq GRU encoder-decoder with Luong dot-product attention and Devanagari vowel-matra-preserving tokenization (`11.01` ChrF++).
- **Experiment 10.3**: Zero-shot vs. end-to-end fine-tuned `Helsinki-NLP/opus-mt-en-mul` (`MarianMT`) prompted with target token `>>brx<<`, boosting BLEU (`0.28 -> 23.69`), ChrF++ (`0.82 -> 51.93`), and mBERT-F1 (`55.04 -> 85.56`) while reducing TER (`113.35 -> 59.64`).
- **Experiment 10.4**: Quantitative benchmarking and visualization across `sacrebleu` BLEU, ChrF++, TER, and mBERT BERTScore-F1 with qualitative Devanagari Bodo translation inspection.

---

## Repository Structure

```text
CLNLP/
├── README.md
├── CLNLP_Lab_Report.pdf
├── upes logo.png
├── Experiment9.ipynb
├── Experiment9_Presentation.pptx
├── Experiment10.ipynb
├── Experiment10_Presentation.pptx
├── Experiment 1/
│   ├── employee_information_100.csv
│   ├── Experiment-1.docx
│   ├── Experiment_1.ipynb
│   └── WALKTHROUGH.md
├── Experiment 2/
│   ├── 2.1_text_data.txt
│   ├── 2.2_tokenization_data.txt
│   ├── 2.3_clean_data.txt
│   ├── Experimemt-2.docx
│   ├── Experiment_2.ipynb
│   └── WALKTHROUGH.md
├── Experiment 3/
│   ├── Experiment-3.docx
│   ├── Experiment_3.ipynb
│   └── WALKTHROUGH.md
├── Experiment 4/
│   ├── 4.1_4.2_input.txt
│   ├── 4.1_term_frequency_toolkit.csv
│   ├── 4.2_term_frequency_pure_python.csv
│   ├── Experiment-4_ Term Frequency and NER.docx
│   ├── Experiment_4.ipynb
│   └── WALKTHROUGH.md
├── Experiment 5/
│   ├── corpus.txt
│   ├── Experiment-5.docx
│   ├── input_sub_word_data.txt
│   ├── Experiment_5.ipynb
│   └── WALKTHROUGH.md
├── Experiment 6/
│   ├── Experiment-6.docx
│   ├── Experiment_6.ipynb
│   ├── glove_corpus.txt
│   └── WALKTHROUGH.md
├── Experiment 7/
│   ├── Experiment-7.docx
│   ├── Experiment_7.ipynb
│   └── WALKTHROUGH.md
└── Experiment 8/
    ├── Experiment-8.pdf
    ├── Experiment_8.ipynb
    ├── WALKTHROUGH.md
    └── xlsum_hindi_sample.jsonl
```

---

## Requirements and Setup

To run the notebooks locally, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy transformers tiktoken sentencepiece gensim torch rouge-score sacrebleu bert-score openpyxl
python -m spacy download en_core_web_sm
```
