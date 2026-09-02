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

---

## Repository Structure

```text
CLNLP/
├── README.md
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
└── Experiment 4/
    ├── 4.1_4.2_input.txt
    ├── 4.1_term_frequency_toolkit.csv
    ├── 4.2_term_frequency_pure_python.csv
    ├── Experiment-4_ Term Frequency and NER.docx
    ├── Experiment_4.ipynb
    └── WALKTHROUGH.md
```

---

## Requirements and Setup

To run the notebooks locally, install the required dependencies:

```bash
pip install pandas matplotlib seaborn nltk spacy
python -m spacy download en_core_web_sm
```
