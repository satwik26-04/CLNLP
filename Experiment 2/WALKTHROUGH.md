# Walkthrough: Experiment 2 - Basic Text Preprocessing

This walkthrough provides a comprehensive, line-by-line explanation of all code, functions, and syntax implemented in [Experiment_2.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%202/Experiment_2.ipynb), covering basic text preprocessing, sentence and word tokenization (across Pure Python, NLTK, and spaCy), and stop word removal.

---

## 1. Setup & Library Imports

### Code Executed
```python
import string
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy

nlp = spacy.load('en_core_web_sm')
```

### Line-by-Line & Syntax Explanation
1. `import string`
   - **Syntax**: `import <module>`
   - **Explanation**: Imports Python's built-in `string` module, providing pre-defined constants such as `string.punctuation` containing all ASCII punctuation characters (`!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`).
2. `import re`
   - **Syntax**: `import <module>`
   - **Explanation**: Imports Python's regular expressions engine for pattern-based searching, splitting, and substring substitution.
3. `import nltk`
   - **Syntax**: `import <library>`
   - **Explanation**: Imports the **Natural Language Toolkit (NLTK)**, a cornerstone library for statistical and symbolic NLP in Python.
4. `from nltk.tokenize import word_tokenize, sent_tokenize`
   - **Syntax**: `from <package>.<module> import <function1>, <function2>`
   - **Explanation**: Selectively imports NLTK's pre-trained Punkt tokenizers: `sent_tokenize` (for sentence splitting) and `word_tokenize` (for token extraction).
5. `from nltk.corpus import stopwords`
   - **Syntax**: `from <package>.<module> import <corpus>`
   - **Explanation**: Imports the NLTK stop words corpus containing standardized lists of high-frequency grammatical words across multiple languages.
6. `import spacy`
   - **Syntax**: `import <library>`
   - **Explanation**: Imports **spaCy**, an industrial-strength NLP framework optimized for production speed and linguistic accuracy.
7. `nlp = spacy.load('en_core_web_sm')`
   - **Syntax**: `spacy.load(name: str) -> Language`
   - **Explanation**: Loads the small English language pipeline model (`en_core_web_sm`) containing tokenizer, tagger, parser, and named entity recognizer components.

---

## Experiment 2.1: Basic Text Preprocessing Pipeline

### Objective
Process raw text from `2.1_text_data.txt` through fundamental normalization stages:
1. Convert text to lowercase (and count uppercase characters).
2. Remove punctuation (and identify distinct punctuation marks).
3. Remove numbers (and count digit/number occurrences).
4. Remove extra whitespace (and measure characters removed).
5. Display the final cleaned text.

### Code Executed
```python
with open('2.1_text_data.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

uppercase_chars = [c for c in raw_text if c.isupper()]
print(f"Total Uppercase Characters: {len(uppercase_chars)}")
print(f"Unique Uppercase Characters: {sorted(list(set(uppercase_chars)))}")
text_lower = raw_text.lower()

puncts_found = [c for c in text_lower if c in string.punctuation]
unique_puncts = sorted(list(set(puncts_found)))
print(f"Total Punctuation Marks: {len(puncts_found)}")
print(f"Different Types of Punctuation ({len(unique_puncts)}): {unique_puncts}")
text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))

digits = [c for c in text_no_punct if c.isdigit()]
numbers = re.findall(r'\d+', text_no_punct)
print(f"Total Digits: {len(digits)}")
print(f"Numbers Found ({len(numbers)}): {numbers}")
text_no_nums = re.sub(r'\d+', '', text_no_punct)

extra_spaces = len(text_no_nums) - len(' '.join(text_no_nums.split()))
print(f"Extra Whitespace Characters Removed: {extra_spaces}")
cleaned_text = ' '.join(text_no_nums.split())

print("\n--- Cleaned Text ---")
print(cleaned_text)
```

### Line-by-Line & Syntax Explanation
1. `with open('2.1_text_data.txt', 'r', encoding='utf-8') as f:`
   - **Syntax**: `with open(file, mode, encoding) as <var>:`
   - **Explanation**: Context manager that opens `2.1_text_data.txt` in read mode (`'r'`) using UTF-8 encoding and ensures automatic file closure.
2. `raw_text = f.read()`
   - **Syntax**: `file.read() -> str`
   - **Explanation**: Reads the entire file content into a single string variable `raw_text`.
3. `uppercase_chars = [c for c in raw_text if c.isupper()]`
   - **Syntax**: List comprehension with `str.isupper() -> bool`
   - **Explanation**: Iterates through each character `c` in `raw_text`, collecting only characters where `isupper()` evaluates to `True`.
4. `print(f"Total Uppercase Characters: {len(uppercase_chars)}")`
   - Prints the total count of uppercase characters (32 characters).
5. `print(f"Unique Uppercase Characters: {sorted(list(set(uppercase_chars)))}")`
   - `set(...)`: Eliminates duplicate characters.
   - `sorted(...)`: Sorts the unique uppercase letters alphabetically (`['A', 'C', 'F', 'I', 'L', 'N', 'P', 'T']`).
6. `text_lower = raw_text.lower()`
   - **Syntax**: `str.lower() -> str`
   - **Explanation**: Converts all alphabetic characters to lowercase, ensuring uniform casing for downstream NLP tasks.
7. `puncts_found = [c for c in text_lower if c in string.punctuation]`
   - Extracts all punctuation characters present in the text by checking membership against `string.punctuation`.
8. `unique_puncts = sorted(list(set(puncts_found)))`
   - Gathers the distinct types of punctuation marks found (`['!', '(', ')', '+', ',', '.']`).
9. `text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))`
   - **Syntax**: `str.maketrans(x, y, z)` and `str.translate(table)`
   - **Explanation**: `str.maketrans('', '', string.punctuation)` builds a 1-to-1 translation mapping table where all punctuation characters in parameter `z` are marked for deletion. `str.translate()` applies this deletion table efficiently in C-level performance.
10. `digits = [c for c in text_no_punct if c.isdigit()]`
    - Counts individual numerical characters using `str.isdigit() -> bool` (13 digits).
11. `numbers = re.findall(r'\d+', text_no_punct)`
    - **Syntax**: `re.findall(pattern: str, string: str) -> list[str]`
    - **Explanation**: Uses regular expression `\d+` (one or more consecutive digit characters) to extract all complete numerical tokens (`['2026', '1000', '12345']`).
12. `text_no_nums = re.sub(r'\d+', '', text_no_punct)`
    - **Syntax**: `re.sub(pattern, replacement, string) -> str`
    - **Explanation**: Substitutes every sequence of digits with an empty string `''`, eliminating all numbers.
13. `extra_spaces = len(text_no_nums) - len(' '.join(text_no_nums.split()))`
    - Measures the exact number of redundant whitespace characters (tabs, repeated spaces, newlines) by subtracting the normalized string length from the current length (12 characters).
14. `cleaned_text = ' '.join(text_no_nums.split())`
    - **Syntax**: `str.split()` + `str.join(iterable)`
    - **Explanation**: `text_no_nums.split()` splits the text on any arbitrary whitespace sequence (collapsing consecutive spaces and newlines). `' '.join(...)` concatenates the resulting words with a single space delimiter.
15. `print(cleaned_text)`
    - Displays the final normalized text.

### Output Metrics Summary
- **Uppercase Characters Found**: 32 (`'A'`, `'C'`, `'F'`, `'I'`, `'L'`, `'N'`, `'P'`, `'T'`)
- **Punctuation Marks Found**: 27 marks across 6 unique types (`'!'`, `'('`, `')'`, `'+'`, `','`, `'.'`)
- **Numbers Found**: 3 number tokens (`2026`, `1000`, `12345`) comprising 13 individual digits
- **Extra Whitespace Removed**: 12 characters

---

## Experiment 2.2: Word & Sentence Tokenization (3 Approaches)

### Objective
Perform sentence tokenization (splitting text into distinct sentences) and word tokenization (splitting text into tokens/words) on `2.2_tokenization_data.txt` using three distinct approaches:
1. **Approach A**: Pure Python (Regular Expressions)
2. **Approach B**: NLTK (`sent_tokenize` & `word_tokenize`)
3. **Approach C**: spaCy (`Doc.sents` & `Token.text`)

---

### Approach A: Pure Python & Regex

#### Code Executed
```python
with open('2.2_tokenization_data.txt', 'r', encoding='utf-8') as f:
    text_2_2 = f.read().strip()

sentences_py = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_2_2) if s.strip()]
words_py = re.findall(r'\w+|[^\w\s]', text_2_2)

print(f"Sentences Count (Pure Python): {len(sentences_py)}")
for idx, sent in enumerate(sentences_py, 1):
    print(f"{idx}. {sent}")

print(f"\nTotal Word/Punctuation Tokens (Pure Python): {len(words_py)}")
print("Tokens Sample:", words_py[:20])
```

#### Line-by-Line & Syntax Explanation
1. `sentences_py = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_2_2) if s.strip()]`
   - **Syntax**: `re.split(pattern, string)` with positive lookbehind `(?<=[.!?])`
   - **Explanation**: The pattern `(?<=[.!?])\s+` matches one or more whitespace characters `\s+` *only when immediately preceded* by sentence-terminating punctuation (`.`, `!`, or `?`). This cleanly splits paragraphs into sentences without stripping the punctuation mark itself.
2. `words_py = re.findall(r'\w+|[^\w\s]', text_2_2)`
   - **Syntax**: `re.findall(pattern, string)`
   - **Explanation**: `\w+` matches alphanumeric word tokens, and `[^\w\s]` matches individual non-word, non-whitespace symbols (punctuation marks like parentheses and commas). This achieves comprehensive tokenization without third-party dependencies.
3. `for idx, sent in enumerate(sentences_py, 1):`
   - **Syntax**: `enumerate(iterable, start=1)`
   - **Explanation**: Yields a 1-indexed tuple `(idx, sent)` for formatted display.

---

### Approach B: NLTK Tokenization

#### Code Executed
```python
sentences_nltk = sent_tokenize(text_2_2)
words_nltk = word_tokenize(text_2_2)

print(f"Sentences Count (NLTK): {len(sentences_nltk)}")
for idx, sent in enumerate(sentences_nltk, 1):
    print(f"{idx}. {sent}")

print(f"\nTotal Word/Punctuation Tokens (NLTK): {len(words_nltk)}")
print("Tokens Sample:", words_nltk[:20])
```

#### Line-by-Line & Syntax Explanation
1. `sentences_nltk = sent_tokenize(text_2_2)`
   - **Syntax**: `nltk.tokenize.sent_tokenize(text: str, language: str = 'english') -> list[str]`
   - **Explanation**: Uses NLTK's pre-trained unsupervised Punkt sentence boundary detector, which handles abbreviations, periods, and initials accurately.
2. `words_nltk = word_tokenize(text_2_2)`
   - **Syntax**: `nltk.tokenize.word_tokenize(text: str, language: str = 'english') -> list[str]`
   - **Explanation**: Uses the Treebank tokenizer to split standard words, contractions (`don't` -> `do`, `n't`), and punctuation into distinct tokens.

---

### Approach C: spaCy Tokenization

#### Code Executed
```python
doc = nlp(text_2_2)
sentences_spacy = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
words_spacy = [token.text for token in doc if not token.is_space]

print(f"Sentences Count (spaCy): {len(sentences_spacy)}")
for idx, sent in enumerate(sentences_spacy, 1):
    print(f"{idx}. {sent}")

print(f"\nTotal Word/Punctuation Tokens (spaCy): {len(words_spacy)}")
print("Tokens Sample:", words_spacy[:20])
```

#### Line-by-Line & Syntax Explanation
1. `doc = nlp(text_2_2)`
   - **Syntax**: `Language.__call__(text: str) -> Doc`
   - **Explanation**: Passes the raw text through spaCy's linguistic pipeline to construct a `Doc` container containing rich linguistic annotations.
2. `sentences_spacy = [sent.text.strip() for sent in doc.sents if sent.text.strip()]`
   - **Syntax**: `Doc.sents -> Generator[Span]`
   - **Explanation**: Iterates over sentence spans generated by spaCy's dependency parser and sentence segmenter.
3. `words_spacy = [token.text for token in doc if not token.is_space]`
   - **Syntax**: `Token.text` and `Token.is_space -> bool`
   - **Explanation**: Iterates over tokens in the document, extracting the raw string while filtering out standalone whitespace tokens.

---

### Comparison of Approaches

| Metric | Pure Python (Regex) | NLTK | spaCy |
| :--- | :--- | :--- | :--- |
| **Sentence Count** | 9 sentences | 9 sentences | 9 sentences |
| **Word/Punct Tokens** | 112 tokens | 110 tokens | 112 tokens |
| **Dependencies** | Standard Library (`re`) | `nltk` + `punkt` data | `spacy` + `en_core_web_sm` |
| **Primary Strength** | Lightweight, zero dependencies | Standard academic benchmark | Fast, production-grade parser |

---

## Experiment 2.3: Stop Words Removal and Token Filtering

### Objective
Tokenize `2.3_clean_data.txt`, identify and isolate all stop words, and display the original vs filtered token collections.

### Code Executed
```python
with open('2.3_clean_data.txt', 'r', encoding='utf-8') as f:
    text_2_3 = f.read()

tokens_2_3 = word_tokenize(text_2_3)
stop_words = set(stopwords.words('english'))

extracted_stopwords = [w for w in tokens_2_3 if w.lower() in stop_words]
unique_stopwords = sorted(list(set(w.lower() for w in extracted_stopwords)))
filtered_tokens = [w for w in tokens_2_3 if w.lower() not in stop_words and w.isalnum()]

print(f"Total Original Tokens: {len(tokens_2_3)}")
print("Original Tokens:", tokens_2_3)

print(f"\nTotal Stop Words Identified: {len(extracted_stopwords)}")
print(f"Unique Stop Words ({len(unique_stopwords)}): {unique_stopwords}")
print("Stop Words Occurrences in Text:", extracted_stopwords)

print(f"\nTotal Filtered Tokens (Stop Words & Punctuation Removed): {len(filtered_tokens)}")
print("Filtered Tokens:", filtered_tokens)
```

### Line-by-Line & Syntax Explanation
1. `tokens_2_3 = word_tokenize(text_2_3)`
   - Splits the entire text into 138 original tokens (including punctuation and words).
2. `stop_words = set(stopwords.words('english'))`
   - **Syntax**: `set(iterable)` over `stopwords.words('english') -> list[str]`
   - **Explanation**: Converts the 179 standard English stop words from NLTK into a Python `set` for $O(1)$ constant-time lookup complexity.
3. `extracted_stopwords = [w for w in tokens_2_3 if w.lower() in stop_words]`
   - Collects every instance where a token's lowercase form matches a word in `stop_words`.
4. `unique_stopwords = sorted(list(set(w.lower() for w in extracted_stopwords)))`
   - Computes the alphabetized list of unique stop words present in the document (22 unique stop words).
5. `filtered_tokens = [w for w in tokens_2_3 if w.lower() not in stop_words and w.isalnum()]`
   - **Syntax**: `w.lower() not in stop_words and w.isalnum()`
   - **Explanation**: Retains only tokens that are **not** stop words AND are alphanumeric (`w.isalnum()` excludes punctuation like `.`, `,`, etc.).
6. Display results:
   - Prints the original tokens, isolated stop words, unique stop words set, and final filtered content tokens.

### Output Metrics Summary
- **Original Tokens Count**: 138 tokens
- **Total Stop Word Occurrences**: 57 stop words
- **Unique Stop Words (22)**: `['a', 'an', 'and', 'are', 'as', 'by', 'can', 'for', 'from', 'in', 'is', 'it', 'of', 'on', 'some', 'such', 'that', 'the', 'these', 'this', 'to', 'with']`
- **Filtered Content Tokens Count**: 56 tokens (`'Natural'`, `'Language'`, `'Processing'`, `'important'`, `'field'`, `'Artificial'`, `'Intelligence'`, `'chatbots'`, `'machine'`, `'translation'`, `'sentiment'`, `'analysis'`, etc.)

---

## Notebook Compliance Verification
- **Code comments**: `0` (Zero comments across all code cells).
- **Markdown cells**: Concise, structured headers for each sub-experiment.
- **Execution state**: All 14 cells executed with all tables, prints, and token lists rendered.
