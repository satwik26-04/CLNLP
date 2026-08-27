# Walkthrough: Experiment 3 - Stemming, Lemmatization, and Regular Expressions

This walkthrough explains the simple, direct implementation used in [Experiment_3.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%203/Experiment_3.ipynb). It is written to help you understand every line of code quickly and explain it confidently during your lab evaluation.

---

## 1. Setup and Imports

### Code
```python
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
```

### Explanation
- `import re`: Imports Python's built-in regular expression module for pattern matching.
- `import nltk`: Imports the Natural Language Toolkit library.
- `from nltk.stem import PorterStemmer, WordNetLemmatizer`: Imports the two core classes:
  - `PorterStemmer`: Implements the Porter stemming algorithm (chops off word suffixes).
  - `WordNetLemmatizer`: Uses the WordNet dictionary database to find canonical base words (lemmas).
- `stemmer = PorterStemmer()`: Creates an instance of the stemmer.
- `lemmatizer = WordNetLemmatizer()`: Creates an instance of the lemmatizer.

---

## Experiment 3.1: Stemming Using the Porter Stemmer

### Goal
Apply the Porter Stemmer to a list of words to remove affixes and see the resulting stems.

### Code
```python
words = ['playing', 'played', 'plays', 'studies', 'studying', 'connected', 'connection', 'computers']

for word in words:
    stem = stemmer.stem(word)
    print(f"{word:12} -> {stem}")
```

### Line-by-Line Explanation
1. `words = [...]`: A simple Python list containing the 8 target words specified in the experiment.
2. `for word in words:`: A standard `for` loop that iterates over each word in the list one at a time.
3. `stem = stemmer.stem(word)`: Calls the `.stem()` method on the word. The algorithm applies suffix-stripping rules (e.g., removing `-ing`, `-ed`, `-s`, `-tion`).
4. `print(f"{word:12} -> {stem}")`: Prints the original word formatted to 12 character spaces followed by an arrow `->` and the resulting stem.

### Results & What to Explain in Evaluation
- `playing`, `played`, `plays` $\to$ `play` (clean suffix removal).
- `studies`, `studying` $\to$ `studi` (notice `studi` is not a real English dictionary word; this demonstrates that stemming is rule-based and does not check a dictionary).
- `connected`, `connection` $\to$ `connect` (demonstrates that both inflections and derivations collapse to the same stem).
- `computers` $\to$ `comput` (chops off `-er` and `-s`).

---

## Experiment 3.2: Lemmatization Using WordNet Lemmatizer

### Goal
Reduce words to valid dictionary root words (lemmas), handling both regular words and irregular words (like `ran`, `went`, `better`) by supplying their Part-of-Speech (POS).

### Code
```python
words = ['cats', 'dogs', 'running', 'runs', 'ran', 'studies', 'studying', 'better', 'children', 'mice', 'went', 'ate', 'leaves', 'caring']

pos_tags = {
    'cats': 'n', 'dogs': 'n', 'running': 'v', 'runs': 'v', 'ran': 'v',
    'studies': 'v', 'studying': 'v', 'better': 'a', 'children': 'n',
    'mice': 'n', 'went': 'v', 'ate': 'v', 'leaves': 'n', 'caring': 'v'
}

for word in words:
    pos = pos_tags[word]
    lemma = lemmatizer.lemmatize(word, pos=pos)
    print(f"{word:12} (POS: {pos}) -> {lemma}")
```

### Line-by-Line Explanation
1. `words = [...]`: The 14 input words given in the experiment sheet.
2. `pos_tags = {...}`: A simple dictionary mapping each word to its Part of Speech tag:
   - `'n'` = Noun (e.g., `cats`, `dogs`, `children`, `mice`, `leaves`)
   - `'v'` = Verb (e.g., `running`, `runs`, `ran`, `studies`, `studying`, `went`, `ate`, `caring`)
   - `'a'` = Adjective (e.g., `better`)
3. `for word in words:`: Iterates through each word.
4. `pos = pos_tags[word]`: Looks up the correct POS tag from the dictionary.
5. `lemma = lemmatizer.lemmatize(word, pos=pos)`: Calls `.lemmatize()` with the word and its POS tag. It looks up the WordNet dictionary and transforms the word into its true base form.
6. `print(...)`: Prints the word, POS tag, and lemma.

### Key Evaluation Concept: Why do we need the `pos` parameter?
- If you call `lemmatize('ran')` without specifying `pos`, WordNet defaults to `pos='n'` (noun). Since `ran` is not a noun, it cannot find a noun base and returns `ran` unchanged.
- When you pass `pos='v'`, WordNet knows it is a verb and correctly returns `run`.
- Similarly: `went` with `pos='v'` $\to$ `go`, `ate` with `pos='v'` $\to$ `eat`, and `better` with `pos='a'` $\to$ `good`.

---

## Experiment 3.3: Information Extraction Using Regular Expressions

### Goal
Extract structured information (emails, URLs, phone numbers, hashtags, mentions) from unstructured text using Python's `re.findall()`.

### Code
```python
text = """Welcome to the Natural Language Processing workshop! For more information, visit https://www.nlpworkshop.com or www.python.org. You can contact the coordinator at nlpworkshop@gmail.com or support@python.org. For registration, call +91-9876543210 or 9123456789. Follow us on social media @NLPWorkshop and @PythonLearner. Share your experience using #NLP, #Python, and #MachineLearning. You can also visit https://github.com/NLPWorkshop for the latest updates."""

emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
urls = re.findall(r'https?://\S+|www\.\S+', text)
urls = [u.rstrip('.,') for u in urls]
mobiles = re.findall(r'\+91-\d{10}|\b\d{10}\b', text)
hashtags = re.findall(r'#\w+', text)
mentions = re.findall(r'(?<!\S)@\w+', text)

print("1. Emails:")
print(emails)

print("\n2. URLs:")
print(urls)

print("\n3. Mobile Numbers:")
print(mobiles)

print("\n4. Hashtags:")
print(hashtags)

print("\n5. Mentions:")
print(mentions)
```

### Pattern Explanation for Each Entity

1. **Emails (`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)**:
   - `[a-zA-Z0-9._%+-]+`: Username part before `@`.
   - `@`: Literal at-sign.
   - `[a-zA-Z0-9.-]+`: Domain name (e.g., `gmail`, `python`).
   - `\.`: Literal period.
   - `[a-zA-Z]{2,}`: Top-level domain with at least 2 letters (e.g., `com`, `org`).
   - Output: `['nlpworkshop@gmail.com', 'support@python.org']`

2. **URLs (`https?://\S+|www\.\S+`)**:
   - `https?://\S+`: Matches `http://` or `https://` followed by any non-whitespace characters (`\S+`).
   - `|`: OR operator.
   - `www\.\S+`: Matches URLs starting with `www.`.
   - `u.rstrip('.,')`: Cleans any trailing punctuation mark from the end of the URL.
   - Output: `['https://www.nlpworkshop.com', 'www.python.org', 'https://github.com/NLPWorkshop']`

3. **Mobile Numbers (`\+91-\d{10}|\b\d{10}\b`)**:
   - `\+91-\d{10}`: Matches numbers starting with country code `+91-` followed by 10 digits (`\d{10}`).
   - `|`: OR operator.
   - `\b\d{10}\b`: Matches standalone 10-digit numbers surrounded by word boundaries `\b`.
   - Output: `['+91-9876543210', '9123456789']`

4. **Hashtags (`#\w+`)**:
   - `#`: Matches the hashtag symbol `#`.
   - `\w+`: Matches one or more word characters (letters/numbers/underscores).
   - Output: `['#NLP', '#Python', '#MachineLearning']`

5. **Mentions (`(?<!\S)@\w+`)**:
   - `(?<!\S)`: Ensures `@` is preceded by whitespace (so it doesn't match `@gmail` inside an email).
   - `@\w+`: Matches `@` followed by the username.
   - Output: `['@NLPWorkshop', '@PythonLearner']`

---

## Quick Comparison Table for Viva / Evaluation

| Concept | Stemming (Porter) | Lemmatization (WordNet) |
| :--- | :--- | :--- |
| **Method** | Chops off suffixes using hardcoded rules | Looks up root words in a dictionary |
| **Output** | May produce non-words (`studi`, `comput`) | Always produces real dictionary words (`study`, `computer`) |
| **POS Sensitivity** | No POS awareness | Requires POS tag to resolve verbs/adjectives |
| **Irregular Words** | Fails (`ran` $\to$ `ran`, `went` $\to$ `went`) | Succeeds (`ran` $\to$ `run`, `went` $\to$ `go`, `better` $\to$ `good`) |
| **Speed** | Very fast | Slightly slower (due to dictionary lookups) |
