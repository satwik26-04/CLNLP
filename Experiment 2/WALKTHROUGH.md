# Walkthrough: Experiment 2 - Basic Text Preprocessing

This walkthrough provides an exhaustive, line-by-line explanation of every code statement, Python function, library method, regular expression pattern, and data transformation implemented in [Experiment_2.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%202/Experiment_2.ipynb). The experiment covers three sub-experiments: basic text preprocessing (case normalization, punctuation removal, number removal, whitespace cleaning), sentence and word tokenization using three independent methodologies (Pure Python with regex, NLTK, and spaCy), and stop word detection, isolation, and removal.

---

## 1. Setup and Library Imports

### Purpose
Before performing any text processing, we must import the necessary Python standard library modules and third-party NLP frameworks. This cell loads everything required across all three sub-experiments so that subsequent cells can reference the libraries without re-importing.

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

### Line-by-Line Explanation

**Line 1: `import string`**
- **Syntax**: `import <module_name>`
- **What it does**: Imports Python's built-in `string` module from the standard library. This module does not need to be installed separately because it ships with every Python installation. The `string` module provides a collection of useful string constants and utility classes. In this experiment, we specifically use `string.punctuation`, which is a pre-defined constant string containing every ASCII punctuation character: `!"#$%&'()*+,-./:;<=>?@[\]^_{|}~`. This constant is 32 characters long and gives us a ready-made reference set of all standard punctuation marks, saving us from manually typing them out or potentially missing one. We use it later in Experiment 2.1 to identify which characters in our raw text are punctuation marks and to build a translation table that strips them out.

**Line 2: `import re`**
- **Syntax**: `import <module_name>`
- **What it does**: Imports Python's built-in `re` (regular expressions) module. Regular expressions are a powerful pattern-matching language that allows you to describe complex text patterns using special metacharacters. The `re` module provides functions like `re.findall()` (find all non-overlapping matches of a pattern in a string and return them as a list), `re.sub()` (find all matches and replace them with a specified string), and `re.split()` (split a string at every position where the pattern matches). We rely on this module heavily across all three sub-experiments: in 2.1 to extract and remove digit sequences, in 2.2 to split text into sentences using lookbehind assertions, and to tokenize words using alternation patterns.

**Line 3: `import nltk`**
- **Syntax**: `import <library_name>`
- **What it does**: Imports the Natural Language Toolkit (NLTK), which is the most widely-used Python library for teaching and prototyping Natural Language Processing tasks. NLTK was created by Steven Bird and Edward Loper at the University of Pennsylvania and provides interfaces to over 50 corpora and lexical resources, along with a suite of text processing libraries for tokenization, stemming, tagging, parsing, classification, and semantic reasoning. Importing the top-level `nltk` package makes all its submodules accessible. NLTK requires separate data downloads (like the Punkt tokenizer models and the stopwords corpus), which must be fetched using `nltk.download()` before first use.

**Line 4: `from nltk.tokenize import word_tokenize, sent_tokenize`**
- **Syntax**: `from <package>.<submodule> import <name1>, <name2>`
- **What it does**: This line selectively imports two specific tokenization functions from the `nltk.tokenize` submodule, making them available as direct function calls without needing the `nltk.tokenize.` prefix:
  - `sent_tokenize`: A sentence-level tokenizer that uses the pre-trained Punkt model (developed by Kiss and Strunk, 2006). The Punkt algorithm is an unsupervised, language-independent system that learns to identify sentence boundaries by analyzing a corpus. It handles edge cases like abbreviations (`Dr.`, `U.S.A.`), decimal numbers (`3.14`), and ellipses (`...`) far better than naive period-based splitting. Under the hood, it computes log-likelihood ratios to determine whether a period-terminated token is an abbreviation or an end-of-sentence marker.
  - `word_tokenize`: A word-level tokenizer that internally calls `sent_tokenize` first and then applies the Penn Treebank tokenization conventions to each sentence. It separates punctuation from adjacent words (e.g., `Intelligence.` becomes `['Intelligence', '.']`), splits contractions according to English morphological rules (e.g., `don't` becomes `['do', "n't"]`, `I'm` becomes `['I', "'m"]`), and handles special cases like quoted strings and possessives.

**Line 5: `from nltk.corpus import stopwords`**
- **Syntax**: `from <package>.<submodule> import <corpus_reader>`
- **What it does**: Imports the `stopwords` corpus reader from NLTK's corpus module. Stop words are extremely common, high-frequency words in a language that typically carry little semantic meaning on their own. Examples include articles (`a`, `an`, `the`), prepositions (`in`, `of`, `to`, `for`, `on`, `with`), conjunctions (`and`, `but`, `or`), pronouns (`it`, `this`, `that`), and auxiliary verbs (`is`, `are`, `was`, `were`). The NLTK stopwords corpus provides curated stop word lists for 23 languages. For English, it contains 179 words. We access them by calling `stopwords.words('english')`, which returns a Python list. These words are used in Experiment 2.3 to filter out non-informative tokens from the text.

**Line 6: `import spacy`**
- **Syntax**: `import <library_name>`
- **What it does**: Imports the spaCy library, which is an industrial-strength, production-grade NLP framework developed by Explosion AI. Unlike NLTK, which is designed primarily for education and research, spaCy is optimized for speed and real-world deployment. It processes text through a linguistic pipeline consisting of multiple components: a tokenizer (rule-based, using prefix/suffix/infix patterns and exception tables), a POS tagger (predicts part-of-speech tags), a dependency parser (builds syntactic parse trees showing how words relate to each other), a named entity recognizer (NER, identifies proper nouns like person names, organizations, and locations), and a lemmatizer. All of these components are packaged into trained pipeline models that can be loaded by name.

**Line 7: `nlp = spacy.load('en_core_web_sm')`**
- **Syntax**: `spacy.load(name: str) -> Language`
- **What it does**: Loads the pre-trained English language pipeline model named `en_core_web_sm` and assigns it to the variable `nlp`. The naming convention breaks down as: `en` = English language, `core` = general-purpose pipeline, `web` = trained on web text data (from the OntoNotes 5 corpus), `sm` = small model size (~12 MB). This is a convention in the spaCy community: the loaded pipeline object is always assigned to a variable called `nlp`, and when you call `nlp(text)`, it processes the text through all pipeline components and returns a `Doc` object containing tokens with rich linguistic annotations. The small model was chosen here because it is lightweight and sufficient for tokenization and sentence segmentation tasks. Larger models (`en_core_web_md`, `en_core_web_lg`, `en_core_web_trf`) include word vectors and transformer-based components for higher accuracy on NER and other tasks, but they are unnecessary for basic tokenization.

---

## Experiment 2.1: Basic Text Preprocessing Pipeline

### Background and Motivation
Text preprocessing is the foundational first stage in any NLP pipeline. Raw text collected from the real world (web pages, user messages, documents) contains noise: inconsistent capitalization, punctuation marks that are irrelevant for semantic analysis, embedded numbers that may or may not be meaningful, and irregular whitespace from formatting artifacts. Before feeding text into downstream NLP models (such as classifiers, clustering algorithms, or language models), we need to normalize it into a clean, consistent format. This experiment demonstrates the standard four-stage preprocessing pipeline and, critically, also measures the extent of noise found at each stage.

### Input Data
The input file [2.1_text_data.txt](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%202/2.1_text_data.txt) contains 5 paragraphs (9 lines including blank separator lines) of text about NLP and text preprocessing. The text deliberately includes mixed casing (`Natural`, `NLP`, `CAPITAL`), various punctuation marks (`(`, `)`, `,`, `.`, `!!!`, `+`), embedded numbers (`2026`, `1000`, `12345`), and irregular whitespace (double spaces, triple spaces, blank lines between paragraphs).

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

### Line-by-Line Explanation

**Line 1: `with open('2.1_text_data.txt', 'r', encoding='utf-8') as f:`**
- **Syntax**: `with open(filepath, mode, encoding=encoding) as variable:`
- **What it does**: Opens the file `2.1_text_data.txt` for reading. The `with` keyword creates a context manager, which is a Python construct that automatically handles resource cleanup. When the indented block under the `with` statement finishes executing (whether normally or due to an exception), the file object `f` is automatically closed, releasing the operating system file handle. This prevents resource leaks. The arguments are:
  - `'2.1_text_data.txt'`: The relative path to the input file (relative to the notebook's working directory, which is the `Experiment 2` folder).
  - `'r'`: The file mode. `'r'` stands for read-only mode, meaning we can read the file's contents but cannot write to or modify it.
  - `encoding='utf-8'`: Specifies the character encoding to use when interpreting the raw bytes from the file. UTF-8 is the dominant encoding on the modern web and can represent every character in the Unicode standard (including accented characters, CJK ideographs, emojis, etc.). Specifying it explicitly prevents encoding errors on systems where the default encoding might differ.

**Line 2: `raw_text = f.read()`**
- **Syntax**: `file_object.read() -> str`
- **What it does**: Reads the entire contents of the opened file as a single string and assigns it to the variable `raw_text`. The `read()` method without arguments reads from the current file position to the end of the file. This means `raw_text` now contains the complete text including all newline characters (`\n` or `\r\n` on Windows), all whitespace, all punctuation, and all casing exactly as written in the file. For this particular file, the string is 795 bytes long and contains 9 lines of text.

**Line 3: `uppercase_chars = [c for c in raw_text if c.isupper()]`**
- **Syntax**: `[expression for variable in iterable if condition]` (list comprehension)
- **What it does**: This is a list comprehension, which is a concise Python syntax for building a new list by iterating over an existing iterable and optionally filtering elements. Here it iterates over every single character `c` in the string `raw_text`, one character at a time. For each character, it calls the `str.isupper()` method, which returns `True` if the character is an uppercase alphabetic letter (A-Z) and `False` for everything else (lowercase letters, digits, spaces, punctuation, newlines). Only characters where `c.isupper()` returns `True` are included in the resulting list. The result is a list of individual uppercase characters found in the text, including duplicates. For example, if `N` appears 6 times in the text as an uppercase letter, it will appear 6 times in this list.

**Line 4: `print(f"Total Uppercase Characters: {len(uppercase_chars)}")`**
- **Syntax**: `print(f"...{expression}...")` (f-string formatted print)
- **What it does**: Prints the total count of uppercase characters found. The `f"..."` syntax is a Python f-string (formatted string literal), where any expression inside curly braces `{}` is evaluated and its result is interpolated into the string. `len(uppercase_chars)` calls the built-in `len()` function, which returns the number of items in the list. The output is `Total Uppercase Characters: 32`, meaning 32 individual uppercase letter occurrences were found across the entire text.

**Line 5: `print(f"Unique Uppercase Characters: {sorted(list(set(uppercase_chars)))}")`**
- **Syntax**: Nested function calls `sorted(list(set(...)))`
- **What it does**: This line chains three operations to find and display the distinct uppercase letters in alphabetical order:
  1. `set(uppercase_chars)`: Converts the list into a Python `set`. A set is an unordered collection of unique elements. If the list contained `['N', 'L', 'P', 'N', 'A', 'I', 'N', ...]`, the set would collapse it to `{'N', 'L', 'P', 'A', 'I', ...}`, eliminating all duplicates.
  2. `list(...)`: Converts the set back into a list, because `sorted()` returns a list and sets have no inherent ordering.
  3. `sorted(...)`: Sorts the list in ascending alphabetical order.
  - The output is `['A', 'C', 'F', 'I', 'L', 'N', 'P', 'T']`, meaning 8 distinct uppercase letters appear in the text. These correspond to words like **N**atural **L**anguage **P**rocessing, **NLP**, **A**rtificial **I**ntelligence, **T**ext, **C**APITAL, **F**or.

**Line 6: `text_lower = raw_text.lower()`**
- **Syntax**: `str.lower() -> str`
- **What it does**: Creates a new string where every uppercase alphabetic character has been converted to its lowercase equivalent, and assigns it to `text_lower`. The `str.lower()` method does not modify the original string (strings in Python are immutable), it returns a brand new string. All non-alphabetic characters (digits, spaces, punctuation, newlines) remain completely unchanged. For example, `"Natural Language Processing (NLP)"` becomes `"natural language processing (nlp)"`. This step is essential for text normalization because without it, a downstream NLP system would treat `Natural` and `natural` as two completely different tokens, inflating the vocabulary size and fragmenting word frequency counts.

**Line 7: `puncts_found = [c for c in text_lower if c in string.punctuation]`**
- **Syntax**: List comprehension with membership test `c in string.punctuation`
- **What it does**: Iterates over every character in the lowercased text and checks whether that character exists within the `string.punctuation` constant string. The expression `c in string.punctuation` performs a substring membership test: it checks if the single character `c` is present anywhere in the 32-character punctuation string. If it is, the character is included in the resulting list. This gives us a complete inventory of every punctuation mark occurrence in the text, including duplicates. For instance, if `!` appears 6 times (as in `!!!` repeated twice), all 6 exclamation marks are captured individually.

**Line 8: `unique_puncts = sorted(list(set(puncts_found)))`**
- **What it does**: Applies the same deduplication-and-sort chain used earlier for uppercase characters, but now on punctuation marks. The result is a sorted list of the distinct punctuation types found in the text: `['!', '(', ')', '+', ',', '.']`. This tells us the text contains 6 different kinds of punctuation marks.

**Line 9: `print(f"Total Punctuation Marks: {len(puncts_found)}")`**
- **What it does**: Prints the total number of individual punctuation character occurrences. The output is `Total Punctuation Marks: 27`, meaning 27 punctuation characters were scattered throughout the text.

**Line 10: `print(f"Different Types of Punctuation ({len(unique_puncts)}): {unique_puncts}")`**
- **What it does**: Prints how many distinct types of punctuation were found (6 types) and lists them. The parenthetical count and the list are both interpolated into the f-string.

**Line 11: `text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))`**
- **Syntax**: `str.maketrans(x, y, z)` creates a translation table; `str.translate(table)` applies it
- **What it does**: This is the most efficient way in Python to delete a set of characters from a string. It works in two steps:
  1. `str.maketrans('', '', string.punctuation)`: The static method `str.maketrans()` creates a translation table (a dictionary mapping Unicode code points to their replacements). When called with three arguments, the first two arguments specify characters to map from and to (here both empty strings, meaning no character-to-character replacements), and the third argument specifies characters to delete entirely. So this creates a table that maps every punctuation character's Unicode code point to `None`, which signals deletion.
  2. `text_lower.translate(table)`: Applies the translation table to the string. For each character in `text_lower`, Python looks up its Unicode code point in the table. If it maps to `None`, the character is deleted from the output. If it is not in the table, the character passes through unchanged. This operation is implemented in C within CPython and executes in a single pass over the string, making it significantly faster than alternatives like `re.sub()` or iterative `str.replace()` calls.
  - After this line, `text_no_punct` contains the text with all 27 punctuation marks stripped out. For example, `"natural language processing (nlp)"` becomes `"natural language processing nlp"`.

**Line 12: `digits = [c for c in text_no_punct if c.isdigit()]`**
- **Syntax**: List comprehension with `str.isdigit() -> bool`
- **What it does**: Iterates over every character in the punctuation-free text and collects those where `c.isdigit()` returns `True`. The `str.isdigit()` method returns `True` for Unicode characters that are classified as decimal digits (0-9 in ASCII, plus digit characters from other scripts like Arabic-Indic numerals). This list captures every individual digit character, including duplicates. The result contains 13 individual digit characters (the digits that make up `2026`, `1000`, and `12345`).

**Line 13: `numbers = re.findall(r'\d+', text_no_punct)`**
- **Syntax**: `re.findall(pattern: str, string: str) -> list[str]`
- **What it does**: Uses the regular expression engine to find all non-overlapping matches of the pattern `\d+` in the string. Breaking down the pattern:
  - `\d`: A regex metacharacter that matches any single decimal digit character (equivalent to the character class `[0-9]`).
  - `+`: A quantifier meaning "one or more of the preceding element." So `\d+` matches a sequence of one or more consecutive digits.
  - `re.findall()` scans the string from left to right, finds every non-overlapping substring that matches the pattern, and returns them all as a list of strings.
  - The result is `['2026', '1000', '12345']`, which are the three complete number tokens embedded in the text. This is more semantically meaningful than the individual digit list because it preserves the numbers as coherent units. The distinction matters: we have 13 individual digits but only 3 actual numbers.

**Line 14: `text_no_nums = re.sub(r'\d+', '', text_no_punct)`**
- **Syntax**: `re.sub(pattern: str, replacement: str, string: str) -> str`
- **What it does**: Finds every substring matching the pattern `\d+` (one or more consecutive digits) and replaces it with the empty string `''`, effectively deleting all numbers from the text. The function returns a new string with all replacements applied. After this line, all three numbers (`2026`, `1000`, `12345`) have been removed, but the spaces that surrounded them remain, which is why we need the whitespace normalization step next.

**Line 15: `extra_spaces = len(text_no_nums) - len(' '.join(text_no_nums.split()))`**
- **Syntax**: Arithmetic on string lengths using `str.split()` and `str.join()`
- **What it does**: Calculates the exact number of extra (redundant) whitespace characters in the text by comparing two measurements:
  1. `len(text_no_nums)`: The current length of the text, including all whitespace as-is (multiple consecutive spaces, tab characters, newline characters `\n`, carriage returns `\r`).
  2. `len(' '.join(text_no_nums.split()))`: The length the text would have if all whitespace were normalized. `text_no_nums.split()` called with no arguments splits the string at every whitespace boundary (spaces, tabs, newlines, carriage returns) regardless of how many consecutive whitespace characters there are, and also strips leading and trailing whitespace. It returns a list of non-whitespace tokens. `' '.join(...)` then reassembles those tokens into a single string with exactly one space between each pair of adjacent tokens.
  - The difference between these two lengths is exactly the number of extra whitespace characters. The output is `12`, meaning 12 characters of unnecessary whitespace (extra spaces, blank lines) were present.

**Line 16: `cleaned_text = ' '.join(text_no_nums.split())`**
- **Syntax**: `str.join(iterable)` with `str.split()`
- **What it does**: Performs the actual whitespace normalization. As described above, `text_no_nums.split()` breaks the text into a list of words by splitting on any whitespace sequence, and `' '.join(...)` reassembles them with single spaces. The result is a single-line string with uniform single-space separators and no leading or trailing whitespace. This is the final cleaned text.

**Lines 17-18: Print the cleaned text**
- **What it does**: Displays the final preprocessed result. The output is a single continuous line of lowercase text with no punctuation, no numbers, and no extra whitespace.

### Output Metrics Summary
| Preprocessing Stage | Metric | Value | Details |
| :--- | :--- | :--- | :--- |
| **Case Conversion** | Uppercase characters found | 32 | 8 distinct uppercase letters: A, C, F, I, L, N, P, T |
| **Punctuation Removal** | Punctuation marks found | 27 | 6 distinct types: `!` `(` `)` `+` `,` `.` |
| **Number Removal** | Individual digits | 13 | Comprising 3 number tokens: `2026`, `1000`, `12345` |
| **Whitespace Normalization** | Extra whitespace removed | 12 characters | From double/triple spaces and blank paragraph separators |

---

## Experiment 2.2: Word and Sentence Tokenization (3 Approaches)

### Background and Motivation
Tokenization is the process of breaking a continuous stream of text into discrete, meaningful units called tokens. It is universally the very first computational step in any NLP pipeline. There are two levels of tokenization:
- **Sentence tokenization** (also called sentence segmentation or sentence boundary detection): Splitting a paragraph or document into individual sentences.
- **Word tokenization**: Splitting a sentence (or entire text) into individual words, numbers, and punctuation marks.

This might seem trivially simple (just split on spaces and periods), but real-world text contains many ambiguities: periods can indicate abbreviations (`Dr.`, `U.S.A.`), decimal numbers (`3.14`), or sentence endings. Contractions (`don't`, `I'm`), hyphenated compounds (`state-of-the-art`), and possessives (`children's`) all require careful handling. This experiment demonstrates and compares three different approaches to tokenization on the same input text to highlight the differences in methodology, accuracy, and token count.

### Input Data
The input file [2.2_tokenization_data.txt](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%202/2.2_tokenization_data.txt) contains 3 paragraphs (5 lines including blank separators) with 9 sentences about NLP and tokenization. The text uses standard English prose with parenthetical expressions `(NLP)`, commas, periods, and no unusual edge cases, making it a good baseline for comparing tokenizer behavior.

---

### Approach A: Tokenization Without External Libraries (Pure Python and Regular Expressions)

#### Purpose
Demonstrate that basic tokenization can be achieved using only Python's built-in `re` module, without any third-party NLP library. This approach is useful in constrained environments where installing NLTK or spaCy is not feasible, and it helps build understanding of what the higher-level libraries are doing under the hood.

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

#### Line-by-Line Explanation

**Line 1: `with open('2.2_tokenization_data.txt', 'r', encoding='utf-8') as f:`**
- **What it does**: Opens the second data file in read mode with UTF-8 encoding using a context manager, identical in mechanism to the file opening in Experiment 2.1.

**Line 2: `text_2_2 = f.read().strip()`**
- **Syntax**: `file.read().strip() -> str`
- **What it does**: Reads the entire file content and then immediately calls `str.strip()`, which removes any leading and trailing whitespace characters (including `\n`, `\r\n`, `\t`, and spaces) from both ends of the string. This is done because text files often have a trailing newline at the end, and we want a clean string that starts and ends with actual text content. The result is assigned to `text_2_2`.

**Line 3: `sentences_py = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_2_2) if s.strip()]`**
- **Syntax**: List comprehension over `re.split(pattern, string)`
- **What it does**: This is the sentence tokenization step. Let us break down the regular expression pattern `(?<=[.!?])\s+` piece by piece:
  - `(?<=...)`: This is a **positive lookbehind assertion**. It is a zero-width assertion, meaning it checks a condition about the characters immediately before the current position in the string, but it does not consume any characters. The regex engine verifies that the characters preceding the current match position satisfy the pattern inside the lookbehind parentheses.
  - `[.!?]`: A **character class** that matches exactly one character that is either a period `.`, an exclamation mark `!`, or a question mark `?`. These are the three standard English sentence-terminating punctuation marks.
  - `\s+`: Matches one or more whitespace characters (spaces, tabs, newlines). The `+` quantifier means "one or more."
  - Combined, the pattern `(?<=[.!?])\s+` means: "match one or more whitespace characters, but only at positions where the character immediately before the whitespace is a period, exclamation mark, or question mark." The key insight is that the lookbehind is zero-width: the punctuation mark is NOT consumed by the match. So when `re.split()` splits the string at these positions, the sentence-ending punctuation stays attached to the preceding sentence.
  - `re.split()` returns a list of substrings. For example, splitting `"Hello. World"` on the pattern would yield `["Hello.", "World"]`.
  - The list comprehension `[s.strip() for s in ... if s.strip()]` applies `str.strip()` to each resulting substring (removing any residual leading/trailing whitespace) and filters out any empty strings (the `if s.strip()` condition ensures that blank results from consecutive delimiters are excluded).
  - **Limitations of this approach**: This naive regex method will incorrectly split on abbreviations like `Dr. Smith` or `U.S.A. is`, because it treats every period followed by whitespace as a sentence boundary. It also struggles with ellipses (`...`) and quoted speech. The NLTK Punkt algorithm and spaCy's parser handle these cases far more gracefully.

**Line 4: `words_py = re.findall(r'\w+|[^\w\s]', text_2_2)`**
- **Syntax**: `re.findall(pattern, string) -> list[str]`
- **What it does**: This is the word tokenization step. The pattern `\w+|[^\w\s]` uses the alternation operator `|` (logical OR) to match two types of tokens:
  - `\w+`: Matches one or more "word characters." In regex, `\w` is shorthand for `[a-zA-Z0-9_]`, matching any letter, digit, or underscore. So `\w+` matches entire words and numbers as single tokens (e.g., `Natural`, `NLP`, `2026`).
  - `[^\w\s]`: The `^` inside square brackets creates a negated character class, matching any character that is NOT a word character (`\w`) and NOT a whitespace character (`\s`). This effectively matches individual punctuation symbols like `(`, `)`, `,`, `.` as separate single-character tokens.
  - `re.findall()` scans through the entire string from left to right, collecting every non-overlapping match of either alternative, and returns them all as a list.
  - The result is a list of 112 tokens, where words, numbers, and punctuation marks are each separate elements.

**Lines 5-6: Display sentences**
- `print(f"Sentences Count (Pure Python): {len(sentences_py)}")`: Prints the total sentence count (9 sentences).
- `for idx, sent in enumerate(sentences_py, 1):`: The `enumerate()` function pairs each element in the list with an index number. The `start=1` argument makes the counting begin at 1 instead of the default 0, which is more natural for displaying numbered sentences to a human reader.

**Lines 7-8: Display word tokens**
- Prints the total token count (112) and a sample of the first 20 tokens to verify correctness.

---

### Approach B: Tokenization Using NLTK

#### Purpose
Demonstrate tokenization using the industry-standard NLTK library, which employs statistically-trained models for sentence boundary detection and linguistically-informed rules for word segmentation.

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

#### Line-by-Line Explanation

**Line 1: `sentences_nltk = sent_tokenize(text_2_2)`**
- **Syntax**: `sent_tokenize(text: str, language: str = 'english') -> list[str]`
- **What it does**: Passes the entire text through NLTK's Punkt sentence tokenizer. The Punkt algorithm (Kiss and Strunk, 2006) is an unsupervised, language-independent method for detecting sentence boundaries. It works by training on a corpus to learn three key statistical models:
  1. **Abbreviation Detection**: It builds a lexicon of abbreviation types by computing whether a period-terminated token is statistically likely to be an abbreviation rather than an end-of-sentence marker. It uses a log-likelihood ratio test to distinguish between these two interpretations.
  2. **Sentence-Starter Detection**: After identifying candidate sentence boundaries (periods followed by a capitalized word), it evaluates whether the capitalized word is a frequent sentence-starter or just a capitalized proper noun mid-sentence.
  3. **Collocation Detection**: It identifies multi-word expressions where a period is part of a collocation (e.g., `et al.`) rather than a boundary.
  - The function returns a list of sentence strings. For our input, it correctly identifies all 9 sentences.

**Line 2: `words_nltk = word_tokenize(text_2_2)`**
- **Syntax**: `word_tokenize(text: str, language: str = 'english') -> list[str]`
- **What it does**: Tokenizes the text into words using a two-stage process:
  1. First, it internally calls `sent_tokenize()` to split the text into sentences.
  2. Then, for each sentence, it applies the **Treebank Word Tokenizer** (also called the Penn Treebank tokenizer), which follows the tokenization conventions established by the Penn Treebank corpus. These rules include:
     - Separating punctuation from adjacent words: `Intelligence.` becomes `['Intelligence', '.']`
     - Splitting contractions: `don't` becomes `['do', "n't"]`, `I'll` becomes `['I', "'ll"]`
     - Treating parentheses and brackets as individual tokens
  - The result is a list of 110 tokens. The slight difference from the 112 tokens produced by the regex approach is due to subtle differences in how the Treebank tokenizer handles certain boundary cases (such as parenthetical expressions or contraction rules).

**Lines 3-8: Display results**
- The display logic is identical to Approach A: enumerate sentences 1-through-9, then print total token count and a sample.

---

### Approach C: Tokenization Using spaCy

#### Purpose
Demonstrate tokenization using spaCy's pipeline-based approach, which processes text through a full linguistic pipeline and uses dependency parsing (not just punctuation patterns) for sentence segmentation.

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

#### Line-by-Line Explanation

**Line 1: `doc = nlp(text_2_2)`**
- **Syntax**: `nlp(text: str) -> Doc` (calling the Language pipeline object)
- **What it does**: Passes the entire text string through spaCy's NLP pipeline. The `nlp` object (loaded earlier as `en_core_web_sm`) processes the text through a sequence of pipeline components:
  1. **Tokenizer**: A rule-based tokenizer that applies prefix rules (characters that should be split from the beginning of a token, like opening quotes `"`), suffix rules (characters at the end, like periods `.`, closing parentheses `)`), and infix rules (characters in the middle, like hyphens `-`, slashes `/`). It also consults a tokenizer exceptions table for known special cases (URLs, emoticons, abbreviations).
  2. **Tagger**: Assigns part-of-speech tags to each token.
  3. **Parser**: Builds a dependency parse tree for each sentence, identifying the syntactic relationships between words (subject, object, modifier, etc.).
  4. **NER**: Identifies named entities (person names, organizations, locations, dates).
  - The result is a `Doc` object, which is a container holding a sequence of `Token` objects, each annotated with linguistic metadata (POS tag, dependency label, lemma, entity label, whitespace information, etc.).
  - Crucially, spaCy's tokenization is **non-destructive**: the original text can be perfectly reconstructed from the tokens because each token stores its trailing whitespace information.

**Line 2: `sentences_spacy = [sent.text.strip() for sent in doc.sents if sent.text.strip()]`**
- **Syntax**: List comprehension over `Doc.sents -> Generator[Span]`
- **What it does**: `doc.sents` is a property that yields `Span` objects, where each `Span` represents one sentence. Unlike NLTK's Punkt algorithm (which uses statistical rules about periods), spaCy determines sentence boundaries using the **dependency parser**. The parser builds a syntactic tree for the text, and sentence boundaries are placed where the parser identifies the root of a new syntactic tree (typically at the start of a new independent clause). This is linguistically more principled than period-based heuristics. The `sent.text` attribute retrieves the raw text string of each sentence span. The `.strip()` call removes any flanking whitespace, and the `if s.strip()` filter excludes empty results.

**Line 3: `words_spacy = [token.text for token in doc if not token.is_space]`**
- **Syntax**: List comprehension with `Token.text` and `Token.is_space` filter
- **What it does**: Iterates over every `Token` object in the `Doc`. Each `Token` has numerous attributes; here we use:
  - `token.text`: A string property returning the verbatim text content of this token (e.g., `"Natural"`, `","`, `"."`).
  - `token.is_space`: A boolean property that returns `True` if the token consists entirely of whitespace characters. This happens when the input text contains things like double newlines between paragraphs, which spaCy preserves as whitespace tokens. We filter these out because they are not meaningful linguistic tokens.
  - The result is a list of 112 non-whitespace tokens.

### Comparison of All Three Approaches

| Metric | Pure Python (Regex) | NLTK | spaCy |
| :--- | :--- | :--- | :--- |
| **Sentence Count** | 9 sentences | 9 sentences | 9 sentences |
| **Word/Punct Tokens** | 112 tokens | 110 tokens | 112 tokens |
| **External Dependencies** | None (standard library `re`) | `nltk` package + `punkt` data download | `spacy` package + `en_core_web_sm` model |
| **Sentence Boundary Method** | Regex lookbehind on `.!?` | Punkt unsupervised statistical model | Dependency parser syntactic analysis |
| **Word Tokenization Method** | Regex alternation `\w+\|[^\w\s]` | Penn Treebank conventions | Rule-based prefix/suffix/infix patterns |
| **Handles Abbreviations** | No (splits on `Dr. Smith`) | Yes (trained abbreviation detector) | Yes (exception tables + parser context) |
| **Primary Use Case** | Lightweight scripts, education | Academic NLP research, prototyping | Production deployments, industrial NLP |
| **Speed** | Very fast (compiled C regex) | Moderate | Fast (Cython-optimized) |
| **Token Count Difference** | The 2-token difference vs NLTK is due to how the Treebank tokenizer handles certain parenthetical and punctuation boundary cases differently from raw regex alternation. |

---

## Experiment 2.3: Stop Words Removal and Token Filtering

### Background and Motivation
In many NLP tasks (text classification, topic modeling, keyword extraction, information retrieval), certain extremely common words add no discriminative value. Words like `the`, `is`, `a`, `an`, `in`, `of`, `and`, `to` appear in virtually every English text regardless of its topic. These are called **stop words**. In a Bag-of-Words or TF-IDF representation, stop words occupy feature dimensions without helping distinguish between documents about, say, machine learning versus cooking. Removing them reduces the vocabulary size, decreases computational costs, and often improves model accuracy for tasks that rely on content word semantics.

However, stop word removal is NOT universally beneficial. In tasks like sentiment analysis (where `not` is a critical negation particle), syntactic parsing (where function words encode grammatical structure), question answering (where `who`, `what`, `which` are stop words but essential query terms), or language modeling (where every word matters for predicting the next token), removing stop words can degrade performance. This experiment demonstrates the mechanics of stop word filtering using NLTK's curated stop word list.

### Input Data
The input file [2.3_clean_data.txt](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%202/2.3_clean_data.txt) contains 7 non-blank lines (9 lines total with paragraph separators) discussing NLP and stop words. The text was chosen because it naturally contains many stop words (articles, prepositions, conjunctions) and also explicitly mentions stop words as a topic, making it pedagogically self-referential.

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

### Line-by-Line Explanation

**Line 1: `with open('2.3_clean_data.txt', 'r', encoding='utf-8') as f:`**
- **What it does**: Opens the third data file in read mode with UTF-8 encoding. Identical mechanism to previous file-opening operations.

**Line 2: `text_2_3 = f.read()`**
- **What it does**: Reads the complete file content into the string `text_2_3`. Note that unlike Experiment 2.2, we do NOT call `.strip()` here. This is acceptable because `word_tokenize()` handles leading/trailing whitespace gracefully.

**Line 3: `tokens_2_3 = word_tokenize(text_2_3)`**
- **Syntax**: `word_tokenize(text: str) -> list[str]`
- **What it does**: Applies NLTK's word tokenizer (Punkt + Penn Treebank rules) to the entire text. The function splits the text into 138 individual tokens, which include content words (`Natural`, `Language`, `Processing`), function words / stop words (`is`, `an`, `of`, `the`), and punctuation marks (`.`, `,`). Each of these token categories will be handled differently in the filtering step.

**Line 4: `stop_words = set(stopwords.words('english'))`**
- **Syntax**: `set(stopwords.words(language: str)) -> set[str]`
- **What it does**: This line performs two operations:
  1. `stopwords.words('english')`: Accesses the NLTK stopwords corpus and retrieves the English stop word list, returning a Python `list` of 179 lowercase strings. These include common articles (`a`, `an`, `the`), prepositions (`in`, `on`, `at`, `to`, `for`, `with`, `from`, `by`, `of`), conjunctions (`and`, `but`, `or`, `nor`), pronouns (`i`, `me`, `my`, `we`, `you`, `he`, `she`, `it`, `this`, `that`, `these`, `those`), auxiliary and modal verbs (`is`, `am`, `are`, `was`, `were`, `be`, `been`, `being`, `have`, `has`, `had`, `do`, `does`, `did`, `will`, `would`, `shall`, `should`, `may`, `might`, `can`, `could`), and various other high-frequency function words.
  2. `set(...)`: Converts the list to a Python `set`. This is a **critical performance optimization**. A `set` is implemented internally as a hash table, which means that checking whether a given word exists in the set (the `in` operator) takes constant time on average, written as $O(1)$. In contrast, checking membership in a `list` requires scanning through the list element by element, which takes linear time $O(M)$ where $M$ is the list length (179 in this case). Since we will be checking every single token in our text against this collection, the performance difference is:
     - With a `list`: $O(N \times M) = O(138 \times 179) = O(24,702)$ comparisons in the worst case.
     - With a `set`: $O(N \times 1) = O(138)$ lookups.
     - For this small dataset the difference is negligible, but for real-world corpora with millions of tokens, the difference becomes enormous.

**Line 5: `extracted_stopwords = [w for w in tokens_2_3 if w.lower() in stop_words]`**
- **Syntax**: List comprehension with `str.lower()` and set membership test
- **What it does**: Iterates over every token `w` in the tokenized text. For each token, it:
  1. Converts the token to lowercase using `w.lower()` (because the stop word list is all lowercase, but our tokens might be capitalized at the start of sentences, e.g., `It`, `The`, `A`).
  2. Checks if this lowercase form exists in our `stop_words` set using the `in` operator.
  3. If the token is a stop word, it is included in the `extracted_stopwords` list.
  - This produces a list of all stop word occurrences in the text, preserving duplicates. If `the` appears 5 times, it will be listed 5 times. The result contains 57 stop word occurrences total.

**Line 6: `unique_stopwords = sorted(list(set(w.lower() for w in extracted_stopwords)))`**
- **Syntax**: `sorted(list(set(generator_expression)))`
- **What it does**: Chains multiple transformations to produce a deduplicated, alphabetically sorted list of the stop words that actually appeared in this specific text:
  1. `w.lower() for w in extracted_stopwords`: A generator expression (similar to a list comprehension but lazily evaluated) that yields the lowercase form of each extracted stop word.
  2. `set(...)`: Eliminates duplicates, collapsing 57 occurrences down to 22 unique stop words.
  3. `list(...)`: Converts the set to a list (required for sorting, since sets are unordered).
  4. `sorted(...)`: Sorts alphabetically.
  - The result is: `['a', 'an', 'and', 'are', 'as', 'by', 'can', 'for', 'from', 'in', 'is', 'it', 'of', 'on', 'some', 'such', 'that', 'the', 'these', 'this', 'to', 'with']` -- 22 unique stop words found in this text.

**Line 7: `filtered_tokens = [w for w in tokens_2_3 if w.lower() not in stop_words and w.isalnum()]`**
- **Syntax**: List comprehension with compound boolean condition
- **What it does**: This is the final content extraction step. For each token `w` in the original token list, it applies two conditions connected with `and` (both must be True for the token to be kept):
  1. `w.lower() not in stop_words`: The token's lowercase form must NOT be in the stop words set. The `not in` operator is the logical negation of `in`. This filters out all 57 stop word occurrences.
  2. `w.isalnum()`: The token must be alphanumeric, meaning it contains only letters and/or digits with no punctuation or special characters. The `str.isalnum()` method returns `True` if every character in the string is either a letter or a digit, and the string has at least one character. For example, `'Natural'` returns `True`, `'NLP'` returns `True`, but `'.'` returns `False`, `','` returns `False`, and `''` (empty string) returns `False`. This effectively strips out all punctuation tokens that `word_tokenize` created.
  - Together, these two conditions retain only content-bearing, non-stop-word, alphanumeric tokens. The result contains 56 filtered content tokens.
  - The distinction between original (138), stop words removed (138 - 57 = 81), and final filtered (56) shows that an additional 25 punctuation tokens were also removed by the `isalnum()` filter ($81 - 56 = 25$ punctuation tokens).

**Lines 8-14: Display results**
- The remaining lines print the original token list, the stop words found (both total occurrences and unique set), and the final filtered content tokens in sequence.

### Output Metrics Summary
| Stage | Count | Description |
| :--- | :--- | :--- |
| **Original Tokens** | 138 | All words + punctuation as produced by `word_tokenize` |
| **Stop Word Occurrences** | 57 | Individual stop word hits across the text |
| **Unique Stop Words** | 22 | Distinct stop words: `a, an, and, are, as, by, can, for, from, in, is, it, of, on, some, such, that, the, these, this, to, with` |
| **Punctuation Tokens Removed** | 25 | Tokens like `.` and `,` filtered by `isalnum()` |
| **Final Content Tokens** | 56 | Meaningful content words: `Natural, Language, Processing, important, field, Artificial, Intelligence, used, many, applications, chatbots, machine, translation, sentiment, analysis, text, summarization, main, purpose, NLP, enable, computers, understand, process, human, language, computer, analyze, large, amount, short, period, time, removal, stop, words, one, common, steps, preprocessing, Stop, commonly, occurring, words, commonly, occurring, reduce, size, improve, efficiency, NLP, applications, ...` |

---

## Notebook Compliance Verification
- **Code comments**: 0 (Zero comments present in any code cell as requested).
- **Markdown cells**: Concise, structured headers and objectives preceding each code cell.
- **Execution state**: All 14 cells (6 markdown + 1 setup code + 5 experiment code cells) fully executed with all terminal outputs, token lists, and metrics rendered inline.
