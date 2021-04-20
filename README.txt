This is the README file for A0194567M and A0194484R's submission

== Python Version ==

I'm (We're) using Python Version 3.7.9 for
this assignment.

== General Notes about this assignment ==

1. Indexing

1.1 Reading input data

The input data is in .csv file, the Python library csv is used to handle processing content
of this file.

The "document_id" field in .csv is used as the actual document ID of the line. The "content"
field is used as the actual content of the corresponding document. Content in "title",
"date_posted", and "court" fields are omitted.

1.2 Normalization of terms

Words are stemmed using NLTK Porter stemmer, and only alphanumerical terms are preserved.
Numbers are not removed, and stop words are not removed. Case-folding to lower case is applied
to all terms.

Notably, there are a number of appearance of enumerating indices in the content of many files,
such as (1), (iii), (c) etc. These terms are also not counted as valid terms.

1.3 Information gathered at indexing stage

All terms, their corresponding document frequency, their term frequency in each document,
and list of positional indices of their appearance in each document are captured at indexing
stage.

The length of the vector of each document is also calculated at this stage.

1.4 Output format

dictionary.txt:
The first line is 2 numbers corresponding to the start position of postings section in postings
file and lengths section in postings file respectively. Each line of the remaining part
represents a term in sorted ascending alphabetical order, and each line consists of the term
itself, document frequency, and pointer to postings file (w.r.t. the postings section)

postings.txt:
There are 3 sections corresponding to positions, documents, and lengths. The file is a binary
file. The positions section contains the positional indices of a term in a specific document
ordered in ascending order by term and document ID as tie breaker. The documents section
is the postings information, containing information corresponding to each line in dictionary.txt
about document ID, term frequency, and positional index pointer (in the postings file). The
lengths section is the vector lengths in double precision (64-bits) of each document. Document
IDs are not the exact same as original when stored in the postings file, but instead, is the
actual document ID subtracted by the smallest document ID.

1.5 Scaling

Since the input data is too big, a technique similar to BSBI is applied. But to simplify the code,
the blocks are separated by the number of documents instead of the actual size of files. This
gives similar performance to BSBI.

Intermediate files are dictionaryN.txt, postingsN.txt, positionsN.txt, lengthsN.txt, where N
denotes the index of block based at 0. They corresponds to the dictionary file of the block
and the three sections of the postings file of the block respectively. The number of files in
each block is 500.


2. Parsing queries

2.1 Handling AND
The input query is split by the key word 'AND'. To compute a final ranking, each part of
the queries after splitting is firstly ranked individually, producing a list of scores by
applying tf-idf and cosine similarity, as described in Section 3.
The scores for documents in each ranking are then combined by calculating their harmonic
average.
For example, if a document has a score a in query A, and a score b in query B. Then its
harmonic average score is compute as 2ab/(a+b).
This average number is chosen because it can balance each side and give final ranking that is
more balanced for each and every query.


2.2 Query Refinement Techniques
query refinement techniques are used to improve the recall of free text searching

2.2.1 Query Extension
NLTK's English WordNet is used for query extension. For every query term, words with similar
meaning to this term is retrieved from the WordNet to form the new query. The new query consists
both the original terms and the extended term, while the original terms having a higher weight.
For now, the weight of original and extended terms are differentiated by adjusting the term
frequency. The extended terms' term frequency is set to be 1, and to make the difference more
obvious, the tf of original terms are set to be (1 + the actual tf) * 2.


3. Searching and ranking

3.1 Preparatory work

Before searching and ranking, information from document files are loaded. Information loaded
to memory at this stage is the full dictionary, including terms and their corresponding
document frequencies and postings pointers. In addition, the two base pointers stored at the
top of the dictionary file, and the vector lengths of all documents are also processed and
loaded to memory.

3.2 Basic ranking scheme

Similar to HW3, lnc.ltc is used to calculate the cosine similarity between the query and each
document.

The algorithm to calculate cosign scores is derived from the one given in
the lecture notes. Vectors for queries are actually not normalized to a unit
vector because this is a shared coefficient for all cosine scores, and thus
it makes no difference in the ranking of score.

Terms in the query that does not appear in the dictionary will not have effect
on the scores of documents. Omitting such terms in the calculation of score
makes sense because, firstly, the result of log-frequency of the term in any
document would be 0 makes the weight of term in any document 0; and, secondly,
actual document frequency is 0, and thus makes idf, and hence weight of term
in the query, undefined as the denominator is 0.


3.3 Phrasal search
Positional indexing is used for phrasal search, and only exact matches (after normalizing) of terms as well as
positions will be returned, and the resulting list is ranked by cosing similarity. Phrasal search is restricted
to 1-3 word/term phrases.

Skip lists with sqrt(N) skip width is applied to both document IDs and positional IDs (if the skip width is large
enough).


== Files included with this submission ==

index.py: required source code of indexing
search.py: required source code of searching
file_io.py: helper functions for reading and writing files
normalize.py: helper functions for normalizing terms
widths.py: constants of widths of each type of number of store.
dictionary.txt: generated dictionary using index.py with given dataset
postings.txt: generated postings list using index.py with given dataset
README.txt: this file

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I/We, A0194567M, A0194484R, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.

[ ] I/We, A0194567M, A0194484R, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

NLTK Porter stemmer help:
https://www.nltk.org/howto/stem.html

Regex to remove non-alphabetical terms help:
https://stackoverflow.com/questions/22520932/python-remove-all-non-alphabet-chars-from-string

NLTK API:
https://www.nltk.org/api/nltk.html

NLTK similar function:
https://www.nltk.org/book/ch01.html

NLTK WordNet:
https://www.nltk.org/book/ch02.html

Python API:
https://docs.python.org/3/library/

Python csv help:
https://www.programiz.com/python-programming/reading-csv-files

CSV field size exception handling:
https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072

Roman numeral regex:
https://www.oreilly.com/library/view/regular-expressions-cookbook/9780596802837/ch06s09.html

Harmonic Mean Wikipedia:
https://en.wikipedia.org/wiki/Harmonic_mean