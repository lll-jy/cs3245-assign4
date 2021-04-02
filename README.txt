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



== Files included with this submission ==

index.py: required source code of indexing
search.py: required source code of searching
shared.py: shared code for index.py and search.py
dictionary.txt: generated dictionary using index.py with data in Reuters
postings.txt: generated postings list using index.py with data in Reuters
lengths.txt: generated length of vector of each document using index.py with data Reuters
README.txt: this file
ESSAY.txt: answers to essay questions

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

Python API:
https://docs.python.org/3/library/

Python csv help:
https://www.programiz.com/python-programming/reading-csv-files

CSV field size exception handling:
https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072