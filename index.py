#!/usr/bin/python3
import csv
import getopt
import sys

from normalize import process_doc


def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print("indexing...")

    # Initializations
    csv.field_size_limit(sys.maxsize)
    dictionary = {}
    doc_freq = {}
    doc_len = {}

    # Open file
    in_file = open(in_dir, 'r')
    reader = csv.DictReader(in_file)

    # Read each file, the files are already in ascending order of document ID
    for row in reader:
        """
        print(dict(row))
        index = index + 1
        if index > 2:
            break
        """
        document_id = int(row['document_id'])
        content = row['content']
        process_res = process_doc(content)
        doc_dict = process_res[0]
        for word in doc_dict:
            if word not in dictionary:
                dictionary[word] = []
                doc_freq[word] = 0
            dictionary[word].append((document_id, doc_dict[word]))
            doc_freq[word] += 1

    # Close file
    in_file.close()


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"


if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
    """
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
