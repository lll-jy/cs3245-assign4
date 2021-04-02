#!/usr/bin/python3
import csv
import getopt
import sys

from normalize import process_doc
from widths import pos_width, post_width, tf_width, pos_pointer_width


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
    read_data(dictionary, doc_freq, doc_len, reader)

    # Write output files
    write_dict_postings(doc_freq, dictionary, doc_len, out_dict, out_postings)

    # Close file
    in_file.close()
    print('end')


def read_data(dictionary, doc_freq, doc_len, reader):
    """
    Reads in data from the file reader of dataset input
    :param dictionary: the dictionary containing the document ID, term frequency, and positional indices list of each
    word of each document that contains this term
    :param doc_freq: the document frequency dictionary
    :param doc_len: the document vector lengths dictionary
    :param reader: the file reader of dataset input
    :return: none
    """
    for row in reader:
        document_id = int(row['document_id'])
        content = row['content']
        process_res = process_doc(content)
        doc_dict = process_res[0]
        doc_positions = process_res[1]
        for word in doc_dict:
            if word not in dictionary:
                dictionary[word] = []
                doc_freq[word] = 0
            dictionary[word].append({
                'id': document_id,
                'tf': doc_dict[word],
                'positions': doc_positions[word]
            })
            doc_freq[word] += 1
        doc_len[document_id] = process_res[2]
        break


def write_dict_postings(doc_freq, dictionary, doc_len, out_dict, out_postings):
    """
    Writes the information from dataset file to dictionary and postings
    :param doc_freq: the document frequency dictionary
    :param dictionary: the dictionary containing the document ID, term frequency, and positional indices list of each
    word of each document that contains this term
    :param doc_len: the dictionary of the vector length of each document
    :param out_dict: the output file for dictionary
    :param out_postings: the output file for postings
    :return: none
    """
    """
    dictionary.txt:
    postings_start_pointer lengths_start_pointer
    term1 df pointer
    term2 df pointer
    ... 
    ------
    postings.txt:
    (of term1 doc1) pos1 pos2 ...
    (of term1 doc2) pos1 pos2 ...
    ...
    (of term2 doc1) pos1 pos2 ...
    ...
    
    (of term 1) doc_id1 term_freq1 positions_pointer1 doc_id2 term_freq2 positions_pointer2 ...
    (of term 2) doc_id1 term_freq1 positions_pointer1 ...
    ...
    
    doc_id1 len1
    doc_id2 len2
    ...
    """
    # Initialization
    dict_writer = open(out_dict, 'w')
    post_writer = open(out_postings, 'w')
    words = sorted(doc_freq)
    pos_pointers = {}
    post_pointers = {}
    pointer = 0

    # Get the positions pointer and print positions to the postings file
    for word in words:
        pos_pointers[word] = {}
        for posting in dictionary[word]:
            pos_pointers[word][posting['id']] = pointer
            for pos in posting['positions']:
                post_writer.write(f'{str(pos).zfill(pos_width)} ')
                pointer += pos_width + 1
            post_writer.write('\n')
            pointer += 1

    # Print the postings
    postings_pointer = pointer
    pointer = 0
    for word in words:
        post_pointers[word] = {}
        for posting in dictionary[word]:
            post_pointers[word] = pointer
            post_writer.write(f"{str(posting['id']).zfill(post_width)} {str(posting['tf']).zfill(tf_width)} "
                              f"{str(pos_pointers[word][posting['id']]).zfill(pos_pointer_width)} ")
            pointer += post_width + tf_width + pos_pointer_width + 3
        post_writer.write('\n')
        pointer += 1

    # Print lengths
    lengths_pointer = pointer + postings_pointer
    for doc in doc_len:
        post_writer.write(f"{doc} {doc_len[doc]}\n")

    # Print dictionary
    dict_writer.write(f"{postings_pointer} {lengths_pointer}\n")
    for word in words:
        dict_writer.write(f"{word} {str(doc_freq[word])} {post_pointers[word]}\n")

    # Close files
    dict_writer.close()
    post_writer.close()


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
