#!/usr/bin/python3
import getopt
import sys

# Global variables
from file_io import load_dict, read_doc_id, read_float_bin_file, read_tf, read_position_pointer, read_positional_index
from widths import doc_width, smallest_doc_id, doc_byte_width, pos_byte_width

pf = None
dictionary = {}
doc_len = {}
postings_base_pointer = None
lengths_base_pointer = None


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')

    # Initial setup
    global pf, postings_base_pointer, lengths_base_pointer
    pf = open(postings_file, 'rb')
    qf = open(queries_file, 'r')
    rf = open(results_file, 'w')

    # Load dictionary
    df = open(dict_file, 'r')
    pointers = df.readline()[:-1].split(' ')
    postings_base_pointer = int(pointers[0])
    lengths_base_pointer = int(pointers[1])
    load_dict(df, dictionary)

    # Load vector lengths
    pf.seek(lengths_base_pointer)
    """
    while True:
        line = pf.readline()
        if not line:
            break
        entries = line[:-1].strip().split(' ')
        doc_len[int(entries[0])] = float(entries[1])
    """
    while True:
        doc = read_doc_id(pf)
        length = read_float_bin_file(pf)
        if not length:
            break
        doc_len[doc] = length[0]

    # print(dictionary['10'])
    # print('here', read_posting('10', 1))
    # print('there', read_position('10', 1, 1))
    # print(doc_len[8891473])
    """
    for i in range(35):
        lf = open(f'lengths{i}.txt', 'rb')
        while True:
            doc = read_doc_id(lf)
            length = read_float_bin_file(lf)
            if not length:
                break
            if doc < 8891473:
                print(length, doc, i)
            print(length, doc, i)
        lf.close()
    """


def read_position(word, doc_index, index):
    """
    Read the positional index
    :param word: the word to search for
    :param doc_index: the document index of the word to search for, not the docID
    :param index: the i-th appearance of the term
    :return: the positional index
    """
    _, _, ptr = read_posting(word, doc_index)
    pf.seek(ptr + pos_byte_width * index)
    return read_positional_index(pf)


def read_posting(word, index):
    """
    Read the document ID, term frequency, and pointer to positions list of the index-th document of given word
    :param word: the term to search
    :param index: the index of the document w.r.t. this term
    :return: tuple (ID, tf, ptr)
    """
    """
    pf.seek(postings_base_pointer + dictionary[word]['ptr'] + doc_width * index)
    info = pf.read(doc_width).strip().split(' ')
    return int(info[0]), int(info[1]), int(info[2])
    """
    pf.seek(postings_base_pointer + dictionary[word]['ptr'] + doc_byte_width * index)
    id = read_doc_id(pf)
    tf = read_tf(pf)
    ptr = read_position_pointer(pf)
    return id, tf, ptr


dictionary_file = postings_file = file_of_queries = output_file_of_results = None


try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"


if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
