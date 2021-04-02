#!/usr/bin/python3
import getopt
import sys

# Global variables
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
    pf = open(postings_file, 'r')
    qf = open(queries_file, 'r')
    rf = open(results_file, 'w')

    # Load dictionary
    df = open(dict_file, 'r')
    pointers = df.readline()
    postings_base_pointer = int(pointers[0])
    lengths_base_pointer = int(pointers[1])
    terms = df.readlines()
    for word_str in terms:
        word_str = word_str[:-1]
        entries = word_str.split(' ')
        dictionary[entries[0]] = {
            'df': entries[1],
            'ptr': entries[2]
        }
    df.close()

    # Load vector lengths
    pf.seek(lengths_base_pointer)
    while True:
        line = pf.readline()
        if not line:
            break
        entries = line[:-1].strip().split(' ')
        doc_len[int(entries[0])] = float(entries[1])



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
