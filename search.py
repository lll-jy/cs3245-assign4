#!/usr/bin/python3
import getopt
import sys
import math
from heapq import *
from nltk.corpus import wordnet as wn

# Global variables
from file_io import load_dict, read_doc_id, read_float_bin_file, read_tf, read_position_pointer, read_positional_index
from widths import doc_width, smallest_doc_id, doc_byte_width, pos_byte_width, number_of_doc_returned
from normalize import process_doc

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
    df = open(dict_file, 'r')
    qf = open(queries_file, 'r')
    rf = open(results_file, 'w')

    # Load dictionary
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

    # Perform search
    query_input = qf.readline()
    # split query by 'AND'
    res = query_input.split('AND')
    queries = []
    for part in res:
        queries.append(part.strip())

    result_lists = []
    for query in queries:
        res = process_query(query)
        result_lists.append(res)

    final_score = compute_harmonic_scores(result_lists)

    result = select_first_k(final_score, 10)

    if not result:
        rf.write('\n')
    rf.write(str(result[0] + smallest_doc_id))
    for doc_id in result[1:]:
        rf.write(' ')
        rf.write(str(doc_id + smallest_doc_id))
    rf.write('\n')

    # Close files
    pf.close()
    df.close()
    qf.close()
    rf.close()


def compute_harmonic_scores(result_lists):
    """
    Compute a harmonic average score for each document with respect to multiple
    queries in query contains 'AND'
    :param result_lists: a list of dictionaries of scores
    :return: a dictionary of the harmonic average scores for each document
    """
    final_score = {}
    if len(result_lists) == 1:
        final_score = result_lists[0]
    else:
        num_of_lists = len(result_lists)
        for document in result_lists[0]:
            acc = 0
            for number in range(1, num_of_lists):
                if document not in result_lists[number] or result_lists[number][document] == 0:
                    final_score[document] = 0
                    break
                else:
                    acc += 1 / result_lists[number][document]
            # Store the harmonic average of each document's scores for different queries
            if acc == 0:
                final_score[document] = 0
            else:
                final_score[document] = num_of_lists / acc
    return final_score


def select_first_k(scores, k):
    """
    Select the first k documents with the highest score
    :param k: number of documents returned
    :param scores: a dictionary of document scores
    :return: a list of k document ids
    """
    res = []
    for doc in scores:
        # Square the scores to make the difference in weight more obvious
        scores[doc] *= scores[doc]
        if len(res) < k:
            heappush(res, (scores[doc], -doc))
        else:
            if res[0] < (scores[doc], -doc):
                heappop(res)
                heappush(res, (scores[doc], -doc))
    res_docs = []
    while res:
        doc = heappop(res)
        if doc[0] > 0:
            res_docs.append(-doc[1])
    res_docs.reverse()
    return res_docs


def process_query(query):
    """
    Determine whether a query is a phrase query or a normal one, and handle
    them accordingly
    :param query: a query string
    :return: a dictionary of scores of documents with respect to their relevance
    for the given query
    """
    if query[0] == "\"":
        query = query.strip("\"")
        return process_phrase_query(query)
    else:
        query_dict = get_query_dict(query)
        # Query extension is used for free text query
        res = process_free_query(query_extension(query_dict))
        return res


def process_phrase_query(query):
    # TODO: phrase query
    return 0


def weight_query(doc_freq, term_freq):
    """
    Calculate the tf-idf weight given tf and df in ltc
    :param doc_freq: df = document frequency > 0
    :param term_freq: tf = term frequency
    :return: the weight calculated using the formula w = (1 + log(tf)) * log(N/df)
    """
    return weight_doc(term_freq) * math.log(len(doc_len) / doc_freq, 10)


def weight_doc(term_freq):
    """
    Calculate the log-frequency weight given tf and df in lnc
    :param term_freq: tf = term frequency
    :return: the weight calculated using the formula w = 1 + log(tf)
    """
    if term_freq == 0:
        return 0
    return 1 + math.log(term_freq, 10)


def query_extension(query_dict):
    """
    Using nltk's WordNet to include words with similar meanings in query
    :param query_dict: a dictionary with query term as key and term
    frequency as value
    :return: another dictionary with query term(including extended terms)
    as key and the default term frequency as value(original terms' tf are
    doubled to increase their weight in searching)
    """
    extension_words = {}
    for term in query_dict:
        for synset in wn.synsets(term):
            for word in synset.lemma_names():
                if word not in query_dict:
                    extension_words[word] = 1

    final_query = {}
    for term in query_dict:
        final_query[term] = 2 * (query_dict[term] + 1)
    for term in extension_words:
        final_query[term] = extension_words[term]
    return final_query


def get_query_dict(query):
    """
    tokenize the input query and return a dictionary with term as key and
    term frequency as value
    :param query: original query
    :return: a dictionary with query term as key and term frequency as value
    """
    res = []
    query_info = process_doc(query)
    query_dict = query_info[0]
    return query_dict


def process_free_query(query_dict):
    """
    Process a free query
    :param query_dict: a dictionary with query term as key and term frequency as value
    :return: a dictionary with scores of the documents
    """
    # Initialize scores of each document to 0
    scores = {}
    for doc in doc_len:
        scores[doc] = 0
    for term in query_dict:
        if term not in dictionary:
            # tf of documents if always 0, doesn't contribute to the score
            continue
        df = dictionary[term]['df']
        qw = weight_query(df, query_dict[term])
        if qw == 0:
            # w(t,q)=0, so additional score for this term is 0
            continue
        doc_count = 0
        while doc_count < df:
            posting_element = read_posting(term, doc_count)
            doc_id = posting_element[0]
            tf = posting_element[1]
            if doc_id not in scores:
                doc_count += 1
                continue
            scores[doc_id] += weight_doc(tf) * qw
            doc_count += 1
        if doc_len[doc] != 0:
            scores[doc] /= doc_len[doc]
        return scores


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
