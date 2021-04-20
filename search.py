#!/usr/bin/python3
import getopt
import queue
import sys
import math
import nltk
from heapq import *
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *

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

    result = select_first_k(final_score, 1000)

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
            for number in range(0, num_of_lists):
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
        return process_phrasal_search(query)
    else:
        # Query extension is used for free text query
        new_query = query_extension(query)
        res = process_free_query(get_query_dict(new_query))
        return res


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


def query_extension(query):
    """
    Using nltk's WordNet to include words with similar meanings in query
    :param query: the original query
    :return: a new query with the original terms occurring twice(to increase
    the weight of the original terms) and extension terms from nltk word net
    occurring once
    """
    words = nltk.word_tokenize(query)
    extension_words = []
    new_query = (query + " ") * 2
    for term in words:
        for synset in wn.synsets(term):
            for word in synset.lemma_names():
                if word not in words:
                    extension_words.append(word)
                    new_query += " " + word
    return new_query

    '''
    final_query = {}
    for term in query_dict:
        final_query[term] = 2 * (query_dict[term] + 1)
    for term in extension_words:
        final_query[term] = extension_words[term]
    return final_query
    '''


def get_query_dict(query):
    """
    tokenize the input query and return a dictionary with term as key and
    term frequency as value
    :param query: original query
    :return: a dictionary with query term as key and term frequency as value
    """
    query_info = process_doc(query)
    query_dict = query_info[0]
    return query_dict


def calculate_scores(query_dict, src_lst):
    """
    Calculates the scores in a list of documents corresponding to a query
    :param query_dict: the query term frequencies mapping
    :param src_lst: the list of documents to consider
    :return: a mapping of docID to scores in the given list
    """
    # Initialize scores of each document to 0
    scores = {}
    for doc in src_lst:
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


def process_free_query(query_dict):
    """
    Process a free query
    :param query_dict: a dictionary with query term as key and term frequency as value
    :return: a dictionary with scores of the documents
    """
    return calculate_scores(query_dict, doc_len)


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
    pf.seek(postings_base_pointer + dictionary[word]['ptr'] + doc_byte_width * index)
    id = read_doc_id(pf)
    tf = read_tf(pf)
    ptr = read_position_pointer(pf)
    return id, tf, ptr


def get_list_for_word(word, pos, details):
    """
    Gets the list for a single word with its corresponding relevant positions
    :param word: the word to search
    :param pos: the positions wanted
    :param details: true if detailed positions also need to be returned
    :return: the list of documents that contain this word in the wanted positions, and the corresponding list of
    positions if needed
    """
    res = []
    res_pos = []
    if len(pos) == 1:
        pf.seek(postings_base_pointer + dictionary[word]['ptr'])
        for _ in range(dictionary[word]['df']):
            res.append(read_doc_id(pf))
            tf = read_tf(pf)
            pp = read_position_pointer(pf)
            if details:
                this_pos = []
                pf.seek(pp)
                for _ in range(tf):
                    this_pos.append(read_positional_index(pf))
                res_pos.append(this_pos)
    pf.seek(postings_base_pointer + dictionary[word]['ptr'])
    for _ in range(dictionary[word]['df']):
        doc_id = read_doc_id(pf)
        tf = read_tf(pf)
        pos_ptr = read_position_pointer(pf)
        if tf < len(pos):
            continue
        pos_reader = open(postings_file, 'rb')
        pos_reader.seek(pos_ptr)
        current_positions = queue.Queue()
        t_id = len(pos)  # term id of next to inspect
        for _ in range(len(pos)):
            current_positions.put(read_positional_index(pos_reader))
        is_valid = False
        this_pos = []
        pos_list = list(current_positions.queue)
        if is_isomorphic(pos_list, pos):
            is_valid = True
            this_pos.append(pos_list[0])
        while t_id < tf:
            if is_valid and not details:
                break
            current_positions.get()
            current_positions.put(read_positional_index(pos_reader))
            t_id += 1
            pos_list = list(current_positions.queue)
            if is_isomorphic(pos_list, pos):
                is_valid = True
                this_pos.append(pos_list[0])
        if is_valid:
            res.append(doc_id)
            res_pos.append(this_pos)
        pos_reader.close()
    return res, res_pos


def is_isomorphic(l1, l2):
    """
    Checks whether the two lists of positions are isomorphic in terms of relative positions, with both lists assumed to
    be non-empty
    :param l1: the first list of positions
    :param l2: the second list of positions
    :return: true if the two lists are isomorphic, e.g. [2, 4] and [1, 3]
    """
    size = len(l1)
    if len(l2) != size:
        return False
    diff = l1[0] - l2[0]
    for i in range(1, size):
        if l1[i] - l2[i] != diff:
            return False
    return True


def intersect_words(w1, w2, pos, details):
    """
    Gets the list of simple intersection of the two given words
    :param w1: the first word to search for
    :param w2: the second word to search for
    :param pos: the relative positions of w1 and w2
    :param details: true if the details of positions of each phrase needed is needed to be returned
    :return: the list of documents that contain the two words in the wanted relative positions, and the corresponding
    list of lists of positions if needed
    """
    res = []
    res_pos = []
    # [w1, w2] for the following
    w = [w1, w2]
    doc_reader = [open(postings_file, 'rb'), open(postings_file, 'rb')]
    base_pointer = [postings_base_pointer + dictionary[w1]['ptr'], postings_base_pointer + dictionary[w2]['ptr']]
    doc_reader[0].seek(base_pointer[0])
    doc_reader[1].seek(base_pointer[1])
    doc_id = [read_doc_id(doc_reader[0]), read_doc_id(doc_reader[1])]
    term_freq = [read_tf(doc_reader[0]), read_tf(doc_reader[1])]
    pos_pointer = [read_position_pointer(doc_reader[0]), read_position_pointer(doc_reader[1])]
    doc_count = [1, 1]  # count = next index to inspect
    doc_freq = [dictionary[w1]['df'], dictionary[w2]['df']]
    skip_width = [math.floor(math.sqrt(doc_freq[0])), math.floor(math.sqrt(doc_freq[1]))]
    while doc_count[0] <= doc_freq[0] and doc_count[1] <= doc_freq[1]:
        found = doc_id[0] == doc_id[1]
        if doc_id[0] != doc_id[1]:
            smaller_index = 0
            if doc_id[0] > doc_id[1]:
                smaller_index = 1
            other_index = 1 - smaller_index
            if skip_width[smaller_index] > 1:
                while doc_count[smaller_index] + skip_width[smaller_index] < doc_freq[smaller_index]:
                    d_id, _, _ = read_posting(w[smaller_index], doc_count[smaller_index] + skip_width[smaller_index])
                    if d_id > doc_id[other_index]:
                        break
                    doc_count[smaller_index] += skip_width[smaller_index]
            doc_reader[smaller_index].seek(base_pointer[smaller_index] + doc_byte_width
                                           * (doc_count[smaller_index] - 1))
            for _ in range(skip_width[smaller_index] + 1):
                d_id = read_doc_id(doc_reader[smaller_index])
                tf = read_tf(doc_reader[smaller_index])
                pp = read_position_pointer(doc_reader[smaller_index])
                doc_count[smaller_index] += 1
                if d_id >= doc_id[other_index]:
                    found = d_id == doc_id[other_index]
                    doc_id[smaller_index] = d_id
                    term_freq[smaller_index] = tf
                    pos_pointer[smaller_index] = pp
                    break
                if doc_count[smaller_index] >= doc_freq[smaller_index]:
                    break
        if found:
            diff = [pos[1] - pos[0], pos[0] - pos[1]]  # diff = other - this
            pos_reader = [open(postings_file, 'rb'), open(postings_file, 'rb')]
            pos_reader[0].seek(pos_pointer[0])
            pos_reader[1].seek(pos_pointer[1])
            position = [read_positional_index(pos_reader[0]), read_positional_index(pos_reader[1])]
            skip_pos_width = [math.floor(math.sqrt(term_freq[0])), math.floor(math.sqrt(term_freq[1]))]
            pos_count = [1, 1]
            found2 = (position[1] - position[0] == diff[0])
            this_pos = []
            if found2:
                this_pos.append(position[0])
            while pos_count[0] <= term_freq[0] and pos_count[1] <= term_freq[1] and (details or not found2):
                smaller_index = 0
                if position[1] - position[0] < diff[0]:
                    smaller_index = 1
                other_index = 1 - smaller_index
                if skip_pos_width[smaller_index] > 1:
                    while pos_count[smaller_index] + skip_pos_width[smaller_index] < term_freq[smaller_index]:
                        pf.seek(pos_pointer[smaller_index] +
                                (pos_count[smaller_index] + skip_pos_width[smaller_index]) * pos_byte_width)
                        p_id = read_positional_index(pf)
                        if position[other_index] - p_id < diff[smaller_index]:
                            break
                        pos_count[smaller_index] += skip_pos_width[smaller_index]
                pos_reader[smaller_index].seek(pos_pointer[smaller_index]
                                               + pos_byte_width * (pos_count[smaller_index] - 1))
                for _ in range(skip_pos_width[smaller_index] + 1):
                    p_id = read_positional_index(pos_reader[smaller_index])
                    pos_count[smaller_index] += 1
                    if position[other_index] - p_id <= diff[smaller_index]:
                        matches = position[other_index] - p_id == diff[smaller_index]
                        found2 = matches or found2
                        position[smaller_index] = p_id
                        if matches:
                            this_pos.append(position[0])
                        break
                    if pos_count[smaller_index] >= term_freq[smaller_index]:
                        break
            if found2:
                res.append(doc_id[0])
                res_pos.append(this_pos)
            pos_reader[0].close()
            pos_reader[1].close()
            for i in range(1):
                doc_id[i] = read_doc_id(doc_reader[i])
                term_freq[i] = read_tf(doc_reader[i])
                pos_pointer[i] = read_position_pointer(doc_reader[i])
                doc_count[i] += 1
    doc_reader[0].close()
    doc_reader[1].close()
    return res, res_pos


def intersect_word_list(word, docs, pos_lst, pos, details):
    """
    Gets the list of simple intersection of the two given words
    :param word: the word to search for
    :param docs: the intermediate list of words
    :param pos_lst: the intermediate list of position lists
    :param pos: the relative positions of word and intermediate list
    :param details: true if the details of positions of each phrase needed is needed to be returned
    :return: the list of documents that contain the two words in the wanted relative positions, and the corresponding
    list of lists of positions if needed
    """
    res = []
    res_pos = []
    # [w1, w2] for the following
    doc_reader = open(postings_file, 'rb')
    base_pointer = postings_base_pointer + dictionary[word]['ptr']
    doc_reader.seek(base_pointer)
    if not docs:
        return [], []
    doc_id = [read_doc_id(doc_reader), docs[0]]
    term_freq = [read_tf(doc_reader), len(pos_lst[0])]
    pos_pointer = read_position_pointer(doc_reader)
    doc_count = [1, 1]  # count = next index to inspect
    doc_freq = [dictionary[word]['df'], len(docs)]
    docs.append(-1)
    pos_lst.append([])
    skip_width = [math.floor(math.sqrt(doc_freq[0])), math.floor(math.sqrt(doc_freq[1]))]
    while doc_count[0] <= doc_freq[0] and doc_count[1] <= doc_freq[1]:
        found = doc_id[0] == doc_id[1]
        if doc_id[0] < doc_id[1]:
            if skip_width[0] > 1:
                while doc_count[0] + skip_width[0] < doc_freq[0]:
                    d_id, _, _ = read_posting(word, doc_count[0] + skip_width[0])
                    if d_id > doc_id[1]:
                        break
                    doc_count[0] += skip_width[0]
            doc_reader.seek(base_pointer + doc_byte_width * (doc_count[0] - 1))
            for _ in range(skip_width[0] + 1):
                d_id = read_doc_id(doc_reader)
                tf = read_tf(doc_reader)
                pp = read_position_pointer(doc_reader)
                doc_count[0] += 1
                if d_id >= doc_id[1]:
                    found = d_id == doc_id[1]
                    doc_id[0] = d_id
                    term_freq[0] = tf
                    pos_pointer = pp
                    break
                if doc_count[0] >= doc_freq[0]:
                    break
        elif doc_id[0] > doc_id[1]:
            if skip_width[1] > 1:
                while doc_count[1] + skip_width[1] < doc_freq[1]:
                    d_id = docs[doc_count[1] + skip_width[1]]
                    if d_id > doc_id[0]:
                        break
                    doc_count[1] += skip_width[1]
            for _ in range(skip_width[1] + 1):
                d_id = docs[doc_count[1]]
                tf = len(pos_lst[doc_count[1]])
                doc_count[1] += 1
                if d_id >= doc_id[0]:
                    found = d_id == doc_id[0]
                    doc_id[1] = d_id
                    term_freq[1] = tf
                    break
                if doc_count[1] >= doc_freq[1]:
                    break
        if found:
            pos_reader = open(postings_file, 'rb')
            pos_reader.seek(pos_pointer)
            position_list = pos_lst[doc_count[1] - 1]
            position_list.append(-1)
            position = [read_positional_index(pos_reader), position_list[0]]
            skip_pos_width = [math.floor(math.sqrt(term_freq[0])), math.floor(math.sqrt(term_freq[1]))]
            pos_count = [1, 1]
            found2 = (position[1] - position[0] == pos[1] - pos[0])
            this_pos = []
            if found2:
                this_pos.append(position[0])
            while pos_count[0] <= term_freq[0] and pos_count[1] <= term_freq[1] and (details or not found2):
                if position[1] - position[0] > pos[1] - pos[0]:
                    if skip_pos_width[0] > 1:
                        while pos_count[0] + skip_pos_width[0] < term_freq[0]:
                            p_id = read_position(word, doc_count[0], pos_count[0] + skip_pos_width[0])
                            if position[1] - p_id < pos[1] - pos[0]:
                                break
                            pos_count[0] += skip_pos_width[0]
                    pos_reader.seek(pos_pointer + pos_byte_width * (pos_count[0] - 1))
                    for _ in range(skip_pos_width[0] + 1):
                        p_id = read_positional_index(pos_reader)
                        pos_count[0] += 1
                        if position[1] - p_id <= pos[1] - pos[0]:
                            matches = position[1] - p_id == pos[1] - pos[0]
                            found2 = matches or found2
                            position[0] = p_id
                            if matches:
                                this_pos.append(position[0])
                            break
                        if pos_count[0] >= term_freq[0]:
                            break
                elif position[1] - position[0] < pos[1] - pos[0]:
                    if skip_pos_width[1] > 1:
                        while pos_count[1] + skip_pos_width[1] < term_freq[1]:
                            p_id = position_list[pos_count[1] + skip_pos_width[1]]
                            if position[0] - p_id < pos[0] - pos[1]:
                                break
                            pos_count[1] += skip_pos_width[1]
                    for _ in range(skip_pos_width[1] + 1):
                        p_id = position_list[pos_count[1]]
                        pos_count[1] += 1
                        if position[0] - p_id <= pos[0] - pos[1]:
                            matches = position[0] - p_id == pos[0] - pos[1]
                            found2 = matches or found2
                            position[1] = p_id
                            if matches:
                                this_pos.append(position[0])
                            break
                        if pos_count[1] >= term_freq[1]:
                            break
            if found2:
                res.append(doc_id[0])
                res_pos.append(this_pos)
            pos_reader.close()
            doc_id[0] = read_doc_id(doc_reader)
            term_freq[0] = read_tf(doc_reader)
            pos_pointer = read_position_pointer(doc_reader)
            doc_count[0] += 1
            doc_id[1] = docs[doc_count[1]]
            term_freq[1] = len(pos_lst[doc_count[1]])
            doc_count[1] += 1
    doc_reader.close()
    return res, res_pos


def process_phrasal_search(text):
    """
    Implement phrasal search given the phrase
    :param text: the phrase before tokenization and normalization to search, the number of words is 1, 2, or 3
    :return: the list of documents ranked by relevance
    """
    tf, pos, _ = process_doc(text)
    lst = get_phrasal_search_list(tf, pos)
    return calculate_scores(tf, lst)


def get_phrasal_search_list(tf, pos):
    """
    Implement phrasal search given the phrase, unordered
    :param tf: the term frequencies of the query
    :param pos: the positional indices of the query
    :return: the list of relevant documents
    """
    # Initialize information needed
    res = []
    words = list(iter(tf))
    count = len(words)
    if count == 1:
        if words[0] not in dictionary:
            return []
        return get_list_for_word(words[0], pos[words[0]], False)[0]
    if count == 2:
        has_duplicates = None
        for word in words:
            if word not in dictionary:
                return []
            if tf[word] > 1:
                has_duplicates = word
        if not has_duplicates:
            return intersect_words(words[0], words[1], [pos[words[0]][0], pos[words[1]][0]], False)[0]
        else:
            list_of_duplicated, duplicated_positions = get_list_for_word(has_duplicates, pos[has_duplicates], True)
            word = words[0]
            if has_duplicates == word:
                word = words[1]
            return intersect_word_list(word, list_of_duplicated, duplicated_positions,
                                       [pos[word][0], pos[has_duplicates][0]], False)[0]
    if count == 3:
        most_frequent_word = words[0]
        highest_frequency = dictionary[words[0]]['df']
        freq1 = dictionary[words[1]]['df']
        if freq1 > highest_frequency:
            most_frequent_word = words[1]
            highest_frequency = freq1
        freq2 = dictionary[words[2]]['df']
        if freq2 > highest_frequency:
            most_frequent_word = words[2]
            highest_frequency = freq2
        first_words = []
        for w in words:
            if w != most_frequent_word:
                first_words.append(w)
        docs, poss = intersect_words(first_words[0], first_words[1],
                                     [pos[first_words[0]][0], pos[first_words[1]][0]], True)
        return intersect_word_list(most_frequent_word, docs, poss,
                                   [pos[most_frequent_word][0], pos[first_words[0]][0]], False)[0]
    return res


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
