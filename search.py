#!/usr/bin/python3
import getopt
import sys
import math
from heapq import *
from nltk.corpus import wordnet as wn

# Global variables
from file_io import load_dict, read_doc_id, read_float_bin_file, read_tf, read_position_pointer, read_positional_index
from widths import doc_byte_width, pos_byte_width, number_of_doc_used_in_relevance_feedback
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
    query = qf.readline()
    res = process_phrasal_search(query)
    # the most 10 relevant result by Cosine similarity is stored in res
    # query_dict = get_query_dict(query)
    # res = process_free_query(query_extension(query_dict))
    if not res:
        rf.write('\n')
    rf.write(str(res[0]))
    for doc_id in res[1:]:
        rf.write(' ')
        rf.write(str(doc_id))
    rf.write('\n')

    # Close files
    pf.close()
    df.close()
    qf.close()
    rf.close()


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
    :return: most relevant 10 documents as a list in descending order of relevance
    """
    res = []
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
    for doc in scores:
        if doc_len[doc] != 0:
            scores[doc] /= doc_len[doc]
        if len(res) < number_of_doc_used_in_relevance_feedback:
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
    pf.seek(postings_base_pointer + dictionary[word]['ptr'] + doc_byte_width * index)
    id = read_doc_id(pf)
    tf = read_tf(pf)
    ptr = read_position_pointer(pf)
    return id, tf, ptr


def process_phrasal_search(text):
    """
    Implement phrasal search given the phrase
    :param text: the phrase before tokenization and normalization to search
    :return: the list of relevant documents ranked by relevance
    """
    # Initialize information needed
    tf, pos, length = process_doc(text)
    indices = {}  # Pointer positions
    skips = {}
    doc_pointers = {}  # Actual pointers, file objects
    doc_content = {}
    final_doc_position = {}  # If >=, end
    position_pointers = {}  # Actual pointers, file objects
    position_indices = {}  # Position pointer positions
    positions = {}  # Position ID
    final_pointer_position = {}
    # doc_heap = []
    valid_list = []
    first_word = None
    for word in pos:
        if word not in dictionary:
            return []
        if not first_word:
            first_word = word
        indices[word] = postings_base_pointer + dictionary[word]['ptr']
        skips[word] = math.floor(math.sqrt(dictionary[word]['df'])) * doc_byte_width
        doc_pointers[word] = open(postings_file, 'rb')
        doc_pointers[word].seek(indices[word])
        doc_content[word] = (read_doc_id(doc_pointers[word]), read_tf(doc_pointers[word]),
                             read_position_pointer(doc_pointers[word]))
        final_doc_position[word] = postings_base_pointer + dictionary[word]['ptr'] \
                                   + doc_byte_width * dictionary[word]['df']
        position_pointers[word] = open(postings_file, 'rb')
        position_indices[word] = 0
        positions[word] = 0
        final_pointer_position[word] = 0
        # heappush(doc_heap, (read_doc_id(doc_pointers[word]), read_tf(), read_position_pointer(), word))
    # Process
    flag = True
    while flag:
        doc_id, _, _ = doc_content[first_word]
        is_valid = True
        for word in doc_content:
            if word == first_word:
                continue
            d_id, _, _ = doc_content[word]
            if d_id == doc_id:
                continue
            elif d_id > doc_id:
                found = get_next_doc_id(first_word, d_id, indices, doc_pointers, doc_content, skips[first_word],
                                final_doc_position[first_word])
                if not found:
                    is_valid = False
                    if indices[first_word] >= final_doc_position[first_word]:
                        flag = False
            else:
                found = get_next_doc_id(word, doc_id, indices, doc_pointers, doc_content, skips[word],
                                        final_doc_position[word])
                if not found:
                    is_valid = False
                    if indices[word] >= final_doc_position[word]:
                        flag = False
        if not is_valid:
            continue
        # Check position
        for word in position_pointers:
            _, term_freq, position_indices[word] = doc_content[word]
            position_pointers[word].seek(position_indices[word])
            positions[word] = read_positional_index(position_pointers[word])
            final_pointer_position[word] = position_indices[word] + term_freq * pos_byte_width
        pos_id = positions[first_word] - pos[first_word]
        for word in positions:
            if word == first_word:
                continue
            p_id = positions[word] - pos[word]
            if p_id == pos_id:
                continue
            elif p_id > pos_id:
                _, term_freq, _ = doc_content[word]
                found = get_next_pos_id(first_word, p_id, position_indices, position_pointers, positions,
                                        math.floor(math.sqrt(term_freq)), final_pointer_position[first_word],
                                        pos[first_word])
        """
        pf.seek(ptr + pos_byte_width * index)
        return read_positional_index(pf)
        """
        valid_list.append(doc_id)
        # Update all
    for word in doc_pointers:
        doc_pointers[word].close()
    for word in position_pointers:
        position_pointers[word].close()
    return valid_list  # rank by process_free_query


def get_next_doc_id(word, target, idx, ptr, ctnt, skip, final):  # True if word can meet target
    """
    Gets the next document ID of the word with skip pointers implemented
    :param word: the word to process
    :param target: the document ID to search for where the word exists
    :param idx: the indices dictionary of word - pointer position
    :param ptr: the pointers dictionary of word - file pointer
    :param ctnt: the content dictionary of word - (id, tf, position ptr)
    :param skip: the skip pointer width
    :param final: the final position of the word
    :return: True if the target ID is found for the word
    """
    if idx[word] + skip < final:
        pf.seek(idx[word] + skip)
        doc_id = read_doc_id(pf)
        if doc_id == target:
            idx[word] += skip
            ctnt[word] = (doc_id, read_tf(pf), read_position_pointer(pf))
            ptr[word].seek(idx[word])
            return True
        elif doc_id < target:
            idx[word] += skip
            return get_next_doc_id(word, target, idx, ptr, ctnt, skip, final)
    bound = int(skip / doc_byte_width)
    for _ in range(bound + 1):
        idx[word] += doc_byte_width
        if idx[word] >= final:
            return False
        doc_id = read_doc_id(ptr[word])
        if doc_id >= target:
            ctnt[word] = (doc_id, read_tf(ptr[word]), read_position_pointer(ptr[word]))
            return doc_id == target
        else:
            read_tf(ptr[word])
            read_position_pointer(ptr[word])


def get_next_pos_id(word, target, idx, ptr, positions, skip, final, offset):  # True if word can meet target
    """
    Gets the next position ID of the word with skip pointers implemented
    :param word: the word to process
    :param target: the position ID to search for where the word exists (with offset processed already)
    :param idx: the indices dictionary of word - position pointer position
    :param ptr: the pointers dictionary of word - file pointer
    :param positions: the positions dictionary of word - pos_id = actual pos_id - offset (pos_id in query dictionary)
    :param skip: the skip pointer width
    :param final: final position of position pointer of the word
    :param offset: the position offset of the word
    :return: True if the target ID is found for the word
    """
    if idx[word] + skip * pos_byte_width < final:
        pf.seek(idx[word] + skip * pos_byte_width)
        pos_id = read_positional_index(pf) - offset
        if pos_id == target:
            idx[word] += skip * pos_byte_width
            positions[word] = pos_id
            ptr[word].seek(idx[word])
            return True
        elif pos_id < target:
            idx[word] += skip
            return get_next_doc_id(word, target, idx, ptr, positions, skip, final)
    bound = int(skip / doc_byte_width)
    for _ in range(bound + 1):
        idx[word] += doc_byte_width
        if idx[word] >= final:
            return False
        pos_id = read_doc_id(ptr[word])
        if pos_id >= target:
            positions[word] = (pos_id, read_tf(ptr[word]), read_position_pointer(ptr[word]))
            return pos_id == target
        else:
            read_tf(ptr[word])
            read_position_pointer(ptr[word])


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
