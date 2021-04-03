#!/usr/bin/python3
import csv
import getopt
import os
import sys
from array import array
from heapq import heappush, heappop

from normalize import process_doc
from file_io import load_dict, read_doc_id, read_tf, read_position_pointer, read_positional_index, write_int_bin_file, \
    read_float_bin_file, write_float_bin_file
from widths import pos_width, post_width, tf_width, pos_pointer_width, doc_width, pos_byte_width, smallest_doc_id, \
    post_byte_width, tf_byte_width, pos_pointer_byte_width, doc_byte_width, double_byte_width


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

    # Open file
    in_file = open(in_dir, 'r')
    reader = csv.DictReader(in_file)

    # Read each file, the files are already in ascending order of document ID
    block_count = read_data(reader)

    # Write output files
    merge_blocks(block_count, out_dict, out_postings)

    # Close file
    in_file.close()


def merge_blocks(block_count, out_dict, out_postings):
    """
    Merges the blocks to write the overall postings file
    :param block_count: the number of blocks
    :param out_dict: the output dictionary file
    :param out_postings: the output postings file
    """
    # Load dictionaries and open files
    index_list = list(range(block_count))
    dictionaries = []
    posting_files = []
    positions_files = []
    lengths_files = []
    for i in index_list:
        df = open(f'dictionary{i}.txt', 'r')
        df.readline()
        dictionary = {}
        load_dict(df, dictionary)
        dictionaries.append(dictionary)
        posting_files.append(open(f'postings{i}.txt', 'rb'))
        positions_files.append(open(f'positions{i}.txt', 'rb'))
        lengths_files.append(open(f'lengths{i}.txt', 'rb'))

    # Prepare to write files
    dict_writer = open(out_dict, 'w')
    post_writer = open(out_postings, 'wb')

    # Print positions
    leading_terms = []
    dictionary_iters = []
    pos_pointers = {}
    pointer = 0
    setup_iters(dictionaries, dictionary_iters, index_list, leading_terms)
    while leading_terms:
        leading_term = heappop(leading_terms)
        update_leading_term(dictionary_iters, leading_term, leading_terms)
        word = leading_term[0]
        collections = [leading_term]
        while leading_terms and leading_terms[0][0] == word:
            term_block_info = heappop(leading_terms)
            collections.append(term_block_info)
            update_leading_term(dictionary_iters, term_block_info, leading_terms)
        pos_pointers[word] = {}
        for block in collections:
            block_index = block[1]
            df = dictionaries[block_index][word]['df']
            ptr = dictionaries[block_index][word]['ptr']
            pf = posting_files[block_index]
            posf = positions_files[block_index]
            pf.seek(ptr)
            for _ in range(df):
                doc = read_doc_id(pf)
                tf = read_tf(pf)
                pp = read_position_pointer(pf)
                posf.seek(pp)
                for _ in range(tf):
                    pos = read_positional_index(posf)
                    write_int_bin_file(post_writer, pos, pos_byte_width)
                pos_pointers[word][doc] = pointer
                pointer += tf * pos_byte_width

    # Print postings
    dictionary_iters = []
    post_pointers = {}
    doc_freq = {}
    postings_base_pointer = pointer
    pointer = 0
    setup_iters(dictionaries, dictionary_iters, index_list, leading_terms)
    while leading_terms:
        leading_term = heappop(leading_terms)
        update_leading_term(dictionary_iters, leading_term, leading_terms)
        word = leading_term[0]
        post_pointers[word] = pointer
        doc_freq[word] = 0
        collections = [leading_term]
        while leading_terms and leading_terms[0][0] == word:
            term_block_info = heappop(leading_terms)
            collections.append(term_block_info)
            update_leading_term(dictionary_iters, term_block_info, leading_terms)
        for block in collections:
            block_index = block[1]
            df = dictionaries[block_index][word]['df']
            ptr = dictionaries[block_index][word]['ptr']
            pf = posting_files[block_index]
            pf.seek(ptr)
            doc_freq[word] += df
            for _ in range(df):
                doc = read_doc_id(pf)
                tf = read_tf(pf)
                pf.read(pos_pointer_byte_width)
                write_int_bin_file(post_writer, doc, post_byte_width)
                write_int_bin_file(post_writer, tf, tf_byte_width)
                write_int_bin_file(post_writer, pos_pointers[word][doc], pos_pointer_byte_width)
                pointer += doc_byte_width

    # Print dictionary
    lengths_base_pointer = pointer + postings_base_pointer
    dictionary_iters = []
    setup_iters(dictionaries, dictionary_iters, index_list, leading_terms)
    dict_writer.write(f'{postings_base_pointer} {lengths_base_pointer}\n')
    while leading_terms:
        leading_term = heappop(leading_terms)
        update_leading_term(dictionary_iters, leading_term, leading_terms)
        word = leading_term[0]
        while leading_terms and leading_terms[0][0] == word:
            term_block_info = heappop(leading_terms)
            update_leading_term(dictionary_iters, term_block_info, leading_terms)
        dict_writer.write(f'{word} {doc_freq[word]} {post_pointers[word]}\n')

    # Print lengths
    for i in index_list:
        lf = lengths_files[i]
        doc = read_doc_id(lf)
        write_int_bin_file(post_writer, doc, post_byte_width)
        length = read_float_bin_file(lf)
        write_float_bin_file(post_writer, length)

    # Close files
    dict_writer.close()
    post_writer.close()
    for f in posting_files:
        f.close()
    for f in positions_files:
        f.close()
    for f in lengths_files:
        f.close()
    for i in index_list:
        os.remove(f'dictionary{i}.txt')
        os.remove(f'postings{i}.txt')
        os.remove(f'positions{i}.txt')
        os.remove(f'lengths{i}.txt')


def update_leading_term(dictionary_iters, term, leading_terms):
    """
    Update the leading terms after popping
    :param dictionary_iters: the list of iterable dictionary terms for each block
    :param term: the term just popped
    :param leading_terms: the heap of leading terms to update
    """
    block_index = term[1]
    next_term = next(dictionary_iters[block_index], None)
    if next_term:
        heappush(leading_terms, (next_term, block_index))


def setup_iters(dictionaries, dictionary_iters, index_list, leading_terms):
    """
    Set up iterable lists for dictionaries for each block
    :param dictionaries: the dictionaries for each block
    :param dictionary_iters: the iterable lists for each block
    :param index_list: the block indices
    :param leading_terms: the heap of terms to initialize
    """
    for i in index_list:
        dict_iter = iter(dictionaries[i])
        dictionary_iters.append(dict_iter)
        heappush(leading_terms, (next(dictionary_iters[i]), i))


def read_data(reader):
    """
    Reads in data from the file reader of dataset input
    :param reader: the file reader of dataset input
    :return the number of blocks
    """
    block_size = 500
    block_index = 0
    block_count = 0
    dictionary = {}
    doc_freq = {}
    doc_len = {}
    for row in reader:
        if block_count >= block_size:
            write_temp(doc_freq, dictionary, doc_len, block_index)
            doc_freq = {}
            dictionary = {}
            doc_len = {}
            block_index += 1
            block_count = 0
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
        block_count += 1
    write_temp(doc_freq, dictionary, doc_len, block_index)
    return block_index + 1


def write_temp(doc_freq, dictionary, doc_len, block_index):
    """
    Writes the information from dataset file to dictionary and postings
    :param doc_freq: the document frequency dictionary
    :param dictionary: the dictionary containing the document ID, term frequency, and positional indices list of each
    word of each document that contains this term
    :param doc_len: the dictionary of the vector length of each document
    :param block_index: the index of the block
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
    index_string = str(block_index)
    dict_writer = open(f'dictionary{index_string}.txt', 'w')
    post_writer = open(f'postings{index_string}.txt', 'wb')
    pos_writer = open(f'positions{index_string}.txt', 'wb')
    len_writer = open(f'lengths{index_string}.txt', 'wb')
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
                pos_writer.write(pos.to_bytes(length=pos_byte_width, byteorder='big', signed=False))
                pointer += pos_byte_width

    # Print the postings
    postings_pointer = pointer
    pointer = 0
    for word in words:
        post_pointers[word] = pointer
        for posting in dictionary[word]:
            post_writer.write((posting['id'] - smallest_doc_id)
                              .to_bytes(length=post_byte_width, byteorder='big', signed=False))
            post_writer.write((posting['tf']).to_bytes(length=tf_byte_width, byteorder='big', signed=False))
            post_writer.write((pos_pointers[word][posting['id']])
                              .to_bytes(length=pos_pointer_byte_width, byteorder='big', signed=False))
            pointer += doc_byte_width

    # Print lengths
    lengths_pointer = pointer + postings_pointer
    for doc in doc_len:
        len_writer.write((doc - smallest_doc_id).to_bytes(length=post_byte_width, byteorder='big', signed=False))
        len_float = array('d', [doc_len[doc]])
        len_float.tofile(len_writer)

    # Print dictionary
    dict_writer.write(f"{postings_pointer} {lengths_pointer}\n")
    for word in words:
        dict_writer.write(f"{word} {str(doc_freq[word])} {post_pointers[word]}\n")

    # Close files
    dict_writer.close()
    post_writer.close()
    pos_writer.close()
    len_writer.close()


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
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
