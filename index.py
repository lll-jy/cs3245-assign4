#!/usr/bin/python3
import csv
import getopt
import sys
from array import array

from normalize import process_doc
from widths import pos_width, post_width, tf_width, pos_pointer_width, doc_width, pos_byte_width, smallest_doc_id, \
    post_byte_width, tf_byte_width, pos_pointer_byte_width, doc_byte_width


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
    # read_data(dictionary, doc_freq, doc_len, reader)
    block_count = read_data(reader)

    # Write output files
    merge_blocks(block_count, out_dict, out_postings)
    # write_dict_postings(doc_freq, dictionary, doc_len, out_dict, out_postings)

    # Close file
    in_file.close()
    print('end')


def merge_blocks(block_count, out_dict, out_postings):
    """
    Merges the blocks to write the overall postings file
    :param block_count: the number of blocks
    :param out_dict: the output dictionary file
    :param out_postings: the output postings file
    """
    for i in list(range(0, block_count)):
        print(1)


#def read_data(dictionary, doc_freq, doc_len, reader):
    """
    Reads in data from the file reader of dataset input
    :param dictionary: the dictionary containing the document ID, term frequency, and positional indices list of each
    word of each document that contains this term
    :param doc_freq: the document frequency dictionary
    :param doc_len: the document vector lengths dictionary
    :param reader: the file reader of dataset input
    """
def read_data(reader):
    """
    Reads in data from the file reader of dataset input
    :param reader: the file reader of dataset input
    :return the number of blocks
    """
    block_size = 10
    block_index = 0
    block_count = 0
    dictionary = {}
    doc_freq = {}
    doc_len = {}
    for row in reader:
        if block_count >= block_size:
            write_temp(doc_freq, dictionary, doc_len, block_index)
            write_dict_postings(doc_freq, dictionary, doc_len,
                                f'dictionary_text{block_index}.txt', f'postings_text{block_index}.txt')
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
        if block_index > 3:
            break
    write_temp(doc_freq, dictionary, doc_len, block_index)
    write_dict_postings(doc_freq, dictionary, doc_len,
                        f'dictionary_text{block_index}.txt', f'postings_text{block_index}.txt')
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
        post_pointers[word] = {}
        for posting in dictionary[word]:
            post_pointers[word] = pointer
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

    max_positional_index = 0
    max_doc_id = 0
    max_term_freq = 0
    max_positions_pointer = 0
    # Get the positions pointer and print positions to the postings file
    for word in words:
        pos_pointers[word] = {}
        for posting in dictionary[word]:
            pos_pointers[word][posting['id']] = pointer
            for pos in posting['positions']:
                post_writer.write(f'{str(pos).zfill(pos_width)} ')
                max_positional_index = max(pos, max_positional_index)
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
            max_term_freq = max(posting['tf'], max_term_freq)
            post_writer.write(f"{str(posting['id']).zfill(post_width)} {str(posting['tf']).zfill(tf_width)} "
                              f"{str(pos_pointers[word][posting['id']]).zfill(pos_pointer_width)} ")
            pointer += doc_width
        post_writer.write('\n')
        pointer += 1

    # Print lengths
    lengths_pointer = pointer + postings_pointer
    for doc in doc_len:
        max_doc_id = max(doc, max_doc_id)
        post_writer.write(f"{doc} {doc_len[doc]}\n")

    # Print dictionary
    dict_writer.write(f"{postings_pointer} {lengths_pointer}\n")
    for word in words:
        dict_writer.write(f"{word} {str(doc_freq[word])} {post_pointers[word]}\n")

    # Close files
    dict_writer.close()
    post_writer.close()

    # Dummy
    max_positions_pointer = postings_pointer
    print(max_positional_index)
    print(max_doc_id)
    print(max_term_freq)
    print(max_positions_pointer)


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
