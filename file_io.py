import os
from array import array
from heapq import heappush, heappop

from widths import post_byte_width, tf_byte_width, pos_pointer_byte_width, pos_byte_width


def load_dict(df, sub_dict):
    """
    Loads the dictionary given the dictionary file
    :param df: the dictionary file reader.
    :param sub_dict: the dictionary result.
    :return: none
    """
    terms = df.readlines()
    for word_str in terms:
        word_str = word_str[:-1]
        entries = word_str.split(' ')
        sub_dict[entries[0]] = {
            'df': int(entries[1]),
            'ptr': int(entries[2])
        }
    df.close()


def read_int_bin_file(file, width):
    """
    Reads an integer using n-byte to represent from file
    :param file: the file to read
    :param width: the number of bytes used to represent the integer
    :return: the integer to read from the current pointer at file
    """
    return int.from_bytes(file.read(width), byteorder='big', signed=False)


def read_doc_id(file):
    """
    Reads the doc ID from a file
    :param file: the file to read
    :return: the document ID from the current pointer
    """
    return read_int_bin_file(file, post_byte_width)


def read_tf(file):
    """
    Reads a term frequency from a file
    :param file: the file to read
    :return: the term frequency from the current pointer
    """
    return read_int_bin_file(file, tf_byte_width)


def read_position_pointer(file):
    """
    Reads a positional index pointer from a file
    :param file: the file to read
    :return: the positional index pointer from the current pointer to file
    """
    return read_int_bin_file(file, pos_pointer_byte_width)


def read_positional_index(file):
    """
    Reads a positional index from a file
    :param file: the file to read
    :return: the positional index from the current pointer
    """
    return read_int_bin_file(file, pos_byte_width)


def write_int_bin_file(file_writer, number, width):
    """
    Writes an integer to a binary file
    :param file_writer: the file to write to
    :param number: the number to write
    :param width: the number of bytes needed to represent the number
    """
    file_writer.write(number.to_bytes(length=width, byteorder='big', signed=False))


"""
file = open('test.txt', 'wb')
a = array('d', [3.33, 1, 9.8])
a.tofile(file)
a2 = array('d', [9.99, 10.5])
a2.tofile(file)
file.close()
file = open('test.txt', 'rb')
b = array('d')
file.seek(16)
b.fromstring(file.read(16))
print(a[1])
print(b)
file.write((2000).to_bytes(length=2, byteorder='big', signed=False))
file.close()
file = open('test.txt', 'rb')
print(int.from_bytes(file.read(1), byteorder='big'))
file.seek(1)
print(int.from_bytes(file.read(), byteorder='big'))
file.seek(0)
"""
