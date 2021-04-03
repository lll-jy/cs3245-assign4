import os
from array import array
from heapq import heappush, heappop


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
