import os
from array import array


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
            'df': entries[1],
            'ptr': entries[2]
        }
    df.close()


file = open('test.txt', 'wb')
a = array('d', [3.33, 1])
a.tofile(file)
file.close()
file = open('test.txt', 'rb')
b = array('d')
b.fromstring(file.read())
print(a[0])
print(b[1])
"""
file.write((2000).to_bytes(length=2, byteorder='big', signed=False))
file.close()
file = open('test.txt', 'rb')
print(int.from_bytes(file.read(1), byteorder='big'))
file.seek(1)
print(int.from_bytes(file.read(), byteorder='big'))
file.seek(0)
file2 = open('test2.txt', 'wb')
file2.write(file.read())
os.remove('test.txt')
file2.close()
"""
