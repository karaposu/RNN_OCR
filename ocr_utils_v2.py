import io
import os
import unicodedata
import string
import glob
import torch
import random
import unique_char_finder_from_csv



Alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
# ALPHABET_OF_DATASET=unique_char_finder_from_csv.find_unique_chars_from_csv( "files_path.csv",3, ALphabet=Alphabet)

ALPHABET_OF_DATASET="bcdefgmnpwxy2345678+"

N_ALPHABET_OF_DATASET =  len(ALPHABET_OF_DATASET)
print("Alphabet used in this dataset is: ", ALPHABET_OF_DATASET)




"""
To represent a single letter, we use a “one-hot vector” of
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.

To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.

That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
"""


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALPHABET_OF_DATASET.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_ALPHABET_OF_DATASET)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_ALPHABET_OF_DATASET)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def line_to_1_tensor(line):
    tensor = torch.zeros(1, 1, N_ALPHABET_OF_DATASET)
    for i, letter in enumerate(line):
        tensor[0][letter_to_index(letter)] = 1
    return tensor