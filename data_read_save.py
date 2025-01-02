import gensim
import numpy as np
import pandas as pd


def read_sequences(file):
    import re, os, sys

    # 读取包含文本内容的txt文件
    with open(file, 'r') as file:
        content = file.read()

    # 按照 '>' 分割文本块
    sequences = content.split('>')[1:]  # 从第一个分割项开始到最后一个项

    # 分割文本块
    seqs = []
    for seq in sequences:
        array = seq.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        sequence = re.sub('U', 'T', sequence)  # replace U as T
        # sequence=sequence[25:176]
        seqs.append([header, sequence])
    return seqs;


def complementary_pairing(dna_sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement_dict[base] for base in dna_sequence)
