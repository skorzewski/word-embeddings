#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys

import numpy as np
import pandas as pd


def get_vec(word, embeddings_df):
    """Find word on the list of word_embeddings and return its vector"""
    return embeddings_df.loc[embeddings_df[0] == word].values[:, 1:]


def vector_len(v):
    return math.sqrt(sum([x * x for x in v]))


def dot_product(v1, v2):
    assert len(v1) == len(v2)
    return sum([x * y for (x, y) in zip(v1, v2)])


def cosine_similarity(v1, v2):
    """
    Returns the cosine of the angle between the two vectors.
    Results range from -1 (very different) to 1 (very similar).
    """
    return dot_product(v1, v2) / (vector_len(v1) * vector_len(v2))


def similarity_calculator(vec):
    return lambda x: cosine_similarity(vec, x)


def find_word(vec, embeddings_df, skip_first=False):
    vecs = embeddings_df.values[:, 1:]
    words = embeddings_df.values[:, 0]
    similarity_to_vec = similarity_calculator(vec)
    similarities = np.apply_along_axis(similarity_to_vec, 1, vecs)
    order = np.argsort(-similarities)
    words_sorted = words[order]
    if skip_first:
        return words_sorted[1:6]
    return words_sorted[:5]


def main(filename):
    """Main function"""
    print("Reading embeddings from file... ", end="", flush=True)
    embeddings = pd.read_csv(filename, sep=" ", header=None)
    print("DONE")
    while True:
        print("> ", end="")
        words = input().split()
        if len(words) == 1:
            a = words[0]
            avs = get_vec(a, embeddings)
            if not avs.any():
                print('No vector for "{}"'.format(a))
                continue
            for av in avs:
                similar = find_word(av, embeddings, skip_first=True)
                print(av)
                print(similar)
        elif len(words) == 2:
            a, b = words
            avs = get_vec(a, embeddings)
            if not avs.any():
                print('No vector for "{}"'.format(a))
                continue
            bvs = get_vec(b, embeddings)
            if not bvs.any():
                print('No vector for "{}"'.format(b))
                continue
            for av in avs:
                for bv in bvs:
                    print(cosine_similarity(av, bv))
        elif len(words) == 3:
            a, b, c = words
            avs = get_vec(a, embeddings)
            if not avs.any():
                print('No vector for "{}"'.format(a))
                continue
            bvs = get_vec(b, embeddings)
            if not bvs.any():
                print('No vector for "{}"'.format(b))
                continue
            cvs = get_vec(c, embeddings)
            if not cvs.any():
                print('No vector for "{}"'.format(c))
                continue
            for av in avs:
                for bv in bvs:
                    for cv in cvs:
                        dv = cv + bv - av
                        d = find_word(dv, embeddings, skip_first=True)
                        print(d)


if __name__ == "__main__":
    main(sys.argv[1])
