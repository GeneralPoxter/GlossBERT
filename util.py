"""
Code based on: https://github.com/tanmaysahay94/SemCorpusReader
"""

import itertools
from collections import defaultdict
from lxml import html
from tqdm import tqdm
import os


def annotated_chunks(path):
    raw = open(path).read()
    parsed = html.fromstring(raw)
    annotated_chunks = [
        (x.tag, x.attrib, x.text.strip().split())
        for x in parsed.getiterator()
        if x.tag in ["wf", "s"]
    ]

    def helper(chunks, splitter="s"):
        return [
            list(g)
            for k, g in itertools.groupby(chunks, lambda x: x[0] == splitter)
            if not k
        ]

    annotated_chunks = helper(annotated_chunks)
    annotated_chunks = [
        [(attrib, text) for tag, attrib, text in annotated_sentence]
        for annotated_sentence in annotated_chunks
    ]
    return annotated_chunks


def read_annotated_chunks(paths):
    files = []
    for path in paths:
        files.extend([os.path.join(path, file) for file in os.listdir(path)])

    chunks = []
    for file in tqdm(files):
        if os.path.isfile(file):
            chunks.extend(annotated_chunks(file))

    return chunks


def tagged_chunks(chunks, tags=["pos"]):
    def helper(attributes):
        return tuple(attributes[t] if t in attributes else None for t in tags)

    return [
        [(*helper(attributes), chunk) for attributes, chunk in annotated_sentence]
        for annotated_sentence in chunks
    ]


def sense_tagged_chunks(chunks, filter_pn=False):
    chunks = tagged_chunks(chunks, tags=["lemma", "wnsn", "pos", "pn"])

    def helper(wnsn):
        if wnsn is None:
            return None
        # Filter wnsn=0 for words dropped from WordNet 1.6
        return [int(n) for n in wnsn.split(";") if int(n) != 0]

    return [
        [
            (lemma, helper(wnsn), pos, text)
            for lemma, wnsn, pos, pn, text in tagged_sentence
            # Filter proper nouns, since they only have one sense
            if not filter_pn or pn is None
        ]
        for tagged_sentence in chunks
    ]


def sense_frequency(path):
    chunks = read_annotated_chunks(path)
    sense_chunks = sense_tagged_chunks(chunks)

    sense_freq = defaultdict(lambda: defaultdict(int))
    for tagged_sentence in sense_chunks:
        for lemma, wnsn, _, _ in tagged_sentence:
            if lemma and wnsn:
                for n in wnsn:
                    # Filter wnsn=0 for words dropped from WordNet 1.6
                    if n:
                        sense_freq[lemma][n] += 1

    return sense_freq


# Split sentence into iterable, ignoring non-alpha characters
# such as punctuation.
def split_alpha(sentence):
    return "".join((c if c.isalpha() else " ") for c in sentence).split()
