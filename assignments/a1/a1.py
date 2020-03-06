import collections
import nltk
import operator
import pandas as pd
import numpy as np

from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('gutenberg')

# Task 1 (1 mark)


def get_top_stems(document, n, stopwords):
    """Return a list of the n most frequent stems, sorted by frequency in descending 
    order. Make sure that the list of stems returned is lowercased, and that the 
    comparison with the list of stop words is not case-sensitive and it is 
    performed before stemming.
    >>> my_document = "This is a sentence. This is another sentence. One more sentence."
    >>> get_top_stems(my_document, 3, ['.', 'is', 'this', 'the'])
    ['sentenc', 'a', 'anoth']
    >>> get_top_stems(emma, 10, my_stopwords)
    ['mr.', "'s", 'emma', 'could', 'would', 'mrs.', 'miss', 'must', 'harriet', 'much']
    """
    porter = PorterStemmer()
    words = word_tokenize(document)

    lower_case_stopwords = [w.lower() for w in stopwords]
    filtered_words = [
        w for w in words if w.lower() not in lower_case_stopwords]

    stems = []
    for word in filtered_words:
        stems.append(porter.stem(word))

    stem_frequency = nltk.FreqDist(stems)
    stem_frequency_list = list(stem_frequency.items())
    stem_frequency_list.sort(key=operator.itemgetter(1), reverse=True)
    top_stems = [key for key, value in stem_frequency_list]
    return top_stems[:n]

# Task 2 (1 mark)


def sentence_to_set(sentence, list_of_stems):
    """Return the set of all stems in a sentence that match the list of stems.
    Make sure that the resulting stems are lowercased, and that the comparison 
    with the list of stems is not case sensitive.

    >>> sentence_to_set("Find the sentence stems of this sentence.", ['sentenc', 'stem']) == {'stem', 'sentenc'}
    True
    >>> long_sentence = "Emma Woodhouse, handsome, clever, and rich, with a comfortable home and happy disposition, seemed to unite some of the best blessings of existence; and had lived nearly twenty-one years in the world with very little to distress or vex her."
    >>> sentence_to_set(long_sentence, ["seem", "chapter", "emma", "the"]) == {'the', 'emma', 'seem'}
    True
    """
    porter = PorterStemmer()
    words = word_tokenize(sentence)

    lower_case_list_of_stems = [s.lower() for s in list_of_stems]
    stems = []
    for word in words:
        stems.append(porter.stem(word).lower())

    filtered_stems = [s for s in stems if s in lower_case_list_of_stems]

    return set(filtered_stems)

# The following function uses your definition of sentence_to_set; do not modify it.
# You will use it in the following tasks.


def compute_jaccard(sentence1, sentence2, list_of_stems):
    """Return the jaccard similarity of two sentences. Refer to this link for the
    formula of the jaccard similarity: https://en.wikipedia.org/wiki/Jaccard_index
    To compute the jaccard similarity, call to your definition of sentence_to_set in
    order to convert a sentence into a set of stems.
    >>> compute_jaccard("This is sentence 1.", "This is another sentence 2.", ['thi', 'sentenc', 'anoth'])
    0.666...
    >>> long_sentence1 = "Emma Woodhouse, handsome, clever, and rich, with a comfortable home and happy disposition, seemed to unite some of the best blessings of existence"
    >>> long_sentence2 = "Emma Woodhouse, handsome, smart, and not poor, with a cosy house and happy disposition, seemed to unite some of the best blessings of existence"
    >>> compute_jaccard(long_sentence1, long_sentence2, ['emma', 'clever', 'seem', 'bless'])
    0.75
    """
    words1 = sentence_to_set(sentence1, list_of_stems)
    words2 = sentence_to_set(sentence2, list_of_stems)

    if len(words1) + len(words2) == 0:
        return 0
    return len(words1 & words2)/len(words1 | words2)

# Task 3 (1 mark)


def get_transition_matrix(list_sentences, list_of_stems, threshold=0.5):
    """Return the transition matrix as a numpy array. To compute the transition
    matrix, use compute_jaccard to find the similarity between two sentences. 
    Sentence 1 links to sentence 2 if their jaccard similarity is larger or equal 
    than the threshold.
    >>> s1 = "This is sentence 1."
    >>> s2 = "This is another sentence 2."
    >>> s3 = "This is another sentence 3."
    >>> s4 = "Another 3 sentences above."
    >>> sentences = [s1, s2, s3, s4]
    >>> get_transition_matrix(sentences, ['thi', 'sentenc', 'anoth', '3'], 0.7)
    array([[1.        , 0.        , 0.        , 0.        ],
           [0.        , 0.5       , 0.33333333, 0.        ],
           [0.        , 0.5       , 0.33333333, 0.5       ],
           [0.        , 0.        , 0.33333333, 0.5       ]])
    >>> get_transition_matrix(sentences, ['thi', 'sentenc', 'anoth', '3'], 0.6)
    array([[0.5       , 0.33333333, 0.        , 0.        ],
           [0.5       , 0.33333333, 0.33333333, 0.        ],
           [0.        , 0.33333333, 0.33333333, 0.5       ],
           [0.        , 0.        , 0.33333333, 0.5       ]])
    >>> get_transition_matrix(sentences, ['thi', '2'], 0.7)
    array([[0.5 , 0.  , 0.5 , 0.25],
           [0.  , 1.  , 0.  , 0.25],
           [0.5 , 0.  , 0.5 , 0.25],
           [0.  , 0.  , 0.  , 0.25]])
    """

    i = 0
    j = 0
    arr = np.array([])

    while i < len(list_sentences) - 1:
        linked = []
        while j < len(list_sentences) - 1:
            if i == j:
                linked.append(0)
            else:
                jaccard_score = compute_jaccard(
                    list_sentences[i], list_sentences[j], list_of_stems)
                if jaccard_score >= threshold:
                    linked.append(1.)
                else:
                    linked.append(0)
            j += 1
        counts = Counter(linked)
        if counts[1.] == 0:
            scaled_links = linked
        else:
            scaled_links = [x/counts[1.] for x in linked]
        if arr.size == 0:
            arr = np.array([scaled_links])
        else:
            # TODO: fix this error "all the input array dimensions except for the concatenation axis must match exactly"
            arr = np.append(arr, [scaled_links], axis=0)
        i += 1

    transition_matrix = arr.transpose()
    return transition_matrix

# Task 4 (1 mark)


def compute_pagerank(list_sentences, list_of_stems, threshold=0.5, damping_factor=0.85, epsylon=0.01):
    """Return the pagerank of the list of sentences.
    >>> s1 = "This is sentence 1."
    >>> s2 = "This is another sentence 2."
    >>> s3 = "This is another sentence 3."
    >>> s4 = "Another 3 sentences above."
    >>> sentences = [s1, s2, s3, s4]
    >>> compute_pagerank(sentences, ['thi', 'sentenc', 'anoth', '3'], 0.7)
    array([[0.25     ],
           [0.218...],
           [0.312...],
           [0.218...]])
    >>> compute_pagerank(sentences, ['thi', 'sentenc', 'anoth', '3'], 0.7, 0.5, 0.001)
    array([[0.25 ...],
           [0.230...],
           [0.288...],
           [0.230...]])
    """
    A = get_transition_matrix(list_sentences, list_of_stems, threshold)
    M = damping_factor * A + 0.15 * (1./4. * np.ones((4, 4)))

    iterations = 0
    PR = 1./4. * np.ones((4, 1))
    oldPR = np.zeros((4.1))
    while max(np.abs(oldPR-PR)) > epsylon:
        oldPR = PR
        PR = np.dot(M, PR)
        iterations += 1
    print(PR)
    return PR

# Task 5 (1 mark)


def summarise(text, list_of_stems, N=3, threshold=0.5, damping_factor=0.85, epsylon=0.01):
    """Return the N most important sentences according to the PageRank algorithm. The sentences must be returned
    in the order of occurrence in the original list of sentences. Your program needs to split the original text into
    its sentences, then compute the PageRank, and return the list of most important sentences.
    >>> s1 = "This is sentence 1. "
    >>> s2 = "This is another sentence 2. "
    >>> s3 = "This is another sentence 3. "
    >>> s4 = "Another 3 sentences above."
    >>> text = s1+s2+s3+s4
    >>> summarise(text, ['thi', 'sentenc', 'anoth', '3'], 2, 0.7)
    ['This is sentence 1.', 'This is another sentence 3.']
    >>> text = emma[:2000]
    >>> top_stems = ['mr.', "'s", 'emma', 'could', 'would', 'mrs.', 'miss', 'must', 'harriet', 'much']
    >>> for s in summarise(text, top_stems , 3, 0.3):
    ...    print(s)
    Even before Miss Taylor had ceased to hold the nominal
    office of governess, the mildness of her temper had hardly allowed
    her to impose any restraint; and the shadow of authority being
    now long passed away, they had been living together as friend and
    friend very mutually attached, and Emma doing just what she liked;
    highly esteeming Miss Taylor's judgment, but directed chiefly by
    her own.
    The real evils, indeed, of Emma's situation were the power of having
    rather too much her own way, and a disposition to think a little
    too well of herself; these were the disadvantages which threatened
    alloy to her many enjoyments.
    It was Miss
    Taylor's loss which first brought grief.
    """
    sents = nltk.sent_tokenize(text)
    PR = compute_pagerank(sents, list_of_stems, threshold,
                          damping_factor, epsylon)
    print(PR)
    return []


# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    nltk.download('stopwords')
    import doctest
    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    my_stopwords = nltk.corpus.stopwords.words(
        'english') + [',', '.', ';', "''", ':', '``', '?', '--', '!']
    doctest.testmod(optionflags=doctest.ELLIPSIS)
