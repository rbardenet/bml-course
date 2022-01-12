import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


class CollapsedGibbs:
    """check out [Murphy 2012, Section 27.3.4]"""

    def __init__(self, corpus, num_topics, num_iterations, seed):
        """ """
        npr.seed(seed)

        # save arguments
        split_corpus = [
            doc.split() for doc in corpus
        ]  # temporarily transform documents to lists of words
        self.num_topics = num_topics
        self.num_iterations = num_iterations  # number of full (systematic) Gibbs sweeps: one iteration is one pass through all variables.

        # process dataset
        self.num_documents = len(split_corpus)
        self.document_lengths = [len(doc) for doc in split_corpus]
        self.maximum_document_length = np.max(self.document_lengths)
        vectorizer = CountVectorizer(
            max_df=0.95,  # I use a sklearn feature extractor to create the vocabulary
            min_df=2,  # remove words that are too infrequent
            max_features=1000,
            stop_words="english",
        )  # remove stop words
        vectorizer.fit_transform(corpus)
        self.vocabulary = (
            vectorizer.get_feature_names()
        )  # a list of words that meet our frequency requirements
        self.num_words = len(self.vocabulary)  # this is V in [Murphy 2012].

        self.corpus = [
            [word.lower() for word in doc if word.lower() in self.vocabulary]
            for doc in split_corpus
        ]  # store corpus as list of words that belong to our dictionary

        # Initialize LDA top-level parameters. If you've solved all questions, try to fit alpha and gamma for bonus points.
        self.alpha = 0.1
        self.gamma = 0.1

        # Initialize counts and the word-topic assignment (the hidden variable z in [Murphy 2012])
        self.count_topic_word = np.zeros((self.num_topics, self.num_words), dtype=int)
        self.count_document_topic = np.zeros(
            (self.num_documents, self.num_topics), dtype=int
        )
        self.assign = -np.ones(
            (self.num_documents, self.maximum_document_length, num_iterations + 1),
            dtype=int,
        )  # -1 codes for an index that is larger than
        # the size of the document
        # Draw an initial assignment for each word
        for i, doc in enumerate(self.corpus):
            for l, word in enumerate(doc):

                # draw and assign
                k = npr.randint(self.num_topics)
                self.assign[i, l, 0] = k  # initialize randomly

                # update counters
                v = self.vocabulary.index(word)
                self.count_topic_word[k, v] += 1
                self.count_document_topic[i, k] += 1

    def run(self):
        """ """
        for t in range(self.num_iterations):
            for i, doc in enumerate(self.corpus):
                for l, word in enumerate(doc):

                    # decrement counters
                    v = self.vocabulary.index(word)  # index of the word
                    k = self.assign[
                        i, l, t
                    ]  # get current topic assignment to that word
                    self.count_topic_word[k, v] -= 1
                    self.count_document_topic[i, k] -= 1

                    # compute conditional probability vector of topic assignment
                    ###### TBC during the exam #######

                    # sample new assignment
                    ###### TBC during the exam #######
                    # k =

                    # Record new assignment and update counters
                    self.assign[i, l, t + 1] = k
                    self.count_topic_word[k, v] += 1
                    self.count_document_topic[i, k] += 1
