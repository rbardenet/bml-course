import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

class CollapsedGibbs:
    """check out [Murphy 2012, Section 27.3.4]
    """
    def __init__(self, corpus, num_topics, num_iterations, seed):
        """
        """
        npr.seed(seed)

        # save arguments
        split_corpus = [doc.split() for doc in corpus] # store documents as lists of words
        self.num_topics = num_topics
        self.num_iterations = num_iterations

        # process dataset
        self.num_documents = len(split_corpus)
        self.document_lengths = [len(doc) for doc in split_corpus]
        self.maximum_document_length = np.max(self.document_lengths)
        vectorizer = CountVectorizer(max_df=0.95,  
                                min_df=2, # select words that appear at least twice
                                max_features=1000,
                                stop_words='english') # output is a sparse matrix
        vectorizer.fit_transform(corpus)
        self.vocabulary = vectorizer.get_feature_names()
        self.num_words = len(self.vocabulary)
        
        self.corpus = [[word.lower() for word in doc if word.lower() in self.vocabulary] 
                       for doc in split_corpus] # store corpus as list of words that belong to our dictionary
        
        # Initialize LDA variables, later on you can fit them
        self.alpha = .1
        self.gamma = .1
        
        # Initialize counts and the word-topic assignment (the hidden variable z in the lectures)
        self.count_topic_word = np.zeros((self.num_topics, 
                                          self.num_words), dtype = int)
        self.count_document_topic = np.zeros((self.num_documents, 
                                              self.num_topics), dtype = int)
        self.assign = -np.ones((self.num_documents, self.maximum_document_length, 
                                num_iterations+1), dtype = int) # -1 codes for a index that is larger than the size of the document
        
        for i, doc in enumerate(self.corpus):
            for l, word in enumerate(doc):

                # update counters
                v = self.vocabulary.index(word)
                k = npr.randint(self.num_topics)
                self.assign[i, l, 0] = k  # initialize randomly
                self.count_topic_word[k, v] += 1
                self.count_document_topic[i, k] += 1        
                
        # print("After initialization", self.count_topic_word)        
        # print("After initialization", self.count_document_topic)        
                
    def run(self):
        """
        """
        for t in range(self.num_iterations):
            for i, doc in enumerate(self.corpus):
                for l, word in enumerate(doc):
                    
                    # decrement counters         
                    v = self.vocabulary.index(word)
                    k = self.assign[i, l, t] # get current topic assignment                    
                    self.count_topic_word[k, v] -= 1
                    self.count_document_topic[i, k] -= 1
                    
                    # compute conditional probability of assignment
                    p = []
                    for k in range(self.num_topics):
                        _1 = (self.count_topic_word[k,v] + self.gamma) / (self.count_topic_word[k,:].sum() + self.num_words*self.gamma)
                        _2 = (self.count_document_topic[i,k] + self.alpha) / (self.document_lengths[i] + self.num_topics*self.alpha)
                        p.append(_1*_2)
                    p = np.array(p)/np.sum(p)
                    
                    # sample new assignment
                    try:
                        k = npr.choice(self.num_topics, p=p)
                    except:
                        print("At iteration", t, ", the vector of probabilities is not valid")
                    
                    # Record new assignment and update counters
                    self.assign[i, l, t+1] = k
                    self.count_topic_word[k, v] += 1
                    self.count_document_topic[i, k] += 1
                    
                        
                    
        