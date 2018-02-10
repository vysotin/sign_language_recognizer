import math
import statistics as st
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
#from functools import lru_cache
import logging
#logging.getLogger().setLevel(logging.INFO)


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False, hmm_iterations=1000):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.hmm_iterations = hmm_iterations
       
    def select(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)    
        best_n = self.find_best_n()
        return self.base_model(best_n) 

    def base_model(self, num_states):
        
        try:
            
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("New model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
        
    """
    Method used by select method implementation for every individual class
    Arguments:
        evaluate_fun  function for evaluation perfomance measure 
        additional_args  object that is passed as additional argument to the evaluate_fun
    return:
        best_n  best choice for the number of states

    """    
    def find_best_n(self):
        max_score = float('-inf')
        best_n = None
        for n_state in range(self.min_n_components, self.max_n_components+1):
            try:
                logL_score_n = self.evaluate_score(n_state)
                logging.debug("LogL score = {} for selector {}".format(logL_score_n,self.__class__.__name__)) 
                if logL_score_n and logL_score_n > max_score:
                    max_score = logL_score_n
                    best_n = n_state
            except Exception as err:
                logging.debug("Error: {} for selector {} n_state={}".format(err,self.__class__.__name__, n_state ))   
                
        logging.debug("Best n_state={}, best logl={}, selector method {} ".format(best_n, max_score,self.__class__.__name__))    
        return best_n
    
    def evaluate_score(self, n_state):
        raise NotImplementedError
    
    def calculate_logl(self, n_states):
      
        # calculate the logarithm of the likelyhood
        try:
            
            model = GaussianHMM(n_components=n_states, n_iter=self.hmm_iterations, covariance_type="diag", random_state=self.random_state).fit(self.X,self.lengths)
            logl = model.score(self.X,self.lengths)
            return (model,logl)
        except Exception as err:
            logging.debug("calculate_logl: evaluation failed for the word = {}, num_states = {}, reason: {}".format(self.this_word, n_states, err))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
       
        
    def evaluate_score(self, n_states):
        
        try:
            model, logl = self.calculate_logl(n_states)
        except Exception as err:
           logging.debug("evaluate_score for BIC: evaluation failed for the word = {0}, num_states = {1}, reason: {2}".format(self.this_word, n_states, err))
           return None
        num_features = len(self.X[0,:])
        logging.debug("evaluate_score for BIC: num_features ={}".format(num_features))
        num_free_param = n_states*n_states + 2*num_features*n_states - 1
        if logl:
            return logl - 0.5*num_free_param*math.log(len(self.sequences))
        else:
            return None

  
   

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def evaluate_score(self, n_states):
        model, logl = self.calculate_logl(n_states)
        if not logl:
            return None

        other_words_score_sum = 0.0
        other_words_count = 0
        for word in self.words:
            if word != self.this_word:
                seq,lengths = self.hwords[word]
                try:
                    logging.debug("evaluate_score DIC: calculating score for the word = {} on the model based on the word {}, num_states = {}".format(word, self.this_word, n_states))
                    word_score = model.score(seq,lengths)
                    logging.debug("evaluate_score DIC:  score for the word = {} on the model based on the word {}  is {}, num_states = {}".format(word,  self.this_word, word_score, n_states))
                    if word_score:
                        other_words_score_sum += word_score
                        other_words_count += 1
                except Exception as err:
                    logging.debug("evaluate_score DIC: score evaluation failed for the word = {}, given model based on the word {},  num_states = {}, reason: {}".format(word, self.this_word, n_states, err))
        penalty = 0.0
        if other_words_count != 0:
            penalty = other_words_score_sum/other_words_count            
        return logl - penalty

  


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False, hmm_iterations=1000):
       
        super(SelectorCV,self).__init__(all_word_sequences, all_word_Xlengths, this_word, 
                 n_constant,  min_n_components, max_n_components,
                 random_state, verbose, hmm_iterations)
        self.n_splits = min(len(self.sequences), 10)
        if self.n_splits > 1:
            self.split_method = KFold(n_splits = self.n_splits )
         
    
    def evaluate_score(self,n_states):
        logging.debug("Entered model_cv_score for n_states= {0}".format(n_states))
        if self.n_splits < 2:
            logging.debug("Number of splits is 1 for the  word {0}".format(self.this_word))
            model, logl = self.calculate_logl(n_states) 
            return logl

        logl_all = list()
        logging.debug("Entered model_cv_score for n_states= {0}".format(n_states))
        for cv_train_idx, cv_test_idx in self.split_method.split(self.sequences):
            try:
                combined_train_seq,train_lengths = combine_sequences(cv_train_idx, self.sequences)
                combined_test_seq,test_lengths = combine_sequences(cv_test_idx, self.sequences)
                model = GaussianHMM(n_components=n_states, n_iter=self.hmm_iterations, random_state=self.random_state, covariance_type="diag").fit(combined_train_seq,train_lengths)
                logl = model.score(combined_test_seq,test_lengths)
                logging.debug("In model_cv_score: LogL= {0}".format(logl))
                logl_all.append(logl)
            except Exception as err:
                logging.debug("In model_cv_score: Error: {0}".format(err))    
                pass
        if not logl_all:
            return None
        return st.mean(logl_all)  
  