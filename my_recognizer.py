import warnings
from asl_data import SinglesData
import logging

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # iterarate through through indexes in the word list
    for ind in range(len(test_set.wordlist)):
        prob_dict = dict()
        X,lengths = test_set.get_item_Xlengths(ind)
        # iterate through the key/values of trained model dictionary
        for word, word_model in models.items():
            try:
                prob_dict[word] = word_model.score(X,lengths)
            except Exception as err:
                #log error and continue
                test_word = test_set.wordlist[ind]
                logging.debug("Error recognizing word {} using model for the word {}. Reason {}".format(test_word,word, err))
                pass
        probabilities.append(prob_dict)
        guesses.append(max(prob_dict, key=prob_dict.get))
    return (probabilities, guesses)