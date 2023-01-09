from simple_corrector import words_loader
from typo_distance import typoDistance, normalized_edit_similarity
from rapidfuzz import process
import rapidfuzz
import joblib
import re

def return_nearest(word, corpus, n=10, 
                    scorer=rapidfuzz.distance.DamerauLevenshtein.normalized_similarity):
    """
    Function to return `n` most similar words based on
    Damerau-Levenshtein normalized similarity from rapidfuzz (default).
    You may change the scorer.

    Args:
    - word(str): input word to be replaced
    - corpus(dict): Corpus dictionary, obtained from Count class
    - n(int): number of candidates

    Return:
    - nearest(list): A nested list consist of tuples of three elements (suggested_word, similarity, index)
    """

    word_list = list(corpus.keys())
    nearest = process.extract(word, word_list, limit=n, 
                            scorer=scorer)
    return nearest

def correction(word, corpus, n_return=1, threshold=0.85, include_score=False, typo_dist=False):
    """
    Wrapper for returning correction candidates.
    n_return = 1 means only highest probability word is returned 
    with minimal threshold of 0.85 (Default).

    Args:
    - word(str): input word to be replaced
    - corpus(dict): Corpus dictionary, obtained from Count class
    - n(int): number of candidates

    Return:
    - corrected: (str) if n_return=1, otherwise return nested list with tuples of suggested words
    """
    # Defense
    if n_return < 1:
        raise ValueError("n_return must be greater than equal to 1")

    candidates = return_nearest(word, corpus)
    cand = sorted(candidates, key= lambda x:x[1], reverse=True) # sort candidates based on score
    if typo_dist:
        typo_sim_cand = [(txt,normalized_edit_similarity(word,txt),score) for txt,score,_ in cand]
        
        # Sort candidates based on typo Similarities
        cand = sorted(typo_sim_cand, key= lambda x:x[1], reverse=True) 

    if n_return == 1:
        # if n_return is 1, then return word with highest score
        if cand[0][1] >= threshold:
            if include_score:
                res = [(cand[0][0],cand[0][1])]
            else:
                res = cand[0][0]
        else:
            # If no candidates have score more than threshold return original word
            if include_score:
                res = [(word, 0)]
            else:
                res = word
    else:
        # Filter candidates that have score more than or equal to threshold
        filtered = [(txt,score) for txt,score,_ in cand if score >= threshold]
        if len(filtered) == 0:
            # if no suggestion pass threshold
            # return the original word
            if include_score:
                res = [(word,0)]
            else:
                res = [word]
        else:
            if len(filtered) <= n_return:
                # If number of suggested correction is less than the number of n_return
                # return as it is
                if include_score:
                    res = filtered
                else:
                    res = [txt for txt,_ in filtered]
            else:
                # If number of suggested correction larger than number of n_return
                # Trim the list
                if include_score:
                    res = filtered[:n_return]
                else:
                    res = [txt for txt,_ in filtered][:n_return]
    
    return res

    # Create a function allowing the app to receive Sentences as input
def correct_sentence(sentence, VOCAB, params, ngrams=3, lm=None):
    # Split the sentence into words
    words = re.findall(r'\b[\w\']+\b', sentence)
    
    # Correct the spelling of each word
    corrected_words = []
    for i,word in enumerate(words):
        corrected_word = correction(word, VOCAB, params['n_candidate'], 
                        params['threshold'], include_score=True, typo_dist=params['typo_dist'])
        if lm is None:
            corrected_words.append(corrected_word[0][0])
        else:
            if i == 0:
                prior_text = ["<s>"]
            elif i < ngrams:
                prior_text = corrected_words[:i]
            else:
                prior_text = corrected_words[i-3:i]

            new_score = [(word[0],lm.score(word[0], prior_text)*word[1]) for word in corrected_word]
            cand = sorted(new_score, key= lambda x:x[1], reverse=True)
            corrected_words.append(cand[0][0]) 
    
    # Join the corrected words into a sentence
    corrected_sentence = ' '.join(corrected_words)
    
    return corrected_sentence

if __name__ == "__main__":
    PATH = "data/big.txt"
    WORDS = words_loader(PATH)
    lm = joblib.load("data/lm_english.pkl")
    params = {
        "n_candidate":3,
        "threshold":0.7,
        "typo_dist":True 
        }

    # print(correction("hafe",WORDS, 3, 0.7, include_score=True, typo_dist=False))
    a = correct_sentence("i habe seen the pokixe", VOCAB=WORDS, params=params, lm=lm)
    print(a)