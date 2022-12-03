from simple_corrector import words_loader
from rapidfuzz import process
import rapidfuzz

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

def correction(word, corpus, n_return=1, threshold=0.85, include_score=False):
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
    if n_return == 1:
        # if n_return is 1, then return word with highest score
        if cand[0][1] >= threshold:
            res = cand[0][0]
        else:
            # If no candidates have score more than threshold return original word
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



if __name__ == "__main__":
    PATH = "data/big.txt"
    WORDS = words_loader(PATH)
    print(correction("barcket",WORDS, 1, 0.7, include_score=False))