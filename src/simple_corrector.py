import re
from collections import Counter

def words(text): 
    return re.findall(r'\w+', text.lower())

def words_loader(path):
    WORDS = Counter(words(open(path).read()))
    return WORDS

def P(word, corpus): 
    "Probability of `word`."
    N=sum(corpus.values())
    return corpus[word] / N

def correction(word, corpus): 
    "Most probable spelling correction for word."
    return max(candidates(word, corpus), key=lambda x:x[1])[0]

def candidates(word, corpus): 
    "Generate possible spelling corrections for word."
    return (known([word], corpus) or known(edits1(word), corpus) or known(edits2(word), corpus) or [word])

def known(words, corpus): 
    "The subset of `words` that appear in the dictionary of corpus."
    word_list = list(set(w for w in words if w in corpus))
    list_pair = [(w,P(w, corpus)) for w in word_list]
    return list_pair

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# Create a function allowing the app to receive Sentences as input
def correct_sentence(sentence):
    # Split the sentence into words
    words = re.findall(r'\b[\w\']+\b', sentence)
    
    # Correct the spelling of each word
    corrected_words = []
    for word in words:
        corrected_word = correction(word)
        corrected_words.append(corrected_word)
    
    # Join the corrected words into a sentence
    corrected_sentence = ' '.join(corrected_words)
    
    return corrected_sentence

if __name__ == "__main__":
    PATH = "data/Sundanese.txt"
    WORDS = words_loader(PATH)