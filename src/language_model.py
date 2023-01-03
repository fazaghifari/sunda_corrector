from nltk.lm import StupidBackoff as sboff
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
import joblib
import time
import nltk

def text_loader(PATH):
    input_text = open(PATH).read()
    tokenizer_words = TweetTokenizer()
    tokens_sentences = [[word.lower() for word in tokenizer_words.tokenize(t) if word.isalpha()] for t in sent_tokenize(input_text)]
    return tokens_sentences

def text_padder(tokenized_text, n=2):
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    return train_data, padded_sents

def model_train(train_data, padded_sents, n=2):
    model = sboff(order=n)
    model.fit(train_data, padded_sents)
    return model

def main_lm(PATH, n):
    WORDS = text_loader(PATH)
    train_data, padded_sents = text_padder(WORDS, n)
    model = model_train(train_data, padded_sents, n)
    return model


if __name__ == "__main__":

    PATH = "data/big.txt"
    t0 = time.time()
    model = main_lm(PATH, 3)
    print(f"time needed = {time.time() - t0} seconds")
    joblib.dump(model, "data/lm_english.pkl")