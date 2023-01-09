from flask import Flask, request
from distance_corrector import correct_sentence
from simple_corrector import words_loader
import joblib

app = Flask(__name__)

PATH = "data/big.txt"
WORDS = words_loader(PATH)
lm = joblib.load("data/lm_english.pkl")
params = {
    "n_candidate":3,
    "threshold":0.7,
    "typo_dist":True 
    }

@app.route('/sentence', methods=['POST'])
def correct():
    sentence = request.form.get('sentence')
    corrected_sentence = correct_sentence(sentence, VOCAB=WORDS, params=params, lm=lm)
    return corrected_sentence

if __name__ == '__main__':
    app.run()