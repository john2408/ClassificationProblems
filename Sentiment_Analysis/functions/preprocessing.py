import re

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


    
def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    
    
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

 
    



def data_cleaning(corpus, sent_tokenizer = False,  text_cleaning = True, use_nltk_cleaning = False ):
    
    """
    """
    
    if text_cleaning:

        corpus = corpus.reset_index(drop=True)
        corpus['text'] = corpus['text'].apply(clean_text)
        corpus['text'] = corpus['text'].str.replace('\d+', '')

    elif use_nltk_cleaning:

        # Step III : Tokenization : In this each entry in the corpus will be broken into set of words
        if sent_tokenizer: 
            corpus['text'] = [sent_tokenize(x) for x in corpus['text']] 
        else:
            #Corpus['text'] = Corpus['text'].apply(lambda x: str(word_tokenize(x)) )
            corpus['text'] = [word_tokenize(x) for x in corpus['text']]

        # Step IV, V, VI : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        # Word Classification for Lemmatizer https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
        # https://www.geeksforgeeks.org/defaultdict-in-python/
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        # Execute Word Tagging
        for index, entry in enumerate(corpus['text']):

            # Declaring Empty List to store the words that follow the rules for this step
            lemma_words = []

            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()

            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            # Posttagging reference : https://www.nltk.org/book/ch05.html 

            for word, tag in pos_tag(entry):

                # Below condition is to check for Stop words and consider only alphabets
                # List of stop words https://gist.github.com/sebleier/554280, https://www.nltk.org/book/ch02.html

                # NLTK check for an alphabetic word https://tedboy.github.io/nlps/generated/generated/nltk.text_type.isalpha.html
                if word not in stopwords.words('english') and word.isalpha():

                    # Reference https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
                    # Use first letter of NLTK Postagging as "pos" parameter mapping it through the dict tag_map
                    lemma_word = word_Lemmatized.lemmatize(word = word,
                                                           pos = tag_map[tag[0]]  )
                    # Append word back to the empty list
                    lemma_words.append(lemma_word)

            # The final processed set of words for each iteration will be stored in 'text_final'
            corpus.loc[index,'text_clean'] = ' '.join(lemma_words)

        corpus.loc[:,'text'] = corpus['text_clean']
        
    return corpus


def prepare_training_data(corpus):
    """
    """
    
    output = {}
    
    # Get training X data
    sentences = corpus['text'].values

    # Use Label encoder for the expected output
    Encoder = LabelEncoder()
    encoded_Y = Encoder.fit_transform(corpus['label'].values)
    Y = pd.get_dummies(encoded_Y).values

    sentences_train, sentences_test, Y_train, Y_test = train_test_split( sentences, Y, test_size=0.25)

    output_label = len(np.unique(encoded_Y))
    
    output['sentences_train'] = sentences_train
    output['sentences_test'] = sentences_test
    output['Y_train'] = Y_train
    output['Y_test'] = Y_test
    output['output_label'] = output_label

    return output

