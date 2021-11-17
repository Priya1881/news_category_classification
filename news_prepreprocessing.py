import nltk
import re
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words= set(stopwords.words('english'))


def clean(text):
         # lowering letters
         text = text.lower()
         # removing html tags
         text = re.sub('<[^>]*>', ' ', text)
         # removing emails
         text = re.sub('\S*@\S*\s?', ' ', text)
         # removing urls
         text = re.sub('https?://[A-Za-z0-9]', ' ', text)
         # removing numbers
         text = re.sub('[^A-Za-z]', ' ', text)

         word_tokens = word_tokenize(text)
         filtered_sentence = []
         for word_token in word_tokens:
             if word_token not in stop_words and len(word_token) > 2:
                 filtered_sentence.append(word_token)
         # joining words
         text = ' '.join(filtered_sentence)
         return text
