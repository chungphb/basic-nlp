import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import word_tokenize, pos_tag
text = "I am learning Natural Language Processing"
tokens = word_tokenize(text)
print (pos_tag(tokens))