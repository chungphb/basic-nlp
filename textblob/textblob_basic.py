# Tokenization
from textblob import TextBlob
blob = TextBlob("This is a great platform to learn data science. \n It helps community through blogs, hackathons, discussions, etc.")
print(blob.sentences)
print(blob.sentences[0])
for word in blob.sentences[0].words:
    print(word)

# Noun Phrase Extraction
blob = TextBlob("Analytics VidHya is a great platform to learn data science.")
for np in blob.noun_phrases:
    print(np)

# Part-of-speech Tagging
for word, tag in blob.tags:
    print(word, tag)

# Words Inflection and Lemmatization
blob = TextBlob("This is a great platform to learn data science. \n It helps community through blogs, hackathons, discussions, etc.")
print(blob.sentences[1].words[1])
print(blob.sentences[1].words[1].singularize())

from textblob import Word
w = Word('Platform')
print(w.pluralize())

for word, pos in blob.tags:
    if pos == 'NN':
        print(word.pluralize())

# N-grams
blob = TextBlob("Analytics VidHya is a great platform to learn data science.")
for ngram in blob.ngrams(2):
    print(ngram)

# Sentiment Analysis
print(blob.sentiment)

# Spelling correction
blob = TextBlob('Analytics Vidhya is a gret platfrm to learn data scence.')
print(blob.correct())
print(blob.words[4].spellcheck())

# Summary of a text
import random

blob = TextBlob('Analytics Vidhya is a thriving community for data driven industry.\
                This platform allows people to know more about analytics from its articles, Q&A forum, and learning paths.\
                Also, we help professionals & amateurs to sharpen their skillsets by providing a platform to participate in Hackathons.')
nouns = list()
for word, tag in blob.tags:
    if tag == 'NN':
        nouns.append(word.lemmatize())

print('This text is about...')
for item in random.sample(nouns, 5):
    word = Word(item)
    print(word.pluralize())

# Text classification using TextBlob
training = [
('Tom Holland is a terrible spiderman.','pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
]
testing = [
('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')
]

from textblob import classifiers
classifier = classifiers.NaiveBayesClassifier(training)
print(classifier.accuracy(testing))
classifier.show_informative_features(3)