#Practical 1: Write a program to convert the given text to speech.
#pip install playsound==1.2.2
#pip install playsound
#pip install gTTS
from playsound import playsound
#import required for text to speech conversion
from gtts import gTTS
mytext = "Welcome to Natural Language programming"
language = "en"
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("myfile.mp3")
playsound("myfile.mp3")

------------------------------------------------------------------------
Practical 2: Write a program to convert speech/audio file to Text.
#pip install SpeechRecognition
import speech_recognition as sr
filename = "audio.wav"
# initialize the recognizer
r = sr.Recognizer()
with sr.AudioFile(filename) as source:
    audio_data = r.record(source)
    text = r.recognize_google(audio_data)
    print(text)


-----------------------------------------------------------------------------
Practical 3: Study of Wordnet Dictionary with methods as synsets, definitions.
#pip3 install inltk
import nltk
from nltk.corpus import wordnet
try:
    syn = wordnet.synsets('car')[0]
    print ("Synset name : ", syn.name())
 # Defining the word
    print ("\nSynset meaning : ", syn.definition())
 # list of phrases that use the word in context
    print ("\nSynset example : ", syn.examples())
except IndexError:
     print('word not found!')


-------------------------------------------------------------------------------
Practical 4: Study of Wordnet Dictionary with methods as part-ofspeech(POS)
import nltk
from nltk.corpus import wordnet

# Getting the first synset (meaning) of the word "hello"
syn = wordnet.synsets('hello')[0]
print("Syn tag for hello:", syn.pos())

# Getting the first synset of the word "doing"
syn = wordnet.synsets('doing')[0]
print("Syn tag for doing:", syn.pos())

# Getting the first synset of the word "beautiful"
syn = wordnet.synsets('beautiful')[0]
print("Syn tag for beautiful:", syn.pos())

# Getting the first synset of the word "quickly"
syn = wordnet.synsets('quickly')[0]
print("Syn tag for quickly:", syn.pos())

#Tag	Meaning
#n	Noun
#v	Verb
#a	Adjective
#r	Adverb

-----------------------------------------------------------------------
# Practical 5: Write a program using python to find synonym and antonym of word "active" using Wordnet
import nltk
from nltk.corpus import wordnet
print( wordnet.synsets("car"))
print(wordnet.lemma('active.a.01.active').antonyms())

--------------------------------------------------------------------------------
# Practical 6: Study of various Brown Corpus with various methods like fileids,sents and categories
import nltk
from nltk.corpus import brown
print("Brown Corpus\n")
print("fileids:",brown.fileids())
print("Categories:",brown.categories())
print("Sentences:",brown.sents())

--------------------------------------------------------------------------------
Practical 7: Write a program to create your own corpora
Code:
# run the commands in shell [ import nltk, nltk.download() ]
import os, os.path
from nltk.corpus.reader import WordListCorpusReader
reader_corpus= WordListCorpusReader('.', ['p1.py'])
print(reader_corpus.words())

-----------------------------------------------------------------------------------
Practical 8: Write a program to implement tokenization.
Code:
# run the commands in shell [ import nltk, nltk.download() ]
import nltk
from nltk import tokenize
para = "How are you? Hope you are fine. I am fine too. Today is Saturday."
sents = tokenize.sent_tokenize(para)
print("\nsentence tokenization\n===================\n",sents)
# word tokenization
print("\nword tokenization\n===================\n")
for index in range(len(sents)):
 words = tokenize.word_tokenize(sents[index])
 print(words)


--------------------------------------------------------------------------------------
Practical 8: Write a program to implement tokenization.
Code:
# run the commands in shell [ import nltk, nltk.download() ]
import nltk
from nltk import tokenize
para = "How are you? Hope you are fine. I am fine too. Today is Saturday."
sents = tokenize.sent_tokenize(para)
print("\nsentence tokenization\n===================\n",sents)
# word tokenization
print("\nword tokenization\n===================\n")
for index in range(len(sents)):
 words = tokenize.word_tokenize(sents[index])
 print(words)

-------------------------------------------------------------------------------------
Practical 9: Write a program to find the most frequent noun tags.
Code:
import nltk
from collections import defaultdict
text=nltk.word_tokenize("Nick does like to play football. Nick does not like to play cricket")
print(text)
tags=nltk.pos_tag(text)
addNounWordS=[]
count=0
for words in tags:
    val=tags[count][1]
    if(val=='NN' or val=='NNS' or val=='NNPS'or val=='NNP'):
        addNounWordS.append(tags[count][0])

        count+=1
print(count)
print(addNounWordS)
temp=defaultdict(int)
for sub in addNounWordS:
    for wrd in sub.split():
        temp[wrd]+=1
res=max(temp,key=temp.get)
print(temp)
print("Word with maximum frequency : " + str(res))

------------------------------------------------------------------------------------------
Practical 10: Maps words to properties using python dictionary
Code: 
# run the commands in shell [ import nltk, nltk.download() ] 
thisdict = {
"brand": "Ford",
"model": "Mustang",
"year": 1964
}
print(thisdict)
print(thisdict["brand"])
print(len(thisdict))
print(type(thisdict))


----------------------------------------------------------------------------------------
Practical 11: Write a program to implement tokenization using Python’s split() function
Code: 
# run the commands in shell [ import nltk, nltk.download() ]
text = """ This tool -is an a beta stage- Alexa developers can use Get-Metrics
API to
seamlessly analyse metric. """
data = text.split('.')
for i in data:
    print (i)


--------------------------------------------------------------------------------------
Practical 12: Define grammar using NLTK analyze a sentence using the same.
Code:
# run the commands in shell [ import nltk, nltk.download() ]
import nltk
from nltk import tokenize
grammar1 = nltk.CFG.fromstring("""
S -> VP
VP -> VP NP
NP -> Det NP
Det -> 'that'
NP -> 'flight'
VP -> 'Book'
""")
sentence = "Book that flight"
for index in range(len(sentence)):
  all_tokens = tokenize.word_tokenize(sentence)
print(all_tokens)
parser = nltk.ChartParser(grammar1)
for tree in parser.parse(all_tokens):
  print(tree)
tree.draw()

----------------------------------------------------------------------------------------
Practical 13: Implement the concept of chunking
Code:
# run the commands in shell [ import nltk, nltk.download() ]
import nltk
from nltk import tokenize
from nltk import tag
from nltk import chunk
para = "Hello! Today you'll be learning NLTK."
sents = tokenize.sent_tokenize(para)
print("\nsentence tokenization\n===================")
print("sentence :",sents)
# word tokenization
print("\nword tokenization\n===================")
for index in range(len(sents)):
  words = tokenize.word_tokenize(sents[index])
  print(words)
# POS Tagging
tagged_words = []
for index in range(len(sents)):
  tagged_words.append(tag.pos_tag(tokenize.word_tokenize(sents[index])))
print("\nPOS Tagging\n===========")
print("Tagged words:",tagged_words)
# Chunking
tree = []
for index in range(len(sents)):
# print("\nchunk:",chunk.ne_chunk(tagged_words[index]))
  tree.append(chunk.ne_chunk(tagged_words[index]))
  print("\nChunking\n========")
for t in tree:
  print(t)


---------------------------------------------------------------------------------
Practical 14: Write a program to use named entity recognition with diagram
using NTK corpus TreeBank
Code:
# run the commands in shell [ import nltk, nltk.download() ]
import nltk
from nltk.corpus import treebank_chunk
print(treebank_chunk.tagged_sents()[0])
print(treebank_chunk.chunked_sents()[0])
treebank_chunk.chunked_sents()[0].draw()

----------------------------------------------------------------------------------
Practical 15: Write a program to implement a concept of Stemming
Code:
# run the commands in shell [ import nltk, nltk.download() ]
import nltk
from nltk.stem import PorterStemmer
word_stemmer = PorterStemmer()
print(word_stemmer.stem('running'))

-----------------------------------------------------------------------------------
Practical 16: Develop a Python function that cleans and normalizes input text
by performing these steps:
Expanding contractions
Converting text to lowercase
Removing punctuation
Removing extra whitespace
Code:
import re
import string
from contractions import fix # Requires: pip install contractions
def clean_text(text):
 # Expand contractions
  text = fix(text)
 # Lowercase the text
  text = text.lower()
 # Remove punctuation
  text = text.translate(str.maketrans('', '', string.punctuation))
 # Remove extra whitespace
  text = re.sub('\s+', ' ', text).strip()
  return text
sample_text = "I'm excited to learn NLP; it's awesome!"
cleaned_text = clean_text(sample_text)
print("Cleaned Text:", cleaned_text)


---------------------------------------------------------------------------
Practical 17: Perform sentiment analysis on movie reviews or tweets using VADER.
Code:
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
texts = [
 "I absolutely loved the movie! It was fantastic.",
 "The movie was okay, not great but not bad either.",
 "I did not enjoy the movie at all. It was boring and too long."
]
for text in texts:
  sentiment = sia.polarity_scores(text)
  print(f"Text: {text}\nSentiment: {sentiment}\n")

--------------------------------------------------------------------------------
Practical 18: Develop an interactive chatbot using Microsoft’s DialoGPT-small
model from Hugging Face. Write a Python program that continuously accepts
user input and generates responses.
Code:
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Chat loop (for demonstration)
for step in range(3):
  user_input = input("User: ")
  new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
  bot_output = model.generate(new_user_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
  bot_response = tokenizer.decode(bot_output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
  print("Bot:", bot_response)


------------------------------------------------------------------------------------
Practical 19: Write a Python program that extracts text from a PDF file. Use a
library of your choice (e.g., PyPDF2 or pdfminer.six) and display the extracted
text.
Code:
#Write a Python program that extracts text from a PDF file. Use a library of
your choice (e.g., PyPDF2 or pdfminer.six) and display the extracted text.
import PyPDF2
def extract_text_from_pdf(pdf_path):
 # Open the PDF file in binary mode
  with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
 # Loop through each page in the PDF
  for page in reader.pages:
    text += page.extract_text() + "\n"
    return text
# Example usage:
pdf_file = 'sample.pdf' # Replace with your PDF file path
extracted_text = extract_text_from_pdf(pdf_file)
print("Extracted Text:\n", extracted_text)


--------------------------------------------------------------------------------------
Practical 20: SMS Spam Collection Dataset — EDA to Model Training
Code:
#SMS Spam Collection Dataset — EDA to Model Training
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('spam.csv', encoding='latin-1')
print(df.shape)
print(df.describe ())
label_encoder = LabelEncoder()
df.rename(columns = {'v1':'label', 'v2':'text'}, inplace = True)
df['label'] = label_encoder.fit_transform(df['label'])
from sklearn.feature_extraction.text import CountVectorizer
x = df['text']
y = df['label']
cv = CountVectorizer()
x= cv.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,
random_state=0)
from sklearn.ensemble import RandomForestClassifier as rf
classifier = rf(n_estimators = 5 ,criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
print('training evaluation', classifier.score(x_train, y_train))
print('testing evaluation ', classifier.score(x_test, y_test))
y_pred = classifier.predict(x_test)
print(y_pred)
