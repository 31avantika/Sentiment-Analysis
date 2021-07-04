import pandas as pd
import numpy as np
import re
import string, nltk, time

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from nltk.probability import FreqDist
import matplotlib.pyplot as plt




reviews = pd.read_csv("C:/Users/Avantika Mishra/Documents/B. Tech/SKILLS LAB 6th Sem/Project/amazon_vfl_reviews.csv")


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
plotPerColumnDistribution (reviews, 10, 5)




# print(reviews.shape)
# print(reviews.isnull())
reviews.isnull().sum()

# the review column, four rows without review text, we drop the rows with the null columns
reviews=reviews.dropna()

#resetting the index
reviews = reviews.reset_index(drop = True)




# removing all characters but not number or alphabets
#print(reviews['review'][150])
def cleanText(input_string):
    modified_string = re.sub('[^A-Za-z0-9]+', ' ', input_string)
    return(modified_string)
reviews['review'] = reviews.review.apply(cleanText)




# From the name we extract the brand
reviews['brandName'] = reviews['name'].str.split('-').str[0]
#print(reviews.head())

#print(reviews['brandName'].value_counts())

reviews['brandName'] = reviews['brandName'].str.title()
#print(reviews.brandName.unique())

# Extracting the product from the name column
products = []
for value in reviews['name']:
    indx = len(value.split('-')[0])+1
    products.append(value[indx:])
reviews['product'] = products
#print(reviews['product'].unique())

#converting to lower case
reviews['clean_review_text']=reviews['review'].str.lower()

#removing punctuations
reviews['clean_review_text']=reviews['clean_review_text'].str.translate(str.maketrans('','',string.punctuation))




#removing stopwords
stopWords = stopwords.words('english')+['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from']
def removeStopWords(stopWords, rvw_txt):
    newtxt = ' '.join([word for word in rvw_txt.split() if word not in stopWords])
    return newtxt
reviews['clean_review_text'] = [removeStopWords(stopWords,x) for x in reviews['clean_review_text']]




#splitting text into words
tokenList=[]
for indx in range(len(reviews)):
       token=word_tokenize(reviews['clean_review_text'][indx])
       tokenList.append(token)
reviews['review_tokens'] = tokenList



#VADER (Valence Aware Dictionary for Sentiment Reasoning) is a model used for text sentiment analysis
#that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion.
#It is available in the NLTK package and can be applied directly to unlabeled text data.
#VADER sentimental analysis relies on a dictionary that maps lexical features to emotion intensities known as sentiment scores.
#The sentiment score of a text can be obtained by summing up the intensity of each word in the text.
#VADER’s SentimentIntensityAnalyzer() takes in a string and returns a dictionary of scores in each of four categories:
#    negative
#    neutral
#    positive
#    compound (computed by normalizing the scores above)
sentiment_model = SentimentIntensityAnalyzer()
sentiment_scores = []
sentiment_score_flag = []
for text in reviews['clean_review_text']:
        sentimentResults = sentiment_model.polarity_scores(text)
        sentiment_score = sentimentResults["compound"]
        #print(sentimentResults)
        #The compound value reflects the overall sentiment ranging from -1 being very negative and +1 being very positive.
        sentiment_scores.append(sentiment_score)
        
		#marking the sentiments as positive, negative and neutral 
        if sentimentResults['compound'] >= 0.05 : 
            sentiment_score_flag.append('positive')
  
        elif sentimentResults['compound'] <= - 0.05 : 
            sentiment_score_flag.append('negative')
  
        else : 
            sentiment_score_flag.append('neutral')
            
reviews['scores']=sentiment_scores
reviews['scoreStatus'] = sentiment_score_flag




#use of countvectorizer
#Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts.
#It also enables the ​pre-processing of text data prior to generating the vector representation.
#This functionality makes it a highly flexible feature representation module for text.
features = CountVectorizer()
features.fit(reviews["clean_review_text"])          #fit() = Learn a vocabulary dictionary of all tokens in the raw documents.
#print(len(features.vocabulary_))                   #vocabulary_: A mapping of terms to feature indices.
#print(features.vocabulary_)




#transform(): Transform documents to document-term matrix.
#A document-term matrix is a mathematical matrix that describes the frequency of terms that occur in a collection of documents. In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms.
#the matrix has document/sample number of rows and total number of unique words in the dataset number of columns
#bagofwords contains [(document_number, frequency of token throughout the dataset) token_frequency_in_that_doc].
bagofWords = features.transform(reviews["clean_review_text"])
#print (bagofWords)
transform_words_matrix = bagofWords.toarray()




positiveReviews = reviews.loc[reviews['scoreStatus'] == "positive"]
negativeReviews = reviews.loc[reviews['scoreStatus'] == "negative"]
df = pd.concat([positiveReviews,negativeReviews])       #entire df with all features
df = df[["clean_review_text","scoreStatus"]]            #a matrix with these two features

#assigning the value of 1 to positive reviews
df['scoreStatus'] = (df['scoreStatus'] == 'positive') * 1




#start training the model
#defining training and testing data as X and Y
X = df["clean_review_text"]
y = df["scoreStatus"]
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 0)

X_train = features.fit_transform (X_train)
X_test = features.transform (X_test)
#transform(): Transform documents to document-term matrix.
#A document-term matrix is a mathematical matrix that describes the frequency of terms that occur in a collection of documents.
#In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms.




#k fold cross validation with k=5
scores = cross_val_score(LogisticRegression(),X_train,y_train, cv = 5)
print('\n\nCross validation score = ', np.mean(scores))
#Cross validation score =  0.9547391007880721



#applying LOGISTIC REGRESSION model
model = LogisticRegression()
model.fit (X_train, y_train)

#printing the score of Logistic Regression
print ('Score of training data = ', model.score (X_train,y_train))                #0.9972067039106145
print ('Score of testing data = ', model.score (X_test,y_test))                   #0.9700520833333334

y_predict = model.predict(X_test)
confusionMatrix = confusion_matrix (y_test, y_predict)             #[[102  22]
                                                                   #[  0 644]]

text1 = "bad"
print ('\n\nPrediction for \'bad\' is ', model.predict(features.transform([text1]))[0])

text2 = "good"
print ('Prediction for \'good\' is ', model.predict(features.transform([text2]))[0])

text3 = "genuine"
print ('Prediction for \'genuine\' is ', model.predict(features.transform([text3]))[0], '\n\n')



tokenized_word = word_tokenize((reviews['clean_review_text'].to_string()))

#Frequency Distribution
fdist = FreqDist(tokenized_word)

# Frequency Distribution Plot
fdist.plot(30,cumulative=False)
#plt.show()


#vectorizing the data using tfidf
#TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
#This is done by multiplying two metrics: how many times a word appears in a document,and the inverse document frequency of the word across a set of documents.
#TFIDF
tfidf = TfidfVectorizer(tokenizer = cleanText)
classifier = LinearSVC()
X = df["clean_review_text"]
y = df["scoreStatus"]
X_train, X_test, y_train, y_test = train_test_split ( X, y, test_size = 0.5, random_state = 0 )

train_vectors = tfidf.fit_transform(X_train)
test_vectors = tfidf.transform(X_test)



#Using SVM
# Perform classification with SVM, kernel=linear
#Linear Kernel is used when the data is Linearly separable, that is, it can be separated using a single Line.
#It is one of the most common kernels to be used.
#It is mostly used when there are a Large number of Features in a particular Data Set.
#One of the examples where there are a lot of features, is Text Classification, as each alphabet is a new feature.
#So we mostly use Linear Kernel in Text Classification.

classifier_linear = svm.SVC (kernel ='linear')
t0 = time.time()
classifier_linear.fit (train_vectors, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1


#A Classification report is used to measure the quality of predictions from a classification algorithm.
#The report shows the main classification metrics precision, recall and f1-score on a per-class basis.
#The metrics are calculated by using true and false positives, true and false negatives.

#results
print("Results for SVC (kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report (y_test, prediction_linear, output_dict = True)
print('positive: ', report['1'])
print('negative: ', report['0'])
#The column 'support' displays how many object of class 0 were in the test set

#testing SVM
test = "I love this product"
review_vector = tfidf.transform([test]) # vectorizing
print(classifier_linear.predict(review_vector))






#Stop word removal is a breeze with CountVectorizer and it can be done in several ways:
#    Use sklearn’s built in English stop word list
#The goal of MIN_DF is to ignore words that have very few occurrences to be considered meaningful.
#Instead of using a minimum term frequency (total occurrences of a word) to eliminate words,
#MIN_DF looks at how many documents contained a term, better known as document frequency.
#The MIN_DF value can be an absolute value (e.g. 1, 2, 3, 4) or a value representing proportion of documents
#(e.g. 0.25 meaning, ignore words that have appeared in 25% of the documents) .
#we can ignore words that are too common with MAX_DF. MAX_DF looks at how many documents contained a term,
#and if it exceeds the MAX_DF threshold, then it is eliminated from consideration.
#The MAX_DF value can be an absolute value (e.g. 1, 2, 3, 4) or a value representing proportion of documents
#(e.g. 0.85 meaning, ignore words appeared in 85% of the documents as they are too common).

#since the dataset is imbalanced, SVM is not producing good results.
#implemented this because we had distunct features of word tokens
#BAYE'S MODEL
class Bayes:
    def _pipeline(self, df):
        cv = CountVectorizer (max_df = 0.95, min_df = 2, stop_words = 'english')
        review = df[['review']]
        Xtrain, Xtest, ytrain, ytest = train_test_split (review, df.rating, random_state = 2)
        cv.fit(pd.concat([Xtrain.review, Xtest.review]))
        Xtrain = cv.transform(Xtrain.review)
        Xtest  = cv.transform(Xtest.review)

        model = MultinomialNB()
        model.fit(Xtrain, ytrain)
        
        ypred = model.predict(Xtest)
        print("\n\n Bayes model accuracy score: ", accuracy_score(ytest, ypred))
                             
Bayes()._pipeline(reviews)                              #Bayes model accuracy score:  0.7956834532374101




#Then implementing Xgboost Classifier
#XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework.
from xgboost import XGBClassifier

class Xgb:
    def _pipeline(self, df):
        cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        review = df[['review']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(review, df.rating, random_state=2)
        cv.fit(pd.concat([Xtrain.review, Xtest.review]))
        Xtrain = cv.transform(Xtrain.review)
        Xtest  = cv.transform(Xtest.review)

        model = XGBClassifier()
        model.fit(Xtrain, ytrain)
        
        ypred = model.predict(Xtest)
        print("\n\n Xgboost classifier accuracy score: ", accuracy_score(ytest, ypred))
                             
        #Xgboost classifier accuracy score:  0.8661870503597122
Xgb()._pipeline(reviews)






#implementing CNN now
#importing keras and all the necessary packages
from keras import layers, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
import logging
import math
#import tensorflow_hub as hub



class Classifier():
  def __init__(self):
    self.train = None
    self.test = None 
    self.model = None
    
  def load_data(self, df):
      """ Load train, test csv files and return pandas.DataFrame
      """
      self.train, self.test = train_test_split(df, test_size=0.2)
      self.train.rename({'review': 'text', 'rating': 'target'}, axis='columns', inplace=True)
      self.test.rename({'review': 'text', 'rating': 'target'}, axis='columns', inplace=True)

  def save_predictions(self, y_preds):
      sub = pd.read_csv(f"sampleSubmission.csv")
      sub['Sentiment'] = y_preds 
      sub.to_csv(f"submission_{self.__class__.__name__}.csv", index=False)
      logging.info(f'Prediction exported to submission_{self.__class__.__name__}.csv')


class C_NN(Classifier):
    def __init__(self, max_features=10000, embed_size=128, max_len=300):
        self.max_features=max_features
        self.embed_size=embed_size
        self.max_len=max_len
    
    def tokenize_text(self, text_train, text_test):
        '''@para: max_features, the most commenly used words in data set
        @input are vector of text
        '''
        tokenizer = Tokenizer(num_words=self.max_features)
        text = pd.concat([text_train, text_test])
        tokenizer.fit_on_texts(text)

        sequence_train = tokenizer.texts_to_sequences(text_train)
        tokenized_train = pad_sequences(sequence_train, maxlen=self.max_len)
        logging.info('Train text tokenized')

        sequence_test = tokenizer.texts_to_sequences(text_test)
        tokenized_test = pad_sequences(sequence_test, maxlen=self.max_len)
        logging.info('Test text tokenized')
        return tokenized_train, tokenized_test, tokenizer
      
    def build_model(self, embed_matrix=[]):
        text_input = Input(shape=(self.max_len, ))
        embed_text = layers.Embedding(self.max_features, self.embed_size)(text_input)
        if len(embed_matrix) > 0:
            embed_text = layers.Embedding(self.max_features, self.embed_size, weights=[embed_matrix], trainable=False)(text_input)
            
        branch_a = layers.Bidirectional(layers.GRU(32, return_sequences=True))(embed_text)
        branch_b = layers.GlobalMaxPool1D()(branch_a)

        #Each neuron in a layer receives an input from all the neurons present in the previous layer—thus, they're densely connected.
        #In other words, the dense layer is a fully connected layer, meaning all the neurons in a layer are connected to those in the next layer.
        
        x = layers.Dense (64, activation='relu')(branch_b)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense (32, activation='relu')(branch_b)
        x = layers.Dropout (0.2)(x)
        branch_z = layers.Dense(6, activation='softmax')(x)
        
        model = Model(inputs=text_input, outputs=branch_z)
        self.model = model

        return model
        
    def embed_word_vector(self, word_index, model='glove-wiki-gigaword-100'):
        glove = api.load(model) # default: wikipedia 6B tokens, uncased
#        zeros = [0] * self.embed_size
        matrix = np.zeros((self.max_features, self.embed_size))
          
        for word, i in word_index.items(): 
            if i >= self.max_features or word not in glove: continue # matrix[0] is zeros, that's also why >= is here
            matrix[i] = glove[word]

        logging.info('Matrix with embedded word vector created')
        return matrix

    def run(self, x_train, y_train):
        checkpoint = ModelCheckpoint('weights_base_best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=3)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=2020)
        BATCH_SIZE = max(16, 2 ** int(math.log(len(X_tra) / 100, 2)))
        logging.info(f"Batch size is set to {BATCH_SIZE}")
        history = self.model.fit(X_tra, y_tra, epochs=30, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), callbacks=[checkpoint, early], verbose=0)

        return history


c = C_NN(max_features = 10000, embed_size = 300, max_len = 300)
c.load_data(reviews)
labels = to_categorical(c.train.target, num_classes = 6)
#labels

vector_train, vector_test, tokenizer = c.tokenize_text(c.train.text, c.test.text)
embed = c.embed_word_vector(tokenizer.word_index, 'word2vec-google-news-300')
c.build_model(embed_matrix=embed)
history = c.run(vector_train, labels)


plt.plot( history.history['acc'] )
#plt.plot( history.history['val_acc'])
#plt.legend(['acc', 'val_acc'])
plt.title("ACCURACY")
plt.xlabel('epoch')

plt.plot( history.history['loss'] )
#plt.plot( history.history['val_loss'])
#plt.legend(['loss', 'val_loss'])
plt.title("LOSS")
plt.xlabel('epoch')

model = load_model('weights_base_best.hdf5')
y_preds = model.predict(vector_test)
final = np.argmax(y_preds, axis=1)
print('\n CNN accuracy score is', accuracy_score(c.test.target, final))
#CNN accuracy score is 0.9118705035971223


#BERT stands for Bidirectional Encoder Representations from Transformers.
#It is a Transformer-based machine learning technique for natural language processing pre-training developed by Google. 
#As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer
#to create state-of-the-art models for a wide range of NLP tasks




















'''
predictions = model.predict(vector_test, batch_size=100, verbose=1)
labels = [1, 0]
prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])

sum(c.test.scoreStatus==prediction_labels)/len(prediction_labels)





def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()
'''
