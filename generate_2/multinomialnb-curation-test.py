import psycopg2
import psycopg2.extras
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from scipy import sparse

try:
  conn = psycopg2.connect("dbname='reddit' user='s140401' host='localhost'")
except:
  print "I am unable to connect to the database"

cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
sql = """select concat(title, content) as text, is_spam, key 
         from spam_data
         order by random()"""

try:
  cur.execute(sql)
except:
  print "Failed to select from spam_data"

word_vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 10, non_negative=True)
bigram_vectorizer = HashingVectorizer(analyzer='char', n_features=2 ** 10, non_negative=True, ngram_range=(1,2))

for i in range(20):
  curr_model = joblib.load('post{}.pkl'.format(i)) 
  data = np.asarray(cur.fetchmany(800))
  content_bigram = bigram_vectorizer.transform(data[:,0])
  content_word = word_vectorizer.transform(data[:,0])
  X_vec = sparse.hstack([content_bigram, content_word])
  y_vec = data[:, 1]
  
  print "Score for {}: {}".format(i, curr_model.score(X_vec, y_vec))
