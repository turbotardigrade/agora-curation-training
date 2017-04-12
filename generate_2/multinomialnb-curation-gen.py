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
  conn2 = psycopg2.connect("dbname='reddit' user='s140401' host='localhost'")
  conn2.set_session(autocommit=True)
except:
  print "I am unable to connect to the database"
  
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
sql = """select concat(title, content) as text, is_spam, key 
         from clean_spam_data
         where not exists (
          select 1 from clean_training_set where clean_spam_data.key = clean_training_set.key ) and not exists (
          select 1 from clean_validation_set where clean_spam_data.key = clean_validation_set.key )
         order by random()"""
sql_insert = """insert into clean_validation_set(text, is_spam, key)
                values """
sql_training_insert = """insert into clean_training_set(text, is_spam, key)
                values """

try:
  cur.execute(sql)
except:
  print "Failed to select from spam_data"

for i in range(50):
  cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
  args_str = ','.join(cur2.mogrify("(%s,%s,%s)", x).decode('utf-8') for x in cur.fetchmany(500))
  cur2.execute(sql_insert + args_str)

def save_training_model():
  clf = MultinomialNB();
  raw_data = cur.fetchmany(1400)
  data = np.asarray(raw_data)
  word_vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 10, non_negative=True)
  bigram_vectorizer = HashingVectorizer(analyzer='char', n_features=2 ** 10, non_negative=True, ngram_range=(1,2))

  content_bigram = bigram_vectorizer.transform(data[:,0])
  content_word = word_vectorizer.transform(data[:,0])
  X_vec = sparse.hstack([content_bigram, content_word])
  y_vec = data[:, 1]
  clf.partial_fit(X_vec, y_vec, classes=['True', 'False']);

  cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
  cur2.execute(sql)

  data = np.asarray(cur2.fetchmany(800))
  content_bigram = bigram_vectorizer.transform(data[:,0])
  content_word = word_vectorizer.transform(data[:,0])
  X_vec = sparse.hstack([content_bigram, content_word])
  y_vec = data[:, 1]
  print "Score for {}: {}".format(i, clf.score(X_vec, y_vec))

  cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
  args_str = ','.join(cur2.mogrify("(%s,%s,%s)", x).decode('utf-8') for x in raw_data)
  cur2.execute(sql_training_insert + args_str)

  joblib.dump(clf, 'post{}.pkl'.format(i)) 
  joblib.dump(clf, 'comment{}.pkl'.format(i))

for i in range(20):
  save_training_model()
