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
sql = """select concat(title, content) as text, hash, alias, timestamp, is_spam, key 
         from spam_data
         where hash is not null and alias is not null and timestamp is not null and length(content) > 100 and not exists (
          select 1 from training_set where spam_data.key = training_set.key ) and not exists (
          select 1 from validation_set where spam_data.key = validation_set.key )
         order by random()"""
sql_insert = """insert into validation_set(text, hash, alias, timestamp, is_spam, key)
                values """
sql_training_insert = """insert into training_set(text, hash, alias, timestamp, is_spam, key)
                values """

try:
  cur.execute(sql)
except:
  print "Failed to select from spam_data"

for i in range(50):
  cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
  args_str = ','.join(cur2.mogrify("(%s,%s,%s,%s,%s,%s)", x).decode('utf-8') for x in cur.fetchmany(500))
  cur2.execute(sql_insert + args_str)

def save_training_model():
  clf = MultinomialNB();
  raw_data = cur.fetchmany(1200)
  data = np.asarray(raw_data)
  word_vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 10, non_negative=True)
  bigram_vectorizer = HashingVectorizer(analyzer='char', n_features=2 ** 10, non_negative=True, ngram_range=(1,2))

  content_bigram = bigram_vectorizer.transform(data[:,0])
  content_word = word_vectorizer.transform(data[:,0])
  hash = word_vectorizer.transform(data[:,1])
  alias = word_vectorizer.transform(data[:,2])
  X_vec = sparse.hstack([content_bigram, content_word, hash, alias])
  y_vec = data[:, 4]
  clf.partial_fit(X_vec, y_vec, classes=['True', 'False']);

  cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
  cur2.execute(sql)

  data = np.asarray(cur2.fetchmany(800))
  content_bigram = bigram_vectorizer.transform(data[:,0])
  content_word = word_vectorizer.transform(data[:,0])
  hash = word_vectorizer.transform(data[:,1])
  alias = word_vectorizer.transform(data[:,2])
  X_vec = sparse.hstack([content_bigram, content_word, hash, alias])
  y_vec = data[:, 4]
  print "Score for {}: {}".format(i, clf.score(X_vec, y_vec))

  cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
  args_str = ','.join(cur2.mogrify("(%s,%s,%s,%s,%s,%s)", x).decode('utf-8') for x in raw_data)
  cur2.execute(sql_training_insert + args_str)

  joblib.dump(clf, 'post{}.pkl'.format(i)) 
  joblib.dump(clf, 'comment{}.pkl'.format(i))

for i in range(20):
  save_training_model()
