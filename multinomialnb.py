import psycopg2
import psycopg2.extras
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from scipy import sparse

try:
  conn = psycopg2.connect("dbname='reddit' user='flyingtardigrades' host='localhost'")
except:
  print "I am unable to connect to the database"
  
cur = conn.cursor('reddit_cursor', cursor_factory=psycopg2.extras.DictCursor)
cur2 = conn.cursor('reddit_cursor2', cursor_factory=psycopg2.extras.DictCursor)


sql = """select concat(title, content) as text, hash, alias, timestamp, is_spam 
         from spam_data
         where hash is not null and alias is not null and timestamp is not null
         order by random()"""
sql2 = """select concat(title, content) as text, hash, alias, timestamp, is_spam 
         from spam_data
         where hash is not null and alias is not null and timestamp is not null and key < 1000"""

try:
  cur.execute(sql)
  cur2.execute(sql2)
except:
  print "Failed to select from spam_data"
  
clf = MultinomialNB();
clf2 = MultinomialNB();

for i in range(1, 2000):
  data = np.asarray(cur.fetchmany(1))
  word_vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 10, non_negative=True)
  bigram_vectorizer = HashingVectorizer(analyzer='char', n_features=2 ** 10, non_negative=True, ngram_range=(1,2))

  content_bigram = bigram_vectorizer.transform(data[:,0])
  content_word = word_vectorizer.transform(data[:,0])
  hash = word_vectorizer.transform(data[:,1])
  alias = word_vectorizer.transform(data[:,2])
  other = np.transpose(sparse.csr_matrix((data[:,3]).astype(np.float)))
  X_vec = sparse.hstack([content_bigram, content_word, hash, alias, other])
  y_vec = data[:, 4]
  clf.partial_fit(X_vec, y_vec, classes=['True', 'False']);

data = np.asarray(cur.fetchmany(2000))
word_vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 10, non_negative=True)
bigram_vectorizer = HashingVectorizer(analyzer='char', n_features=2 ** 10, non_negative=True, ngram_range=(1,2))

content_bigram = bigram_vectorizer.transform(data[:,0])
content_word = word_vectorizer.transform(data[:,0])
hash = word_vectorizer.transform(data[:,1])
alias = word_vectorizer.transform(data[:,2])
other = np.transpose(sparse.csr_matrix((data[:,3]).astype(np.float)))
X_vec = sparse.hstack([content_bigram, content_word, hash, alias, other])
y_vec = data[:, 4]
clf2.partial_fit(X_vec, y_vec, classes=['True', 'False']);

print "Fitted"

for i in range(0, 5):
  validate = np.asarray(cur.fetchmany(500))
  content_bigram_ver = word_vectorizer.transform(validate[:,0])
  content_word_ver = word_vectorizer.transform(validate[:,0])
  hash_ver = word_vectorizer.transform(validate[:,1])
  alias_ver = word_vectorizer.transform(validate[:,2])
  other_ver = np.transpose(sparse.csr_matrix((validate[:,3]).astype(np.float)))
  X_vec_ver = sparse.hstack([content_bigram_ver, content_word_ver, hash_ver, alias_ver, other_ver])
  y_vec_ver = validate[:, 4]
  print "CLF: {}".format(clf.score(X_vec_ver, y_vec_ver))
  print "CLF2: {}".format(clf2.score(X_vec_ver, y_vec_ver))

validate = np.asarray(cur2.fetchall())
content_bigram_ver = word_vectorizer.transform(validate[:,0])
content_word_ver = word_vectorizer.transform(validate[:,0])
hash_ver = word_vectorizer.transform(validate[:,1])
alias_ver = word_vectorizer.transform(validate[:,2])
other_ver = np.transpose(sparse.csr_matrix((validate[:,3]).astype(np.float)))
X_vec_ver = sparse.hstack([content_bigram_ver, content_word_ver, hash_ver, alias_ver, other_ver])
y_vec_ver = validate[:, 4]
print "CLF: {}".format(clf.score(X_vec_ver, y_vec_ver))
print "CLF2: {}".format(clf2.score(X_vec_ver, y_vec_ver))

joblib.dump(clf, 'model2.pkl') 