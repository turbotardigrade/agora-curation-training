import psycopg2
import psycopg2.extras
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation
from scipy import sparse

try:
  conn = psycopg2.connect("dbname='reddit' user='s140401' host='localhost'")
except:
  print "I am unable to connect to the database"
  
cur = conn.cursor('reddit_cursor', cursor_factory=psycopg2.extras.DictCursor)
cur2 = conn.cursor('reddit_cursor2', cursor_factory=psycopg2.extras.DictCursor)


sql = """select concat(title, content) as text, hash, alias, timestamp, 
          case when is_spam = true then 1 else 0 end as is_spam, 
          case when is_spam = false then 1 else 0 end as is_ham
         from spam_data
         where hash is not null and alias is not null and timestamp is not null
         order by random()"""
sql2 = """select concat(title, content) as text, hash, alias, timestamp, 
           case when is_spam = true then 1 else 0 end as is_spam, 
           case when is_spam = false then 1 else 0 end as is_ham
         from spam_data
         where hash is not null and alias is not null and timestamp is not null 
         and length(content) < 100 and title is null"""

try:
  cur.execute(sql)
  cur2.execute(sql2)
except:
  print "Failed to select from spam_data"
  
model = Sequential()
model.add(Dense(output_dim=2048, input_dim=4097))
model.add(Activation("relu"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

for i in range(1, 20):
  data = np.asarray(cur.fetchmany(100))
  word_vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 10, non_negative=True)
  bigram_vectorizer = HashingVectorizer(analyzer='char', n_features=2 ** 10, non_negative=True, ngram_range=(1,2))

  content_bigram = bigram_vectorizer.transform(data[:,0])
  content_word = word_vectorizer.transform(data[:,0])
  hash = word_vectorizer.transform(data[:,1])
  alias = word_vectorizer.transform(data[:,2])
  other = np.transpose(sparse.csr_matrix((data[:,3]).astype(np.float)))
  X_vec = sparse.hstack([content_bigram, content_word, hash, alias, other])
  model.train_on_batch(X_vec.toarray(), (data[:, 4:]).astype(np.float))

data = np.asarray(cur.fetchmany(1000))
word_vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 10, non_negative=True)
bigram_vectorizer = HashingVectorizer(analyzer='char', n_features=2 ** 10, non_negative=True, ngram_range=(1,2))

content_bigram = bigram_vectorizer.transform(data[:,0])
content_word = word_vectorizer.transform(data[:,0])
hash = word_vectorizer.transform(data[:,1])
alias = word_vectorizer.transform(data[:,2])
other = np.transpose(sparse.csr_matrix((data[:,3]).astype(np.float)))
X_vec = sparse.hstack([content_bigram, content_word, hash, alias, other])
y_vec = (data[:, 4:]).astype(np.float)

prediction = model.predict(X_vec.toarray(), batch_size=100)
num_correct = 0
number = 0
for i in range(0, 1000):
  number += 1
  if prediction[i][0] > 0.5 and y_vec[i][0] > 0.5 and prediction[i][1] < 0.5:
    num_correct += 1
  elif prediction[i][0] < 0.5 and y_vec[i][0] < 0.5 and prediction[i][1] > 0.5:
    num_correct += 1

print model.evaluate(X_vec.toarray(), y_vec, batch_size=100)
print float(num_correct) / float(number)
