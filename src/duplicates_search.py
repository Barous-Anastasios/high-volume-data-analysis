from gensim import corpora, models, similarities
from collections import defaultdict
import pandas
import os
import sys

dataframe = pandas.read_csv("./../data/train_set.csv", sep="\t", error_bad_lines = False)
df = pandas.DataFrame(columns = ['DOCUMENT ID1', 'DOCUMENT ID2', 'SIMILARITY'])
documents = list(dataframe["Content"])

stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
frequency = defaultdict(int)
for text in texts:
	for token in text:
		frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]

if (os.path.exists('./gensim/dictionary.dict')):
	dictionary = corpora.Dictionary.load('./gensim/dictionary.dict')
	corpus = corpora.MmCorpus('./gensim/corpus.mm')
else:
	dictionary = corpora.Dictionary(texts)
	dictionary.save('./gensim/dictionary.dict')
	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize('./gensim/corpus.mm', corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=1)
for ind, row in dataframe.iterrows():
	print(ind)
	mylist = []
	vec_bow = dictionary.doc2bow(row['Content'].lower().split())
	vec_lsi = lsi[vec_bow]
	index = similarities.MatrixSimilarity(lsi[corpus])
	sims = index[vec_lsi]

	for i, item in enumerate(sims):
		if(item >= 0.7):
			mylist.append([row['Id'], list(dataframe["Id"])[i], item])

	df = df.append(pandas.DataFrame(mylist, columns = ['DOCUMENT ID1', 'DOCUMENT ID2', 'SIMILARITY']))
df.to_csv('./../dist/duplicatePairs.csv', sep='\t')