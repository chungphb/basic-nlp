doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."

doc_complete = [doc1, doc2, doc3]                                       # chungphb: Kho ngu lieu (Corpus)
doc_clean = [doc.split() for doc in doc_complete]                       # chungphb: Danh sach cac term (tu) trong kho ngu lieu.

from gensim import corpora, models

# Creating the term dictionary of our corpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)                              # chungphb: Tao tu dien term tu danh sach cac term (Moi term se duoc gan voi mot chi so)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]        # chungphb: Chuyen kho ngu lieu thanh ma tran cac term dua tren phep danh chi so o tren.

# Creating the object for LDA model using gensim library
Lda = models.ldamodel.LdaModel                                   # chungphb: Tao doi tuong LDA tu thu vien gensim.

# Running and Training LDA model on the document term matrix
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

# Results
print(ldamodel.print_topics())