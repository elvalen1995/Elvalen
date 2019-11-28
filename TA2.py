import nltk
import pandas
import csv
import re
#from nltk.stem.porter import PorterStemmer

#stemmer = PorterStemmer()
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

def custom_preprocessor(text):
    text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations and replace to the space W  itu karakter d number + untuk setiap string
    text = nltk.word_tokenize(text)       #tokenizing
    text = [word for word in text if not word in stop_words] #English Stopwords
    text = [lemmatizer.lemmatize(word) for word in text]              #Lemmatising
    return text

array = []
array2 = []
with open ('data_review.csv', 'r', encoding = 'ISO-8859-1'  ) as csv_file :
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for line in csv_reader:
        #print(line[1])
        text = line [1].lower()#casefolding
        
        pre_processing = custom_preprocessor(text)
        array2.append(pre_processing)
        if 'pos' in line [0]:
            array.append(1)
        else:
            array.append(0)
        #array.append(line[0])
        print(pre_processing)
        
raw ={'class':array, 'text':array2}
dataf = pandas.DataFrame(raw,columns=['class','text'])      
dataf.to_csv('dataclean.csv', index = False)
        
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer

dataraw=pandas.read_csv('dataclean.csv')
txt=dataraw['text']
hitungKata = CountVectorizer() #menghitung jumlah kemunculan kata
hasilhitung = hitungKata.fit_transform(txt)

hitungidf=TfidfTransformer(use_idf=True, smooth_idf=True)
hasilidf=hitungidf.fit(hasilhitung)#menghitung idf dg inputan jumlah kemunculan data

idf = pandas.DataFrame(hitungidf.idf_, index=hitungKata.get_feature_names(), columns = ['Hasil IDF'])

tfidf = hitungidf.transform(hasilhitung)
kata =  hitungKata.get_feature_names()

#get tfidf vector for first document
kata_pertama=tfidf[0]

#print the score

datasimpan = pandas.DataFrame(kata_pertama.T.todense(), index=kata, columns=["hasil_TFIDF"])
sorting=datasimpan.sort_values(by=["hasil_TFIDF"], ascending = False)
sorting.to_csv('hasil_TFIDF.csv')
#print(datasimpan)

tfidf_vectorizer=TfidfVectorizer(use_idf=True)

#just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(txt)

#get the first vector out (for the first documents)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]

df1 = pandas.DataFrame (first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
sorting = df1.sort_values(by=["tfidf"], ascending = False)
print(sorting)

from sklearn.model_selection import KFold 

dataraw=pandas.read_csv('dataclean.csv')
X = dataraw ['text']
Y = dataraw ['class']

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

import time
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#from sklearn.metrics import precision_recall_fscore_support

cv = KFold (n_splits = 10, random_state = 42, shuffle=False )

#train dan test indek untuk kfold
i=0
for train_index, test_index in cv.split(Y):
    print ("Train Index:", train_index,"\n")
    print ("Test_Index:", test_index)
    
    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    
    train_vectors = vectorizer.fit_transform (X_train)
    test_vectors = vectorizer.transform (X_test)
    
    #perform classification SVM 
    classifier_linear = svm.SVC(kernel='poly' , d = 10, gamma = 0.01) 
    t0 = time.time()
    classifier_linear.fit(train_vectors,Y_train)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict= t2-t1
    
    #result
    print ("training time: %fs; Prediction time %fs" % (time_linear_train, time_linear_predict))
    report = classification_report (Y_test, prediction_linear)
    confus = confusion_matrix (Y_test, prediction_linear)
    acc = accuracy_score (Y_test, prediction_linear)
    prec = precision_score (Y_test, prediction_linear)
    rec = recall_score (Y_test, prediction_linear)
    f1 = f1_score (Y_test, prediction_linear)
   
    print(report)
    print (confus)
    print ("Akurasi: ", acc)
    print("Precission:", prec)
    print("recall:", rec)
    print ("F1 measure: ",f1)






 #print ('positive:', report [1])
    #print ('negative:' , report [0]) 








    #akurasi=accuracy_score(Y_test,prediction_linear)
   # clf_rep = precision_recall_fscore_support(Y_test, prediction_linear)
    #OUTPUT TO EXCEL DALAM PERULANGAN SAVE PER K-FOLD
    
  #  filename = 'Output K-Fold'+str(i)
 #   filepath = filename+'.csv'
#    raw_data = {'Asli' : Y_test,
  #              'Prediksi' : prediction_linear,
   #             'Review' : X_test,
    #            'Akurasi' : akurasi}
#                'Precision' : clf_rep[0].round(2),
 #               'Recall' : clf_rep[1].round(2),
  #              'F1-Score' : clf_rep[2].round(2),
   #             'Support' : clf_rep[3].round(2)} 
  #  df = pandas.DataFrame(raw_data, columns = ['Asli','Prediksi','Review','Akurasi'])#,'Precision','Recall','F1-Score','Support'])
   # df.to_csv(filepath, index=False)
    #i=i+1
    
    #from nltk.classify.scikitlearn import SklearnClassifier
    #from sklearn.svm import SVC
    #out_dict = {
     #        "precision" :clf_rep[0].round(2)
      #      ,"recall" : clf_rep[1].round(2)
       #     ,"f1-score" : clf_rep[2].round(2)
        #    ,"support" : clf_rep[3]
   
    
    #train = [(X_train, Y_train)]
    #SVC_classifier = SklearnClassifier(SVC(kernel='linear'))
    #SVC_classifier.train(train)
    #print("SVC_classifier accuracy on Train:", (nltk.classify.accuracy(SVC_classifier,train))*100)
    #print("SVC_classifier accuracy on Test:", (nltk.classify.accuracy(SVC_classifier, Y_test, X_test))*100)
    #print(X_train)
   # print(Y_train)
   # print(Y_train.shape(1,1172))
    
   
    
#klasifikasi itu x_train dan Y_train, nah klo prediksi itu Y_test, y tes digunakan sbg pembanding
    
#print (pre_processing)        
#filepath = 'data_review.csv'
#text = line[1] 
#pre_processing = custom_preprocessor(text)
#print(pre_processing.Get_filepath())
#output = pre_processing.Output_Program()