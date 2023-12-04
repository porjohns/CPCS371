import re
import string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import RocCurveDisplay
import matplotlib as plt

#Import Data
def getData():
    columns = ["sentiment", "id", "date", "query", "user_id", "text"]
    df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin", names=columns)
    df["sentiment"] = df["sentiment"].replace(4,1)
    return df.loc[790000:810000]

#Apply Preprocessesing (except Feature Extraction)
def preprocessTxt(line):
    
    # Lower Case
    text = ""
    text = text + line
    text = text.lower()
    
    #Remove urls
    text = re.sub(r'((www.\S+)|(https?://\S+))', r"", text)
    
    #Remove tags
    text = re.sub(r'(@\S+) | (#\S+)', r'', text)
    
    # Remove Numbers
    text = re.sub(r'\d+','',text)
    
    # Remove Punctuation
    text = text.translate(str.maketrans("","", string.punctuation))
    
    # Remove Whitespaces
    text = text.strip()
    
    #Stop Words
    stopWords = set(stopwords.words("english"))
        
    #Tokenization
    tokens = word_tokenize(text)
        
    #Lemmatization 
    wnl = WordNetLemmatizer() 
        
    #print(tokens)
    stems = []
    for word in tokens:
        stems.append(wnl.lemmatize(word))
            
    text = ""
    for word in stems:
        if word not in stopWords:
            text = text + " " + word 
    
    return text

#Feature Extraction
def featureExtraction():
    
    countVect = CountVectorizer(preprocessor=preprocessTxt, tokenizer=word_tokenize)
    totalCount = CountVectorizer(tokenizer=word_tokenize)
    
    df = getData()
    
    totalCountModel = totalCount.fit_transform(df['text'])
    
    countVectModel = countVect.fit_transform(df['text'])
    
    counts = countVectModel.toarray()
    
    counts = pd.DataFrame(counts,
                      columns=countVect.get_feature_names_out())
    
    countsOfFeatures = []
    
    for feature in counts.columns:
        cnt = sum(counts.loc[:,feature])
        countsOfFeatures.append(cnt)
       
    length = len(countsOfFeatures)
    mean = sum(countsOfFeatures)/length
    print("Mean: " , mean)
    
    variance = sum([((x - mean) ** 2) for x in countsOfFeatures]) / len(countsOfFeatures) 
    res = variance ** 0.5
    print("Standard Deviation: ", str(res))
        
    X_train, X_test, y_train, y_test = train_test_split(countVectModel,df["sentiment"],test_size = 0.2, random_state=1234)
     
    return X_train, X_test, y_train, y_test

#Run all models
def runModels():
    
    X_train, X_test, y_train, y_test = featureExtraction()
    
    MultiNB(X_train, X_test, y_train, y_test)
    
    LinSVC(X_train, X_test, y_train, y_test)
    
    LogRegression(X_train, X_test, y_train, y_test)

#Multinominal Naive Bayes Model
def MultiNB(X_train, X_test, y_train, y_test):
      
    model = MultinomialNB()
    
    model = model.fit(X_train,y_train)
    
    svc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    print("Multinomial NB:")
    print()
    print(classification_report(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    print("FPR :", fpr)
    print("TPR :", tpr)
    print("Thresholds :", thresholds)
    svc_disp.plot(alpha=0.8)

#Linear SVC Model
def LinSVC(X_train, X_test, y_train, y_test):
    
    model = LinearSVC()
    
    model.fit(X_train,y_train)
    
    svc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    print("Linear SVC:")
    print()
    print(classification_report(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    print("FPR :", fpr)
    print("TPR :", tpr)
    print("Thresholds :", thresholds)
    svc_disp.plot(alpha=0.8)

#Logistic Regression Model
def LogRegression(X_train, X_test, y_train, y_test): 
    
    model = LogisticRegression()
    
    model = model.fit(X_train,y_train)
    
    svc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    print("Logistic Regression:")
    print()
    print(classification_report(y_test, y_pred)) 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    print("FPR :", fpr)
    print("TPR :", tpr)
    print("Thresholds :", thresholds)
    svc_disp.plot(alpha=0.8)
    
runModels()
