import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import wordcloud
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay




df = pd.read_csv('fifa_world_cup_2022_tweets.csv')



df.head()



df.info()



df.columns




df.isnull().sum()



df['Sentiment'].value_counts()   #counts of sentiments


100*df['Sentiment'].value_counts()/22524    #percentage of sentiments



df.describe()


df['Number of Likes'].value_counts()


text_df = df.drop(['Unnamed: 0', 'Date Created', 'Number of Likes', 'Source of Tweet','Sentiment'],axis=1)

text_df.head()

print(text_df['Tweet'].iloc[0])
print(text_df['Tweet'].iloc[1])
print(text_df['Tweet'].iloc[2])
print(text_df['Tweet'].iloc[3])

text_df.info()


# Data pre-processing

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


text_df['Tweet'] = text_df['Tweet'].apply(data_processing)


text_df = text_df.drop_duplicates('Tweet')


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


text_df['Tweet'] = text_df['Tweet'].apply(lambda x: stemming(x))


text_df.head()

print(text_df['Tweet'].iloc[0])
print(text_df['Tweet'].iloc[1])
print(text_df['Tweet'].iloc[2])
print(text_df['Tweet'].iloc[3])


text_df.info()


def polarity(text):
    return TextBlob(text).sentiment.polarity



text_df['polarity'] = text_df['Tweet'].apply(polarity)


text_df.head(10)

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


text_df['sentiment'] = text_df['polarity'].apply(sentiment)


text_df.head()

text_df['sentiment'].value_counts()      #count of sentiments

100*text_df['sentiment'].value_counts()/21345    #percentage of sentiments


# Data visualization

fig = plt.figure(figsize=(6,3),dpi=150)
sns.countplot(x='sentiment', data = text_df);
#plt.savefig('count_sentiment.png',bbox_inches='tight')



fig = plt.figure(figsize=(6,3),dpi=200)
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = text_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments');
# plt.savefig('pie_chart_dist_of_sentiments.png',bbox_inches='tight')


pos_tweets = text_df[text_df.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'], ascending= False)
pos_tweets.head()



text = ' '.join([word for word in pos_tweets['Tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive tweets', fontsize=19)
plt.show();
wordcloud.to_file('wc_for_positive_tweet.png')


from PIL import Image


mask = np.array(Image.open('happy_emoji.png'))


mwc = WordCloud(background_color='black',mask=mask, width=800, height=400)
mwc.generate(text)
plt.figure(figsize=(8,4),dpi=200, facecolor='white')
plt.imshow(mwc)
plt.axis('off')
plt.show()
# mwc.to_file('mwc_for_positive_tweet.png')



neg_tweets = text_df[text_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['polarity'], ascending= False)
neg_tweets.head()



text = ' '.join([word for word in neg_tweets['Tweet']])
plt.figure(figsize=(20,15), facecolor='None',dpi=200)
wordcloud = WordCloud(max_words=500, width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative tweets', fontsize=19)
plt.show()
# wordcloud.to_file('wc_for_negative_tweet.png');


mask2 = np.array(Image.open('angry_emoji.png'))
mwc2 = WordCloud(background_color='black',mask=mask2, width=800, height=400)
mwc2.generate(text)
plt.figure(figsize=(8,4),dpi=200, facecolor='white')
plt.imshow(mwc2)
plt.axis('off')
plt.show()
# mwc2.to_file('mwc_for_negative_tweet.png')


neutral_tweets = text_df[text_df.sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending= False)
neutral_tweets.head()


text = ' '.join([word for word in neutral_tweets['Tweet']])
plt.figure(figsize=(20,15),dpi=200,facecolor='None')
wordcloud = WordCloud(max_words=500, width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral tweets', fontsize=19)
plt.show()
# wordcloud.to_file('wc_for_neutral_tweet.png');



mask3 = np.array(Image.open('neutral_emoji.png'))
mwc3 = WordCloud(background_color='black',mask=mask3, width=800, height=400)
mwc3.generate(text)
plt.figure(figsize=(8,4),dpi=200, facecolor='None')
plt.imshow(mwc3)
plt.axis('off')
plt.show()
# mwc3.to_file('mwc_for_neutral_tweet.png')


vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['Tweet'])


feature_names = vect.get_feature_names_out()
print(f"Number of features: {len(feature_names)}\n")
print(f"First 20 features:\n {feature_names[:20]}")


#  building the model using Logistic Regression

X = text_df['Tweet']
Y = text_df['sentiment']
X = vect.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))

import warnings
warnings.filterwarnings('ignore')



logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))




style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=logreg.classes_)
disp.plot()

#  Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(x_train, y_train)


print("Best parameters:", grid.best_params_)


y_pred = grid.predict(x_test)


logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))


style.use('classic')
cm = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=logreg.classes_)
disp.plot()


# Using another algorithm (Support vector machine)

from sklearn.svm import LinearSVC

SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)


svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("test accuracy: {:.2f}%".format(svc_acc*100))


print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))


# Hyperparameter Tuning for SVM


grid = {
    'C':[0.01, 0.1, 1, 10],
    'kernel':["linear","poly","rbf","sigmoid"],
    'degree':[1,3,5,7],
    'gamma':[0.01,1]
}
grid = GridSearchCV(SVCmodel, param_grid)
grid.fit(x_train, y_train)


print("Best parameter:", grid.best_params_)


y_pred = grid.predict(x_test)



logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))


style.use('classic')
cm = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=logreg.classes_)
disp.plot()


data = [['Logistic Reg', 89.72], ['Tuned Logistic Reg', 91.38], ['SVC', 92.48],['Tuned SVC', 92.60]]


relation_df = pd.DataFrame(data=data,columns=['model','Accuracy'])


relation_df



fig = plt.figure(figsize=(5,3),dpi=150)
sns.barplot(data=relation_df,x='model',y='Accuracy')
plt.xticks(rotation =75);
plt.ylim(85,100)
percentage = [89.72,91.38,92.48,92.60]
for rect1 in p1:
    height = rect1.get_height()
    plt.annotate( "{}%".format(height),(rect1.get_x() + rect1.get_width()/2, height+.05),ha="center",va="bottom",fontsize=15)
plt.show()
# fig.savefig('barplot_of_percent_wise_accuracy.png',bbox_inches='tight')








