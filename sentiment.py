import re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# -------------------data preparation--------------
data = open("files/classified_tweets.txt")
# data = open("files/sample.txt")
raw_data = data.read().splitlines()
data.close()

random.shuffle(raw_data)

X_data = []
y_data = []

for line in raw_data:
    indiv_data = line.split(' ', 1)
    X_data.append(indiv_data[1])
    y_data.append(indiv_data[0])

X_data_array = np.array(X_data)
y_data_array = np.array(y_data)

X_sample = X_data[0::20]  #[start:end:step] in list slice
y_sample = y_data[0::20]

X_sample_array = np.array(X_sample)
y_sample_array = np.array(y_sample)

print ("---Done with data preparation----")


#------------------ modeling training for algorithms testing (NB and LR)------------------
def pred_transfrom(a):
    b = []
    for i in a:
        if(i == '4'):
            e = 1
        else: e = 0
        b.append(e)

    return b

def kfoldtest (dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def kfoldtrain (dataset, kfoldtest):
    train = []
    for item in kfoldtest:
        train_element = [x for x in dataset if x not in item]
        train.append(train_element)
    return train

count_vect = CountVectorizer(max_features=3000, stop_words="english")

fpr_nb = []
tpr_nb = []
roc_auc_nb = []
accu_nb = []

fpr_lr = []
tpr_lr = []
roc_auc_lr = []
accu_lr = []

kfold = 10
indexList = range(len(X_sample))
test_index = kfoldtest(indexList,kfold)
train_index = kfoldtrain(indexList,test_index)



for index in range(len(test_index)):

	X_train, X_test = X_sample_array[train_index[index]], X_sample_array[test_index[index]]
	y_train, y_test = y_sample_array[train_index[index]], y_sample_array[test_index[index]]

     X_train_count = count_vect.fit_transform(X_train).toarray()
     X_test_count = count_vect.transform(X_test).toarray()

     y_true = pred_transfrom(y_test)

     nb_clf = GaussianNB().fit(X_train_count, y_train)
     y_pred_nb_temp = nb_clf.predict(X_test_count)
     y_pred_nb = pred_transfrom(y_pred_nb_temp)

     fpr_nb_fold, tpr_nb_fold, thresholds_nb_fold = roc_curve(y_true, y_pred_nb)
     roc_auc_nb_fold = auc(fpr_nb_fold, tpr_nb_fold)
     fpr_nb += fpr_nb_fold.tolist()
     tpr_nb += tpr_nb_fold.tolist()
     roc_auc_nb.append(roc_auc_nb_fold)

     accu_nb.append(accuracy_score(y_true, y_pred_nb))

     lr_clf = LogisticRegression().fit(X_train_count, y_train)
     y_pred_lr_temp = lr_clf.predict(X_test_count)
     y_pred_lr = pred_transfrom(y_pred_lr_temp)

     fpr_lr_fold, tpr_lr_fold, thresholds_lr_fold = roc_curve(y_true, y_pred_lr)
     roc_auc_lr_fold = auc(fpr_lr_fold, tpr_lr_fold)
     fpr_lr += fpr_lr_fold.tolist()
     tpr_lr += tpr_lr_fold.tolist()
     roc_auc_lr.append(roc_auc_lr_fold)

     accu_lr.append(accuracy_score(y_true, y_pred_lr))

# --------------Computate the fpr and tpr for two algorithm and plot ROC curve-------------
fpr_nb_mean = (sum(fpr_nb)-10.0)/10.0
tpr_nb_mean = (sum(tpr_nb)-10.0)/10.0
roc_auc_nb_mean = sum(roc_auc_nb)/10.0

fpr_lr_mean = (sum(fpr_lr)-10.0)/10.0
tpr_lr_mean = (sum(tpr_lr)-10.0)/10.0
roc_auc_lr_mean = sum(roc_auc_lr)/10.0

plt.figure(1)
plt.title('Receiver Operating Characteristic')
plt.plot([0,fpr_nb_mean,1], [0,tpr_nb_mean,1], 'b', label = 'NB_AUC = %0.4f'% roc_auc_nb_mean)
plt.plot([0,fpr_lr_mean,1], [0,tpr_lr_mean,1], 'r', label = 'LR_AUC = %0.4f'% roc_auc_lr_mean)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.ylabel('True Positive Rate(TPR)')
plt.xlabel('False Positive Rate(FPR)')

accu = [accu_nb, accu_lr]

plt.figure(2)
plt.title('Accuracy based on two algorithms')
f = plt.boxplot(accu,0,'', widths=(0.35, 0.35), patch_artist=True)
colors = ['lightblue', 'lightgreen']
for patch, color in zip(f['boxes'], colors):
         patch.set_facecolor(color)
 plt.xticks([1,2],('Naive Bayes','Logistic Regression'))
 plt.ylabel('Accuracy')

# --------------Now LR algorithm is used for the formal model training and prediction-------------

data = open("files/unclassified_tweets.txt")
raw_data = data.read().splitlines()
data.close()

X_test = []

for line in raw_data:
    X_test.append(line)


X_test_array = np.array(X_test)

X_train_count = count_vect.fit_transform(X_data_array)
X_test_count = count_vect.transform(X_test_array)

lr_clf = LogisticRegression().fit(X_train_count, y_data_array)
y_pred = lr_clf.predict(X_test_count).tolist()

positive = y_pred.count('4')
negative = y_pred.count('0')

plt.figure(3)
plt.title('Sentiment Analysis with Logistic Regression')
labels = 'positive', 'negative'
sentiment = [positive, negative]
colors = ['lightcoral', 'lightskyblue']
plt.pie(sentiment,  labels=labels, colors=colors, autopct='%1.3f%%', shadow=True, startangle=100)
plt.axis('equal')

# --------------Party classifier and Political sentiment analysis-------------

def party(tw):

    liberal_keywords = ['liberal', 'realchange', 'trudeau', 'freedom', 'responsibility', 'samesex', 'cannabis',
                        'justin', 'marijuana', 'refugee', 'trudeaujustin']

    conservative_keywords = ['conservative', 'stephen', 'harper', 'tory', 'economics', 'government',
                             'diversity', 'security', 'trade', 'democracy', 'federalism', 'stephenharper']

    NDP_keywords = ['tommulcair', 'ndp', 'peace', 'environmental','newdemocracy', 'broadbent', 'taxincreases', 'socialist']

    num_liberal = 0
    num_conservative = 0
    num_NDP = 0
    num_other = 0

    hashtags = re.findall(r"#(\w+)", tw)

    for word in hashtags:
        if (word in liberal_keywords):
            num_liberal += 1
        elif (word in conservative_keywords):
            num_conservative += 1
        elif (word in NDP_keywords):
            num_NDP += 1
        else:
            num_other += 1
    # count the number of hashtags in different party

    party = 'party'

    result = [num_liberal, num_conservative, num_NDP, num_other]
    finalresult = max(result)

    if (finalresult == num_liberal):
        party = 'Liberal'
    elif (finalresult == num_conservative):
        party = 'Conservative'
    elif (finalresult == num_NDP):
        party = 'NDP'
    else:
        party = 'other'
    # indicate the party with most number of key word

    return party

partycounter = {'Liberal':0, 'Conservative':0, 'NDP':0, 'other':0}

polsent = {'Liberal':{'4':0, '0':0}, 'Conservative':{'4':0, '0':0}, 'NDP':{'4':0, '0':0}, 'other':{'4':0, '0':0}}

for i in range(len(X_test)):
    pol = party(X_test[i])
    partycounter[pol] += 1
    polsent[pol][y_pred[i]] += 1

polsent_pos = []
polsent_neg = []
polsent_label = []

for par in polsent.keys():
    polsent_label.append(par)
    pos = polsent[par]['4']
    neg = polsent[par]['0']
    pos_frac = float((pos+0.0)/(pos + neg))
    neg_frac = float((neg+0.0)/(pos + neg))
    polsent_pos.append(pos_frac)
    polsent_neg.append(neg_frac)

plt.figure(4)
plt.title('political party classfication')
labels = []
sizes = []
for key in partycounter.keys():
    labels.append(key)
    sizes.append(partycounter[key])
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
plt.pie(sizes, colors=colors, autopct='%1.3f%%', shadow=True, startangle=160)
plt.legend(labels, loc="best")
plt.tight_layout()
plt.axis('equal')

plt.figure(5)
n_groups = 4
index = np.arange(n_groups)
bar_width = 0.35
plt.bar(index, polsent_pos, bar_width,
        color='lightcoral', label='Positive')

plt.bar(index + bar_width, polsent_neg, bar_width,
        color='lightskyblue', label='Negative')

plt.title('political sentiment analysis')
plt.xlabel('political parties')
plt.ylabel('political sentiment composion')
plt.xticks(index + bar_width, polsent_label)
plt.legend(loc="best")

plt.tight_layout()
plt.show()