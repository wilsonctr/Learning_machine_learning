#RandomForest

scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)
scaled_features = pd.DataFrame(scaled_features, columns=X.columns)

X = scaled_features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
RF = RandomForestClassifier(n_estimators = 50, 
                            min_samples_split = 4,
                            criterion = 'entropy',
                            max_features = 'log2',
                            min_samples_leaf = 4)  
RF.fit(X_train, y_train) 
#svclassifier.fit(X, y) 
pred = RF.predict(X_test)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Print out classification report and confusion matrix
#print(classification_report(y_test, pred))
print accuracy_score(pred, y_test)

# calculate the fpr and tpr for all thresholds of the classification
probs = RF.predict_proba(X_test)
preds = probs[:,1]

# calculate the fpr and tpr for all thresholds of the classification
probs = RF.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, lw=2, )
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

scores = cross_val_score(RF, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#RandomForest params search

n_estimators = range(50, 200, 50)
param_grid = {'n_estimators': n_estimators,
             'criterion' : ["gini", "entropy"],
             "min_samples_split" : [2,3,4,5],
             "min_samples_leaf" : [1,2,3,4,5,6],
             "max_features" : ["auto", "log2"],
             }
grid_search = GridSearchCV(RF, param_grid, cv=5)
grid_search.fit(X, y)
grid_search.best_params_
print grid_search.best_params_ , grid_search.best_score_

# SVM params search
from sklearn.model_selection import GridSearchCV

Cs = [0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1, 5, 10]
gammas = [0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1,]
kernel = ['rbf','linear','poly','sigmoid']
degree = [1,2,3]
param_grid = {'C': Cs, 'gamma' : gammas, 'kernel' : kernel, 'degree' : degree}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, )
grid_search.fit(X, y)
grid_search.best_params_
print grid_search.best_params_ , grid_search.best_score_

#Count total charge of peptides/proteins

# Total Charge
pos_res = ['R', 'K', 'H']
neg_res = ['E', 'D']
charges = []
for i in data.Sequence:
    temp_charge = 0
    for j in i:
        if j in pos_res:
            temp_charge += 1
        if j in neg_res:
            temp_charge -= 1
    
    charges.append(temp_charge)

data['charges'] = charges

#Count amino acids in protein sequences
#Amino acid count
aa_columns = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
amino_aa_count_df = pd.DataFrame(columns=aa_columns)
temp_df = pd.DataFrame(columns=aa_columns)

for i in counter:
    temp_list = []
    for j in aa_columns:
        if j in i:
            temp_list.append(i[j])
        else:
            temp_list.append(0)

    temp_df = temp_df.append(pd.Series(temp_list,index=aa_columns), ignore_index=True)

data = data.join(temp_df)

counter = []
for i in data.Sequence:
    counter.append(Counter(i.replace(' ','')))
