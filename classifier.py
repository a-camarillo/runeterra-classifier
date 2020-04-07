from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.svm import LinearSVC 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('first_round.csv',index_col=0)
y = df[['card_region']]
X = df.drop(columns=['card_region','card_name','card_type'])

print(df['card_type'].unique())
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# rand_forest = RandomForestClassifier(random_state=1)
# rand_forest.fit(X_train,y_train.values.ravel())
# rf_pred = rand_forest.predict(X_test)

# rf_f1_score = f1_score(y_test,rf_pred,average='weighted')

# decision_tree = DecisionTreeClassifier(random_state=1)
# decision_tree.fit(X_train,y_train)
# dt_pred = decision_tree.predict(X_test)

# dt_f1_score = f1_score(y_test,dt_pred,average='weighted')

# svc = LinearSVC(random_state=1,max_iter=10000)
# svc.fit(X_train,y_train.values.ravel())
# svc_pred = svc.predict(X_test)

# svc_f1_score = f1_score(y_test,svc_pred,average='weighted')

# rf_cfm = plot_confusion_matrix(rand_forest,X_test,y_test,xticks_rotation=90)
# plt.title('Random Forest')
# plt.gcf().savefig('./images/first_rf_cfm.png')
# dt_cfm = plot_confusion_matrix(decision_tree,X_test,y_test,xticks_rotation=90)
# plt.title('Decision Tree')
# plt.gcf().savefig('./images/first_dt_cfm.png')
# svc_cfm = plot_confusion_matrix(svc,X_test,y_test,xticks_rotation=90)
# plt.title('Support Vector Classifier')
# plt.gcf().savefig('./images/first_svc_cfm.png')

