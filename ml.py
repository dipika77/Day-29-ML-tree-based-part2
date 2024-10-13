#Bagging and Random Forest

#Bagging
#instructions
'''Import DecisionTreeClassifier from sklearn.tree and BaggingClassifier from sklearn.ensemble.
Instantiate a DecisionTreeClassifier called dt.
Instantiate a BaggingClassifier called bc consisting of 50 trees'''

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator= dt, n_estimators= 50, random_state=1)


#instructions
'''Fit bc to the training set.
Predict the test set labels and assign the result to y_pred.
Determine bc's test set accuracy.'''

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 



#Out of Bag Evaluation
#instructions
'''Import BaggingClassifier from sklearn.ensemble.
Instantiate a DecisionTreeClassifier with min_samples_leaf set to 8.
Instantiate a BaggingClassifier consisting of 50 trees and set oob_score to True.'''

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf= 8, random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator= dt, 
            n_estimators= 50,
            oob_score= True,
            random_state=1)



#instructions
'''Fit bc to the training set and predict the test set labels and assign the results to y_pred.
Evaluate the test set accuracy acc_test by calling accuracy_score.
Evaluate bc's OOB accuracy acc_oob by extracting the attribute oob_score_ from bc.'''

# Fit bc to the training set 
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))


#Random Forest
#instructions
'''Import RandomForestRegressor from sklearn.ensemble.
Instantiate a RandomForestRegressor called rf consisting of 25 trees.
Fit rf to the training set.'''

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators= 25,
            random_state=2)
            
# Fit rf to the training set    
rf.fit(X_train, y_train) 


#instructions
'''Import mean_squared_error from sklearn.metrics as MSE.
Predict the test set labels and assign the result to y_pred.
Compute the test set RMSE and assign it to rmse_test.'''

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


#instructions
'''Call the .sort_values() method on importances and assign the result to importances_sorted.
Call the .plot() method on importances_sorted and set the arguments:
kind to 'barh'
color to 'lightgreen'''

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind ='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
