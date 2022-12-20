# %%
"""
The code for testing the classifiers. Uses binary.csv and multiclass.csv

Loads feature lists and tuning ranges from pickles in CWD.

Change model types in models to tune on specific ones.

Writes output files.

Author: Wesley
"""

# Accelerates tuning of some classifiers
#from sklearnex import patch_sklearn
#patch_sklearn()

import pandas as pd
import numpy as np

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from sklearn.feature_selection import RFECV

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)

from time import time

from sklearn.metrics import classification_report

from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.base import clone


# %%
binary = pd.read_csv("binary_train.csv")
binary_test = pd.read_csv("binary_test.csv")

# %% [markdown]
# Preprocessing (make labels numeric)

# %%
# Encode attack labels to int and save as array to be used later.
binary[" Label"] = binary[" Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
binary_test[" Label"] = binary_test[" Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# %% [markdown]
# Load feature sets and search spaces and enumerate their contents.

# %%
feature_sets = pickle.load(open("feature_sets.pickle", 'rb'))
search_spaces = pickle.load(open("hyperparameter_search_spaces.pickle", 'rb'))

print(f"Available Tuning Ranges: {search_spaces.keys()}")

print("Feature Sets for Binary Dataset:")
for key, value in feature_sets["Binary"].items():
    if key == "RFE Sets":
        print(value.keys())

    elif key == "PCA":
        print(f"{key}, suggested variance threshold is {value}")
        
    else:
        print(key)

print("Feature Sets for Multiclass Dataset:")
for key, value in feature_sets["Multiclass"].items():
    if key == "RFE Sets":
        print(value.keys())

    elif key == "PCA":
        print(f"{key}, suggested variance threshold is {value}")

    else:
        print(key)

# %%

"""
This is a helper method to place our performance results in a DataFrame for future analysis.
"""
def format_results(y_test, predicted_values, fold_index, fitTime):
    # get scores
    accuracy = accuracy_score(y_test,predicted_values)
    recall_pos = recall_score(y_test, predicted_values)
    precision_pos = precision_score(y_test,predicted_values)
    f1 = f1_score(y_test,predicted_values)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_values).ravel()

    # This avoids divide by zero errors in some cases of no predicted samples.
    if (tn + fp) > 0:
        recall_neg = tn / (tn + fp)
    else:
        recall_neg = 0

    if (tn + fn) > 0:
        precision_neg = tn / (tn + fn)
    else:
        precision_neg = 0

    cols = ["Fitting Time", "accuracy", "TP", "TN", "FP", "FN", "Precision: 0", "Precision: 1", "Recall: 0", "Recall: 1", "F1 Score"]
    results = [fitTime, accuracy, tp, tn, fp, fn, precision_neg, precision_pos, recall_neg, recall_pos, f1]

    outFrame = pd.DataFrame([results], columns=cols, index=[fold_index])

    return outFrame

# %% [markdown]
# This is for tuning on the multiclass set

# %%
models = {
    #"Decision Tree": DecisionTreeClassifier(random_state=42),
    #"Random Forest": RandomForestClassifier(random_state=42),
    #"XGBoost": xgb.XGBClassifier(random_state=42, num_class=12, objective='multi:softmax'),
    #"Linear SVC": make_pipeline(StandardScaler(), LinearSVC(random_state=42)),
    #"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
    #"KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

score_methods = ['accuracy']

feature_set = feature_sets["Multiclass"]

y = binary[" Label"].copy()
X = binary.drop([" Label"], axis=1)

y_test_f = binary_test[" Label"].copy()
X_test_f = binary_test.drop([" Label"], axis=1)

# %%
def fold_run(opt, feature_key, feature_val, train_index, test_index, counter):

    # PCA requires different logic to create X.
    if feature_key != "PCA":
        current_X = X.loc[:, feature_val]
        X_train, X_test = current_X.iloc[train_index,:], current_X.iloc[test_index,:]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]

        current_X_f = X_test_f.loc[:, feature_val]
                        
    else:
        pca_trans = PCA(n_components=feature_val, random_state=42)
        current_X = X.loc[:, X.columns]
        X_train, X_test = current_X.iloc[train_index,:], current_X.iloc[test_index,:]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]

        # Apply PCA to training set and use it to transform test set.
        X_train = pca_trans.fit_transform(X_train)
        X_test = pca_trans.transform(X_test)
        current_X_f = pca_trans.transform(X_test_f)

        # Convert back to DataFrames
        pca_cols = ["PC"+str(i) for i in list(range(1, len(X_train[0])+1))]
        X_train = pd.DataFrame(data=X_train, columns=pca_cols)
        X_test = pd.DataFrame(data=X_test, columns=pca_cols)
        current_X_f = pd.DataFrame(data=current_X_f, columns=pca_cols)

    startTime = time()

    opt.fit(X_train,Y_train)

    endTime = time()
    fitTime = endTime - startTime

    predicted_values = opt.predict(X_test)

    # get metrics for this fold.
    foldFrame = format_results(Y_test, predicted_values, counter, fitTime)

    # Print a classification report on the testing results.
    #print(f"Validation Results {counter}: ")
    #print(classification_report(Y_test, predicted_values, target_names=multiclass_labels, digits=6))

    #print(f"Testing Results {counter}: ")
    # Print a classification report on the testing results.
    pred_test = opt.predict(current_X_f)
    #print(classification_report(y_test_f, pred_test, target_names=multiclass_labels, digits=6))

    f1 = f1_score(y_test_f,pred_test, average='macro')

    return foldFrame, counter, opt.best_params_, f1

# %%
if __name__ == '__main__':
    params = []

    executor = ProcessPoolExecutor(max_workers=3)

    # This will hold all of our results.
    runFrame = None

    for name, model in models.items():
            for feature_key, feature_val in feature_set.items():

                # If we're on the RFE sets, check if we have one for this classifier. If not, skip it.
                if feature_key == "RFE Sets":
                    if name in feature_val.keys():
                        feature_val = feature_val[name]
                    else:
                        continue
            
                for score_method in score_methods:
                    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

                    counter = 0

                    # Used to hold data for a single run (performance metric)
                    frameList = [None, None, None, None, None]
                    perfFrame = None

                    futures = []

                    for train_index, test_index in kf.split(X, y):                   

                        opt = BayesSearchCV(estimator=clone(model),search_spaces=search_spaces[name],n_iter=50,scoring=score_method,cv=5,n_jobs=5)

                        futures.append(executor.submit(fold_run, opt, feature_key, feature_val, train_index, test_index, counter))

                        counter += 1

                    for future in as_completed(futures):
                        foldFrame, counter, best_param, f1 = future.result()

                        # Add them to our lists of metric.
                        frameList[counter] = foldFrame

                        # Add tuple with the best params as well as the related model/config
                        params.append((f"Binary {name} {feature_key} {score_method} Fold {counter}", f1, best_param))

                    perfFrame = pd.concat(frameList)
                    # Create a new line in the results table that averages all the folds
                    perfFrame.loc["fold average"] = perfFrame.mean()

                    # Mark the results table with the chosen classifier and the current performance metric.
                    perfFrame['metric'] = [score_method for j in range(0,6)]
                    perfFrame['Classifier'] = [name for j in range(0,6)]
                    perfFrame['Feature Set'] = [feature_key for j in range(0,6)]
                    perfFrame['Dataset'] = ["binary" for j in range(0,6)]
                    print(f"{name} with {feature_key} and {score_method} completed.")

                    # Add this run to the table with all runs.
                    if runFrame is None:
                        runFrame = perfFrame
                    else:
                        runFrame = pd.concat([runFrame, perfFrame])

    # Write output file, best parameters, and best models to be used later.
    runFrame.to_csv(f"binary_results_NB_{time()}.csv")
    pickle.dump(params, open(f"binary_params_NB_{time()}.pickle", "wb"))
    executor.shutdown(wait=False)


