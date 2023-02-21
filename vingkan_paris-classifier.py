#!/usr/bin/env python
# coding: utf-8



from shutil import copyfile
copyfile(src = "../input/fairness.py", dst = "../working/fairness.py")
import pandas as pd




import pickle
from fairness import read_skills

SEED = 0
WEIGHTS = pickle.load(open("../input/PARiS.pickle", "rb"))
all_skills = read_skills("../input/skills.txt")
target = "Interview"
predictors = all_skills
demographics = ["Veteran", "Female", "URM", "Disability"]
data = pd.read_csv("../input/resumes_development.csv", index_col=0)
data.head()




data[[target] + demographics].corr()




from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from fairness import PARiSClassifier, evaluate_model, rank_models




d_train, d_test = train_test_split(data, test_size=0.5, stratify=data[target], shuffle=True, random_state=SEED)
X_train = d_train[predictors]
y_train = d_train[target]
X_test = d_test[predictors]
y_test = d_test[target]
print("Train: N = {0}, P(Interview) = {1:.5f}".format(len(X_train), y_train.mean()))
print("Test:  N = {0}, P(Interview) = {1:.5f}".format(len(X_test), y_test.mean()))




logres = LogisticRegression(solver="liblinear", penalty="l2", fit_intercept=True)
logres.fit(X_train, y_train)
evaluate_model(y_test, logres.predict(X_test))




paris = PARiSClassifier(WEIGHTS)
paris.fit(X_train, y_train)
evaluate_model(y_test, paris.predict(X_test))




models = []
models.append(PARiSClassifier(WEIGHTS))
models.append(LogisticRegression(solver="liblinear", penalty="l2", fit_intercept=True))
models.append(DecisionTreeClassifier())
models.append(KNeighborsClassifier(n_neighbors=3))
print("{} models".format(len(models)))




rdf, cols, clfs = rank_models(models, d_train, y_train, d_test, y_test, predictors, demographics)




rdf[cols].sort_values(by="F1", ascending=False).round(3)




pilot = pd.read_csv("../input/resumes_pilot.csv", index_col=0)
pilot.head()




pilot[[target] + demographics].corr()




Compare the models on the pilot data:




y_pilot = pilot[target]
d_pilot = pilot[predictors + demographics]
rdf, cols, clfs = rank_models(models, d_train, y_train, d_pilot, y_pilot, predictors, demographics)




rdf[cols].sort_values(by="F1", ascending=False).round(3)




from fairness import unvectorize

pclf = clfs[0]
for i, (yt, pa, x) in enumerate(zip(pilot[target], pclf.predict_proba(pilot[predictors])[:,1], pilot.values)):
    if yt == 1 and pa < pclf.threshold:
        print("Applicant {0}, P(I|X) = {1:.3f}".format(i, pa))
        skills = unvectorize(pilot.columns, x)
        print("{}".format(", ".join(skills)))
        print()




sports = [
    "Basketball",
    "Football",
    "Baseball",
    "Swimming",
    "Soccer",
    "Diving"
]
data[[target] + sports].corr()




sports_idx = [predictors.index(sport) for sport in sports]
sports_idx




pd.DataFrame(WEIGHTS[0][sports_idx], sports)

