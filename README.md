Highly interpretable, sklearn-compatible classifier based on decision rules
===============

This is a scikit-learn compatible wrapper for the Bayesian Rule List classifier 
developed by [Letham et al., 2015](http://projecteuclid.org/euclid.aoas/1446488742) (see [Letham's original code](http://lethalletham.com/)), 
extended by a minimum description length-based discretizer ([Fayyad &
Irani, 1993](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf)) for continuous data.

It produces rule lists, which makes trained classifiers **easily interpretable 
to human experts**, and is competitive with state of the art classifiers such as 
random forests or SVMs.

For example, an easily understood Rule List model of the well-known Titanic dataset:

```
IF male AND adult THEN survival probability: 21% (19% - 23%)
ELSE IF 3rd class THEN survival probability: 44% (38% - 51%)
ELSE IF 1st class THEN survival probability: 96% (92% - 99%)
ELSE survival probability: 88% (82% - 94%)
``` 

Letham et al.'s approach only works on discrete data. However, this approach can still be used
on continuous data after discretization. The RuleListClassifier class also includes a discretizer 
that can deal with continuous data (using [Fayyad & Irani's](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf) 
minimum description length principle criterion, based on an implementation by 
[navicto](https://github.com/navicto/Discretization-MDLPC)).

Usage
===============

The project requires [pyFIM](http://www.borgelt.net/pyfim.html), [scikit-learn](http://scikit-learn.org/stable/install.html), and [pandas](http://pandas.pydata.org/) to run.

Usage example:

```python
feature_labels = ["#Pregnant","Glucose concentration test","Blood pressure(mmHg)","Triceps skin fold thickness(mm)","2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]
    
data = fetch_mldata("diabetes") # get dataset
y = (data.target+1)/2 # target labels (0 or 1)
Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y) # split

# train classifier (allow more iterations for better accuracy)
clf = RuleListClassifier(max_iter=10000, class1label="diabetes", verbose=False)
clf.fit(Xtrain, ytrain, feature_labels=feature_labels)

print "RuleListClassifier Accuracy:", clf.score(Xtest, ytest), "Learned interpretable model:\n", clf
print "RandomForestClassifier Accuracy:", sklearn.ensemble.RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest)
"""
**Output:**
RuleListClassifier Accuracy: 0.776041666667 Learned interpretable model:
Trained RuleListClassifier for detecting diabetes
==================================================
IF Glucose concentration test : 157.5_to_inf THEN probability of diabetes: 81.1% (72.5%-72.5%)
ELSE IF Body mass index : -inf_to_26.3499995 THEN probability of diabetes: 5.2% (1.9%-1.9%)
ELSE IF Glucose concentration test : -inf_to_103.5 THEN probability of diabetes: 14.4% (8.8%-8.8%)
ELSE IF Age (years) : 27.5_to_inf THEN probability of diabetes: 59.6% (51.8%-51.8%)
ELSE IF Glucose concentration test : 103.5_to_127.5 THEN probability of diabetes: 15.9% (8.0%-8.0%)
ELSE probability of diabetes: 44.7% (29.5%-29.5%)
=================================================

RandomForestClassifier Accuracy: 0.729166666667
"""
```