Highly interpretable, sklearn-compatible classifier based on decision rules
===============

This is a scikit-learn compatible wrapper for the Bayesian Rule List classifier 
developed by [Benjamin Letham et al., 2015](http://projecteuclid.org/euclid.aoas/1446488742), 
parallelized for faster inference, and extended by a discretizer for continuous data.

It produces rule lists, which makes trained classifiers **easily interpretable 
to human experts**, and is competitive with state of the art classifiers such as 
random forests or SVMs.

Letham et al.'s approach only works on discrete data. The RuleListClassifier class
here also includes a discretizer that can deal with continuous data (using [Fayyad \&
Irani's](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf) minimum 
description length principle criterion, based on an implementation by 
[navicto](https://github.com/navicto/Discretization-MDLPC)).

Dependencies
- numpy, scipy, pandas, scikit-learn
- [pyFIM](http://www.borgelt.net/pyfim.html)


