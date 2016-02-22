from sklearn.base import BaseEstimator
import sklearn.metrics
import sys
import numpy as np
from LethamBRL import *

class RuleListClassifier(BaseEstimator):
    """
    This is a scikit-learn compatible wrapper for the Bayesian Rule List
    classifier developed by Benjamin Letham. It produces a highly
    interpretable model (a list of decision rules) of the same form as
    an expert system. 

    Parameters
    ----------
    listlengthprior : int, optional (default=3)
        Prior hyperparameter for expected list length (excluding null rule)

    listwidthprior : int, optional (default=1)
        Prior hyperparameter for expected list length (excluding null rule)
        
    maxcardinality : int, optional (default=1)
        Maximum cardinality of an itemset
        
    minsupport : int, optional (default=10)
        Minimum support (%) of an itemset

    alpha : array_like, shape = [n_classes]
        prior hyperparameter for multinomial pseudocounts

    n_chains : int, optional (default=3)
        Number of MCMC chains for inference

    max_iter : int, optional (default=50000)
        Maximum number of iterations
    """
    
    def __init__(self, listlengthprior=3, listwidthprior=1, maxcardinality=2, minsupport=10, alpha = array([1.,1.]), n_chains=3, max_iter=50000):
        self.listlengthprior = listlengthprior
        self.listwidthprior = listwidthprior
        self.maxcardinality = maxcardinality
        self.minsupport = minsupport
        self.alpha = alpha
        self.n_chains = n_chains
        self.max_iter = max_iter
        
        self.thinning = 1 #The thinning rate
        self.burnin = self.max_iter//2 #the number of samples to drop as burn-in in-simulation
        
        
    def fit(self, X, y): # -1 for unlabeled
        """Fit rule lists to data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data 

        y : array_like, shape = [n_samples]
            Labels

        Returns
        -------
        self : returns an instance of self.
        """
        permsdic = defaultdict(default_permsdic) #We will store here the MCMC results
        
        data = np.copy(X)
        #Now find frequent itemsets
        #Mine separately for each class
        data_pos = [x for i,x in enumerate(data) if y[i]==0]
        data_neg = [x for i,x in enumerate(data) if y[i]==1]
        assert len(data_pos)+len(data_neg) == len(data)
        try:
            itemsets = [r[0] for r in fpgrowth(data_pos,supp=self.minsupport,zmax=self.maxcardinality)]
            itemsets.extend([r[0] for r in fpgrowth(data_neg,supp=self.minsupport,zmax=self.maxcardinality)])
        except TypeError:
            itemsets = [r[0] for r in fpgrowth(data_pos,supp=self.minsupport,max=self.maxcardinality)]
            itemsets.extend([r[0] for r in fpgrowth(data_neg,supp=self.minsupport,max=self.maxcardinality)])
        itemsets = list(set(itemsets))
        print len(itemsets),'rules mined'
        #Now form the data-vs.-lhs set
        #X[j] is the set of data points that contain itemset j (that is, satisfy rule j)
        X = [ set() for j in range(len(itemsets)+1)]
        X[0] = set(range(len(data))) #the default rule satisfies all data
        for (j,lhs) in enumerate(itemsets):
            X[j+1] = set([i for (i,xi) in enumerate(data) if set(lhs).issubset(xi)])
        #now form lhs_len
        lhs_len = [0]
        for lhs in itemsets:
            lhs_len.append(len(lhs))
        nruleslen = Counter(lhs_len)
        lhs_len = array(lhs_len)
        itemsets_all = ['null']
        itemsets_all.extend(itemsets)
        
        Xtrain,Ytrain,nruleslen,lhs_len,itemsets = (X,np.vstack((y, 1-y)).T,nruleslen,lhs_len,itemsets_all)
            
        #Do MCMC
        res,Rhat = run_bdl_multichain_serial(self.max_iter,self.thinning,self.alpha,self.listlengthprior,self.listwidthprior,Xtrain,Ytrain,nruleslen,lhs_len,self.maxcardinality,permsdic,self.burnin,self.nchains,[None]*self.nchains)
            
        #Merge the chains
        permsdic = merge_chains(res)
        
        ###The point estimate, BRL-point
        self.d_star = get_point_estimate(permsdic,lhs_len,Xtrain,Ytrain,self.alpha,nruleslen,self.maxcardinality,self.listlengthprior,self.listwidthprior) #get the point estimate
            
        return self
        

