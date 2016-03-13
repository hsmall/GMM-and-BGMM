# Learning: GMM and EM
# Author : Jinjing Zhou, Gunaa A V, Isaac C
# Date : 2 Feb 2016

import numpy as np
from pprint import pprint
import copy
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
from GMM_EM import *

LABELED_FILE = "surveylabeled.dat"
UNLABELED_FILE = "surveyunlabeled.dat"


def perform_em(X, N, M, params, max_iters=50, eps=1e-2):
    """Estimate Model 2 paramters using EM

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        init_params: parameters of the model given as a dictionary
            Dictionary shoudl contain: params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']

    Output:
        params: parameters of the trained model encoded as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
        log_likelihood: array of log-likelihood values across iterations
    """

    def compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd):
        """Compute the log-likelihood of the data given our parameters

        Input:
            mu_0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
            mu_1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
            sigma_0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
            sigma_1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1
            phi: hyperparameter for princinct preferences
            lambd: hyperparameter for princinct preferences

        Output:
            ll: (float), value of the log-likelihood
        """
        ll = 0.0
        # -------------------------------------------------------------------------
        for (i,j),x_ij in sorted(X.items()):
            if j%M == 0:
                y_0 = np.log(1 - phi)
                y_1 = np.log(phi)
            p0 = p_xij_given_zij(x_ij, mu_0, sigma_0)
            p1 = p_xij_given_zij(x_ij, mu_1, sigma_1)
            p_y0 = lambd * p0 + (1-lambd) * p1
            p_y1 = lambd * p1 + (1-lambd) * p0
            y_0 += np.log(p_y0)
            y_1 += np.log(p_y1)
            if j%M == M-1:
                ll += np.log(np.exp(y_0)+np.exp(y_1))
        # -------------------------------------------------------------------------
        return ll

    mu_0 = params['mu_0']
    mu_1 = params['mu_1']
    sigma_0 = params['sigma_0']
    sigma_1 = params['sigma_1']
    phi = params['phi']
    lambd = params['lambda']

    log_likelihood = [compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd)]
    for iter in xrange(max_iters):
        # -------------------------------------------------------------------------
        # TODO: E step
        summary = estimate_leanings_of_precincts(X, N, M, params)
        weight = {}
        for (i,j),x_ij in X.items():
            p0 = p_xij_given_zij(x_ij, mu_0, sigma_0)
            p1 = p_xij_given_zij(x_ij, mu_1, sigma_1)
            #for each weight, (y0z0, y0z1, y1z0, y1z1)
            res = ((1-summary[i][1])*lambd*p0, (1-summary[i][1])*(1-lambd)*p1, summary[i][1]*(1-lambd)*p0, summary[i][1]*lambd*p1)
            weight[(i,j)] = (res[0]/sum(res), res[1]/sum(res), res[2]/sum(res), res[3]/sum(res))
        # -------------------------------------------------------------------------
        phi, lambd = 0.0, 0.0
        mu_0 = np.matrix([0.0, 0.0])
        mu_1 = np.matrix([0.0, 0.0])
        sigma_0 = np.matrix([[0.0,0.0],[0.0,0.0]])
        sigma_1 = np.matrix([[0.0,0.0],[0.0,0.0]])


        # -------------------------------------------------------------------------
        # TODO: M step
        weightSum = np.sum(weight.values(),axis=0)
        phi = (weightSum[2] + weightSum[3])/(N*M)
        lambd = (weightSum[0] + weightSum[3])/(N*M)

        for k in weight.keys():
            mu_0 += X[k]*(weight[k][0]+weight[k][2])
            mu_1 += X[k]*(weight[k][1]+weight[k][3])
        mu_0/=(weightSum[0]+weightSum[2])
        mu_1/=(weightSum[1]+weightSum[3])
        for k in weight.keys():
            sigma_0 += X[k].T*X[k] * (weight[k][0]+weight[k][2])
            sigma_1 += X[k].T*X[k] * (weight[k][1]+weight[k][3])
        sigma_0/=(weightSum[0]+weightSum[2])
        sigma_1/=(weightSum[1]+weightSum[3])
        sigma_0-=mu_0.T*mu_0
        sigma_1-=mu_1.T*mu_1
        # -------------------------------------------------------------------------

        # Check for convergence
        this_ll = compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd)
        log_likelihood.append(this_ll)
        if np.abs((this_ll - log_likelihood[-2]) / log_likelihood[-2]) < eps:
            break
        params['mu_0'] = mu_0
        params['mu_1'] = mu_1
        params['sigma_0'] = sigma_0
        params['sigma_1'] = sigma_1
        params['lambda'] = lambd
        params['phi'] = phi

    # pack the parameters and return

    params['mu_0'] = mu_0
    params['mu_1'] = mu_1
    params['sigma_0'] = sigma_0
    params['sigma_1'] = sigma_1
    params['lambda'] = lambd
    params['phi'] = phi

    return params, log_likelihood

def random_covariance():
    P = np.matrix(np.random.randn(2,2))
    D = np.matrix(np.diag(np.random.rand(2) * 0.5 + 1.0))
    return P*D*P.T


if __name__ == '__main__':
    X, N, M = read_unlabeled_matrix(UNLABELED_FILE)

    #===============================================================================
    # Run model 2
    # choosing initial values of pi,mu and sigma based on the labelled data
    X, N, M = read_unlabeled_matrix(UNLABELED_FILE)
    params = {}
    pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
    MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    params['pi'] = pi
    params['mu_0'] = mu_0
    params['mu_1'] = mu_1
    params['sigma_0'] = sigma_0
    params['sigma_1'] = sigma_1
    params['phi'] = MLE_phi
    params['lambda'] = MLE_lambda
    params, log_likelihood = perform_em(X, N, M, params)
    params_list = [params]
    log_likelihood_list = [log_likelihood]

    # choosing random initial values of pi,mu and sigma
    for _ in range(2):
        params = {}
        params['pi'] = np.random.rand()
        params['mu_0'] = np.random.randn(1,2)
        params['mu_1'] = np.random.randn(1,2)
        params['sigma_0'] = random_covariance()
        params['sigma_1'] = random_covariance()
        params['phi'] = np.random.rand()
        params['lambda'] = np.random.rand()
        params, log_likelihood = perform_em(X, N, M, params)
        params_list.append(params)
        log_likelihood_list.append(log_likelihood)


    plt.figure()
    for i, params in enumerate(params_list):
        colorprint('For initialization parameter %s: '%i,'teal')
        colorprint('%s'%log_likelihood_list[i],'red')
        colorprint('%s'%params,'yellow')
        plt.plot(log_likelihood_list[i])
    plt.legend(['MLE initialization', 'Random initialization', 'Random initialization'], loc=4)
    plt.xlabel('Iteration')
    plt.ylabel('Log likelihood')
    plt.show()

    #===============================================================================
    # Try 30 initial values and choose the parameters with highest MLE (unsupervised learning) and Plot the MLE of model 2 for Z values (individual inclinations)

    MAX_MLE = float('-inf')
    MAX_params = None
    for _ in range(30):
        print 'Random start %s'%_
        params = {}
        params['pi'] = np.random.rand()
        params['mu_0'] = np.random.randn(1,2)
        params['mu_1'] = np.random.randn(1,2)
        params['sigma_0'] = random_covariance()
        params['sigma_1'] = random_covariance()
        params['phi'] = np.random.rand()
        params['lambda'] = np.random.rand()
        try:
            params, log_likelihood = perform_em(X, N, M, params)
            if log_likelihood[-1]>MAX_MLE:
                MAX_MLE = log_likelihood[-1]
                MAX_params = params
                print 'Update: MLE = ',MAX_MLE
        except:
            print 'Invalid random start!'
            pass
    colorprint('The final maximum likelihood estimation likelihood is: %s'%MAX_MLE,'teal')
    colorprint('The final maximum likelihood estimation parameters are:\n%s'% MAX_params,'red')
    plot_individual_inclinations(X, N, M, params=MAX_params)