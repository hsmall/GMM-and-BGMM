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

LABELED_FILE = "surveylabeled.dat"
UNLABELED_FILE = "surveyunlabeled.dat"


#===============================================================================
# General helper functions

def colorprint(message, color="rand"):
    """Prints your message in pretty colors! 

    So far, only the colors below are available.
    """
    if color == 'none': print message; return
    if color == 'demo':
        for i in range(99):
            print '%i-'%i + '\033[%sm'%i + message + '\033[0m\t',
    print '\033[%sm'%{
        'neutral' : 99,
        'flashing' : 5,
        'underline' : 4,
        'magenta_highlight' : 45,
        'red_highlight' : 41,
        'pink' : 35,
        'yellow' : 93,   
        'teal' : 96,     
        'rand' : np.random.randint(1,99),
        'green?' : 92,
        'red' : 91,
        'bold' : 1
    }.get(color, 1)  + message + '\033[0m'

def read_labeled_matrix(filename):
    """Read and parse the labeled dataset.

    Output:
        Xij: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        Zij: dictionary of party choices.
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a float.
        N, M: Counts of precincts and voters.
    """
    Zij = {} 
    Xij = {}
    M = 0.0
    N = 0.0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            i, j, Z, X1, X2 = line.split()
            i, j = int(i), int(j)
            if i>N: N = i
            if j>M: M = j

            Zij[i-1, j-1] = float(Z)
            Xij[i-1, j-1] = np.matrix([float(X1), float(X2)])
    return Xij, Zij, N, M


def read_unlabeled_matrix(filename):
    """Read and parse the unlabeled dataset.
    
    Output:
        Xij: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters.
    """
    Xij = {}
    M = 0.0
    N = 0.0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            i, j, X1, X2 = line.split()
            i, j = int(i), int(j)
            if i>N: N = i
            if j>M: M = j

            Xij[i-1, j-1] = np.matrix([float(X1), float(X2)])
    return Xij, N, M


#===============================================================================
# Functions that define the probability distribution

def p_yi(y_i, phi):
    """Probability of y_i.

    Bernouilli distribution with parameter phi.
    """
    return (phi**y_i) * ((1-phi)**(1-y_i))

def p_zij(z_ij, pi):
    """Probability of z_ij.

    Bernouilli distribution with parameter pi.
    """
    return (pi**z_ij) * ((1-pi)**(1-z_ij))


def p_zij_given_yi(z_ij, y_i, lambd):
    """Probability of z_ij given yi.

    Bernouilli distribution with parameter lambd that targets
    the variable (z_ij == y_i).
    """
    if z_ij == y_i:
        return lambd
    return 1-lambd

def z_marginal(z_ij, lambd, phi):
    """Marginal probability of z_ij with yi marginalized out."""
    return p_zij_given_yi(z_ij, 1, lambd) * p_yi(1, phi) \
         + p_zij_given_yi(z_ij, 0, lambd) * p_yi(0, phi)

def p_xij_given_zij(x_ij, mu_zij, sigma_zij):
    """Probability of x_ij.

    Given by multivariate normal distribution with params mu_zij and sigma_zij.
    
    Input:
        x_ij: (1,2) array of continuous variables
        mu_zij: (1,2) array representing the mean of class z_ij
        sigma_zij: (2,2) array representing the covariance matrix of class z_ij

    All arrays must be instances of numpy.matrix class.
    """
    assert isinstance(x_ij, np.matrix)
    k = x_ij.shape[1]; assert(k==2)

    det_sigma_zij = sigma_zij[0, 0]*sigma_zij[1, 1] - sigma_zij[1, 0]*sigma_zij[0, 1]
    assert det_sigma_zij > 0

    sigma_zij_inv = -copy.copy(sigma_zij); sigma_zij_inv[0, 0] = sigma_zij[1, 1]; sigma_zij_inv[1, 1] = sigma_zij[0, 0]
    sigma_zij_inv /= det_sigma_zij

    # print "better be identity matrix:\n", sigma_zij.dot(sigma_zij_inv)

    multiplicand =  (((2*math.pi)**k)*det_sigma_zij)**(-0.5)
    exponentiand = -.5 * (x_ij-mu_zij).dot(sigma_zij_inv).dot((x_ij-mu_zij).T)
    exponentiand = exponentiand[0,0]
    return multiplicand * np.exp(exponentiand)

def p_xij_given_yi(x_ij, y_i, mu_0, sigma_0, mu_1, sigma_1, lambd):
    """Probability of x_ij given y_i.
    
    To compute this, marginalize (i.e. sum out) over z_ij.
    """

    # -------------------------------------------------------------------------
    
    return p_zij_given_yi(0, y_i, lambd) * p_xij_given_zij(x_ij, mu_0, sigma_0) + p_zij_given_yi(1, y_i, lambd) * p_xij_given_zij(x_ij, mu_1, sigma_1)
    # -------------------------------------------------------------------------
    



#===============================================================================
# MLE estimation for Model 1

def MLE_Estimation():
    """Perform MLE estimation of model 1 parameters.

    Output:
        pi: (float), estimate of party proportions
        mean0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
        mean1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
        sigma0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
        sigma1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1
    """
    Xij, Zij, N, M = read_labeled_matrix(LABELED_FILE)
    assert(len(Zij.items()) == M*N)

    meanZ = 0.0
    # -------------------------------------------------------------------------
    # TODO: Code to compute meanZ
    # The probability of Z == 1
    meanZ = sum(Zij.values())/len(Zij.values())
    # -------------------------------------------------------------------------
    mean0 = np.matrix([0.0, 0.0])
    mean1 = np.matrix([0.0, 0.0])
    # -------------------------------------------------------------------------
    # TODO: Code to compute mean0, mean1
    # z0 is a list of xij with zij == 1
    z0 = [Xij[_] for _ in Xij.keys() if Zij[_]==0.0]
    z1 = [Xij[_] for _ in Xij.keys() if Zij[_]==1.0]
    mean0 = np.mean(z0,axis=0)
    mean1 = np.mean(z1,axis=0)
    # -------------------------------------------------------------------------
    sigma0 = np.matrix([[0.0,0.0],[0.0,0.0]])
    sigma1 = np.matrix([[0.0,0.0],[0.0,0.0]])
    # -------------------------------------------------------------------------
    # TODO: Code to compute sigma0, sigma1
    sigma0 = np.cov(np.array(z0).T[0],np.array(z0).T[1], bias=1)
    sigma1 = np.cov(np.array(z1).T[0],np.array(z1).T[1], bias=1)
    # An alternative to compute sigma0 and sigma1
    '''
    count0 = 0.0
    count1 = 0.0
    sigma0Xijtotal = np.matrix([[0.0,0.0],[0.0,0.0]])
    sigma1Xijtotal = np.matrix([[0.0,0.0],[0.0,0.0]])
    for i in xrange(0,N):
        for j in xrange(0,M):
            if Zij[(i,j)] == 0:
                tmp0 = Xij[(i,j)] - mean0
                sigma0Xijtotal += np.dot(tmp0.T,tmp0)
                count0 += 1.0
            else:
                tmp1 = Xij[(i,j)] - mean1
                sigma1Xijtotal += np.dot(tmp1.T,tmp1)
                count1 += 1.0
    sigm0 = sigma0Xijtotal/(count0)
    sigm1 = sigma1Xijtotal/(count1)
    '''
    # -------------------------------------------------------------------------
    return meanZ, mean0, mean1, sigma0, sigma1
#===============================================================================
# Parameter estimation for Model 2

def MLE_of_phi_and_lamdba():
    """Perform MLE estimation of model 2 parameters.

    Assumes that Z variables have been estimated using heuristic proposed in
    the question.

    Output:
        MLE_phi: estimate of phi
        MLE_lambda: estimate of lambda
    """
    Xij, Zij, N, M = read_labeled_matrix(LABELED_FILE)
    assert(len(Zij.items()) == M*N)

    MLE_phi, MLE_lambda = 0.0, 0.0
    # -------------------------------------------------------------------------
    # TODO: Code to compute MLE_phi, MLE_lambda
    Z=[[] for _ in range(N)]
    for _ in Zij:
        Z[_[0]].append(Zij[_])

    Y=[0]*N
    dif = 0
    for i in range(N):
        Y[i] = 1.0 if sum(Z[i])>=M/2 else 0.0
        dif+=(M-sum(Z[i]))**(1-Y[i]) * sum(Z[i])**Y[i]
    MLE_phi = np.mean(Y)
    MLE_lambda = dif/(M*N)
    # -------------------------------------------------------------------------
    return MLE_phi, MLE_lambda
#===============================================================================
# Estimate the leanings of the precincts

def estimate_leanings_of_precincts(X, N, M, params=None):
    """Estimate the leanings y_i given data X.

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        params: parameters of the model given as a dictionary
            Dictionary should contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']

    Output:
        Summary: length-N list summarizing the leanings
            Format is: [(i, prob_i, y_i) for i in range(N)]
    """
    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']

    posterior_y = [None for i in range(N)] 
    # -------------------------------------------------------------------------
    # TODO: Code to compute posterior_y
    for i in range(N):
        p_y0 = p_yi(0,MLE_phi)
        p_y1 = p_yi(1,MLE_phi)
        for key in sorted(X.keys())[i*M:(i+1)*M]:
            p_y0 *= p_xij_given_yi(X[key], 0, mu_0, sigma_0, mu_1, sigma_1, MLE_lambda)
            p_y1 *= p_xij_given_yi(X[key], 1, mu_0, sigma_0, mu_1, sigma_1, MLE_lambda)
        posterior_y[i]=p_y1/(p_y0+p_y1)
    summary = [(i, p, 1 if p>=.5 else 0) for i, p in enumerate(posterior_y)]
    return summary

def plot_individual_inclinations(X, N, M, params=None):
    """Generate 2d plot of inidivudal statistics in each class.

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        params: parameters of the model given as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
    """

    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']

    domain0 = []
    range0 = []
    domain1 = []
    range1 = []
    summary = estimate_leanings_of_precincts(X, N, M)
    for (i, j), x_ij in sorted(X.items()):
        posterior_z = [0.0, 0.0]
        # -------------------------------------------------------------------------
        # TODO: Code to compute posterior_z
        p0 = p_xij_given_zij(x_ij, mu_0, sigma_0)
        p1 = p_xij_given_zij(x_ij, mu_1, sigma_1)
        phi = summary[i][1]
        posterior_z[0] = p0*((1-phi)*MLE_lambda+phi*(1-MLE_lambda))
        posterior_z[1] = p1*((phi)*MLE_lambda+(1-phi)*(1-MLE_lambda))
        if posterior_z[1] >= posterior_z[0]:
            domain0.append(x_ij[0, 0])
            range0.append(x_ij[0, 1])
        else:
            domain1.append(x_ij[0, 0])
            range1.append(x_ij[0, 1]) 
            # -------------------------------------------------------------------------
    plt.plot(domain1, range1, 'r+')          
    plt.plot(domain0, range0, 'b+')
    p1,  = plt.plot(mu_0[0,0], mu_0[0,1], 'kd')
    p2,  = plt.plot(mu_1[0,0], mu_1[0,1], 'kd')
    plt.show()  

def perform_em_model1(X, N, M, params, max_iters=50, eps=1e-2):
    """Estimate Model 1 paramters using EM

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        init_params: parameters of the model given as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 

    Output:
        params: parameters of the trained model encoded as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
        log_likelihood: array of log-likelihood values across iterations
    """

    def compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi):
        """Compute the log-likelihood of the data given our parameters

        Input:
            pi: (float), estimate of party proportions
            mu_0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
            mu_1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
            sigma_0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
            sigma_1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1

        Output:
            ll: (float), value of the log-likelihood
        """
        ll = 0.0
        # -------------------------------------------------------------------------
        for (i,j),x_ij in X.items():
            p0 = (1-pi)*p_xij_given_zij(x_ij, mu_0, sigma_0)
            p1 = (pi)*p_xij_given_zij(x_ij, mu_1, sigma_1)
            ll += np.log(p0+p1)
        # -------------------------------------------------------------------------
        return ll

    mu_0 = params['mu_0']
    mu_1 = params['mu_1']
    sigma_0 = params['sigma_0']
    sigma_1 = params['sigma_1']
    pi = params['pi']

    log_likelihood = [compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi)]

    for iter in xrange(max_iters):
        # -------------------------------------------------------------------------
        # TODO: E step
        weight={}
        for (i,j),x_ij in X.items():
            #z = 0
            z_0 = (1-pi)*p_xij_given_zij(x_ij, mu_0, sigma_0)
            #z = 1
            z_1 = (pi)*p_xij_given_zij(x_ij, mu_1, sigma_1)
            weight[(i,j)] = (z_0/(z_0+z_1), z_1/(z_0+z_1))
        weightSum = np.sum(weight.values(),axis=0)
        # -------------------------------------------------------------------------
        pi = 0.0
        mu_0 = np.matrix([0.0, 0.0])
        mu_1 = np.matrix([0.0, 0.0])
        sigma_0 = np.matrix([[0.0,0.0],[0.0,0.0]])
        sigma_1 = np.matrix([[0.0,0.0],[0.0,0.0]])
        # -------------------------------------------------------------------------
        # TODO: M step
        pi = weightSum[1]/(M*N)
        for k in weight.keys():
            mu_0 += X[k]*weight[k][0]
            mu_1 += X[k]*weight[k][1]
        mu_0/=weightSum[0]
        mu_1/=weightSum[1]
        for k in weight.keys():
            sigma_0 += X[k].T*X[k] * weight[k][0]
            sigma_1 += X[k].T*X[k] * weight[k][1]
        sigma_0/=weightSum[0]
        sigma_1/=weightSum[1]
        sigma_0-=mu_0.T*mu_0
        sigma_1-=mu_1.T*mu_1
        # -------------------------------------------------------------------------
        # Check for convergence
        this_ll = compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi)
        log_likelihood.append(this_ll)
        if np.abs((this_ll - log_likelihood[-2]) / log_likelihood[-2]) < eps:
            break
    # pack the parameters and return

    params['mu_0'] = mu_0
    params['mu_1'] = mu_1
    params['sigma_0'] = sigma_0
    params['sigma_1'] = sigma_1
    params['pi'] = pi

    return params, log_likelihood

def random_covariance():
    P = np.matrix(np.random.randn(2,2))
    D = np.matrix(np.diag(np.random.rand(2) * 0.5 + 1.0))
    return P*D*P.T
    
if __name__ == '__main__':
    X, N, M = read_unlabeled_matrix(UNLABELED_FILE)
    #===============================================================================
    # Run model 1
    # choosing initial values of pi,mu and sigma based on the supervised data
    params = {}
    pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
    MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    params['pi'] = pi
    params['mu_0'] = mu_0
    params['mu_1'] = mu_1
    params['sigma_0'] = sigma_0
    params['sigma_1'] = sigma_1
    params, log_likelihood = perform_em_model1(X, N, M, params)
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
        params, log_likelihood = perform_em_model1(X, N, M, params)
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
