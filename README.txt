An implementation of Gaussian Mixture Model with an example of election analysis involving voter registration, to model the voter preferences in a city.

In the dataset, i is the precinct of the city, j is the individual respondent within its precinct, Zij represents which party this respondent prefers, Xij(1) and Xij(2) are the scores of the overall social conservatism/liberalism and economic conservatism/liberalism of the j-th respondent in district i.

Model 1 is a simple Gussian Mixture Model, and it performs the same as open source library GMM tools (e.g. Sklearn)

Model 2 is a more sophisticated geography-aware mixture model. By running two different algorithms, we can see the latter one has higher MLE probability. This is not available in open source libraries.

Run GMM.py to see the simple Gaussian Mixture Model.

Run GMM_GA.py to see the integrated Bernoulli-GMM algorithm.