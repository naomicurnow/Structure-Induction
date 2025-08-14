
# All params are set to the same value as in Kemp and Tenenbaum (2008)

EPSILON = 1e-3 # for structure fitting: when new score - old score is less, end the loop
N_RESTARTS = 1 # number of times to repeat the complete search
THETA = 1e-3 # complexity prior: P(S|F) ∝ θ^{|S|}

EFF_M = 1000 # effective number of features for similarity data (m)

PARTNERS_PER = 2 # number of members to pair with each when creating seed pairs during search

# prior on edge lengths and sigma 
# paper uses exponential(BETA) (equivalent to gamma(1, BETA))
# this is insufficent for constraining edge lengths for the tree form, so swapped to gamma(2, BETA)
PRIOR = 'gamma' 
BETA = 0.4 # rate
ALPHA = 2 # shape