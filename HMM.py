from itertools import product, chain
import numpy as np
from scipy.special import logsumexp
import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    Rewritten using log probabilities.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A, A_log:   The transition matrix.
            
            O, O_log:   The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = np.asarray(A)
        self.O = np.asarray(O)
        self.A_start = np.ones(self.L)/self.L
        self.A_log = np.log(self.A)
        self.O_log = np.log(self.O)
        self.A_start_log = np.log(self.A_start)


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        logprobs = np.zeros((M+1, self.L))
        seqs = np.asarray([['' for _ in range(self.L)] for _ in range(M + 1)])

        logprobs[1] = self.A_start_log + self.O_log[:, x[0]]
        seqs[1] = np.array([str(st) for st in range(self.L)])

        for t in range(1, M):
            for st in range(self.L):
                max_tr_prob = 0
                max_prev_st = 0
                for prev_st in range(self.L):
                    tr_prob = logprobs[t, prev_st] + self.A_log[prev_st, st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        max_prev_st = prev_st
                max_prob = max_tr_prob + self.O_log[st, x[t]]
                logprobs[t+1, st] = max_prob
                seqs[t+1, st] = seqs[t, max_prev_st] + str(max_prev_st)

        max_idx = logprobs[M].index(max(logprobs[M]))
        max_seq = seqs[M][max_idx]
        return max_seq

 
    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = np.zeros((M+1, self.L))

        alphas[1] = self.A_start_log + self.O_log[:, x[0]]
        if normalize:
            alphas[1] -= logsumexp(alphas[1])

        for t in range(1, M):
            for st in range(self.L):
                prev_logprob = logsumexp(alphas[t, :]+self.A_log[:, st])
                alphas[t+1, st] = prev_logprob + self.O_log[st, x[t]]
            if normalize:
                alphas[t+1] -= logsumexp(alphas[t+1])

        return alphas ### alphas[0] is zero vector


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = np.zeros((M+1, self.L))

        if normalize:
            betas[M] = np.log(np.ones(self.L)/self.L)

        for t in reversed(range(1, M+1)):
            for st in range(self.L):
                next_logprob = logsumexp(betas[t, :]+self.A_log[st, :]+self.O_log[:, x[t-1]])
                betas[t-1, st] = next_logprob
            if normalize:
                betas[t-1] -= logsumexp(betas[t-1])

        return betas


    def unsupervised_learning(self, X, N_iters, threshold=0.001, verbose=False):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.

            threshold:  Threshold value determining early stop of learning.
                        Roughly speaking, the iteration ends when every entries
                        of A and O are updated by less than the threshold.
        '''
        patience = 2    # Num of epochs to be waited before early stopping criterion applies.

        if verbose:
            print('Train unsupervised HMM:')
        for _ in range(N_iters):
            if ((_+1) % (N_iters//10) == 0) and verbose:
                print('.', end='')
            A_num = np.zeros((self.L, self.L))
            A_denom = np.zeros(self.L)
            O_num = np.zeros((self.L, self.D))
            O_denom = np.zeros(self.L)

            A_log_prev = self.A_log
            O_log_prev = self.O_log

            for j in range(len(X)):
                alphas, betas = self.forward(X[j], normalize=True)[1:,:], self.backward(X[j], normalize=True)[1:,:]
                logprob_curr = np.zeros((len(X[j]), self.L))
                logprob_curr_nxt = np.zeros((len(X[j])-1, self.L, self.L))
                for i in range(len(X[j])):
                    logprob_curr[i] = alphas[i,:]+betas[i,:]
                    logprob_curr[i] -= logsumexp(logprob_curr[i])
                    if i != len(X[j])-1:
                        logprob_curr_nxt[i] = self.A_log+alphas[i,:].reshape((self.L,1))+betas[i+1,:]+self.O_log[:,X[j][i+1]]
                        logprob_curr_nxt[i] -= logsumexp(logprob_curr_nxt[i])
                for a in range(self.L):
                    A_denom[a] = np.logaddexp(A_denom[a], logsumexp(logprob_curr[:-1,a]))
                    O_denom[a] = np.logaddexp(O_denom[a], logsumexp(logprob_curr[:,a]))
                    for b in range(self.L):
                        A_num[a,b] = np.logaddexp(A_num[a,b], logsumexp(logprob_curr_nxt[:,a,b]))
                    for w in range(self.D):
                        logprob_w = []
                        for i in range(len(X[j])):
                            if X[j][i]==w:
                                logprob_w.append(logprob_curr[i,a])
                        O_num[a,w] = np.logaddexp(O_num[a,w], np.logaddexp.reduce(logprob_w))

            self.A_log = A_num - A_denom.reshape((self.L,1))
            self.O_log = O_num - O_denom.reshape((self.L,1))

            delta_A = np.abs(self.A_log - A_log_prev)
            delta_O = np.abs(self.O_log - O_log_prev)

            if _>patience and (np.amax(delta_A*np.exp(A_log_prev))+np.amax(delta_O*np.exp(O_log_prev)) < threshold):
                if verbose:
                    print('Converged at', _, 'iterations')
                break

        self.A = np.exp(self.A_log)
        self.O = np.exp(self.O_log)


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = np.sum(np.exp(alphas[-1,:]))
        return prob


    def generate_sonnet(self, df, numLines=14):
        '''
        Generates a sonnet, assuming each lines have 10 syllables.
        The syllable dictionary must be reindexed using the observation index.
        '''
        A = (self.A).copy()
        A_sum = A.sum(axis=1, keepdims=True)
        A = A/A_sum
        O = (self.O).copy()
        O_sum = O.sum(axis=1, keepdims=True)
        O = O/O_sum
        A_start = (self.A_start).copy()
        A_start_sum = np.sum(A_start)
        A_start = A_start/A_start_sum

        sonnet = []
        for _ in range(numLines):
            line = []
            maxsyllensum = 0
            minsyllensum = 0
            current_st = np.random.choice(self.L, p=A_start)    
            while maxsyllensum<10:
                current_word = np.random.choice(self.D, p=O[current_st,:])
                length1 = abs(df.iloc[current_word, 0])
                length2 = abs(df.iloc[current_word, 1])
                isend1 = (df.iloc[current_word, 0]<0)
                isend2 = (df.iloc[current_word, 1]<0)
                
                if minsyllensum + length1>10:
                    pass
                else:
                    line.append(df.index[current_word])
                    if maxsyllensum + max(length1, length2)>=10:
                        minsyllensum += length1
                        maxsyllensum += max(length1, length2)
                    else:
                        if isend1==True:
                            minsyllensum += length2
                            maxsyllensum += length2
                        elif isend2==True:
                            minsyllensum += length1
                            maxsyllensum += length1
                        else:
                            minsyllensum += length1
                            maxsyllensum += max(length1, length2)
                    current_st = np.random.choice(self.L, p=A[current_st,:])
            sonnet.append(line)
        return sonnet

    def generate_sonnet_with_stress(self, df_syl, df_stress, strict=False, numLines=14):
        '''
        Generates a sonnet, assuming each lines have 10 syllables and follow iambic pentameter.
        The syllable and stress dictionaries must be reindexed using the observation index.
        '''
        A = (self.A).copy()
        A_sum = A.sum(axis=1, keepdims=True)
        A = A/A_sum
        O = (self.O).copy()
        O_sum = O.sum(axis=1, keepdims=True)
        O = O/O_sum
        A_start = (self.A_start).copy()
        A_start_sum = np.sum(A_start)
        A_start = A_start/A_start_sum

        sonnet = []
        while len(sonnet)<numLines:
            line = []
            maxsyllensum = 0
            minsyllensum = 0
            current_st = np.random.choice(self.L, p=A_start)    
            while maxsyllensum<10:
                current_word = np.random.choice(self.D, p=O[current_st,:])
                length1 = abs(df_syl.iloc[current_word, 0])
                length2 = abs(df_syl.iloc[current_word, 1])
                isend1 = (df_syl.iloc[current_word, 0]<0)
                isend2 = (df_syl.iloc[current_word, 1]<0)
                
                if minsyllensum + length1>10:
                    pass
                else:
                    line.append(df_syl.index[current_word])
                    if maxsyllensum + max(length1, length2)>=10:
                        minsyllensum += length1
                        maxsyllensum += max(length1, length2)
                    else:
                        if isend1==True:
                            minsyllensum += length2
                            maxsyllensum += length2
                        elif isend2==True:
                            minsyllensum += length1
                            maxsyllensum += length1
                        else:
                            minsyllensum += length1
                            maxsyllensum += max(length1, length2)
                    current_st = np.random.choice(self.L, p=A[current_st,:])
            stress = [df_stress.loc[word][0] for word in line]
            comb = list(product(*stress))
            isregular = False
            for x in comb:
                stressList = list(chain.from_iterable(x))
                stressChng = [stressList[i]-stressList[i-1] for i in range(1, len(stressList))]
                isregular_temp = True
                for i, y in enumerate(stressChng):
                    if not strict and ((i%2 == 0 and y<0) or (i%2 == 1 and y>0)):
                        isregular_temp = False
                        break
                    elif strict and ((i%2 == 0 and y<=0) or (i%2 == 1 and y>=0)):
                        isregular_temp = False
                        break
                if isregular_temp==True:
                    isregular = True
                    break
            if isregular:
                sonnet.append(line)
        return sonnet

def unsupervised_HMM(X, n_states, N_iters, threshold=0.001, verbose=False):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Keep the reindexing table for emission.
    obs_idx = {}
    for i, obs in enumerate(list(observations)):
        obs_idx[obs] = i

    # Reindex the training dataset from 0 to D-1.
    X_reidxd = []
    for x in X:
        x_reidxd = []
        for obs in x:
            x_reidxd.append(obs_idx[obs])
        X_reidxd.append(x_reidxd)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X_reidxd, N_iters, threshold=threshold, verbose=verbose)

    return HMM, obs_idx

def unsupervised_HMM_CV(X, n_states, N_iters, threshold=0.001, n_folds=5, verbose=False):
    '''
    Unsupervised HMM with k-fold cross validation.
    Assess the trained model in terms of the log likelihood on the validation set.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Keep the reindexing table for emission.
    obs_idx = {}
    for i, obs in enumerate(list(observations)):
        obs_idx[obs] = i

    # Reindex the training dataset from 0 to D-1. 
    X_reidxd = []
    for x in X:
        x_reidxd = []
        for obs in x:
            x_reidxd.append(obs_idx[obs])
        X_reidxd.append(x_reidxd)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Split dataset into train-validation pair.
    divider = len(X_reidxd)//n_folds
    random.shuffle(X_reidxd)
    X_val = X_reidxd[:divider]
    X_train = X_reidxd[divider:]

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X_train, N_iters, threshold=threshold, verbose=verbose)
    
    # Calculate log likelihood on validation set.
    loglikelihoods = []
    for x in X_val:
        prob_alpha = HMM.probability_alphas(x)
        if prob_alpha != 0:
            loglikelihoods.append(np.log(prob_alpha))
    loglikelihood = np.mean(loglikelihoods)

    return HMM, loglikelihood, obs_idx