import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
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
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


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
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for st in range(self.L):
            probs[1][st] = self.A_start[st] * self.O[st][x[0]]
            seqs[1][st] = str(st)

        for t in range(1, M):
            for st in range(self.L):
                max_tr_prob = 0
                max_prev_st = 0
                for prev_st in range(self.L):
                    tr_prob = probs[t][prev_st]*self.A[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        max_prev_st = prev_st
                max_prob = max_tr_prob * self.O[st][x[t]]
                probs[t+1][st] = max_prob
                seqs[t+1][st] = seqs[t][max_prev_st] + str(max_prev_st)

        max_idx = probs[M].index(max(probs[M]))
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
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        denom = 0
        for st in range(self.L):
            alphas[1][st] = self.A_start[st]*self.O[st][x[0]]
            if normalize:
                denom += alphas[1][st]
        if normalize:
            alphas[1][st] = alphas[1][st] / denom

        for t in range(1, M):
            denom = 0
            for st in range(self.L):
                prev_prob = 0
                for prev_st in range(self.L):
                    prev_prob += alphas[t][prev_st]*self.A[prev_st][st]
                alphas[t+1][st] = prev_prob*self.O[st][x[t]]
                if normalize:
                    denom += alphas[t+1][st]
            if normalize:
                alphas[t+1][st] = alphas[t+1][st] / denom

        return alphas


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
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        denom = 0
        for st in range(self.L):
            betas[M][st] = 1    
            if normalize:
                denom += betas[M][st]
        if normalize:
            betas[M][st] = betas[M][st] / denom

        for t in reversed(range(1, M+1)):
            denom = 0
            for st in range(self.L):
                next_prob = 0
                for next_st in range(self.L):
                    next_prob += betas[t][next_st]*self.A[st][next_st]*self.O[next_st][x[t-1]]
                betas[t-1][st] = next_prob
                if normalize:
                    denom += betas[t-1][st]
            if normalize:
                betas[t-1][st] = betas[t-1][st] / denom

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        Count_denom = [0 for _ in range(self.L)]
        for a in range(self.L):
            num_ab = [0 for _ in range(self.L)]
            for j in range(len(Y)):
                for i in range(len(Y[j])-1):
                    if Y[j][i]==a:
                        Count_denom[a] += 1
                        for b in range(self.L):
                            if Y[j][i+1]==b:
                                num_ab[b] += 1
            self.A[a] = [num_ab[b] / Count_denom[a] for b in range(self.L)]

        # Calculate each element of O using the M-step formulas.

        Count_denom = [0 for _ in range(self.L)]
        for a in range(self.L):
            num_aw = [0 for _ in range(self.D)]
            for j in range(len(X)):
                for i in range(len(X[j])):
                    if Y[j][i]==a:
                        Count_denom[a] += 1
                        for w in range(self.D):
                            if X[j][i]==w:
                                num_aw[w] += 1
            self.O[a] = [num_aw[w] / Count_denom[a] for w in range(self.D)]


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        for _ in range(N_iters):
            A_num = [[0 for _ in range(self.L)] for _ in range(self.L)]
            A_denom = [[0 for _ in range(self.L)] for _ in range(self.L)]
            O_num = [[0 for _ in range(self.D)] for _ in range(self.L)]
            O_denom = [[0 for _ in range(self.D)] for _ in range(self.L)]
            for j in range(len(X)):
                alphas, betas = self.forward(X[j], normalize=True)[1:], self.backward(X[j], normalize=True)[1:]
                for i in range(len(X[j])):
                    if i != len(X[j]) - 1:
                        A_num_norm = sum(sum(alphas[i][c]*betas[i+1][d]*self.A[c][d]*self.O[d][X[j][i+1]] for c in range(self.L)) for d in range(self.L))
                    A_denom_norm = sum(alphas[i][c]*betas[i][c] for c in range(self.L))
                    for a in range(self.L):
                        for b in range(self.L):
                            if i != len(X[j]) - 1:
                                A_num[a][b] += alphas[i][a]*betas[i+1][b]*self.A[a][b]*self.O[b][X[j][i+1]] / A_num_norm
                                A_denom[a][b] += alphas[i][a]*betas[i][a] / A_denom_norm
                        for w in range(self.D):
                            O_denom[a][w] += alphas[i][a]*betas[i][a] / A_denom_norm
                            if X[j][i]==w:
                                O_num[a][w] += alphas[i][a]*betas[i][a] / A_denom_norm
            for a in range(self.L):
                for b in range(self.L):
                    self.A[a][b] = A_num[a][b] / A_denom[a][b]
                for w in range(self.D):
                    self.O[a][w] = O_num[a][w] / O_denom[a][w]


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        RV_states = [random.random() for _ in range(M)]
        RV_emission = [random.random() for _ in range(M)]

        A_start_sum = sum(self.A_start)
        A_sum = [sum(self.A[a]) for a in range(self.L)]
        O_sum = [sum(self.O[a]) for a in range(self.L)]

        for i in range(M):
            if i==0:
                a = 0
                stopper = True
                while stopper:
                    a += 1
                    if RV_states[i] < (sum(self.A_start[:a]) / A_start_sum):
                        stopper = False
                        a -= 1
                        break
                states.append(a)
                w = 0
                stopper = True
                while stopper:
                    w += 1
                    if RV_emission[i] < (sum(self.O[a][:w]) / O_sum[a]):
                        stopper = False
                        w -= 1
                        break
                emission.append(w)
            else:
                a = 0
                stopper = True
                while stopper:
                    a += 1
                    if RV_states[i] < (sum(self.A[states[i-1]][:a]) / A_sum[states[i-1]]):
                        stopper = False
                        a -= 1
                        break
                states.append(a)
                w = 0
                stopper = True
                while stopper:
                    w += 1
                    if RV_emission[i] < (sum(self.O[a][:w]) / O_sum[a]):
                        stopper = False
                        w -= 1
                        break
                emission.append(w)

        return emission, states


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
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

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

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
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
    HMM.unsupervised_learning(X, N_iters)

    return HMM
