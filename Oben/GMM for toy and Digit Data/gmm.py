import numpy as np
from kmeans import KMeans

def pdf_multivariate(X, mu, s):
    if np.linalg.det(s) == 0:
        s =  s + 1e-3*np.identity(s.shape[0])
    return np.linalg.det(s) ** -0.5 * (2 * np.pi) ** (-X.shape[1]/2.0) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(s) , (X - mu).T).T ) ) 

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means = KMeans(self.n_cluster,self.max_iter,self.e)
            self.means, R, _ = k_means.fit(x)
            r = np.zeros((R.size, self.n_cluster))
            r[np.arange(R.size),R] = 1
            self.pi_k = np.zeros(self.n_cluster)
            self.variances = np.zeros((self.n_cluster,D,D))
            for k in range(self.n_cluster):
                Nk = (R == k).sum()
                self.pi_k[k] = Nk/N
                m = r[:,k].reshape(-1,1)*(x-self.means[k])
                self.variances[k] = m.T.dot(x-self.means[k]) / Nk

                        
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.pi_k = np.array([1/self.n_cluster for _ in range(self.n_cluster)])
            self.means = x[np.random.randint(0,N,self.n_cluster)]
            self.variances = np.zeros((self.n_cluster,D,D))
            for k in range(self.n_cluster):
                self.variances[k] = np.identity(D)
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        l = self.compute_log_likelihood(x)

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        for it in range(self.max_iter):
            # E step, get responsibilities
            r = np.zeros((N,self.n_cluster))
            for k in range(self.n_cluster):
                r[:,k] = self.pi_k[k]*pdf_multivariate(x,self.means[k],self.variances[k])
                
            r = (r.T / np.sum(r, axis=1)).T
            
            # M - step
            for k in range(self.n_cluster):
                Nk = r[:,k].sum()
                self.means[k] = (r[:,k].reshape(-1,1) * x).sum(0) / Nk
                m = r[:,k].reshape(-1,1)*(x-self.means[k])
                self.variances[k] = m.T.dot(x-self.means[k]) / Nk
                self.pi_k[k] = Nk/N

            lnew = self.compute_log_likelihood(x)
            if np.abs(l-lnew) <= self.e:
                return it
            l = lnew        
        return self.max_iter
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        ks = np.random.choice(self.n_cluster, size=N, p=self.pi_k)
        data = []
        for k in ks:
            data.append(np.random.multivariate_normal(self.means[k],self.variances[k]))
        return np.array(data)
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        N = x.shape[0]
        r =  np.zeros((N,self.n_cluster))
        for k in range(self.n_cluster):
            r[:,k] = self.pi_k[k] * pdf_multivariate(x,self.means[k],self.variances[k]) 
        ll = np.sum(np.log(np.sum(r,axis=1)))
        return float(ll)
        # DONOT MODIFY CODE BELOW THIS LINE
