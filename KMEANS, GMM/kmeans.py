import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE

        uk = np.zeros((self.n_cluster,D))

       # for i in range(0,self.n_cluster):
        uk = x[np.random.randint(0,N,self.n_cluster),:]

        #for all points calculate the euclidean distance from each uk. For each point find the minimum and assign that cluster. Also, J is sum of them all.

        J = 0
        Jnew = 0
        R = np.zeros((N,self.n_cluster))
        Membership = np.zeros(N)
        d_old = np.zeros(self.n_cluster)

        for i in range(0,self.max_iter):
            Jnew = 0
            for j in range(0,N):
                dist = 0
                for k in range(0,self.n_cluster):
                    dist = np.linalg.norm(x[j]-uk[k])
                    #Jnew +=dist
                    d_old[k]=dist
                min_index = np.argmin(d_old)
                Jnew += d_old[min_index]
                R[j][min_index] = 1
                Membership[j] = min_index

            Jnew = Jnew/N

            if np.abs(J-Jnew)<=self.e:
                return tup
            else:
                J = Jnew
                #calculate mean uk
                for l in range(0,self.n_cluster):
                    s = 0
                    count = 0
                    for j in range(0,N):
                        if R[j][l] == 1:
                            s += x[j]
                            count += 1
                    if count==0:
                        uk[l]=0
                    else:
                        uk[l] = s/count
                tup = (uk,Membership,i)


        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        clusters = self.n_cluster
        iter = self.max_iter
        k_means = KMeans(clusters,iter, self.e)

        tup = k_means.fit(x)

        centroids = tup[0]
        centroids = np.asarray(centroids)
        membership = tup[1]
        membership = np.asarray(membership)
        max_c = 0
        centroid_labels = []

        for i in range(0,clusters):
            for j in y:
                max_now=0
                check = np.where(membership==centroids[i])
                for k in check:
                    if j == membership[k]:
                        max_now+=1
                if(max_now>max_c):
                    centroid_labels[i]=j
                    max_c=max_now







        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = np.asarray(centroid_labels)
        self.centroids = np.asarray(centroids)



        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
           self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        dist = np.zeros(len(self.centroids))
        for i in range(0,len(self.centroids)):
            for j in range(0,N):
                dist[j] = np.linalg.norm(x[j] - self.centroids[i])

            k = np.argmin(dist)

            return self.centroid_labels[k]



        # DONOT CHANGE CODE BELOW THIS LINE
