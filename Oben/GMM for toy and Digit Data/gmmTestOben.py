import numpy as np
from data_loader import toy_dataset, load_digits, load_gmmdata
from gmm import GMM
from utils import Figure
from matplotlib.patches import Ellipse


def compute_elipse_params(variance):
    '''
        Compute elipse params for plotting from variance
    '''

    # http://www.cs.cornell.edu/cv/OtherPdf/Ellipse.pdf Slide 17
    # https://stackoverflow.com/a/41821484

    variance_inv = np.linalg.inv(variance)
    a = variance_inv[0, 0]
    c = variance_inv[1, 1]
    b = variance_inv[0, 1] + variance_inv[1, 0]

    M = (variance_inv + variance_inv.T) / 2
    eig, _ = np.linalg.eig(M)
    if (np.abs(eig[0] - a) < np.abs(eig[0] - c)):
        lambda1, lambda2 = eig
    else:
        lambda2, lambda1 = eig

    angle = np.arctan(b / (a - c)) / 2
    return np.sqrt(1 / lambda1), np.sqrt(1 / lambda2), angle



################################################################################
# GMM on oben's dataset
# We fit a gaussian distribution on digits dataset and show generate samples from the distribution
# Complete implementation of sample function for GMM class in gmm.py
################################################################################
init = ['k_means', 'random']

x_train = load_gmmdata()
k_val = np.zeros(10)

#x can only be 2 dimensional so i run PCA over z
for i in init:
    for j in range(1,11):
        n_cluster = j
        gmm = GMM(n_cluster=n_cluster, max_iter=1000, init=i, e=1e-10)
        iterations = gmm.fit(x_train)
        ll = gmm.compute_log_likelihood(x_train)
        print('GMM for oben dataset with {} initialisation converged in {} iterations for k = {}. Final log-likelihood of data: {}'.format(i, iterations, n_cluster,ll))

        # plot cluster means
        means = gmm.means
    print ("\n")
    '''
    from matplotlib import pyplot as plt
    l = int(np.ceil(np.sqrt(n_cluster)))

    im = np.zeros((10 * l, 10 * l))
    for m in range(l):
        for n in range(l):
            if (m * l + n < n_cluster):
                im[10 * m:10 * m + 8, 10 * n:10 * n +
                    8] = means[m * l + n].reshape([8, 8])
    im = (im > 0) * im
    plt.imsave('plots/means_{}.png'.format(i), im, cmap='Greys')

    # plot samples
    N = 100
    l = int(np.ceil(np.sqrt(N)))
    samples = gmm.sample(N)

    assert samples.shape == (
        N, x_train.shape[1]), 'Samples should be numpy array with dimensions {}X{}'.format(N, x_train.shape[1])

    im = np.zeros((10 * l, 10 * l))
    for m in range(l):
        for n in range(l):
            if (m * l + n < N):
                im[10 * m: 10 * m + 8, 10 * n: 10 * n +
                    8] = samples[m * l + n].reshape([8, 8])
    im = (im > 0) * im
    plt.imsave('plots/samples_{}.png'.format(i), im, cmap='Greys')

    np.savez('results/gmm_obne_{}.npz'.format(i), iterations=np.array(
        [iterations]), variances=gmm.variances, pi_k=gmm.pi_k, means=gmm.means, samples=samples, log_likelihood=ll, x=x_test, y=y_test)
'''