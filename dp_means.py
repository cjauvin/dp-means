# Christian Jauvin
# http://christianjauv.in
# 2012-07-13
# Python translation of this R code: https://github.com/johnmyleswhite/bayesian_nonparametrics/tree/master/code/dp-means
# Found on: http://www.johnmyleswhite.com/notebook/2012/06/26/bayesian-nonparametrics-in-r/
# Based on: http://arxiv.org/pdf/1111.0352.pdf

from numpy import *

def dp_means(data, Lambda, max_iters=100, tolerance=10e-3):

    n = len(data)
    k = 1
    assignments = ones(n)
    mu_x = array((mean(data[:,0]),)) 
    mu_y = array((mean(data[:,1]),))
    is_converged = False
    n_iters = 0
    ss_old = float('inf')
    ss_new = float('inf')
    while not is_converged and n_iters < max_iters:
        n_iters += 1
        for i in range(n):
            distances = repeat(None, k)
            for j in range(k):
                distances[j] = (data[i, 0] - mu_x[j]) ** 2 + (data[i, 1] - mu_y[j]) ** 2
            if min(distances) > Lambda:
                k += 1
                assignments[i] = k-1
                mu_x = append(mu_x, data[i, 0])
                mu_y = append(mu_y, data[i, 1])
            else:
                assignments[i] = argmin(distances)
        for j in range(k):
            if len(where(assignments == j)) > 0:
                mu_x[j] = mean(data[assignments == j, 0])
                mu_y[j] = mean(data[assignments == j, 1])
        ss_new = 0      
        for i in range(n):
            ss_new <- ss_new + (data[i, 0] - mu_x[assignments[i]]) ** 2 + (data[i, 1] - mu_y[assignments[i]]) ** 2    
        ss_change = ss_old - ss_new
        is_converged = ss_change < tolerance  
    return {'centers': array((mu_x, mu_y)).transpose(),
            'assignments': assignments,
            'k': k, 'n_iters': n_iters}
                        
if __name__ == '__main__':

    import matplotlib.pyplot as plt, random as rnd

    def generate_data(n=100):
        mu_x = array((0, 5, 10, 15))
        mu_y = array((0, 5, 0, -5))
        data = empty(shape=(n, 3))
        for i in range(n):
            c = rnd.choice(range(4))
            data[i] = (random.normal(mu_x[c], 1), random.normal(mu_y[c], 1), c)
        return data

    data = generate_data(100)    
    results = dp_means(data, 50)
    print 'found %d clusters' % len(set(results['assignments']))
    plt.scatter(data[:,0], data[:,1], c=results['assignments'])
    plt.show()
