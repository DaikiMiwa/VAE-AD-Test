import numpy as np
from non_gaussian_noise import standardized_rv_at_distance

def naive_ar1(w,h,rho):
    adj_indexies_d_1 = [[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]
    covariance_matrix = np.identity(w*h)

    for i in range(w):
        for j in range(h):
            print("at",i,j)
            for adj_idx in adj_indexies:
                if i+adj_idx[0] >= 0 and i+adj_idx[0] < w and j+adj_idx[1] >= 0 and j+adj_idx[1] < h:
                    print("adj",i+adj_idx[0],j+adj_idx[1])
                    covariance_matrix[i*h+j,(i+adj_idx[0])*h+j+adj_idx[1]] = rho
                    covariance_matrix[(i+adj_idx[0])*h+j+adj_idx[1],i*h+j] = rho

    return covariance_matrix

def ar1_image_covariance(w,h,rho):
    col_ar1 = np.zeros([w,w])
    for i in range(w):
        for j in range(w):
            col_ar1[i,j] = rho**abs(i-j)

    row_ar1 = np.zeros([w,w])
    for i in range(w):
        for j in range(w):
            row_ar1[i,j] = rho**abs(i-j)

    return np.kron(col_ar1,row_ar1)

def multivariate_normal_sample(covariance,n_samples,shape):
    L = np.linalg.cholesky(covariance)
    X_test = []
    n = covariance.shape[0]

    for i in range(n_samples):
        X_test.append(np.dot(L,np.random.normal(size=n)).reshape(shape))

    return np.array(X_test)

def nongaussian_sample(distribution,ws_distance,n_samples,shape,seed=1234):
    X_test = []
    rvs = standardized_rv_at_distance(distribution, distance=ws_distance)

    for i in range(n_samples):
        X_test.append(rvs.rvs(size=shape))

        return np.array(X_test)
