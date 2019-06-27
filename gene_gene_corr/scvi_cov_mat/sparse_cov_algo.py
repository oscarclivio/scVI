import numpy as np

from sklearn.covariance import graphical_lasso

def build_emp_cov_matrix(X):

    n = X.shape[0]

    return X.T.dot(X) / n


def objective(sigma, emp_cov, weights_mat, lmb_weights):

    _, log_det_term = np.linalg.slogdet(sigma)
    trace_term = np.trace(np.linalg.solve(sigma,emp_cov))
    l1_term = lmb_weights * np.sum(np.abs(weights_mat * sigma))
    return log_det_term + trace_term + l1_term


def find_lower_bound_eigenvalues(emp_cov, weights_mat, lmb_weights, dichotomy=True, n_iters_max = 1000, eps=1e-5):

    p = emp_cov.shape[0]

    # Find lowest eigenvalue of the empirical cov matrix
    lmb_min = np.min( np.linalg.eig(emp_cov)[0] )

    # Define sigma hat and compute evaluation of objective on it
    sigma_hat = emp_cov # Possible change here
    obj_sigma_hat = objective(sigma_hat, emp_cov, weights_mat, lmb_weights)

    # Threshold for finding delta
    threshold = obj_sigma_hat - (p-1)*(np.log(lmb_min) + 1)

    # Function h
    def h(lmb):
        return np.log(lmb) + lmb_min / lmb

    # Dichotomy
    delta_min = lmb_min * 1e-4
    while h(delta_min) <= threshold:
        delta_min = (delta_min ** 2) / lmb_min

    if not(dichotomy):
        return delta_min

    delta_max = lmb_min
    n_iters = 0
    while delta_max - delta_min > eps and n_iters < n_iters_max:
        delta_new = (delta_min + delta_max) / 2
        diff = h(delta_new) - threshold
        if diff < 0:
            delta_max = delta_new
        elif diff > 0:
            delta_min = delta_new
        else:
            return delta_new
        n_iters += 1
        if n_iters == n_iters_max:
            print("Dichotomy : max iters reached")

    return delta_min

def soft_thresholding_operator(A, B):

    return np.sign(A) * np.maximum(0., A - B)

def update_sigma(sigma_old, sigma_zero_new, emp_cov, weights_mat, lmb_weights, delta, lr=1e-3,
                 lr_alt_dir=1e-1, eps=1e-4, n_iters_max = 1000, debug_force = False):

    sigma_old_inv = np.linalg.inv(sigma_old)
    sigma_zero_new_inv = np.linalg.inv(sigma_zero_new)

    if lr_alt_dir is None:
        lr_alt_dir = lr

    # First, update sigma using the simple soft-thresholding function and check if the minimal eigenvalue is
    # greater or equal than delta

    left_term_soft = sigma_old - lr*(sigma_zero_new_inv - sigma_old_inv.dot(emp_cov).dot(sigma_old_inv))
    right_term_soft = lmb_weights * lr * weights_mat

    #print("Old")
    #print(sigma_old)
    #print("Left")
    #print(left_term_soft)
    #print("Right")
    #print(right_term_soft)

    sigma_new = soft_thresholding_operator(left_term_soft, right_term_soft)

    #print("New")
    #print(sigma_new)

    lmb_min = np.min( np.linalg.eig(sigma_new)[0] )

    print(lmb_min, delta)

    # If the above condition is verified, return the new sigma as is, else perform
    # alternating direction method of multipliers

    if debug_force:
        sigma_new = (delta / (2*lmb_min)) * sigma_new
        lmb_min = np.min( np.linalg.eig(sigma_new)[0] )

    if lmb_min > delta:
        return sigma_new

    print("Alternating direction method of multipliers has to be performed")

    Id = np.identity(sigma_old.shape[0])
    sigma_new_old = sigma_old + Id / eps
    sigma_new_new = sigma_old

    theta_old = 2*Id
    theta_new = Id

    Y_old = 2*Id
    Y_new = Id

    n_iters = 0

    while np.max(np.abs(sigma_new_new - sigma_new_old)) > eps and n_iters < n_iters_max:

        sigma_new_old = sigma_new_new
        Y_old = Y_new
        theta_old = theta_new

        sigma_new_old_inv = np.linalg.inv(sigma_new_old)
        to_diagonalize = (sigma_new_old - \
                          lr*(sigma_zero_new_inv - sigma_new_old_inv.dot(emp_cov).dot(sigma_new_old_inv)) \
                          + lr_alt_dir * theta_old - Y_old ) / (1 + lr_alt_dir)
        D, U = np.linalg.eig(to_diagonalize)

        sigma_new_new = U.dot(np.diag(np.maximum(D, delta))).dot(U.T)
        if debug_force:
            print("Test : ", np.min(np.linalg.eig(sigma_new_new)[0]), delta)
        theta_new = soft_thresholding_operator(sigma_new_new + Y_old / lr_alt_dir,
                                               lmb_weights * weights_mat / lr_alt_dir)
        Y_new = Y_old + lr_alt_dir*(sigma_new_new - theta_new)

        n_iters += 1
        if n_iters >= n_iters_max:
            print("Alt dir method : max iters reached")

        if debug_force:
            print("Mean diff :", np.mean(np.abs(sigma_new_new - sigma_new_old)))


    sigma_new = sigma_new_new
    lmb_min = np.min(np.linalg.eig(sigma_new)[0])
    print(lmb_min, delta)

    return sigma_new


def estimate_cov_matrix(X, weights_mat, lmb_weights, lr=1e-3,
                        lr_alt_dir=1e-1, eps=1e-4, n_iters_max = 1000, dichotomy=True):

    emp_cov = build_emp_cov_matrix(X)

    delta = find_lower_bound_eigenvalues(emp_cov, weights_mat, lmb_weights, dichotomy=dichotomy)

    Id = np.identity(emp_cov.shape[0])
    sigma_zero_old = emp_cov + Id / eps
    sigma_zero_new = emp_cov

    n_iters_zero = 0

    while np.max(np.abs(sigma_zero_new - sigma_zero_old)) > eps and n_iters_zero < n_iters_max:

        sigma_zero_old = sigma_zero_new

        sigma_old = sigma_zero_old + Id / eps
        sigma_new = sigma_zero_old

        n_iters_sigma = 0
        while np.max(np.abs(sigma_new - sigma_old)) > eps and n_iters_sigma < n_iters_max:
            sigma_old = sigma_new
            sigma_new = update_sigma(sigma_old, sigma_zero_old, emp_cov, weights_mat, lmb_weights, delta, lr=lr,
                                     lr_alt_dir=lr_alt_dir, eps=eps, n_iters_max = n_iters_max)
            n_iters_sigma += 1
            if n_iters_sigma >= n_iters_max:
                print("Sigma : max iters reached")
            else:
                print("Iters sigma :", n_iters_sigma)
                print("Diff sigma : ", np.max(np.abs(sigma_new - sigma_old)))


        sigma_zero_new = sigma_new

        n_iters_zero += 1
        if n_iters_zero >= n_iters_max:
            print("Sigma zero : max iters reached")
        else:
            print("Current iter zero:", n_iters_zero)
            print("Diff zero:", np.max(np.abs(sigma_zero_new - sigma_zero_old)))
            print(sigma_zero_new)

    print(n_iters_sigma, n_iters_zero)

    return sigma_zero_new


# TODO : test function !
if __name__ == '__main__':

    np.random.seed(1)

    n = 200
    p = 100

    sigma_gt = np.identity(p) + 0.1 * np.random.binomial(1, 0.1, (p,p))
    sigma_gt[sigma_gt > 1.] = 1.
    assert(np.min(np.linalg.eig(sigma_gt)[0]) > 1e-4)
    means_gt = np.zeros((p,))

    X = np.random.multivariate_normal(means_gt, sigma_gt, size=(n,))

    emp_cov = build_emp_cov_matrix(X)

    weights_mat = np.ones((p, p)) - np.identity(p)
    lmb_weights = 0.1

    TO_TEST = [0, 2]

    if 0 in TO_TEST:


        delta_min_1 = find_lower_bound_eigenvalues(emp_cov,
                                                   weights_mat,
                                                   lmb_weights,
                                                   dichotomy=False)

        print("\nDELTA TEST")

        print(delta_min_1)

        delta_min_2 = find_lower_bound_eigenvalues(emp_cov,
                                                   weights_mat,
                                                   lmb_weights,
                                                   dichotomy=True)

        print(delta_min_2)

    if 1 in TO_TEST:

        print("\nSimple sigma update test - S only")

        delta_min = find_lower_bound_eigenvalues(emp_cov,
                                                 weights_mat,
                                                 lmb_weights,
                                                 dichotomy=True)

        sigma_new_1 = update_sigma(emp_cov, emp_cov, emp_cov, weights_mat,
                                   lmb_weights, delta_min, lr=1e-3)

        print(sigma_new_1)

        sigma_new_2 = update_sigma(emp_cov, emp_cov, emp_cov, weights_mat,
                                   lmb_weights, delta_min, lr=1e-3, debug_force=True)

        print(sigma_new_2)

    if 2 in TO_TEST:

        print("\nComplete algo : ")

        sigma = estimate_cov_matrix(X, weights_mat, lmb_weights, lr=1e-2,
                                    lr_alt_dir=1e-2, eps=6e-6, n_iters_max = 2000, dichotomy=True)

        print("GROUND TRUTH")
        print(sigma_gt)

        print("\n\nRESULT")
        print(sigma)


        diff = np.abs(sigma_gt - sigma)
        print("Max abs diff :")
        print(np.max(diff))
        print("Mean abs diff :")
        print(np.mean(diff[sigma_gt > 0.]))

        print("Ratio of non-zeros (gt vs predicted) :", (sigma_gt > 0.).sum() / (p*p), (sigma > 0.).sum() / (p*p))

        print("Average value of diagonal predicted non-zeros :", np.trace(sigma) / p)
        sigma_bis = sigma
        for ind in range(p):
            sigma_bis[ind,ind] = 0.
        print("Average value of non-diagonal predicted non-zeros :", np.mean(sigma_bis[sigma_bis > 0.]))


        print("GRAPHICAL LASSO COVARIANCE MATRIX :")

        sigma_gl = graphical_lasso(emp_cov=emp_cov, alpha=0.2)[0]
        print(sigma_gl)
        print(sigma_gl.shape)

        diff = np.abs(sigma_gt - sigma_gl)
        print("Max abs diff :")
        print(np.max(diff))
        print("Mean abs diff :")
        print(np.mean(diff[sigma_gt > 0.]))

        print("Ratio of non-zeros (gt vs predicted) :", (sigma_gt > 0.).sum() / (p*p), (sigma_gl > 0.).sum() / (p*p))

        print("Average value of diagonal predicted non-zeros :", np.trace(sigma_gl) / p)
        sigma_bis = sigma_gl
        for ind in range(p):
            sigma_bis[ind,ind] = 0.
        print("Average value of non-diagonal predicted non-zeros :", np.mean(sigma_bis[sigma_bis > 0.]))






        print("EMPIRICAL COVARIANCE MATRIX :")
        sigma_emp = build_emp_cov_matrix(X)
        print(sigma_emp)

        diff = np.abs(sigma_gt - sigma_emp)
        print("Max abs diff :")
        print(np.max(diff))
        print("Mean abs diff :")
        print(np.mean(diff[sigma_gt > 0.]))

        print("Ratio of non-zeros (gt vs predicted) :", (sigma_gt > 0.).sum() / (p*p), (sigma_emp > 0.).sum() / (p*p))

        print("Average value of diagonal predicted non-zeros :", np.trace(sigma_emp) / p)
        sigma_bis = sigma_emp
        for ind in range(p):
            sigma_bis[ind,ind] = 0.
        print("Average value of non-diagonal predicted non-zeros :", np.mean(sigma_bis[sigma_bis > 0.]))



