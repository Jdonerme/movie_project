import prob2utils as HW5utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import util as ut
import numpy as np

# Make Y: mxn matrix of ratings by user and movie id
def create_u_v_matrices():
    Y, _, num_ratings, _ = ut.get_rating_data([])
    Y = np.array(Y)
    M = int(max([x[0] for x in Y]))
    N = int(max(num_ratings))

    K = 20
    eta = 0.001
    reg = 0.01
    eps = 0.01
    max_epochs = 100

    result = HW5utils.train_model(M, N, K, eta, reg, Y, eps, max_epochs)
    U, V, error = result

    np.savetxt('U_matrix.txt', U, delimiter=',')
    np.savetxt('V_matrix.txt', V, delimiter=',')

def apply_svd():
    #create_u_v_matrices()
    U = np.loadtxt('U_matrix.txt', delimiter=',')
    V = np.loadtxt('V_matrix.txt', delimiter=',')

    A,s,B = np.linalg.svd(V, full_matrices=True)
    V_tilde = np.dot(A[:, [0, 1]].transpose(), V)
    U_tilde = np.dot(A[:, [0, 1]].transpose(), V)
    return V_tilde, U_tilde

def part_a_visual():
    V_tilde, U_tilde = apply_svd()

    get_top_ten_movies = function()
    # Rescale dimensions


    # Plot projected  V
    print(len(V_tilde))
    print(len(V_tilde[0]))
    print(V_tilde)

    x = []
    y = []

    for movie in top_ten_movies:
        new_x = V_tilde[0]:
        new_y = V_tile[1]

        x.append(new_x)
        y.append(new_y)


    #plt.scatter()
    #plt.show()




part_a_visual()
