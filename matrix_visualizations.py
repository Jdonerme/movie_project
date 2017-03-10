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

    U, V, error = HW5utils.train_model(M, N, K, eta, reg, Y, eps, max_epochs)

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

def make_visual(movies, title='Matrix Visualization'):
    V_tilde, U_tilde = apply_svd()

    # Plot projected  V
    print(len(V_tilde))
    print(len(V_tilde[0]))
    print(V_tilde)

    x = []
    y = []
    print(len(movies))
    for movie in movies:
        print(movie)

        new_x = V_tilde[0][movie]
        new_y = V_tilde[1][movie]

        x.append(new_x)
        y.append(new_y)

    plt.title(title)
    plt.xlabel('X') # Should probably be a better name
    plt.ylabel('Y') # Same
    plt.scatter(x, y)
    plt.show()


def main():
    movies, ids, titles = ut.get_category_ids(genres_desired=['Animation', 'Comedy', 'Musical'])
    top, pop, g1, g2, g3 = ids

    # Part A
    make_visual(movies, title='Ten Random Movies Matrix Factorization Visualization')

    # Part B
    make_visual(pop, title='Ten Most Popular Movies Matrix Factorization Visualization')

    # Part C
    make_visual(top, title='Ten Best Movies Matrix Factorization Visualization')

    # Part D, genre 1
    make_visual(g1, title='Ten Animation Movies Matrix Factorization Visualization')

    # Part D, genre 2
    make_visual(g2, title='Ten Comedy Movies Matrix Factorization Visualization')

    # Part D, genre 3
    make_visual(g3, title='Ten Musical Movies Matrix Factorization Visualization')

main()
