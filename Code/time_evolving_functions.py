import networkx as nx
import scipy as _sp
import numpy as np
import matplotlib.pyplot as plt
from d3s.algorithms import dinv, sortEig


def evolving_eigs(As, evs=5, plot=True, which="LM", fb=True):
    if type(As)==np.ndarray:
        evs_array = np.zeros((evs, As.shape[-1]))
        for i in range(As.shape[-1]):
            A = As[:, :, i]
            if fb:
                D = np.diag(np.sum(A, 1))
                P = dinv(D) @ A
                D_nu = np.diag(np.sum(P, 0))  # uniform density mapped forward
                A = P @ dinv(D_nu) @ P.T

            d, v = sortEig(A, evs=evs, which=which)
            evs_array[:, i] = d

    elif type(As)==dict:
        evs_array = np.zeros((evs, len(As.keys())))
        i = 0
        for key in As.keys():
            d, v = sortEig(As[key], evs=evs, which=which)
            evs_array[:, i] = d
            i += 1

    if plot:
        plt.plot(evs_array.T)
        plt.xlabel("time")
        plt.ylabel("eigenvalue")
        plt.show()

    return evs_array


def simple_cluster(n_vertices):
    A = np.zeros((n_vertices, n_vertices))
    for v in range(n_vertices):
        A[v, np.mod(v-1,n_vertices)] = 1
        A[v, np.mod(v+1, n_vertices)] = 1

    return A


def ladder(n_clusters, cluster_size, t, T, eps_edges):
    """

    :param n_clusters:
    :param cluster_size:
    :param t:
    :param T:
    :param eps_edges: list of tuples for edges with time dependent weight
    :return:
    """
    A = np.zeros((cluster_size*n_clusters, cluster_size*n_clusters)).astype(float)

    for c in range(n_clusters):
        A[c * cluster_size:(c + 1) * cluster_size, c * cluster_size:(c + 1) * cluster_size] = simple_cluster(cluster_size)

        # add connection to other clusters
        # function of t
        eps = t * (-1) / T + 1
        for i in range(len(eps_edges)):
            k = eps_edges[i][0]
            j = eps_edges[i][1]

            A[k, j] = eps

    return A, eps


def evolving_ladder(n_clusters, cluster_size, times, eps_edges, diag="on"):
    # makes ladder of two clusters with reducing edge weights over time
    As = np.zeros((cluster_size*n_clusters, cluster_size*n_clusters, len(times))).astype(float)
    T = times[-1]
    eps = np.zeros(len(times))
    for i in range(len(times)):
        A, e = ladder(n_clusters, cluster_size, t=times[i], T=T, eps_edges=eps_edges)
        As[:,:,i] = A

    if diag=="off" or diag=="mixed":
        A_copy = np.zeros_like(As)
        for i in range(n_clusters):
            if diag=="off":
                s = np.random.randint(1,n_clusters)
            elif diag=="mixed":
                s = np.random.randint(0, n_clusters)
            for j in range(As.shape[-1]):
                temp = np.roll(As[i*cluster_size:(i+1)*cluster_size,:,j], shift=s*cluster_size, axis=1)
                A_copy[i*cluster_size:(i+1)*cluster_size,:,j] = temp
        As = A_copy
    return As, eps