import netcomp as nc
import networkx as nx
import numpy as np
from numpy.ma.core import append
import itertools
import random
from scipy.stats import norm, beta, uniform, bernoulli
from scipy.special import comb


def P_List(a, b, n):
    """Returns a Numpy arrary of length n of ranbom probabilities generated from a beta distribution.

    Parameters
    ----------
    a : integer
        Beta parameter.

    b : integer
        Beta parameter

    n : integer
        desired length of list

    Returns
    -------
    p_list : NumPy array
        A list of random probabilities generated from a beta distribution

    Notes
    -----

    References
    ------
    """
    p_list = np.random.beta(a, b, size=n)
    return p_list


def bern(nodes, p_list):
    """Returns a bernoulli random graph with edge probabilities p_list.

    Parameters
    ----------
    nodes : integer
        The number of nodes for the graph.

    p_list : list
        The the respective edge probabilities for each node

    Returns
    -------
    A : NumPy array
        A Bernoulli random graph adjency matrix

    Notes
    -----

    References
    ----------
    """
    n = nodes
    items = n * n
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = itertools.combinations(range(n), 2)
    p_index = 0
    for e in edges:
        x = bernoulli.rvs(p_list[p_index], size=1)
        if x == 1:
            G.add_edge(*e)
        p_index = p_index + 1

    A = nx.to_numpy_array(G)
    return A


def known_mean(sample=None, dist_function=None, k=None, nodes=None, p_list=None):
    """Returns the 'known' emperical sample Frechiet mean for a given sample.

    Parameters
    ----------
    sample : list of Numpy Arrays
        A sample of random graphs generated from a specific random model.

    dist_function : String , in {'lambda_laplacian','lambda_laplacian_norm','lambda_adjacency', 'deltacon', 'bern'}
        The matrix for which eigenvalues will be calculated.

    k : integer , optional (default=None)
        An optional parameter for the lambda distance specifying the number of the eigenvalues

    nodes : integer , optional (default=None)
        An optional parameter for the Bernoulli distance specifying the number of nodes in the graph

    p_list : Numpy Array , optional (default=None)
        An optional parameter for the Bernoulli distance specifying edge probiblities used to generate the sample graphs


    Returns
    -------
    mean : NumPy array
        Theoretical known Frechiet mean of the sample

    Notes
    -----

    References
    ----------
    """
    dist = []
    N = len(sample)

    if dist_function == 'lambda_adjacency':
        for i in range(len(sample)):
            dist2 = []
            for j in range(len(sample)):
                dist2.append(nc.lambda_dist(sample[j], sample[i], k=k, kind='djacency') ** 2)
            dist.append(sum(dist2)/N)

    elif dist_function == 'lambda_laplacian_norm':
        for i in range(len(sample)):
            dist2 = []
            for j in range(len(sample)):
                dist2.append(nc.lambda_dist(sample[j], sample[i], k=k, kind='laplacian_norm') ** 2)
            dist.append(sum(dist2)/N)

    elif dist_function == 'lambda_laplacian':
        for i in range(len(sample)):
            dist2 = []
            for j in range(len(sample)):
                dist2.append(nc.lambda_dist(sample[j], sample[i], k=k, kind='laplacian') ** 2)
            dist.append(sum(dist2)/N)

    elif dist_function == 'deltacon':
        for i in range(len(sample)):
            dist2 = []
            for j in range(len(sample)):
                dist2.append(nc.deltacon0(sample[j], sample[i]) ** 2)
            dist.append(sum(dist2)/N)

    elif dist_function == 'bern':
        n = nodes
        G = nx.Graph()
        G.add_nodes_from(range(n))
        edges = itertools.combinations(range(n), 2)
        p_index = 0
        for e in edges:
            if p_list[p_index] > (1 / 2):
                G.add_edge(*e)
            p_index = p_index + 1

        mean = nx.to_numpy_array(G)
        return (mean)

    else:
        for i in range(len(sample)):
            dist2 = []
            for j in range(len(sample)):
                dist2.append(nc.resistance_distance(sample[j], sample[i]) ** 2)
            dist.append(sum(dist2)/N)

    mean = sample[np.argmin(dist)]

    return (mean)


def Random_Sample_Graphs(generator, nodes, N, ps=None, sb_sizes=None, sb_p=None, pref_m=None):
    """Returns a list of sample graphs with n nodes, generated from a specific random model generator

    Parameters
    ----------
    generator : string
        the random genrator used to genertae the random sample:
        'bern': Bernoulli random graph
        'sb': Strochastic Block Random Graph
        'pref': Prefrential Attachment random graph

    nodes : integer
        desired number of nodes for each graph

    N: integer
        Number of graphs for each sample (sample length)

    Hyperparatmeters:
      ps: list (default None)
          parameters for the 'bern' model genrator. See 'bern' for details

      sb_sizes: list of list (default None)
          parameter of the size of each community in a stochastic block model. Networkx for details

      sb_p: list of float (default None)
          parameter for the connection proabilities for a stochastic block model. See Networkx for details

      pref_m: int (default None)
        Number of edges to attach from a new node to existing nodes for the preffrential attachment model





    Returns
    -------
    sample_graphs : list of NumPy array
        list of graph adjencey matrcies

    Notes
    -----

    References
    ------
    """

    if generator == 'bern':
        temp_list = []
        num_edges = comb(nodes, 2).astype(int)  # nodes choose 2
        for i in range(N):
            temp_list.append(bern(nodes, ps))

        sample_graphs = temp_list

    elif generator == 'sb':
        temp_list = []
        while len(temp_list) < N:
            G = nx.stochastic_block_model(sb_sizes, sb_p)
            if nx.is_connected(G):
                A = nx.to_numpy_array(G)
                temp_list.append(A)

        sample_graphs = temp_list

    else:
        temp_list = []
        sizes = sb_sizes
        for i in range(N):
            p = sb_p
            temp_list.append(nx.to_numpy_array(nx.generators.random_graphs.barabasi_albert_graph(nodes, pref_m)))

        sample_graphs = temp_list

    return (sample_graphs)


def Data_Generator(sample_x_list, sample_y_list, sample_size, generator, nodes, N, a=2.3, b=3, sb_sizes=None, sb_p=None,
                   m=5):
    """Returns a list of sample graphs with n nodes, generated from a specific random model generator

    Parameters
    ----------
    sample_x_list : list
        a list to store the summed sample graphs. Can be empty of filled

    sample_y_list : list
        a list to store the known mean graphs for each summed sample. Can be empty of filled

    sample_size : Int
        number of data points to generate

    generator : string in {'bern', 'sb', 'pref'}
        the random genrator used to genertae the random sample:
        'bern': Bernoulli random graph
        'sb': Strochastic Block Random Graph
        'pref': Prefrential Attachment random graph

    nodes : integer
        desired number of nodes for each graph

    N: integer
        Number of graphs for each sample (sample length)

    Hyperparatmeters:
      a, b: float (default a=2.3, b=3)
          parameters for the 'bern' model genrator. See 'bern' for details

      sb_sizes: list of list (default None)
          parameter of the size of each community in a stochastic block model. Networkx for details

      sb_p: list of float (default None)
          parameter for the connection proabilities for a stochastic block model. See Networkx for details

      m: int (default 5)
        parameter of the pref attachment model





    Returns
    -------
    sample_graph : NumPy array
        an element wise sum of the random sample

    Notes
    -----

    References
    ------
    """

    if generator == 'bern':
        for i in range(sample_size):
            num_edges = comb(nodes, 2).astype(int)
            ps = P_List(a, b, num_edges)
            x_sample = Random_Sample_Graphs(generator='bern', nodes=nodes, N=N, ps=ps)
            x = sum(x_sample)
            x = x / (N / 255)  # constant to make the maximum 255 = #of samples/255
            x = x.round()  # rounding to whole number
            sample_x_list.append(x)

            sample_y_list.append(known_mean(sample=None, dist_function='bern', nodes=nodes, p_list=ps))

    elif generator == 'sb':
        for i in range(sample_size):
            x_sample = Random_Sample_Graphs(generator='sb', nodes=nodes, N=N, sb_sizes=sb_sizes, sb_p=sb_p)
            x = sum(x_sample)
            x = x / (N / 255)  # constant to make the maximum 255 = #of samples/255
            x = x.round()  # rounding to whole number
            sample_x_list.append(x)

            sample_y_list.append(known_mean(sample=x_sample, dist_function='lambda_adjency'))

    else:
        for i in range(sample_size):
            x_sample = Random_Sample_Graphs(generator='pref', nodes=nodes, N=N, pref_m=m)
            x = sum(x_sample)
            x = x / (N / 255)  # constant to make the maximum 255 = #of samples/255
            x = x.round()  # rounding to whole number
            sample_x_list.append(x)

            sample_y_list.append(known_mean(sample=x_sample, dist_function='lambda_laplacian', nodes=nodes))

    return (sample_x_list, sample_y_list)