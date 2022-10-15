import numpy as np
import networkx as nx
import collections


def spectrum(adjacency, reshape=True,  round=True):
    """A function to compute the eigenvalues for a set graph adjacency matrices.

            Parameters
            ----------
            adjacency : Numpy array
                    Contains the adjacency matrices in the form of numpy arrys

            reshape : Bool
                Argument specifying whether or not to reshape the adjacency matrices

            round : Bool
                Argument specifying whether or not to round the round the values in the adjacency matrices to
                either 0 or 1

            Returns
            -------
            eigenvals : list
                A list of numpy arrays which contain the eigenvalues for each of the provided adjacency matrices

            Notes
            -----

            References
            ------
            """

    a_new = []
    eigenvals = []
    eigenbasis = []

    if round & reshape:
        # Rounding the values to either 0 or 1 and changing the shape from (x,x,1) to (x,x)
        for i in adjacency:
            a_new.append(np.squeeze(i, axis=2).round(decimals=0, out=None))
        # Computing the eigenvalues and basis
        for j in range(len(adjacency)):
            a, b = np.linalg.eigh(a_new[j])
            eigenvals.append(a)
            eigenbasis.append(b)

    else:
        # Computing the eigenvalues and basis
        for i in range(len(adjacency)):
            a, b = np.linalg.eigh(adjacency[i])
            eigenvals.append(a)
            eigenbasis.append(b)

    return eigenvals


def baseline_model(sample):
    """A simple function to compute a "baseline" model for comparison. Which are just rounded to 0 or one. This acts
        like taking the simple "average" of each sample graph in the sense that we divide each entry by the matrix by
        the number of samples used to generate it. Note that because we normalized our entries to between 0 and 1, we
        only need to round to the nearest value.

             Parameters
             ----------
             sample : numpy array
                     Contains the normalized sample graphs adjacency matrices in the form of numpy arrays

             Returns
             -------
             x : list of numpy array
                 A list of numpy arrays which contain the adjacency matrices for the baseline model

             Notes
             -----

             References
             ------
             """

    x = []
    for a in sample:
        x.append(np.squeeze(a, axis=2).round(decimals=0, out=None))
    return x


def spectral_diff(predicted, target, percent = False):
    """A simple function to compute a the absolute spectral difference between the model predicted spectrum and the
        target spectrum as well as the difference between the baseline model predicted spectrum and the target spectrum.

            Parameters
            ----------
            predicted : list
                    Contains the spectrum from the predicted adjacency matrices in the form of numpy arrays

            target : list
                    Contains the spectrum from the target adjacency matrices in the form of numpy arrays

            percent: Bool
                    Default False: a boolean operator for percentage or not

            Returns
            -------
            diff_pred : list
                    A list of numpy arrays which contain spectral difference of between the predicted spectrum and the
                    target spectrum

            Notes
            -----

            References
            ------
            """
    diff_pred = []
    if percent:
        np.seterr(divide='raise')
        np.seterr(invalid='raise')
        for i in range(len(target)):
            # sorting the absolute values of the eigenvalues from largest to smallest
            a = np.sort(abs(predicted[i]))[::-1]
            c = np.sort(abs(target[i]))[::-1]
            try:
                diff = abs(a - c) / c
            except FloatingPointError:
                diff = 0

            diff_pred.append(diff)

    else:
        for i in range(len(target)):
            # sorting the absolute values of the eigenvalues from largest to smallest
            a = np.sort(abs(predicted[i]))[::-1]
            c = np.sort(abs(target[i]))[::-1]

            diff_pred.append(abs(a-c))
    return diff_pred


def average_spectral_diff(spec_diff):
    """A simple helper function to compute a the average spectral distance for each eigenvalue for a set of
        eigen differences .

        Parameters
        ----------
        spec_diff : list
                Contains the spectral distances in the form of numpy arrays for which to compute the average

        Returns
        -------
        avg_diff : numpy array
                Contains the average spectral distance for each eigenvalue

        Notes
        -----

        References
        ------
        """
    avg_diff = sum(spec_diff)/len(spec_diff)

    return avg_diff


def avg_degree(adjacency, reshape=True, round=True):
    """A function to compute the average degree distribution of each node in a graph.

                Parameters
                ----------
                adjacency : Numpy array
                        Contains the adjacency matrices in the form of numpy arrays

                reshape : Bool
                    Argument specifying whether or not to reshape the adjacency matrices

                round : Bool
                    Argument specifying whether or not to round the round the values in the adjacency matrices to
                    either 0 or 1

                Returns
                -------
                degree : list
                    A list of the degrees

                avg_cnt: list
                    A list of the average number of nodes that have the specific degree

                Notes
                -----

                References
                ------
                """
    a_new = []
    degree_list = []
    degree_seq = []
    if round & reshape:
        # Rounding the values to either 0 or 1 and changing the shape from (x,x,1) to (x,x)
        for i in adjacency:
            a_new.append(np.squeeze(i, axis=2).round(decimals=0, out=None))
        # Computing the degrees of each adjacency matrix and storing them all in a list
        for j in range(len(adjacency)):
            g = nx.from_numpy_array(a_new[j])
            s = sorted([d for n, d in g.degree()], reverse=True)
            degree_list.append(s)
        # Merging the list of lists from above into one list
        for deg in degree_list:
            degree_seq += deg
        # Calculating the counts for each degree
        degreeCount = collections.Counter(degree_seq)

        # The counts from the degree
        deg, cnt = zip(*degreeCount.items())

        # Converting the tuples to list and find the avg for cnt
        deg = list(deg)
        cnt = list(cnt)
        avg_cnt = [c / len(adjacency) for c in cnt]

    else:
        # Computing the degrees of each adjacency matrix and storing them all in a list
        for j in range(len(adjacency)):
            g = nx.from_numpy_array(adjacency[j])
            s = sorted([d for n, d in g.degree()], reverse=True)
            degree_list.append(s)
        # Merging the list of lists from above into one list
        for deg in degree_list:
            degree_seq += deg
            # Calculating the counts for each degree
        degreeCount = collections.Counter(degree_seq)

        # The counts from the degree
        deg, cnt = zip(*degreeCount.items())

        # Converting the tuples to list and find the avg for cnt
        deg = list(deg)
        cnt = list(cnt)
        avg_cnt = [c / len(adjacency) for c in cnt]

    return deg, avg_cnt


def degree_dist(adjacency, nodes=28, reshape=True, round=True):
    """A function to compute the average degree distribution of each node in a graph.

                Parameters
                ----------
                adjacency : Numpy array
                        Contains the adjacency matrices in the form of numpy arrys

                adjacency : int 
                        Number of nodes in each adjacency matrix 

                reshape : Bool
                    Argument specifying whether or not to reshape the adjacency matrices

                round : Bool
                    Argument specifying whether or not to round the round the values in the adjacency matrices to
                    either 0 or 1

                Returns
                -------
                degree : list
                    A list of the degrees

                avg_cnt: list
                    A list of the average number of nodes that have the specific degree

                Notes
                -----

                References
                ------
                """
    a_new = []
    degree_list = []
    cnt_list = []
    pk_list = []
    if round & reshape:
        # Rounding the values to either 0 or 1 and changing the shape from (x,x,1) to (x,x)
        for i in adjacency:
            a_new.append(np.squeeze(i, axis=2).round(decimals=0, out=None))
        # Computing the degrees of each adjacency matrix and storing them all in a list
        for j in range(len(adjacency)):
            g = nx.from_numpy_array(a_new[j])
            s = sorted([d for n, d in g.degree()], reverse=True) # degree sequence
            # Calculating the counts for each degree
            degreeCount = collections.Counter(s)

            # The counts from the degree and converting to list 
            deg, cnt = zip(*degreeCount.items())
            deg = list(deg)
            cnt = list(cnt)
            #calculating p(k)
            pk = [k / nodes for k in cnt]
            # appending to output lists  
            degree_list.append(deg) 
            cnt_list.append(cnt) 
            pk_list.append(pk)

        
  

    else:
        # Computing the degrees of each adjacency matrix and storing them all in a list
        for j in range(len(adjacency)):
            g = nx.from_numpy_array(adjacency[j])
            s = sorted([d for n, d in g.degree()], reverse=True)
            
      
            # Calculating the counts for each degree
            degreeCount = collections.Counter(s)

            # The counts from the degree and converting to list 
            deg, cnt = zip(*degreeCount.items())
            deg = list(deg)
            cnt = list(cnt)
            #calculating p(k)
            pk = [k / nodes for k in cnt]
            # appending to output lists  
            degree_list.append(deg) 
            cnt_list.append(cnt) 
            pk_list.append(pk)


    return degree_list, cnt_list, pk_list


def make_same(a_degree,count_a,b_degree,count_b):
    """A helper function to make the probability space of each distribution the same.

                Parameters
                ----------
                a_degree : list
                    Contains the degrees of the fist distribution 

                count_a : list 
                    Contains the respective count for each degree in 'a_degree' 

                b_degree : list
                    Contains the degrees of the second distribution

                count_b : list
                    Contains the respective count for each degree in 'a'

                Returns
                -------
                new_degree_a : list
                    Contains the modified degrees for the first distribution 

                new_count_a: list
                    Contains the modified counts for each degree in 'new_degree_a'
                    
                new_degree_b : list
                    Contains the modified degrees for the second distribution. Same as 'new_degree_a'

                new_count_b: list
                    Contains the modified counts for each degree in 'new_degree_b'

                Notes
                -----

                References
                ------
                """
    if a_degree==b_degree: 
        return(a_degree, count_a, b_degree, count_b)
    #Make the new count lists and degree lists 
    else: 
        c = np.unique(a_degree+b_degree)[::-1]
        new_count_a = [0]*len(c)
        new_count_b = [0]*len(c)
        for k in range(len(c)):
            try: 
                aindex = a_degree.index(c[k])
                apk = count_a[aindex]
                new_count_a[k] = apk
            except ValueError: 
                new_count_a[k] = 0
        for k in range(len(c)): 
            try: 
                bindex = b_degree.index(c[k])
                bpk = count_b[bindex]
                new_count_b[k] = bpk
            except ValueError: 
                new_count_b[k] = 0
    
    degree = list(set(a_degree+b_degree))
    degree.sort(reverse=True)
    new_degree_a = degree
    new_degree_b = degree 

    return(new_degree_a, new_count_a, new_degree_b, new_count_b)
    

def AdKL(CountP,CountQ):
    """A function to compute the adjusted KL Divergance between two distributions. 
       Uniformallty adds 1 to every count in the case either counain a 0. 

                Parameters
                ----------
                CountP : list
                    Contains the respective count for each degree in the P distribtuion

                CountQ : list 
                    Contains the respective count for each degree in the Q distribtuion 

                Returns
                -------
                sum(a) : int
                    The adjusted KL Divergance between P and Q 

                Notes
                -----

                References
                ------
                """

    nodes = sum(CountP)
    if 0 in CountP or 0 in CountQ: 
        NewCountP = [x+1 for x in CountP]
        NewCountQ = [x+1 for x in CountQ]
    else: 
        NewCountP = CountP 
        NewCountQ = CountQ

    #calculating the probabilities
    P = [k/nodes for k in NewCountP]
    Q = [k/nodes for k in NewCountQ]
    a = [0]*len(P)
    for i in range(len(P)): 
        a[i] = P[i]* (np.log(P[i])-np.log(Q[i]))

    return(sum(a))

def AdjustedKL(target_degree, target_counts, pred_degree, pred_counts): 
    """A function to compute the adjusted KL Divergance for a list of degree distributions. 
       Reshapes the degree space so each distribiton has the same bounds and uniformally 
       adds 1 to each count when needed. 

                Parameters
                ----------
                target_degree : list of lists
                    Contains all the degrees for each of the target graphs

                target_counts : list of lists
                    Contains all counts for each degree list in 'target_counts'

                pred_degree : list of lists
                    Contains all the degrees for each of the predicted graphs

                pred_counts : list of lists
                    Contains all counts for each degree list in 'pred_counts'  

                Returns
                -------
                kls : list
                    The adjusted KL Divergances between the target graphs and predicted 

                adjusted_tcounts: list of lists 
                    Contains the modified counts for the target degree distributions 
                
                adjusted_pcounts: list of lists 
                    Contains the modified counts for the predicted degree distributions 
                
                degrees: list of lists 
                    Contains the modified degrees for each  



                Notes
                -----

                References
                ------
                """
    kls = []
    degrees = []
    adjusted_tcounts = []
    adjusted_pcounts = []
    for i in range(len(target_degree)): 
        t_degree = target_degree[i]
        t_count = target_counts[i] 
        p_degree = pred_degree[i] 
        p_count = pred_counts[i]
        td_new, adjusted_target, pd_new, adjusted_pred = make_same(t_degree, t_count, p_degree, p_count)
        kls.append(AdKL(adjusted_target, adjusted_pred))
        adjusted_tcounts.append(adjusted_target)
        adjusted_pcounts.append(adjusted_pred)
        degrees.append(td_new)
  
    return(kls, adjusted_tcounts, adjusted_pcounts, degrees)
    


