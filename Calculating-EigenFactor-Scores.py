#!/usr/bin/env python

# In[1]:


# Import the relevant libraries 
import numpy as np
import pandas as pd
import time


# In[2]:


# Functions needed to load file with edge list and calculate corresponding eigenfactor 
def num_node(file):
    """Reads in a file where the first column (jrnl) is the original journal and the 
    second column (jrnl_cited) is the referenced journal. Will find the number of 
    unique journals from both columns and return the larger unique number 
    (i.e., the greatest number of unique journals from the file)."""
    
    df = pd.read_csv(file, sep=',',header=None,names=['jrnl','jrnl_cited','ref_count'])
    if df['jrnl'].nunique() > df['jrnl_cited'].nunique():
        return df['jrnl'].nunique()
    else:
        return df['jrnl_cited'].nunique()


# In[3]:


def create_adj_matrix(file, node_count):
    """Creates an adjacency matrix from a file with edge list. The number of journals is used to set the 
    shape of the matrix (n*n). Number of journals (nodes) was determined by num_journal function.
    Returns a dataframe."""
 
    # Create empty n*n matrix (where n = num_node returned from num_journal function)
    # Using np.zero function so journals that don't have a match will have a citation count of 0
    adj_matrix = np.zeros((node_count, node_count))
    
    # Open file with edge list
    with open(file) as fp:
        for index, line in enumerate(fp):
            # Splits each line by the comma 
            line = line.split(',')   
            # This is the journal that is doing the citing (using INT to convert to int data type from string)
            jrnl = int(line[0])
            # This is the journal that is being cited 
            jrnl_cited = int(line[1])
            # This is the number of times that jrnl cited jrnl_cited
            ref_count = int(line[2])
            # Set the values of the adjacency matrix 
            # Index of the jrnl (column) matches index of the jrnl_cited (row) and the value is set to ref_count
            adj_matrix[jrnl_cited][jrnl] = ref_count 
    
    # returns adj_matrix converted to a dataframe
    return adj_matrix


# In[4]:


def mod_matrix_retrieve_dnode(adj_matrix):
    """Modifies an existing adjacency matrix (dataframe) by:
    - Setting the diagonal to 0
    - Normalizing the columns (dividing each entry in the column by the sum of the column)"""
    
    # Create empty dangling node vector 
    dnode_vect = []

    # Checks if adj_matrix is a dataframe 
    if isinstance(adj_matrix, pd.DataFrame):
        adj_matrix = adj_matrix.values
    
    # Set the diagonal of the adjacency matrix to 0 
    # Prevents journals (nodes) from receiving credit for referencing themselves
    np.fill_diagonal(adj_matrix, 0)
    
    # Convert to dataframe for manipulating values
    df=pd.DataFrame(adj_matrix)
    
    # Normalize the columns by dividing each value by the sum of each column 
    # Will append values to dangling node vector 
    # Append 0 to dangling list if sum of columns > 0
    # Append 1 to dangling list to identify dangling node (where sum of column = 0)
    for column in df:
        # Check to see if the sum of the column is zero
        # Do nothing to prevent dividing by zero
        if abs(df[column].sum()) > 0:
            df[column] = df[column]/(df[column].sum())
            dnode_vect.append(0)
        else:
            df[column] = df[column]
            dnode_vect.append(1)

    # Convert to matrix 
    h_matrix=df.values

    return h_matrix, dnode_vect


# In[5]:


def calc_article_vect(art_dict={}, num_node=0, dict_present=False):
    """Calculates article vector. If dict_present is False, it will create an article vector 
    where all articles publish one paper. 
    If dict_present is True, takes a dict with journal (key) and 
    number of articles published (value) for the journal. 
    Divides number of articles of each journal by the total articles published from all journals."""
    
    art_vect = []
    
    if dict_present == False:
        for _ in range(num_node):
            art_vect.append(1/num_node)
    else:
        a_tot = sum(art_dict.values())
        for key, value in art_dict.items():
            art_vect.append(value/a_tot)
    
    return art_vect


# In[6]:


def calc_start_vect(matrix_h, node_count):
    """Calculates the initial start vector. Iterates the influence vector. 
    Vector will equal 1 divided by n (number of unique nodes)."""
    
    start_vect = []
    
    for column in matrix_h:
        start_vect.append(1/node_count)
    
    return start_vect


# In[7]:


def calc_inf_vector(mat_h, start_vct, art_vct, alpha, d): 
    """Calculates the influence vector. The following equation is used to compute
    the influence vector rapidly: 
    pi^(k+1) = 
    alpha * matrix_h * start_vect + 
    [alpha * d_node * start_vect + (1-alpha)] * art_vect"""
    
    # multiply alpha with the h matrix using the MULTIPLY function (alpha * H)
    alpha_h_start = np.multiply(alpha, mat_h)
    # multiply start vector (pi^k) with alpha * H
    alpha_h_start = np.dot(alpha_h_start, start_vct)
    
    # alpha * dangling node * start vector + (1 - alpha)
    alpha_dnode_startvect = np.multiply(alpha, d)
    alpha_dnode_startvect = np.dot(alpha_dnode_startvect, start_vct)
    alpha_dnode_startvect = alpha_dnode_startvect + (1 - alpha)
    
    # multiply article vector with previous reuslt
    alpha_dnode_startvect = np.dot(alpha_dnode_startvect, art_vct)

    inf_vector = alpha_h_start + alpha_dnode_startvect
    
    return inf_vector


# In[8]:


def iterate_converge(mat_h, start_vct, art_vct, alpha, epsilon, d):
    """Checks to see if iteration has converged. 
    Convergence is found via checking if residual is less than epsilon (calculating the L1 norm). 
    If convergence has not been reached, iterate again. Counts number of iterations till convergence."""
    
    # Calculate the initial influence vector using the starting vector 
    inf_vector_k = calc_inf_vector(mat_h=mat_h, start_vct=start_vct, art_vct=art_vct, alpha=alpha, d=d)
    # Calculate the first iteration of the influence vector 
    inf_vector_k1 = calc_inf_vector(mat_h=mat_h, start_vct=inf_vector_k, art_vct=art_vct, alpha=alpha, d=d)
    iterate_count = 1
    # Calculating the initial L1 norm
    l1 = np.linalg.norm((inf_vector_k1 - inf_vector_k), ord=1)
    l1 = np.round(l1, 5)
    
    # If the L1 norm is greater than the epsilon value, continue to iterate 
    while l1 > epsilon:
        # Count the iteration 
        iterate_count += 1
        # Set the reference influence vector to be the iterated influence vector
        inf_vector_k = inf_vector_k1
        # Calculate next iteration of the influence vector 
        inf_vector_k1 = calc_inf_vector(mat_h=mat_h, start_vct=inf_vector_k1, art_vct=art_vct, alpha=alpha, d=d)
        # Calculate and set the new L1 norm
        l1 = np.linalg.norm((inf_vector_k1 - inf_vector_k), ord=1)

    return inf_vector_k1, iterate_count


# In[9]:


def calc_eigenfactor(mat_h, inf_vector):
    """Calculates the eigenfactor values for each journal. Takes the dot product of the H matrix and the 
    converged influence vector and normalizes it to sum to 1. Multiplied by 100 to convert to percentage.
    Eigenfactor = 100 * ((matrix_h*influence_vector)/sum(matrix_h * influence_vector))"""
    
    # Multiply H matrix and converged influence vector 
    h_inf = np.dot(mat_h, inf_vector)
    # Calculate sum of the above result for calculating the normalized sum to 1 
    sum_h_inf = np.sum(np.dot(mat_h, inf_vector))
    
    # Multiply by 100 to caclculate the eigenvalue in percentage 
    eigenfactor = 100 * (h_inf/sum_h_inf)
    
    return eigenfactor


# In[29]:


def top_20_eigenfactor(mat_h, eigenfactor):
    """Returns the top 20 journals based on their Eigenfactor scores."""
    
    # Create a copy of matrix H
    eigen_df = pd.DataFrame(mat_h)
    # Rename dataframe's index to Journal
    eigen_df.index.name = 'Journal'
    # Create an 'Eigenfactor' column with eigenfactor values (calculated from calc_eigenfactor function)
    eigen_df['Eigenfactor'] = eigenfactor
    # Drop all other columns except 'Eigenfactor'
    eigen_df.drop(eigen_df.columns.difference(['Eigenfactor']), 1, inplace=True)
    # Pull top 20 results (sorted by Eigenfactor descending)
    top_20 = eigen_df.sort_values(by = 'Eigenfactor', ascending=False).head(20)
    
    return top_20, eigen_df


# In[15]:


def main():
    # Import libraries
    import numpy as np
    import pandas as pd
    import time
    
    # Setting the constants needed for the algorithm
    alpha = 0.85
    epsilon = 0.00001

    # Set the filepath to a edge list file ('links.txt' in this case)
    filepath = 'links.txt'

    # Begin timing the code
    start_time = time.time()

    # Calculate the number of nodes from the edge list
    node_count = num_node(filepath)    
    # Create an adjacency matrix from the edge list 
    adj_matrix = create_adj_matrix(file=filepath, node_count=node_count)
    # Create the H matrix (which has been modified) and a vector indicating the dangling nodes
    h_matrix, d_node= mod_matrix_retrieve_dnode(adj_matrix=adj_matrix)
    # Create the article vector, in this case a vector of (1/node_count) for all articles
    article_vector = calc_article_vect(num_node=node_count, dict_present=False)
    # Create the start vector, the vector used for iterating the influence vector 
    start_vector = calc_start_vect(matrix_h=h_matrix, node_count=node_count)
    # Find the converged influence vector and the number of iterations it took for convergence
    influence_vector, iterate_num = iterate_converge(mat_h=h_matrix, start_vct=start_vector, art_vct=article_vector, alpha=alpha, epsilon=epsilon, d=d_node)
    # Create a vector with the eigenfactors (for each journal)
    eigenfactor_vector = calc_eigenfactor(mat_h=h_matrix, inf_vector=influence_vector)
    # Return the scores for the top 20 journals 
    top20, df = top_20_eigenfactor(h_matrix, eigenfactor_vector)

    # End the timer for the code
    end_time = time.time()

    print("Top 20 journals and calculated scores for the input file:")
    # Print top 20 journals and corresponding eigenfactors
    article_count=1
    for index, value in top20.stack().iteritems():
        print(article_count, '. Journal: {0[0]}, Eigenfactor: {1}'.format(index, value), sep='')
        article_count+=1

    print("Total time (seconds):", end_time-start_time)
    print("Total number of iterations:", iterate_num)


if __name__ == "__main__":
    main()

# Sample output: 
# Top 20 journals and calculated scores for the input file:
# 1. Journal: 4408, Eigenfactor: 1.4481186906767658
# 2. Journal: 4801, Eigenfactor: 1.4127186417103397
# 3. Journal: 6610, Eigenfactor: 1.2350345744039224
# 4. Journal: 2056, Eigenfactor: 0.6795023571614872
# 5. Journal: 6919, Eigenfactor: 0.6648791185697115
# 6. Journal: 6667, Eigenfactor: 0.6346348415047293
# 7. Journal: 4024, Eigenfactor: 0.5772329716737367
# 8. Journal: 6523, Eigenfactor: 0.48081511644794395
# 9. Journal: 8930, Eigenfactor: 0.47777264655981166
# 10. Journal: 6857, Eigenfactor: 0.4397348022988983
# 11. Journal: 5966, Eigenfactor: 0.4297177536469516
# 12. Journal: 1995, Eigenfactor: 0.3862065206886991
# 13. Journal: 1935, Eigenfactor: 0.3851202633995659
# 14. Journal: 3480, Eigenfactor: 0.37957760331596935
# 15. Journal: 4598, Eigenfactor: 0.372789008691295
# 16. Journal: 2880, Eigenfactor: 0.3303062827175173
# 17. Journal: 3314, Eigenfactor: 0.32750789522300316
# 18. Journal: 6569, Eigenfactor: 0.319271668905672
# 19. Journal: 5035, Eigenfactor: 0.3167790348824174
# 20. Journal: 1212, Eigenfactor: 0.3112570455380745
# Total time (seconds): 50.19530510902405
# Total number of iterations: 31