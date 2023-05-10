from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# Question 5.1, load image data into a matrix
def load_and_center_dataset(filename):
    x = np.load(filename) 
    mean = np.mean(x, axis=0) # find the mean using this method
    x = x - mean # center the data, easy for PCA
    return x

# Question 5.2, get the covariance matrix from the centered matrix
def get_covariance(dataset):
    # this is a matrix multiplication way of doing this problem
    x_trans = np.transpose(dataset) # finding the transpose and matrix multiplication
    result = (1/(len(dataset)-1)) * np.dot(x_trans, dataset)

    # this is the old way of doing this problem- more mathematically obvious
    #result = np.zeros((len(dataset[0]), len(dataset[0])))
    #for row in range(len(result)):
     #   result  += np.reshape(np.transpose(dataset[row]), (1024,1)) @ np.reshape(dataset[row], (1, 1024))
    #result = (1/(len(dataset)-1)) * result
    return result

# Question 5.3, find the largest eigne values and corresponding eigenvectors
def get_eig(S, m):
    # find the highest 2 eigenvalue and corresponding eigen vectors. They are aligned in ascending order
    v, w = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])

    # matrix to store eigen values in a matrix
    ev = np.zeros((len(v),len(v)))

    # we reverse the eigenvalues so they are placed in descending order
    v = v[::-1]

    for i in range(len(v)):
        ev[i][i] = v[i]

    # we'll reverse the eigenvector column, since we did it for their corresponding eigenvalues, doing this will
    # make sure our eigenvectors are correctly aligned.
    for i in range(len(w)):
        w[i] = w[i][::-1]

    return ev, w

#Question 5.4, find eigenvalues and eigenvectors base on proportion
def get_eig_prop(S, prop):
    #this part is to calculate trace
    v, w = eigh(S)
    ev_mat = np.zeros((len(v), len(v)))
    for i in range(len(v)):
        ev_mat[i][i] = v[i]

    # find eigenvalues and eigenvectors with proportion higher the prop. Do so by using subset_by_values
    # all eigenvectors should be higher than trace*prop, since eval/trace = prop => eval = trace*prop
    v2, w2 = eigh(S, subset_by_value = [np.trace(ev_mat)*prop, v[-1]])

    # reverse the eigen values, so they are in descending order
    ev = np.zeros((len(v2), len(v2)))
    v2 = v2[::-1]

    for i in range(len(v2)):
        ev[i][i] = v2[i]

    # since we revered eigenvalues, we reverse corresponding eigen vectors so it works.
    for i in range(len(w2)):
        w2[i] = w2[i][::-1]
    return ev, w2

#Question 5.5 find the projection matrix
def project_image(image, U):
    # Variable that stores the result vector
    # result is a 1d vector, but it is actually a column vector. 
    projection = np.zeros((len(U))) 

    # calculate a - dot between image vector and Uj column
    for i in range(len(U[0])):
        a = np.dot(np.transpose(U)[i], image) 
        scalar_mult = a * np.transpose(U)[i]
        projection = projection + scalar_mult
    
    return projection

# Q5.6, display the projected and original image
def display_image(orig, proj):
    # reshape it to a square matrix of (32, 32), need to traponse so it is correct orientation
    image = np.reshape(orig, (32, 32))
    image = np.transpose(image)
    project = np.reshape(proj, (32, 32))
    project = np.transpose(project)

    # plotting part, plot the result
    plot = plt.subplots(figsize= (10,3), nrows=1, ncols=2)
    fig = plot[0]
    ax_list = list(plot[1])
    ax_list[0].set_title("Original")
    ax_list[1].set_title("Projection")
    plt1 = ax_list[0].imshow(image, aspect='equal')
    plt2 = ax_list[1].imshow(project, aspect='equal')
    fig.colorbar(plt1, ax=ax_list[0])
    fig.colorbar(plt2, ax=ax_list[1])
    plt.show()

x = load_and_center_dataset("YaleB_32x32.npy")

S = get_covariance(x)

Lambda, U = get_eig(S, 2)

Lambda2, U2 = get_eig_prop(S, 0.01)
#print(Lambda)
#print(U)
print(Lambda2)
print(U2)

projection = project_image(x[150], U2)

#display_image(x[150], projection)