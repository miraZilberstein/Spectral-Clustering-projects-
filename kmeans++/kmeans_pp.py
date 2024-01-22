import sys
import mykmeanssp  # Import the custom C extension for K-means.
import pandas as pd, numpy as np

# Validate if a string can be considered a valid integer.
def is_valid_int(string):
    # Single character string must be a digit.
    if len(string) == 1:
        return string[0].isdigit()
    # For longer strings, check if the first character is digit or '+' and the rest are digits.
    elif string[0].isdigit() or string[0] == "+":
        return string[1:].isdigit()
    else:
        return False

# Validate if a string represents a non-negative float (used for epsilon validation).
def is_valid_eps(num):
    try:
        float_num = float(num)
        return float_num >= 0
    except ValueError:
        return False

# Check if a string represents the number zero.
def is_zero(num):
    return num == "0" or num == "+0"

# Check if a string represents the number one.
def is_one(num):
    return num == "1" or num == "+1"

# Validate if the filename ends with .txt or .csv, indicating a valid file type.
def is_valid_file(name):
    return len(name) > 4 and (name[-4:] == ".txt" or name[-4:] == ".csv")

# Handle termination of the program with an error message.
def termination(error):
    if error == 0:
        print("Invalid Input!")
    else:
        print("An Error Has Occurred")
    sys.exit()

# Validate the command-line arguments provided to the script.
def is_valid(args):
    # Ensure the correct number of arguments are provided.
    if len(args) > 5 or len(args) < 4:
        termination(0)

    # Validation logic for various configurations of command-line arguments.
    if not is_valid_int(args[0]) or is_zero(args[0]) or is_one(args[0]):
        termination(0)

    if len(args) == 4:
        if not is_valid_file(args[2]) or not is_valid_file(args[3]) or not is_valid_eps(args[1]):
            termination(0)
    else:
        if not is_valid_int(args[1]) or is_zero(args[1]) or not is_valid_file(args[3]) or not is_valid_file(args[4]) or not is_valid_eps(args[2]):
            termination(0)

    return True

# K-means++ algorithm for selecting initial centroids in an optimized manner.
def kmeans_pp(k, n, data):
    np.random.seed(0)  # For reproducibility.
    centroids_index = np.array([np.random.choice(n)])
    for i in range(1, k):
        D = np.array([np.min(np.sum(np.square(data[l, 1:] - data[centroids_index, 1:]), axis=1)) for l in range(n)])
        Dm = np.sum(D)
        P = D / Dm
        centroids_index = np.append(centroids_index, np.random.choice(n, p=P))
    return centroids_index

# Print the final solution: indices of chosen centroids and their coordinates.
def print_solution(vals, indices, d, k):
    indices_str = ",".join(str(index) for index in indices)
    print(indices_str)

    for i in range(k):
        ctrd = ",".join("{:.4f}".format(vals[i * d + j]) for j in range(d))
        print(ctrd)

# The main function orchestrates the reading of input data, executing K-means++, and printing results.
def main():
    args = sys.argv[1:]
    is_valid(args)  # Validate command-line arguments.
    # Reading input files.
    if len(args) == 4:
        f1 = pd.read_csv(args[2], header=None)
        f2 = pd.read_csv(args[3], header=None)
    else:
        f1 = pd.read_csv(args[3], header=None)
        f2 = pd.read_csv(args[4], header=None)

    # Merging and sorting input data.
    f_join = pd.merge(f1, f2, on=0, how='inner')
    f_join.sort_values(by=[0], inplace=True)
    data = f_join.to_numpy()

    # Extracting dimensions and data size.
    k, n = int(args[0]), len(f_join)
    d = data.shape[1] - 1

    # Handling invalid K.
    if k >= n:
        termination(0)

    # Setting up parameters and calling the custom C extension for K-means.
    epsilon = float(args[1] if len(args) == 4 else args[2])
    max_iter = 300 if len(args) == 4 else int(args[1])
    centroids_index = kmeans_pp(k, n, data)
    data_points = data[:, 1:].flatten().tolist()
    centroids = data[centroids_index, 1:].flatten().tolist()
    res = mykmeanssp.fit(d, k, n, max_iter, epsilon, data_points, centroids)

    # If K-means C extension fails, terminate.
    if res is None:
        termination(1)

    # Print the final centroids and their indices.
    print_solution(res, centroids_index, d, k)

if __name__ == "__main__":
    main()
