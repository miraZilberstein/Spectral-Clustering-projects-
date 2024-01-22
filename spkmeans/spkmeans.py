import sys
import spkmeansmodule  # Custom module for spectral k-means and related operations
import pandas as pd, numpy as np

# Validates if input is a positive integer or zero
def is_valid_int(string):
    # Single character must be a digit
    # Longer strings must start with a digit or '+' and the rest be digits
    return string.isdigit() or (string[0] in "+0123456789" and string[1:].isdigit())

# Validates if the file has a .txt or .csv extension
def is_valid_file(name):
    return name.endswith(".txt") or name.endswith(".csv")

# Prints an error message and exits the program
def termination(error):
    print("Invalid Input!" if error == 0 else "An Error Has Occurred")
    sys.exit()

# Validates command line arguments
def is_valid(args):
    # Requires exactly 3 arguments
    # Checks for valid 'k', goal, and file extension
    if len(args) != 3 or not is_valid_int(args[0]) or args[1] not in ("spk", "wam", "ddg", "lnorm", "jacobi") or not is_valid_file(args[2]):
        termination(0)

# K-means++ algorithm for initial centroids selection
def kmeans_pp(k, n, data):
    np.random.seed(0)
    centroids_index = [np.random.choice(n)]
    for _ in range(1, k):
        D = [np.min([np.dot(x - data[ci], x - data[ci]) for ci in centroids_index]) for x in data]
        centroids_index.append(np.random.choice(n, p=np.array(D) / np.sum(D)))
    return centroids_index

# Converts a flat list to a 2D matrix
def flat_to_matrix(flat, n, m):
    return [flat[i*m:(i+1)*m] for i in range(n)]

# Eigengap heuristic to determine optimal k
def eigengap_heuristic(eigen_vals):
    return np.argmax(np.diff(eigen_vals[:len(eigen_vals)//2])) + 1

# Normalize rows of a matrix
def normalize(mat):
    return np.array([row / np.linalg.norm(row) if np.linalg.norm(row) else row for row in mat])

# Prints indices of selected centroids and their coordinates
def print_solution(vals, indices, d, k):
    print(",".join(map(str, indices)))
    for i in range(k):
        print(",".join(f"{vals[i*d + j]:.4f}" for j in range(d)))

# Main program execution
def main():
    args = sys.argv[1:]
    is_valid(args)
    
    k = int(args[0])
    goal = args[1]
    filename_py = args[2]
    file_py = pd.read_csv(filename_py, header=None)
    n = len(file_py)

    if k >= n:
        termination(0)

    # Execute corresponding function based on the goal
    if goal == "wam":
        spkmeansmodule.pyWam(filename_py)
    elif goal == "ddg":
        spkmeansmodule.pyDdg(filename_py)
    elif goal == "lnorm":
        spkmeansmodule.pyLnorm(filename_py)
    elif goal == "jacobi":
        spkmeansmodule.pyJacobi(filename_py)
    elif goal == "spk":
        # Special handling for spectral k-means
        spk_res = spkmeansmodule.pySpk(filename_py, n)
        spk_mat = np.array(flat_to_matrix(spk_res, n+1, n))
        eigen_vals = spk_mat[0,:]
        k = k or eigengap_heuristic(eigen_vals)
        U_mat = spk_mat[1:][:, np.argsort(-eigen_vals)[:k]]
        T_mat = normalize(U_mat)
        centroids_index = kmeans_pp(k, n, T_mat)
        res = spkmeansmodule.fit(len(T_mat[0]), k, n, 300, 0, T_mat.flatten().tolist(), T_mat[centroids_index].flatten().tolist())
        if res is None:
            termination(1)
        print_solution(res, centroids_index, len(T_mat[0]), k)

if __name__ == "__main__":
    main()
