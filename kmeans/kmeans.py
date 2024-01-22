import sys

# Checks if a string represents a valid integer.
def is_valid_int(string):
    if len(string) == 1:
        return string[0].isdigit()
    elif string[0].isdigit() or string[0] == "+":
        return string[1:].isdigit()
    else:
        return False

# Checks if the string represents zero.
def is_zero(num):
    return num == "0" or num == "+0"

# Checks if the string represents one.
def is_one(num):
    return num == "1" or num == "+1"

# Validates if the string is a valid .txt filename.
def is_valid_txt(name):
    return len(name) > 4 and name[-4:] == ".txt"

# Handles termination with an appropriate error message.
def termination(error):
    if error == 0:
        print("Invalid Input!")
    else:
        print("An Error Has Occurred")
    sys.exit()

# Validates the command line arguments for the program.
def is_valid(args):
    if (len(args) > 4 or len(args) < 3):  # Validates the number of arguments.
        termination(0)
        
    if not is_valid_int(args[0]) or is_zero(args[0]) or is_one(args[0]):  # Validates k.
        termination(0)
    elif (len(args) == 3):
        if (not is_valid_txt(args[1])) or (not is_valid_txt(args[2])):  # Validates file names.
            termination(0)
    else:  # len(args) == 4
        if (not is_valid_int(args[1])) or is_zero(args[1]) or \
        (not is_valid_txt(args[2])) or (not is_valid_txt(args[3])):  # Validates max_iter and file names.
            termination(0)

    return True 

# Calculates Euclidean distance between two vectors.
def euclidean_distance(v1, v2, d):
    curr_sum = 0
    for i in range(d):
        curr_sum += ((v1[i] - v2[i])**2)
    return curr_sum  # Returns squared distance for efficiency.

# Calculates the Euclidean norm of a vector.
def euclidean_norm(vector, d):
    norm = 0
    for i in range(d):
        norm += (vector[i]**2)
    return norm**0.5  # Returns the square root of the sum of squares.

# Checks for convergence between the current and previous centroids.
def convergence(prev, curr, k, d):
    epsilon = 0.001
    for i in range(k):  # Checks each centroid for convergence.
        prev_norm = euclidean_norm(prev[i], d)
        curr_norm = euclidean_norm(curr[i], d)
        cent_dist = abs(curr_norm - prev_norm)
        if cent_dist >= epsilon:  # If any centroid hasn't converged, return False.
            return False
    return True  # All centroids have converged.

# The K-means algorithm implementation.
def kmeans(k, max_iter, input_filename, output_filename):
    inp = open(input_filename, 'r')
    data_points = inp.readlines()
    n = len(data_points)
    if k >= n:  # Ensures there are more data points than clusters.
        termination(0)

    try:
        for i in range(n):
            tmp_row = data_points[i].split(",")
            dp = [float(elem) for elem in tmp_row]
            data_points[i] = dp

        prev_centroids = data_points[:k]
        curr_centroids = data_points[:k]
        d = len(data_points[0])
        iter_cnt = 0
        stop = False

        while not stop and iter_cnt < max_iter:
            clusters = [[] for i in range(k)]
            for dp in data_points:  # Assign data points to the nearest centroid.
                min_dist = euclidean_distance(dp, curr_centroids[0], d)
                ret_j = 0
                for j in range(k):
                    dist = euclidean_distance(dp, curr_centroids[j], d)
                    if dist < min_dist:
                        min_dist = dist
                        ret_j = j
                clusters[ret_j].append(dp)

            for i in range(k):  # Update centroids to the mean of assigned data points.
                new_cntr = [sum(vector[j] for vector in clusters[i]) / len(clusters[i]) for j in range(d)]
                curr_centroids[i] = new_cntr

            if convergence(prev_centroids, curr_centroids, k, d):  # Check for convergence.
                stop = True
            iter_cnt += 1
            prev_centroids = curr_centroids.copy()

        out = open(output_filename, 'w')
        for vector in curr_centroids:  # Write the final centroids to the output file.
            str_vec = ",".join("{:.4f}".format(num) for num in vector)
            out.writelines(str_vec + "\n")
        inp.close()
        out.close()
        return 0
    except:
        termination(1)
        return 1

# Main function that processes command line arguments and calls the kmeans function.
def main():
    args = sys.argv[1:]
    is_valid(args)
    if len(args) == 3:
        kmeans(int(args[0]), 200, args[1], args[2])
    else:
        kmeans(int(args[0]), int(args[1]), args[2], args[3])

if __name__ == "__main__":
    main()
