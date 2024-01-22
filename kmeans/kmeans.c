#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Function to check if a character is a digit.
int is_digit(char ch) {
    return (ch >= '0' && ch <= '9');
}

// Function to validate if a string represents a valid integer.
int is_valid_int(char *string) {
    int str_len = strlen(string);
    if (str_len == 1) {
        return is_digit(string[0]);
    }
    else if (is_digit(string[0]) || string[0] == '+') {
        for (int i = 1; i < str_len; i++) {
            if (!is_digit(string[i])) {
                return 0; // Not a valid digit, return false.
            }
        }
        return 1; // All characters are digits, return true.
    }
    return 0; // Not a valid integer format.
}

// Function to check if the string represents zero.
int is_zero(char *num) {
    int str_len = strlen(num);
    return ((str_len == 1 && num[0] == '0') || (str_len == 2 && num[0] == '+' && num[1] == '0'));
}

// Function to check if the string represents one.
int is_one(char *num) {
    int str_len = strlen(num);
    return ((str_len == 1 && num[0] == '1') || (str_len == 2 && num[0] == '+' && num[1] == '1'));
}

// Function to check if a string is a valid .txt filename.
int is_valid_text(char *name) {
    int str_len = strlen(name);
    return str_len > 4 && name[str_len-4] == '.' && name[str_len-3] == 't' && name[str_len-2] == 'x' && name[str_len-1] == 't';
}

// Function to handle program termination with an error message.
int termination(int error) {
    if (error == 0) {
        printf("Invalid Input!");
    }
    else {
        printf("An Error Has Occurred");
    }
    exit(1); // Terminate the program.
}

// Function to validate command line arguments.
int is_valid(int argc, char *args[]) {
    if (argc > 4 || argc < 3) {
        termination(0); // Incorrect number of arguments.
    }
    if (!is_valid_int(args[0]) || is_zero(args[0]) || is_one(args[0])) {
        termination(0); // First argument is not a valid number.
    }
    else if (argc == 3) {
        if (!is_valid_text(args[1]) || !is_valid_text(args[2])) {
            termination(0); // File names are not valid.
        }
    }
    else {
        if (!is_valid_int(args[1]) || is_zero(args[1]) || !is_valid_text(args[2]) || !is_valid_text(args[3])) {
            termination(0); // Arguments are not valid.
        }
    }
    return 1; // Arguments are valid.
}

// Function to calculate Euclidean distance between two vectors.
double euclidean_distance(double *v1, double *v2, int d) {
    double sum = 0;
    for (int i = 0; i < d; i++) {
        sum += pow((v1[i] - v2[i]), 2); // Summing the square of differences.
    }
    return sqrt(sum); // Returning the square root of the sum.
}

// Function to calculate the Euclidean norm of a vector.
double euclidean_norm(double *vector, int d) {
    double sum = 0;
    for (int i = 0; i < d; i++) {
        sum += pow(vector[i], 2); // Summing the squares of each component.
    }
    return sqrt(sum); // Returning the square root of the sum.
}

// Function to check for convergence between previous and current centroids.
int convergence(double *prev[], double *curr[], int k, int d) {
    double epsilon = 0.001; // Convergence threshold.
    for (int i = 0; i < k; i++) {
        if (fabs(euclidean_norm(curr[i], d) - euclidean_norm(prev[i], d)) >= epsilon) {
            return 0; // Not converged.
        }
    }
    return 1; // Converged.
}

// The K-means algorithm implementation.
int kmeans(int k, int max_iter, char *input_filename, char *output_filename) {
    // Variables for file operations, data points, centroids, etc.
    FILE *inp, *out;
    char c;
    int d = 0, n = 0, i, j, iter_cnt = 0, stop = 0;
    double *space_data_points, **data_points;
    double *space_prev_centroids, **prev_centroids;
    double *space_curr_centroids, **curr_centroids;

    // Opening the input file.
    inp = fopen(input_filename, "r");
    if (inp == NULL) {
        termination(1); // File opening failed.
    }

    // Determining the dimensionality (d) by counting commas until a newline.
    while ((c = fgetc(inp)) != '\n') {
        if (c == ',') d++;
    }
    d += 1; // Incrementing d for the actual number of dimensions.

    // Counting the number of data points (n).
    rewind(inp); // Resetting file pointer to start.
    while ((c = fgetc(inp)) != EOF) {
        if (c == '\n') n++;
    }

    // Allocating memory for data points and centroids.
    space_data_points = (double*) calloc(n * d, sizeof(double));
    data_points = (double**) calloc(n, sizeof(double*));
    space_prev_centroids = (double*) calloc(k * d, sizeof(double));
    prev_centroids = (double**) calloc(k, sizeof(double*));
    space_curr_centroids = (double*) calloc(k * d, sizeof(double));
    curr_centroids = (double**) calloc(k, sizeof(double*));

    // Memory allocation checks.
    if (!space_data_points || !data_points || !space_prev_centroids || !prev_centroids || !space_curr_centroids || !curr_centroids) {
        fclose(inp);
        termination(1); // Memory allocation failed.
    }

    // Initializing data points array.
    for (i = 0; i < n; i++) {
        data_points[i] = space_data_points + i * d;
    }

    // Reading data points from the file.
    rewind(inp); // Resetting file pointer to start to read data.
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            fscanf(inp, "%lf,", &data_points[i][j]);
        }
    }

    // Initializing centroids arrays.
    for (i = 0; i < k; i++) {
        prev_centroids[i] = space_prev_centroids + i * d;
        curr_centroids[i] = space_curr_centroids + i * d;
        // Initially setting current and previous centroids to the first k data points.
        for (j = 0; j < d; j++) {
            curr_centroids[i][j] = data_points[i][j];
            prev_centroids[i][j] = data_points[i][j];
        }
    }

    // K-means algorithm main loop.
    while (!stop && iter_cnt < max_iter) {
        // Variables for cluster assignment and centroid updating.
        double *space_clusters_sum, **clusters_sum;
        int *size_cluster = (int*) calloc(k, sizeof(int));
        space_clusters_sum = (double*) calloc(k * d, sizeof(double));
        clusters_sum = (double**) calloc(k, sizeof(double*));

        // Memory allocation checks.
        if (!size_cluster || !space_clusters_sum || !clusters_sum) {
            fclose(inp);
            termination(1); // Memory allocation failed.
        }

        // Initializing clusters sum array.
        for (i = 0; i < k; i++) {
            clusters_sum[i] = space_clusters_sum + i * d;
        }

        // Assigning data points to the nearest centroid and updating cluster sums.
        for (i = 0; i < n; i++) {
            double min_dist = euclidean_distance(data_points[i], curr_centroids[0], d);
            int closest_centroid = 0;
            for (j = 1; j < k; j++) {
                double dist = euclidean_distance(data_points[i], curr_centroids[j], d);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            size_cluster[closest_centroid]++;
            for (j = 0; j < d; j++) {
                clusters_sum[closest_centroid][j] += data_points[i][j];
            }
        }

        // Updating centroid positions.
        for (i = 0; i < k; i++) {
            if (size_cluster[i] == 0) continue; // Avoid division by zero.
            for (j = 0; j < d; j++) {
                curr_centroids[i][j] = clusters_sum[i][j] / size_cluster[i];
            }
        }

        // Checking for convergence.
        if (convergence(prev_centroids, curr_centroids, k, d)) {
            stop = 1; // Convergence achieved.
        }

        // Preparing for the next iteration.
        iter_cnt++;
        for (i = 0; i < k; i++) {
            for (j = 0; j < d; j++) {
                prev_centroids[i][j] = curr_centroids[i][j];
            }
        }

        // Freeing memory for clusters.
        free(size_cluster);
        free(space_clusters_sum);
        free(clusters_sum);
    }

    // Writing the final centroids to the output file.
    out = fopen(output_filename, "w");
    if (out == NULL) {
        fclose(inp);
        termination(1); // File opening failed.
    }
    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
            fprintf(out, "%.4f", curr_centroids[i][j]);
            if (j < d-1) fprintf(out, ",");
            else fprintf(out, "\n");
        }
    }

    // Cleanup: closing files and freeing allocated memory.
    fclose(inp);
    fclose(out);
    free(space_data_points);
    free(data_points);
    free(space_prev_centroids);
    free(prev_centroids);
    free(space_curr_centroids);
    free(curr_centroids);

    return 0; // Successful completion.
}

// Main function: parses command line arguments and invokes the k-means algorithm.
int main(int argc, char* argv[]) {
    int k; // Number of clusters.
    is_valid(argc-1, ++argv); // Validates the command line arguments.
    sscanf(argv[0], "%d", &k); // Parses the number of clusters.
    
    // Distinguishes between default and user-defined maximum iterations.
    if (argc-1 == 3) {
        // Calls kmeans with default maximum iterations.
        kmeans(k, 200, argv[1], argv[2]);
    } else {
        // Calls kmeans with specified maximum iterations.
        int max_iter;
        sscanf(argv[1], "%d", &max_iter);
        kmeans(k, max_iter, argv[2], argv[3]);
    }
    return 0; // Indicates successful execution.
}
