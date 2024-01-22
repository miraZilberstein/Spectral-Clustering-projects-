#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Python.h>

// Global variables for data points and centroids
double *space_data_points;
double **data_points;
double *space_prev_centroids;
double **prev_centroids;
double *space_curr_centroids;
double **curr_centroids;
PyObject *final_list;

// Function to handle termination and print an error message
int termination(int error) {
    printf("An Error Has Occurred");
    exit(1);
    return 1;
}

// Function to calculate the Euclidean distance between two vectors
double euclidean_distance(double *v1, double *v2, int d) {
    double curr_sum = 0;
    for (int i = 0; i < d; i++) {
        curr_sum += pow((v1[i] - v2[i]),2);
    }
    return curr_sum;
}

// Function to calculate the Euclidean norm of a vector
double euclidean_norm(double *vector, int d) {
    double norm = 0;
    for (int i = 0; i < d; i++) {
        norm += pow(vector[i],2);
    }
    return pow(norm,0.5);
}

// Function to check for convergence between iterations
int convergence(double *prev[], double *curr[], int k, int d, double epsilon) {
    for (int i = 0; i < k; i++) {
        double prev_norm = euclidean_norm(prev[i], d);
        double curr_norm = euclidean_norm(curr[i], d);
        double cent_dist = fabs(curr_norm - prev_norm);
        if (cent_dist >= epsilon) {
            return 0;
        }
    }
    return 1;
}

// Main K-means algorithm function
PyObject *kmeans(int d, int k, int n, int max_iter, double epsilon, PyObject *data_points_py, PyObject *centroids_py) {
    int iter_cnt = 0;
    int stop = 0;
    PyObject *tmp_list;

    // Memory allocation for data points and centroids
    space_data_points = calloc(n*d, sizeof(double)); 
    data_points = calloc(n, sizeof(double*));
    if (space_data_points == NULL || data_points == NULL) {
        free(space_data_points);
        free(data_points);
        termination(1);
    }
    for (int i = 0; i < n; i++) {
        data_points[i] = space_data_points + i*d;
    }

    // Parsing data points from Python object to C array
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            data_points[i][j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i*d + j));
        }
    }

    // Memory allocation and initialization for centroids
    space_prev_centroids = calloc(k*d, sizeof(double)); 
    prev_centroids = calloc(k, sizeof(double*));
    space_curr_centroids = calloc(k*d, sizeof(double)); 
    curr_centroids = calloc(k, sizeof(double*));
    if (space_curr_centroids == NULL || curr_centroids == NULL) {
        free(space_data_points);
        free(data_points);
        free(space_prev_centroids);
        free(prev_centroids);
        free(space_curr_centroids);
        free(curr_centroids);
        termination(1);
    }

    // Parsing initial centroids from Python object to C array
    for (int i = 0; i < k; i++) {
        curr_centroids[i] = space_curr_centroids + i*d;
        prev_centroids[i] = space_prev_centroids + i*d;
        for (int j = 0; j < d; j++) {
            curr_centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(centroids_py, i*d + j));
            prev_centroids[i][j] = curr_centroids[i][j];
        }
    }

    // K-means iteration loop
    while (!stop && iter_cnt < max_iter) {
        // Assignment step
        int *size_cluster = calloc(k, sizeof(int));
        double *space_clusters_sum = calloc(k*d, sizeof(double)); 
        double **clusters_sum = calloc(k, sizeof(double*));
        for (int i = 0; i < k; i++) {
            clusters_sum[i] = space_clusters_sum + i*d;
        }

        // Calculating distances and updating clusters
        for (int i = 0 ; i < n; i++) {
            double min_dist = euclidean_distance(data_points[i], curr_centroids[0], d);
            int ret_j = 0;
            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(data_points[i], curr_centroids[j], d);
                if (dist < min_dist) {
                    min_dist = dist;
                    ret_j = j;
                }
            }
            size_cluster[ret_j]++;
            for (int j = 0; j < d; j++) {
                clusters_sum[ret_j][j] += data_points[i][j];
            }
        }

        // Update step for centroids
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                curr_centroids[i][j] = clusters_sum[i][j] / size_cluster[i]; 
            }
        }

        // Checking for convergence
        if (convergence(prev_centroids, curr_centroids, k, d, epsilon)) {
            stop = 1;
        }
        iter_cnt++;

        // Preparing for next iteration
        for (int i = 0; i < k; i++) {
            for(int j = 0; j < d; j++) {
                prev_centroids[i][j] = curr_centroids[i][j];
            }
        }
        free(size_cluster);
        free(space_clusters_sum);
        free(clusters_sum);
    }

    // Creating final list of centroids to return to Python
    tmp_list = PyList_New(d*k);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            PyList_SetItem(tmp_list, i*d + j, PyFloat_FromDouble(curr_centroids[i][j]));
        }
    }

    // Freeing allocated memory
    free(space_data_points);
    free(data_points);
    free(space_prev_centroids);
    free(prev_centroids);
    free(space_curr_centroids);
    free(curr_centroids);

    return tmp_list; // Returning the final list of centroids
}

// Python interface for the kmeans function
static PyObject *fit(PyObject *self, PyObject *args) {
    int d_py, k_py, n_py, max_iter_py;
    double epsilon_py;
    PyObject *data_points_py, *centroids_py;

    // Parsing Python arguments
    if (!PyArg_ParseTuple(args, "iiiidOO", &d_py, &k_py, &n_py, &max_iter_py, &epsilon_py, &data_points_py, &centroids_py)) {
        return NULL;
    }

    // Calling the kmeans function and returning the result
    final_list = kmeans(d_py, k_py, n_py, max_iter_py, epsilon_py, data_points_py, centroids_py);
    return final_list;
}

// Method definitions for this module
static PyMethodDef myMethods[] = {
    {"fit", fit, METH_VARARGS, "Executes the K-Means Clustering Algorithm."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef mykmeanssp = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "K-Means Module for Python",
    -1,
    myMethods
};

// Initialization function for this module
PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&mykmeanssp);
}
