#ifndef SPKMEANS_H_
#define SPKMEANS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Python.h>

typedef int bool;
#define true 1
#define false 0

int tremination(int error);
double** build_matrix(int n,int m);
double sign(double num);
void free_matrix(double **matrix);
double** unit_matrix(int n);
void print_matrix(double *A[],int n,int m);
double** read_X(char *input_filename,int *n,int *d);
double w_ij(double *v1, double *v2, int d);
double** return_wam(double *X[],int n,int d);
void wam(char *input_filename);
double d_ij(double *w_i, int n);
double** return_ddg(double *W[],int n);
void ddg(char *input_filename);
double** D_new(double *D[],int n);
double** multiply(double *A[],double *B[],int m);
double** return_lnorm(double *D[],double *W[],int n);
void lnorm(char *input_filename);
double** return_jacobi(double** A,int n);
void jacobi(char *input_filename);
PyObject *wam_to_jacobi(char *input_filename);
double euclidean_distance(double *v1, double *v2, int d);
double euclidean_norm(double *vector, int d);
int convergence(double *prev[], double *curr[], int k, int d, double epsilon);
PyObject *kmeans(int d, int k, int n, int max_iter, double epsilon, PyObject *data_points_py, PyObject *centroids_py);
int is_valid_fileName(char *name);
int is_valid_goal(char *goal);

#endif
