#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"

static PyObject *pyWam(PyObject *self, PyObject *args) {
    char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    wam(filename);
    return PyList_New(0);
}

static PyObject *pyDdg(PyObject *self, PyObject *args) {
    char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    ddg(filename);
    return PyList_New(0);
}

static PyObject *pyLnorm(PyObject *self, PyObject *args) {
    char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    lnorm(filename);
    return PyList_New(0);
}

static PyObject *pyJacobi(PyObject *self, PyObject *args) {
    char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    jacobi(filename);
    return PyList_New(0);
}

static PyObject *pySpk(PyObject *self, PyObject *args) {
    int n_py = 0;
    char *filename;
    if (!PyArg_ParseTuple(args, "si", &filename, &n_py)) {
        return NULL;
    }

    PyObject *final_list = PyList_New((n_py+1)*n_py);
    final_list = wam_to_jacobi(filename);
    return final_list;
}

static PyObject *fit(PyObject *self, PyObject *args) {
    int d_py = 0;
    int k_py = 0;
    int n_py = 0;
    int max_iter_py = 0;
    double epsilon_py = 0.0;
    PyObject *data_points_py;
    PyObject *centroids_py;
    if (!PyArg_ParseTuple(args, "iiiidOO", &d_py, &k_py, &n_py, &max_iter_py, &epsilon_py, &data_points_py, &centroids_py)) {
        return NULL;
    }

    PyObject *final_list = PyList_New(d_py*k_py);
    final_list = kmeans(d_py, k_py, n_py, max_iter_py, epsilon_py, data_points_py, centroids_py);
    return final_list;
}

static PyMethodDef myMethods[] = {
    {"pyWam", pyWam, METH_VARARGS, "Weighted Adjacency Matrix."},
    {"pyDdg", pyDdg, METH_VARARGS, "Diagonal Degree Matrix."},
    {"pyLnorm", pyLnorm, METH_VARARGS, "Normalized Graph Laplacian."},
    {"pyJacobi", pyJacobi, METH_VARARGS, " Jacobi Algorithm."},
    {"pySpk", pySpk, METH_VARARGS, "SPK-Means Flow."},
    {"fit", fit, METH_VARARGS, "K-Means Clustering Algorithm."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef spkmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "spkmeansmodule",
    "K-Means Module",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_spkmeansmodule(void) {
    return PyModule_Create(&spkmeansmodule);
}
