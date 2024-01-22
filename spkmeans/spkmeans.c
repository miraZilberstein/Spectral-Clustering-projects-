#include "spkmeans.h"

double *space_data_points;
double **data_points;
double *space_prev_centroids;
double **prev_centroids;
double *space_curr_centroids;
double **curr_centroids;
PyObject *tmp_list;

// terminates if needed.
int tremination(int error) {
    if (error == 0) {
        printf("Invalid Input!");
    }
    else {
        printf("An Error Has Occurred");
    }
    exit(1);
}

// build matrix n*m
double** build_matrix(int n,int m) {
    int row;
    double *space_matrix = calloc(n*m,sizeof(double)); 
    double **matrix = calloc(n,sizeof(double*)); 
    if(space_matrix == NULL || matrix == NULL){tremination(1);} 
    for(row = 0; row < n; row++) {matrix[row] = space_matrix + row*m;}
    return matrix;
}

// sign(num) = num < 0 ? -1 : 1
double sign(double num) {
 return num < 0 ? -1 : 1;
}

// free matrix
void free_matrix(double **matrix) {
    free(*matrix);
    free(matrix);
}

// return unit matrix with size n*n
double** unit_matrix(int n) {
    int row,column;
    double **unit_matrix = build_matrix(n,n);
    for (row = 0; row < n; row++) // fill unit_matrix
    {
      for (column = 0; column < n; column++) 
      {
        unit_matrix[row][column] = (row==column)? 1 : 0;
      }
    }
    return unit_matrix;
}

// print matrix A(n*m) 
void print_matrix(double *A[],int n,int m) {
    int i,j;
    for (i = 0; i < n; i++) 
    {
      for (j = 0; j < m; j++) 
      {
        printf("%0.4f",A[i][j]);
        if(j!=m-1) {printf("%c",',');}
      }
      printf("\n");
    }
}

// read X from file and find n,d
double** read_X(char *input_filename,int *n,int *d) {
    FILE *inp;
    inp = fopen(input_filename,"r");
    if(inp == NULL){tremination(1);}
    *d = 1; // number of dimensions in vector = (num of ',') +1
    char c = fgetc(inp);
    while ((c = fgetc(inp)) != '\n')  // count num of ','
    {
        if(c == ',') {*d += 1;}
    }
    *n = 1; // number of vectors = num of lines (+1, because we already read a line)
    while ((c = fgetc(inp)) != EOF) //count num of lines
    {
      if (c == '\n')
        *n += 1;
    }
    //read X
    double **X = build_matrix(*n,*d);
    int i,j;
    rewind(inp); // go to beginning of file
    for (i = 0; i < *n; i++) // fill matrix of vectors from file
    {// i = num of vector, j = num of dimention in vector i.
      for (j = 0; j < *d; j++) 
      {
        fscanf(inp, "%lf,", &(X[i][j]));
      }
    } 
    fclose(inp);
    return X;
}

// calculate w_i,j for wam
double w_ij(double *v1, double *v2, int d) {
    double curr_sum = 0;
    int i;
    for (i = 0; i < d; i++) {
        curr_sum += pow((v1[i] - v2[i]),2);

    }
    double w = exp(-pow(curr_sum,0.5)/2); // exp(x)= e^x
    return w; 
}

// return Weighted Adjacency Matrix. 
double** return_wam(double *X[],int n,int d) {  
    double **W = build_matrix(n,n);
    int i,j;

    for (i = 0; i < n; i++) // fill W
    {
      for (j = 0; j < n; j++) 
      {
        W[i][j] = (i==j)? 0 : w_ij(X[i],X[j],d);
      }
    }
    return W;
}

// print Weighted Adjacency Matrix.
void wam(char *input_filename) {
    int n,d;
    double** X = read_X(input_filename,&n,&d);
    double** W = return_wam(X,n,d);
    print_matrix(W,n,n);
    free_matrix(X);
    free_matrix(W);
}

// calculate d_i,j for ddg
double d_ij(double *w_i, int n) {
    double d = 0;
    int j;
    for (j = 0; j < n; j++) {
        d += w_i[j];
    }
    return d; 
}

// return Diagonal Degree Matrix.
double** return_ddg(double *W[],int n) {
    double **D = build_matrix(n,n);
    int i,j;
    for (i = 0; i < n; i++) // fill D
    {
      for (j = 0; j < n; j++) 
      {
        D[i][j] = (i==j)? d_ij(W[i],n) : 0;
      }
    }
    return D;
}

// print Diagonal Degree Matrix.         
void ddg(char *input_filename) {
    int n,d;
    double** X = read_X(input_filename,&n,&d);
    double** W = return_wam(X,n,d);
    double** D = return_ddg(W,n);
    print_matrix(D,n,n);
    free_matrix(X);
    free_matrix(W);
    free_matrix(D);
}

// calculate D^(-0.5)
double** D_new(double *D[],int n) {
    double **D_new = build_matrix(n,n);
    int i,j;
    for (i = 0; i < n; i++) // fill D_new
    {
      for (j = 0; j < n; j++) 
      {
        D_new[i][j] = (i==j)? pow(D[i][j],-0.5) : 0;
      }
    }
    return D_new;
}

// multiply 2 matrix m*m
double** multiply(double *A[],double *B[],int m) {
    double **AB = build_matrix(m,m);
    int i,j,k;
    for (i = 0; i < m; i++) // fill AB
    {
      for (j = 0; j < m; j++) 
      {
        AB[i][j] = 0;
        for(k = 0; k < m; k++){AB[i][j] += A[i][k]*B[k][j];};
      }
    }
    return AB;
}

// Return Normalized Graph Laplacian.
double** return_lnorm(double *D[],double *W[],int n) {
    double **Dnew = D_new(D,n); //Dnew = D^(-0.5)
    double **DW = multiply(Dnew,W,n); //DW = Dnew*W = [D^(-0.5)]*W
    double **DWD = multiply(DW,Dnew,n); //DWD = Dnew*W*Dnew = [D^(-0.5)]*W*[D^(-0.5)]
    double **lnorm_matrix = build_matrix(n,n);
    int i,j;
    for (i = 0; i < n; i++) // fill lnorm
    {
      for (j = 0; j < n; j++) 
      {
        lnorm_matrix[i][j] = (i==j)? 1-DWD[i][j] : -DWD[i][j];
      }
    }
    free_matrix(Dnew);
    free_matrix(DW);
    free_matrix(DWD);

    return lnorm_matrix;
}

// print Normalized Graph Laplacian.
void lnorm(char *input_filename) {
    int n,d;
    double** X = read_X(input_filename,&n,&d);
    double** W = return_wam(X,n,d);
    double** D = return_ddg(W,n);
    double** lnorm_matrix = return_lnorm(D,W,n);
    print_matrix(lnorm_matrix,n,n);
    free_matrix(X);
    free_matrix(W);
    free_matrix(D);
    free_matrix(lnorm_matrix);
}

// return eigenvalues and eigenvectors. (first line are eigenvalues,everything else is eigenvectors)
double** return_jacobi(double** A,int n) {
    int r,row,column;
    double max,curr,c,s,t,offA,offAtag,theta;
    double epsilon = 0.00001;
    int num_rotations = 0; 
    double** V = unit_matrix(n);
    bool Atag_diagonal = false;
    bool convergence = false;
    int i = 0;
    int j = 0;

     //1c jacobi- repeat a,b until A' is diagonal matrix or as discribed in 5.
    double **Atag = build_matrix(n,n);
    for(row = 0; row < n; row++) // Atag = A
    {
        for(column = 0; column < n; column ++)
        {
            Atag[row][column] = A[row][column];
        }
    }
    while(!convergence && num_rotations <= 100 &&!Atag_diagonal ) 
    {
    num_rotations += 1;
    //1a jacobi- build a rotation matrix P (as described in 2,3,4)
    max = -1;
    for(row = 0; row < n; row ++) //find i,j
    {
        for(column = row + 1; column < n; column ++)
        {
            curr = fabs(A[row][column]);
            if(row != column && curr > max)
            {
                max = curr;
                i = row;
                j = column;
            }
        }
    }
    double** P = unit_matrix(n); //build P
    theta = (A[j][j] - A[i][i])/(2*A[i][j]);
    t = sign(theta)/(fabs(theta) + sqrt(pow(theta,2) + 1)) ; 
    c = 1/sqrt(pow(t,2) + 1);
    s = t*c;
    P[i][i] = P[j][j] = c;
    P[i][j] = s;
    P[j][i] = -s;
    
    //1b jacobi- transfer A to A' (as described in 6)
    Atag[i][j] = Atag[j][i] = 0;
    Atag[j][j] = pow(s,2)*A[i][i] + pow(c,2)*A[j][j] + 2*s*c*A[i][j];
    Atag[i][i] = pow(c,2)*A[i][i] + pow(s,2)*A[j][j] - 2*s*c*A[i][j];
   
    for (r = 0; r < n; r++) 
    {
        if(r != i && r != j) 
        {
            Atag[r][i] = Atag[i][r] = c*A[r][i] - s*A[r][j];
            Atag[r][j] = Atag[j][r] = c*A[r][j] + s*A[r][i];
        }
    }
    offA = 0;
    offAtag = 0;
    Atag_diagonal = true;  
    //check convergence(5) and diagonal
    for (row = 0; row < n; row++) 
    {
      for (column = 0; column < n; column++) 
      {
        if(row == column){continue;}
        if(Atag[row][column] != 0){Atag_diagonal = false;} 
        offA += pow(A[row][column],2);
        offAtag += pow(Atag[row][column],2);
      }
    }
    if(fabs(offA - offAtag) <= epsilon) {convergence = true;}
    for (row = 0; row < n; row++) // A = A'
    {
      for (column = 0; column < n; column++) 
      {
        A[row][column] = Atag[row][column];
      }
    }
    double** temp = V;

    V = multiply(V,P,n);
    free(temp);
    free(P);
    }

    double **to_return = build_matrix(n + 1,n);
    for(column = 0; column < n; column++){to_return[0][column] = A[column][column];} // add eigenvalues
    for(row = 1; row < n + 1 ; row++) // add eigenvectors
    {
        for(column = 0; column < n; column++)
        {
            
            to_return[row][column] = V[row - 1][column];
        }
    }
    free_matrix(Atag);
    free_matrix(V);
    return to_return;
}
//prints: first line eigenvalues,second line onward the corresponding eigenvectors
void jacobi(char *input_filename) {
    int n;
    double** A = read_X(input_filename,&n,&n);
    double** eigen_vectors_values = return_jacobi(A,n);
    print_matrix(eigen_vectors_values,n+1,n);
    free_matrix(eigen_vectors_values);
    free_matrix(A);
}

// do all steps - wam,ddg,lnorm,jacobi
PyObject *wam_to_jacobi(char *input_filename) {
    int i,j,k,n,d;
    double** X = read_X(input_filename,&n,&d);
    double** W = return_wam(X,n,d);
    double** D = return_ddg(W,n);
    double** L = return_lnorm(D,W,n);
    double** J = return_jacobi(L,n);
    for(i = 0; i < n+1; i++)
    {
        for(j = 0; j < n; j++){
            if(J[i][j] == -0){J[i][j] = 0;}
        }
    }

    tmp_list = PyList_New(n*(n+1));
    for (i = 0; i < n+1; i++) {
        for (k = 0; k < n; k++) {
            PyList_SetItem(tmp_list, i*n + k,PyFloat_FromDouble(J[i][k]));
        }
    }

    free_matrix(X);
    free_matrix(W);
    free_matrix(D);
    free_matrix(L);
    free_matrix(J);
    for(i = 0; i < n + 1;i++)
    {
        for(j = 0;j < n;j++)
        {

        }
    }

    return tmp_list;
}

// calculate euclidean distance.
double euclidean_distance(double *v1, double *v2, int d) {
    double curr_sum = 0;
    int i;
    for (i = 0; i < d; i++) {
        curr_sum += pow((v1[i] - v2[i]),2);
    }
    return curr_sum;
}

// calculate euclidean norm.
double euclidean_norm(double *vector, int d) {
    double norm = 0;
    int i;
    for (i = 0; i < d; i++) {
        norm += pow(vector[i],2);
    }
    return pow(norm,0.5);
}

// check convergence. 
int convergence(double *prev[], double *curr[], int k, int d, double epsilon) {
    int i;
    for (i = 0; i < k; i++) {
        double prev_norm = euclidean_norm(prev[i], d);
        double curr_norm = euclidean_norm(curr[i], d);
        double cent_dist = fabs(curr_norm - prev_norm);
        if (cent_dist >= epsilon) {
            return 0;
        }
    }
    return 1;
}

// kmeans algo.
PyObject *kmeans(int d, int k, int n, int max_iter, double epsilon, PyObject *data_points_py, PyObject *centroids_py) {
    int i,j;
    int iter_cnt = 0;
    int stop = 0;

    space_data_points = calloc(n*d, sizeof(double)); 
    data_points = calloc(n, sizeof(double*));
    if (space_data_points == NULL || data_points == NULL) {
        free(space_data_points);
        free(data_points);
        tremination(1);
    }
    for (i = 0; i < n; i++) {
        data_points[i] = space_data_points + i*d;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            data_points[i][j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i*d + j));
        }
    }

    space_prev_centroids = calloc(k*d, sizeof(double)); 
    prev_centroids = calloc(k, sizeof(double*));
    if (space_prev_centroids == NULL || prev_centroids == NULL) {
        free(space_data_points);
        free(data_points);
        free(space_prev_centroids);
        free(prev_centroids);
        tremination(1);
    }
    for (i = 0; i < k; i++) {
        prev_centroids[i] = space_prev_centroids + i*d;
    }

    space_curr_centroids = calloc(k*d, sizeof(double)); 
    curr_centroids = calloc(k, sizeof(double*));
    if (space_curr_centroids == NULL || curr_centroids == NULL) {
        free(space_data_points);
        free(data_points);
        free(space_prev_centroids);
        free(prev_centroids);
        free(space_curr_centroids);
        free(curr_centroids);
        tremination(1);
    }
    for (i = 0; i < k; i++) {
        curr_centroids[i] = space_curr_centroids + i*d;
    }

    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
            curr_centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(centroids_py, i*d + j));
            prev_centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(centroids_py, i*d + j));
        }
    }

    while (!stop && iter_cnt < max_iter) {
        int *size_cluster;
        double *space_clusters_sum; 
        double **clusters_sum;
        double min_dist;
        double dist;
        size_cluster = calloc(k, sizeof(int));
        space_clusters_sum = calloc(k*d, sizeof(double)); 
        clusters_sum = calloc(k, sizeof(double*));
        if (space_clusters_sum == NULL || clusters_sum == NULL || size_cluster == NULL) {
            free(space_data_points);
            free(data_points);
            free(space_prev_centroids);
            free(prev_centroids);
            free(space_curr_centroids);
            free(curr_centroids);
            free(size_cluster);
            free(space_clusters_sum);
            free(clusters_sum);
            tremination(1);
        }
        for (i = 0; i < k; i++) {
            clusters_sum[i] = space_clusters_sum + i*d;
        }

        for (i = 0 ; i < n; i++) {
            int ret_j = 0;
            min_dist = euclidean_distance(data_points[i], curr_centroids[0], d);
            for (j = 0; j < k; j++) {
                dist = euclidean_distance(data_points[i], curr_centroids[j], d);
                if (dist < min_dist) {
                    min_dist = dist;
                    ret_j = j;
                }
            }
            size_cluster[ret_j]++;
            for (j = 0; j < d; j++) {
                clusters_sum[ret_j][j] += data_points[i][j];
            }
        }
        for (i = 0; i < k; i++) {
            for (j = 0; j < d; j++) {
                if (size_cluster == 0) {
                    free(space_data_points);
                    free(data_points);
                    free(space_prev_centroids);
                    free(prev_centroids);
                    free(space_curr_centroids);
                    free(curr_centroids);
                    free(size_cluster);
                    free(space_clusters_sum);
                    free(clusters_sum);
                    tremination(1);
                }
                curr_centroids[i][j] = clusters_sum[i][j] / size_cluster[i]; 
            }
        }

        if (convergence(prev_centroids, curr_centroids, k, d, epsilon)) {
            stop = 1;
        }
        iter_cnt++;

        for (i = 0; i < k; i++) {
            for(j = 0; j < d; j++) {
                prev_centroids[i][j] = curr_centroids[i][j];
            }
        }
        free(size_cluster);
        free(space_clusters_sum);
        free(clusters_sum);
    }

    tmp_list = PyList_New(d*k);
    for (i = 0; i < k; i++) {
        for (j = 0; j <d; j++) {
            PyList_SetItem(tmp_list, i*d + j,PyFloat_FromDouble(curr_centroids[i][j]));
        }
    }

    free(space_data_points);
    free(data_points);
    free(space_prev_centroids);
    free(prev_centroids);
    free(space_curr_centroids);
    free(curr_centroids);

    return tmp_list;
}

// check if file name input is valid
int is_valid_fileName(char *name) {
    int str_len = strlen(name);
    if(!(str_len > 4 && name[str_len-4] == '.')) {return false;}
    if(name[str_len-3] == 't' && name[str_len-2] == 'x' && name[str_len-1] == 't') {return true;}
    if(name[str_len-3] == 'c' && name[str_len-2] == 's' && name[str_len-1] == 'v') {return true;} 
    return false;
}
// check if goal input is valid(wam/ddg/lnorm/jacobi) 
int is_valid_goal(char *goal) {  
    if(strcmp(goal, "wam") == 0){return 0;}
    if(strcmp(goal, "ddg") == 0){return 1;}
    if(strcmp(goal, "lnorm") == 0){return 2;}
    if(strcmp(goal, "jacobi") == 0){return 3;}
    return -1;
}

// should allow calls to wam(), ddg(), lnorm() and jacobi(). Does not allow to call spk().
int main(int argc, char* argv[]) {
    if(argc != 3){tremination(0);}
    int goal = is_valid_goal(argv[1]);
    char *fileName = argv[2]; 
    if(!is_valid_fileName(fileName) || goal == -1){tremination(0);}
    if(goal == 0){wam(fileName);}
    else if(goal == 1){ddg(fileName);}
    else if(goal ==2){lnorm(fileName);}
    else if(goal == 3){jacobi(fileName);}
    return 0; 
}
