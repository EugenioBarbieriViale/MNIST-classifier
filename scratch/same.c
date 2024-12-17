#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "MNIST.h"

#define square(x) (x) * (x)

#define input_size 784
#define hidden_size 128

#define epochs 10
#define rate 1e-2

// TOOLS
double rand_double() {
    return ((double)rand() / (double)RAND_MAX);
}

void dot(int M, int N, int K, double matA[M][N], double matB[N][K], double matC[M][K]) {
    // dot(MxN, NxK)
    for(int m = 0; m < M; m++){
        for(int k = 0; k < K; k++){
            matC[m][k] = 0;
            for(int n = 0; n < N; n++){
                matC[m][k] += matA[m][n] * matB[n][k];
            }
        }
    }
}

double sigmoid(double x) {
    return (double)(1.f/(1.f + exp(-x)));
}

double d_sigmoid(double x) {
    return (double)(exp(-x))/(square(exp(-x)+1.f));
}

void applay_sigm(int l, double arr[1][l]) {
    for (int i=0; i<l; i++) {
        double temp = arr[0][i];
        arr[0][i] = sigmoid(temp);
    }
}

// INITIALIZATION
void alloc_input(int index, double input[1][input_size]) {
    for (int i=0; i<input_size; i++)
        input[0][i] = train_image[index][i];
}

void init_weights(int rows, int cols, double w[rows][cols]) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            w[r][c] = rand_double();
        }
    }
}

// COMPUTATION
int *process_label(int label) {
    int *ans = (int*)malloc(10 * sizeof(int));
    if (!ans) return NULL;

    for (int i=0; i<10; i++) {
        if ((label-1) == i)
            ans[i] = 1;
        else
            ans[i] = 0;
    }
    return ans;
}

double *calc_error(double output[1][10], int label) {
    double *e = (double*)malloc(10 * sizeof(double));
    int *target = process_label(label);

    for (int i=0; i<10; i++) {
        e[i] = (output[0][i] - (double)target[i]) * d_sigmoid(output[0][i]);
    }
    return e;
}

double get_cost(double error[10]) {
    double cost = 0.f;
    for (int i=0; i<10; i++)
        cost += error[i];
    return cost;
}

void hidden_error(double hidden_layer[1][hidden_size], double error[10], double output[1][10], double w2[hidden_size][10]) {
    /* error = (weight_k * error_j) * d_sigmoid(output) */
    error = (error[j] * )
}

void forward(double input[1][input_size], double w1[input_size][hidden_size], double w2[hidden_size][10], int label) {
    double hidden_layer[1][hidden_size];
    dot(1, input_size, hidden_size, input, w1, hidden_layer);
    applay_sigm(hidden_size, hidden_layer);

    double output[1][10];
    dot(1, hidden_size, 10, hidden_layer, w2, output);
    applay_sigm(10, output);

    double *error = calc_error(output, label);

    printf("%f\n", get_cost(error));
}

int main() {
    srand(time(NULL));
    load_mnist();

    int index = 0;

    int label = train_label[index];

    double input[1][input_size];
    alloc_input(index, input);

    double w1[input_size][hidden_size];
    init_weights(input_size, hidden_size, w1);

    double w2[hidden_size][10];
    init_weights(hidden_size, 10, w2);

    forward(input, w1, w2, label);

    return 0;
}
