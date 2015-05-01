#ifndef UTIL_H
#define UTIL_H
#include "math.h"
#include "iostream"
#include "vector"
#include "fstream"
#include "time.h"
#include "Layer.h"
using namespace std;
class size;
#define SIGMOID(x) (1/(1+exp(-x)))

// template <class T>
// int getArrayLen(T& array)
// {//使用模板定义一个函数getArrayLen,该函数将返回数组array的长度
// 	return (sizeof(array) / sizeof(array[0]));
// }
int ReverseInt (int i);
//void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double> > &arr);//save input data as vector
void ReadMNIST(int NumberOfImages, int DataOfAnImage,double ***arr, string name);//save input data as array
//void ReadMNIST_Label(int NumberOfImages, vector<vector<double> > &arr);
void ReadMNIST_Label(int NumberOfImages, double **arr, string name);
double cpu_time();
void randomMatrix(int x, int y, double** outmatrix);
void randperm(int n, int* out);
void convnValid(double **matrix,double **kernel,int m, int n, int km, int kn, double** outmatrix);// m n is the dimension of matrix and km kn is the dimension of kernel, outmatrix is result
void Sigmoid(double** matrix,double bias,int m, int n);// m n is the dimension of matrix
void ArrayPlus(double **x,double **y, int m, int n);//矩阵加法 结果放在y里；
void scaleMatrix(double** lastMap,size scaleSize, int m, int n, double** sampling);//sampling
void rot180(double** matrix, int m, int n, double** rotMatrix);
void convnFull(double** matrix, double** kernel, int m, int n, int km, int kn, double** outmatrix, double** extendMatrix);// convn full mode
void matrixDsigmoid(double** matrix, int m, int n, double** outmatrix);// calculate derivation of sigmoid function with matrix
void kronecker(double** matrix,size scale,int m, int n, double** outmatrix);
void matrixMultiply(double** matrix1, double** matrix2, int m, int n);//inner product of matrix 1 and matrix 2, result is matrix1
void sum(double**** errors, int j, int m, int n, int batchSize, double **outmatrix);
double sum(double** error,int m, int n);
void ArrayDivide(double** matrix, int batchSize, int m, int n);// result is matrix;
void ArrayMultiply(double** matrix, double val, int m, int n);// array multiply a double value, result in matrix
void setValue(double** maps, double** sum, int m, int n);
int findIndex(double*** p);
int findIndex(double* p);
#endif
