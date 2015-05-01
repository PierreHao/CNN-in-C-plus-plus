#include "iostream"
#include "vector"
#include "fstream"
#include "util.h"
using namespace std;

int ReverseInt (int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
// void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double> > &arr)
// {
// 	arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
// 	ifstream file ("G:\\t10k-images.idx3-ubyte",ios::binary);
// 	if (file.is_open())
// 	{
// 		int magic_number=0;
// 		int number_of_images=0;
// 		int n_rows=0;
// 		int n_cols=0;
// 		file.read((char*)&magic_number,sizeof(magic_number));
// 		magic_number= ReverseInt(magic_number);
// 		file.read((char*)&number_of_images,sizeof(number_of_images));
// 		number_of_images= ReverseInt(number_of_images);
// 		file.read((char*)&n_rows,sizeof(n_rows));
// 		n_rows= ReverseInt(n_rows);
// 		file.read((char*)&n_cols,sizeof(n_cols));
// 		n_cols= ReverseInt(n_cols);
// 		for(int i=0;i<number_of_images;++i)
// 		{
// 			for(int r=0;r<n_rows;++r)
// 			{
// 				for(int c=0;c<n_cols;++c)
// 				{
// 					unsigned char temp=0;
// 					file.read((char*)&temp,sizeof(temp));
// 					arr[i][(n_rows*r)+c]= (double)temp/255.0;
// 				}
// 			}
// 		}
// 	}
// }

void ReadMNIST(int NumberOfImages, int DataOfAnImage,double ***arr, string name)
{
	//arr = new double**[NumberOfImages];
	ifstream file (name,ios::binary);
	if (file.is_open())
	{
		int magic_number=0;
		int number_of_images=0;
		int n_rows=0;
		int n_cols=0;
		file.read((char*)&magic_number,sizeof(magic_number));
		magic_number= ReverseInt(magic_number);
		file.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images= ReverseInt(number_of_images);
		file.read((char*)&n_rows,sizeof(n_rows));
		n_rows= ReverseInt(n_rows);
		file.read((char*)&n_cols,sizeof(n_cols));
		n_cols= ReverseInt(n_cols);
		for(int i=0;i<NumberOfImages;++i)
		{
			//arr[i] = new double*[DataOfAnImage];
			for(int r=0;r<n_rows;++r)
			{
				//arr[i][r] = new double[DataOfAnImage];
				for(int c=0;c<n_cols;++c)
				{
					unsigned char temp=0;
					file.read((char*)&temp,sizeof(temp));
					arr[i][r][c]= (double)temp/255.0;
				}
			}
		}
	}
}

// void ReadMNIST_Label(int NumberOfImages,vector<vector<double> > &arr)
// {
// 	arr.resize(NumberOfImages,vector<double>(10));
// 	ifstream file ("G:\\t10k-labels.idx1-ubyte",ios::binary);
// 	if (file.is_open())
// 	{
// 		int magic_number=0;
// 		int number_of_images=0;
// 		//int n_rows=0;
// 		//int n_cols=0;
// 		file.read((char*)&magic_number,sizeof(magic_number));
// 		magic_number= ReverseInt(magic_number);
// 		file.read((char*)&number_of_images,sizeof(number_of_images));
// 		number_of_images= ReverseInt(number_of_images);
// 		//file.read((char*)&n_rows,sizeof(n_rows));
// 		//n_rows= ReverseInt(n_rows);
// 		//file.read((char*)&n_cols,sizeof(n_cols));
// 		//n_cols= ReverseInt(n_cols);
// 		for(int i=0;i<NumberOfImages;++i)
// 		{
// 			unsigned char temp=0;
// 			file.read((char*)&temp,sizeof(temp));
// 			int label = (int) temp;		
// 			for (int j =0; j<10; j++)
// 			{
// 				if (j == label)
// 					arr[i][j] = 1.0;
// 				else
// 					arr[i][j] = 0.0;
// 			}
// 		}
// 	}
// }

void ReadMNIST_Label(int NumberOfImages,double **arr, string name)
{
	/*arr = new double *[NumberOfImages];*/
	ifstream file (name,ios::binary);
	if (file.is_open())
	{
		int magic_number=0;
		int number_of_images=0;
		
		file.read((char*)&magic_number,sizeof(magic_number));
		magic_number= ReverseInt(magic_number);
		file.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images= ReverseInt(number_of_images);
		
		for(int i=0;i<NumberOfImages;++i)
		{
			//arr[i] = new double[10];
			unsigned char temp=0;
			file.read((char*)&temp,sizeof(temp));
			int label = (int) temp;		
			for (int j =0; j<10; j++)
			{
				if (j == label)
					arr[i][j] = 1.0;
				else
					arr[i][j] = 0.0;
			}
		}
	}
}

double cpu_time()
{
	return clock()/CLOCKS_PER_SEC;
}

void randomMatrix(int x, int y, double** matrix)
{
	for(int i=0; i<x; i++)
	{
		//matrix[i] = new double[y];
		for(int j=0; j<y; j++)
		{
			matrix[i][j]=(double)(rand()%1000)*0.001-0.5;
		}
	}
}

void randperm(int n, int *perm)
{
	int i, j, t;
	for(i=0; i<n; i++)
		perm[i] = i;
	for(i=0; i<n; i++) {
		j = rand()%(n-i)+i;
		t = perm[j];
		perm[j] = perm[i];
		perm[i] = t;
	}
}

void convnValid(double **matrix,double **kernel,int m, int n, int km, int kn, double** outmatrix)
{
// 	int m = getArrayLen(matrix);
// 	int n = getArrayLen(matrix[0]);
// 	int km = getArrayLen(kernel);
// 	int kn = getArrayLen(kernel[0]);
	// the number of column of convolution
	int kns = n-kn+1;
	// row
	int kms = m-km+1;
// 	double **outmatrix;
// 	outmatrix = new double *[kms];
	for (int i=0; i<kms; i++)
	{
		/*outmatrix[i] = new double[kns];*/
		for(int j=0; j<kns; j++)
		{
			double sum=0.0;
			for(int ki=0; ki<km; ki++)
			{
				for (int kj=0; kj<kn; kj++)
				{
					sum+=matrix[i+ki][j+kj]*kernel[ki][kj];
				}
			}
			outmatrix[i][j] = sum;
		}
	}
}

void Sigmoid(double **matrix,double bias,int m, int n)
{
// 	int m = getArrayLen(matrix);
// 	int n = getArrayLen(matrix[0]);
	
	for (int i=0; i<m; i++)
	{
		for(int j=0; j<n; j++)
		{
			matrix[i][j] = SIGMOID(matrix[i][j]+bias);
		}
	}
}

void ArrayPlus(double **x,double **y, int m, int n)
{
// 	int m = getArrayLen(x);
// 	int n = getArrayLen(x[0]);
	for (int i=0; i<m; i++)
	{
		for(int j=0; j<n; j++)
		{
			y[i][j]=x[i][j]+y[i][j];
		}
	}
}

void scaleMatrix(double** matrix, size scale, int m, int n, double** outMatrix)
{
// 	int m = getArrayLen(matrix);
// 	int n = getArrayLen(matrix[0]);
	int sm =m/scale.x;
	int sn = n/scale.y;
	if (sm*scale.x!=m || sn*scale.y!=n)
	{
		cout<<"scale can not divide by matrix";
	}
	int s = scale.x*scale.y;
	for (int i = 0; i < sm; i++) {
		for (int j = 0; j < sn; j++) {
			double sum = 0.0;
			for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
				for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++) {
					sum += matrix[si][sj];
				}
			}
			outMatrix[i][j] = sum / s;
		}
	}
}

void rot180(double** matrix, int m, int n, double** M)
{
	// 按列对称交换
	for (int i=0; i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			M[i][j] = matrix[i][n-1-j];
		}
	}
	// 按行对称交换
	for(int j=0; j<n; j++)
	{
		for(int i=0; i<m/2; i++)
		{
			double tmp = M[i][j];
			M[i][j] = M[m-1-i][j];
			M[m-1-i][j] = tmp;
		}
	}
}

void convnFull(double** matrix, double** kernel, int m, int n, int km, int kn, double** outmatrix, double **extendMatrix)
{
// 	double **extendMatrix; //为什么这种方式内存无限被吃？
	// 扩展 并初始化 全0矩阵
// 	extendMatrix = new double*[m + 2 * (km - 1)];
// 	for (int k=0; k<m+2*(km-1); k++)
// 	{
// 		extendMatrix[k] = new double[n+2*(kn-1)];
// 		for (int a=0; a<n+2*(kn-1); a++)
// 		{
// 			extendMatrix[k][a] = 0.0;
// 		}
// 	}
	//对应部分赋值
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
	}

	convnValid(extendMatrix, kernel, m + 2 * (km - 1), n+2*(kn-1), km, kn, outmatrix);
//	delete []extendMatrix;
}

void matrixDsigmoid(double** matrix, int m, int n, double** M)
{
	for (int i=0;i<m;i++)
	{
		for (int j=0; j<n;j++)
		{
			M[i][j] = matrix[i][j]*(1-matrix[i][j]);
		}
	}
}

void kronecker(double** matrix,size scale,int m, int n, double** OutMatrix)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
				for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++) {
					OutMatrix[ki][kj] = matrix[i][j];
				}
			}
		}
	}
}

void matrixMultiply(double** matrix1, double** matrix2, int m, int n)
{
	for (int i=0; i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			matrix1[i][j] = matrix1[i][j]*matrix2[i][j];
		}
	}
}

void sum(double**** errors, int j, int m, int n, int batchSize, double **M)
{
	for (int mi = 0; mi < m; mi++) {
		for (int nj = 0; nj < n; nj++) {
			double sum = 0;
			for (int i = 0; i < batchSize; i++){
				sum += errors[i][j][mi][nj];
			}
			M[mi][nj] = sum;
		}
	}
}

double sum(double** error, int m, int n)
{
	double sum = 0.0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			sum += error[i][j];
		}
	}
	return sum;
}

void ArrayDivide(double** M, int batchSize, int m, int n)
{
	for (int i=0; i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			M[i][j] = M[i][j]/batchSize;
		}
	}
}

void ArrayMultiply(double** matrix, double val, int m, int n)
{

	for (int i=0; i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			matrix[i][j] = matrix[i][j]*val;
		}
	}
}

int findIndex(double*** p)
{
	int index = 0;
	double v;
	double Max = p[0][0][0];
	for(int i=1; i<10; i++)
	{
		v = p[i][0][0];
		if(p[i][0][0] > Max)
		{
			Max = p[i][0][0];
			index = i;
		}
	}
	return index;
}

int findIndex(double* p)
{
	int index = 0;
	double Max = p[0];
	for(int i=1; i<10; i++)
	{
		double v = p[i];
		if(p[i] > Max)
		{
			Max = p[i];
			index = i;
		}
	}
	return index;
}

void setValue(double** maps, double** sum, int m, int n)
{
	for (int i = 0; i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			maps[i][j] = sum[i][j];
		}
	}
}