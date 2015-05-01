#include "iostream"
#include "util.h"
#include "vector"
#include "CNN.h"
#include "time.h"
#include "stdlib.h"
using namespace std;
int main()
{
	// initialize input data
 	double*** train_x, ***test_x;
	double **train_label, **test_label;
	int NumberOfImages = 60000;
	int length = 28;
	train_x = new double**[NumberOfImages];
	test_x = new double**[NumberOfImages/6];
	train_label = new double *[NumberOfImages];
	test_label = new double *[NumberOfImages/6];
	for (int i=0; i<NumberOfImages; i++)
	{
		train_x[i]=new double*[28];
		train_label[i]=new double[10];
		if (i<NumberOfImages/6)
		{
			test_x[i]=new double*[28];
			test_label[i] = new double [10];
		}
		for (int j=0; j<28; j++)
		{
			train_x[i][j]=new double[28];
			if (i<NumberOfImages/6)
			{
				test_x[i][j]=new double[28];
			}
		}
	}
 	ReadMNIST(NumberOfImages,length,train_x,"G:\\train-images.idx3-ubyte");
 	ReadMNIST_Label(NumberOfImages,train_label, "G:\\train-labels.idx1-ubyte");
	ReadMNIST(NumberOfImages/6,length,test_x,"G:\\t10k-images.idx3-ubyte");
	ReadMNIST_Label(NumberOfImages/6,test_label, "G:\\t10k-labels.idx1-ubyte");

    // constructor CNN
	LayerBuilder builder;
    Layer layer;
    builder.addLayer(layer.buildInputLayer(size(28,28)));
    builder.addLayer(layer.buildConvLayer(6, size(5, 5)));
    builder.addLayer(layer.buildSampLayer( size(2, 2)));
    builder.addLayer(layer.buildConvLayer(12, size(5, 5)));
    builder.addLayer(layer.buildSampLayer( size(2, 2)));
    //builder.addLayer(layer.buildConvLayer(20, size(4, 4)));
    builder.addLayer(layer.buildOutputLayer(10));
    CNN cnn = CNN(builder, 10);// biuder batchsize
    for (int i=0; i<2; i++)
    {
		double t1 = cpu_time();
		cnn.train(train_x,train_label, NumberOfImages);
		double t2 = cpu_time();
		cout<<t2-t1<<" s"<<endl;
		cnn.test(test_x,test_label, NumberOfImages/6);
    }
	// delete data
	for (int i=0; i<NumberOfImages; i++)
	{
		delete []train_label[i];
		for (int j=0; j<28; j++)
		{
			delete []train_x[i][j];
		}
		delete []train_x[i];
	}

	for (int i=0; i<NumberOfImages/6; i++)
	{
		delete []test_label[i];
		for (int j=0; j<28; j++)
		{
			delete []test_x[i][j];
		}
		delete []test_x[i];
	}
	delete []train_label;
	delete []train_x;
	delete []test_x;
	delete []test_label;

	return 0;
}

