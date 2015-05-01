#include "Layer.h"
#include "util.h"
Layer Layer:: buildInputLayer(size mapsize)
{
	Layer layer;
	layer.type = 'I';
	layer.outMapNum = 1;
	layer.mapSize = mapsize;
	return layer;
}

Layer Layer::buildConvLayer(int outMapNum, size kernelSize)
{
	Layer layer;
	layer.type = 'C';
	layer.outMapNum = outMapNum;
	layer.kernelSize = kernelSize;
	return layer;
}

Layer Layer::buildSampLayer(size scaleSize)
{
	Layer layer;
	layer.type = 'S';
	layer.scaleSize = scaleSize;
	return layer;
}

Layer Layer::buildOutputLayer(int classNum)
{
	Layer layer;
	layer.classNum = classNum;
	layer.type = 'O';
	layer.mapSize = size(1,1);
	layer.outMapNum = classNum;
	return layer;

}


void Layer::initKernel(int frontMapNum)
{
	kernel = new double***[frontMapNum];
	for (int i =0; i<frontMapNum;i++)
	{
		kernel[i]= new double**[outMapNum];
		for (int j=0; j<outMapNum; j++)
		{
			kernel[i][j] = new double*[kernelSize.x];
			for(int ii=0; ii<kernelSize.x; ii++)
			{
				kernel[i][j][ii] = new double[kernelSize.y];
			}
			randomMatrix(kernelSize.x,kernelSize.y,kernel[i][j]);
		}
	}
	
}

void Layer::initOutputKernel(int frontMapNum, size s)
{
	kernelSize = s;
	kernel = new double***[frontMapNum];
	for (int i =0; i<frontMapNum;i++)
	{
		kernel[i]=new double**[outMapNum];
		for (int j=0; j<outMapNum; j++)
		{
			kernel[i][j] = new double*[kernelSize.x];
			for(int ii=0; ii<kernelSize.x; ii++)
			{
				kernel[i][j][ii] = new double[kernelSize.y];
			}
			randomMatrix(kernelSize.x,kernelSize.y, kernel[i][j]);
		}
	}
	
}

void Layer::initErros(int batchSize)
{
	errors = new double***[batchSize];
	for (int i=0;i<batchSize;i++)
	{
		errors[i] = new double**[outMapNum];
		for (int m=0; m<outMapNum; m++)
		{
			errors[i][m] = new double*[mapSize.x];
			for (int n=0; n<mapSize.x;n++)
			{
				errors[i][m][n] = new double[mapSize.y];
			}
		}
	}
}

void Layer::setError(int num,int mapNo, int mapX, int mapY, double v)
{
	errors[num][mapNo][mapX][mapY] = v;
}

void Layer::setError(int numBatch, int mapNo, double** matrix, int m, int n)
{
	for(int i=0; i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			errors[numBatch][mapNo][i][j] = matrix[i][j];
		}
	}
}