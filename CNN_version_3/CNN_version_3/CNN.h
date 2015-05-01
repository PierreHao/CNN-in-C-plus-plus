#ifndef CNN_H
#define CNN_H
#include "iostream"
#include "util.h"
#include "vector"
#include "Layer.h"
using namespace std;
typedef vector<Layer > layers;
//Builder some layers that you want
class LayerBuilder
{
public: 
	layers mLayers;

	LayerBuilder(){};
	LayerBuilder(Layer layer){
		mLayers.push_back(layer);
	}
	void addLayer(Layer layer)
	{
		mLayers.push_back(layer);
	}
};

class CNN
{
private:
	layers m_layers;
	int layerNum;
	int batchSize;
	double ALPHA;

public:
	CNN(LayerBuilder layerBuilder, int batchSize)
	{
		ALPHA = 1.0;
		m_layers =  layerBuilder.mLayers;
		layerNum = m_layers.size();
		this->batchSize = batchSize;
		setup(batchSize);
		//initPerator();
	};
	~CNN(){};
	void setBatchsize(int batchsize){
		this->batchSize = batchsize;
	}
	void train(double ***train_x,double **train_label, int numofimage);
	void test(double*** test_x,double** test_label, int numOfImage);
	
	void setup(int batchSize);// builder CNN with batchSize and initialize some parameters of each layer
	void forward(double*** x);
	void backPropagation(double*** x,double** y);
	//back-propagation
	void setOutLayerErrors(double*** x,double **y);
	void setHiddenLayerErrors();
	void setSampErrors(Layer &layer, Layer &nextLayer);
	void setConvErrors(Layer &layer, Layer &nextLayer);
	void updateKernels(Layer &layer, Layer &lastLayer);
	void updateBias(Layer &layer);
	void updateParas();
	//forward
	void setInLayerOutput(double*** x);
	void setConvOutput(Layer &layer, Layer &lastLayer);
	void setSampOutput(Layer &layer, Layer &lastLayer);
};


#endif