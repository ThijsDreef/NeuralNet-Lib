#ifndef NEURON__
#define NEURON__
#include "layer.h"
#include <rand>
#include <math.h>

class Neuron
{
private:
  float learnRate = 0.15f;
  float momentum = 0.4f;
  float gradient;
  float output;
  int index;
  float transferFunction(float x);
  float sumDOW(Layer nextlayer);
  float derevativeTransferFunction(float x);

public:
  std::vector<float> outputWeights;
  std::vector<float> deltaWeights;
  Neuron (int numOutPuts, int index);
  void feedForward(Layer & prevLayer);
  void calcHiddenGradients(Layer & nextlayer);
  void calcOutputGradients(float target);
  void setOutput(float output);
  void updateInputWeights(Layer & prevLayer);
  void setIndex(int index);
  void getGradient();
  int getIndex();
  float getGradient();
  float getOutput();
  virtual ~Neuron ();
};
#endif
