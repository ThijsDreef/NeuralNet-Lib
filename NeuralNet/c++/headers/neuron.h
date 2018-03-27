#ifndef NEURON__
#define NEURON__
#include <cstdlib>
#include <math.h>
#include <vector>

class Neuron
{
private:
  static const float learnRate = 0.15f;
  static const float momentum = 0.4f;
  float gradient;
  float output;
  int index;
  float transferFunction(float x);
  float sumDOW(std::vector<Neuron> & nextlayer);
  float derevativeTransferFunction(float x);

public:
  std::vector<float> outputWeights;
  std::vector<float> deltaWeights;
  Neuron (int numOutPuts, int index);
  void feedForward(std::vector<Neuron> & prevLayer);
  void calcHiddenGradients(std::vector<Neuron> & nextlayer);
  void calcOutputGradients(float target);
  void setOutput(float output);
  void updateInputWeights(std::vector<Neuron> & prevLayer);
  void setIndex(int index);
  void setGradient(float gradient);
  int getIndex();
  float getGradient();
  float getOutput();
  virtual ~Neuron ();
};
#endif
