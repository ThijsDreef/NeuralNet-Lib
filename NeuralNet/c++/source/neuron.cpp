#include "../headers/neuron.h"

Neuron::Neuron(int numOutPuts, int index)
{
  this->index = index;
  outputWeights.reserve(numOutPuts);
  deltaWeights.reserve(numOutPuts);
  for (unsigned int i = 0; i < numOutPuts; i++)
  {
    outputWeights.push_back(rand() / (float)RAND_MAX);
    deltaWeights.push_back(0);
  }
}

void Neuron::feedForward(std::vector<Neuron> & prevLayer)
{
  float sum = 0;
  for (unsigned int n = 0; n < prevLayer.size(); n++)
    sum += prevLayer[n].getOutput() * prevLayer[n].outputWeights[index];
  output = transferFunction(sum);
}

void Neuron::calcHiddenGradients(std::vector<Neuron> & nextLayer)
{
  float dow = sumDOW(nextLayer);
  gradient = dow * derevativeTransferFunction(output);
}

void Neuron::calcOutputGradients(float target)
{
  float delta = target - output;
  gradient = delta * derevativeTransferFunction(output);
}

void Neuron::setOutput(float output)
{
  this->output = output;
}

float Neuron::transferFunction(float x)
{
  return tanhf(x);
}

float Neuron::sumDOW(std::vector<Neuron> & nextLayer)
{
  float sum = 0.0f;
  for (unsigned int n = 0; n < nextLayer.size(); n++)
    sum += outputWeights[n] * nextLayer[n].gradient;
  return sum;
}

float Neuron::derevativeTransferFunction(float x)
{
  return 1.0f - x * x;
}

void Neuron::updateInputWeights(std::vector<Neuron> & prevLayer)
{
  for (unsigned int n = 0; n < prevLayer.size(); n++)
  {
    Neuron & neuron = prevLayer[n];
    float oldDeltaWeight = neuron.deltaWeights[index];
    float newDeltaWeight = learnRate * neuron.getOutput() * gradient + momentum * oldDeltaWeight;
    neuron.deltaWeights[index] = newDeltaWeight;
    neuron.outputWeights[index] += newDeltaWeight;
  }
}

void Neuron::setGradient(float gradient)
{
  this->gradient = gradient;
}

int Neuron::getIndex()
{
  return index;
}

float Neuron::getGradient()
{
  return gradient;
}

float Neuron::getOutput()
{
  return output;
}

Neuron::~Neuron()
{

}
