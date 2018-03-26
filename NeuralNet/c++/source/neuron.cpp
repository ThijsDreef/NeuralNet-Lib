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

void Neuron::feedForward(Layer & prevLayer)
{
  float sum = 0;
  for (unsigned int n = 0; n < prevLayer.neurons.size(); n++)
    sum += prevLayer.neurons[n].getOutput() * prevLayer.neurons[n].getOutputWeights[index];
  output = transferFunction(sum);
}

void Neuron::calcHiddenGradients(Layer & nextLayer)
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

float Neuron::getOutput()
{
  return output;
}

float transferFunction(float x)
{
  return tanhf(x);
}

float Neuron::sumDOW(Layer & nextLayer)
{
  float sum = 0.0f;
  for (unsigned int n = 0; n < nextLayer.neurons.size(); n++)
    sum += outputWeights[n] * nextLayer.neurons[n].gradient;
  return sum;
}

float Neuron::derevativeTransferFunction(float x)
{
  return 1.0f - x * x;
}

void updateInputWeights(Layer & prevLayer)
{
  for (unsigned int n = 0; n < prevLayer.neurons.size(); n++)
  {
    Neuron & neuron = prevLayer.neurons[n];
    float oldDeltaWeight = neuron.deltaWeights[index];
    float newDeltaWeight = learnRate * neuron.getOutput() * gradient + momentum * oldDeltaWeight;
    neuron.deltaWeights[index] = newDeltaWeight;
    neuron.outputWeights[index] += newDeltaWeight;
  }
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
