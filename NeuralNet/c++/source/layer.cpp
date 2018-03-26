#include "../headers/layer.h"

Layer::Layer(unsigned int size, unsigned int numOutPuts)
{
  neurons.reserve(size + 1);
  for (unsigned int i = 0; i <= size; i++)
    neurons.push_back(Neuron(numOutPuts, i));
  neurons[size].setOutput(1.0f);
}

Layer::~Layer()
{
  
}
