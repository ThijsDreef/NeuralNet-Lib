#ifndef LAYER__
#define LAYER__
#include "neuron.h"

class Layer
{
public:
  std::vector<Neuron> neurons;
  Layer (unsigned int size, unsigned int numOutPuts);
  virtual ~Layer ();
};
#endif
