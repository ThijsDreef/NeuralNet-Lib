#ifndef NETWORK__
#define NETWORK__
#include "neuron.h"
#include "layer.h"
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

class Network
{
private:
  std::string vecToString(std::vector<int> top);
public:
  float vEpoch;
  float error;
  float recentAverageError;
  float recentAverageSmoothingFactor;
  long runTime;
  std::vector<int> topology;
  std::vector<Layer> layers;
  int loadPercentage;
  void feedForward(std::vector<float> input);
  void backProp(std::vector<float> target);
  std::vector<float> getResults();
  std::vector<std::string>> split(std::string str, std::string sep);
  void trainBackProp(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> target, float minEpochError)
  void save(std::string filename);
  void load(std::string filename);
  Network (std::string file);
  Network (std::vector<int> topology);
  virtual ~Network ();
};
#endif
