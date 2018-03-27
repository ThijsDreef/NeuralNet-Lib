#ifndef NETWORK__
#define NETWORK__
#include "neuron.h"
#include "layer.h"
#include <string.h>
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
  std::vector<std::string> split(std::string str, std::string sep);
  void save(std::string filename);
  void load(std::string filename, int & percentage);
  Network (std::string file);
  Network (std::vector<int> topology);
  virtual ~Network ();
};
#endif
