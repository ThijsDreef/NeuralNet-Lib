#include "../headers/network.h"

Network::Network(std::string file)
{
  // load(file);
}

Network::Network(std::vector<int> topology)
{
  this->topology = topology;
  layers.reserve(topology.size());
  for (unsigned int i = 0; i < topology.size(); i++)
    layers.push_back(Layer(topology[i], (i == topology.size() - 1) ? 0 : topology[i + 1]));
}

Network::~Network()
{

}

void Network::feedForward(std::vector<float> input)
{
    if (input.size() != layers[0].neurons.size())
      return;
    for (unsigned int i = 0; i < input.size(); i++)
      layers[0].neurons[i].setOutput(input[i]);

    for (unsigned int layernum = 1; layernum < layers.size(); layernum++)
      for (unsigned int n = 0; layers[layernum].neurons.size() - 1; n++)
        layers[layernum].neurons[n].feedForward(layers[layernum - 1].neurons);
}

void Network::backProp(std::vector<float> target)
{
  Layer & output = layers[layers.size() -1];

  error = 0;
  for (unsigned int n = 0; n < output.neurons.size() -1; n++)
  {
    float delta = target[n] - output.neurons[n].getOutput();
    error += delta * delta;
  }
  error /= output.neurons.size() -1;
  error = sqrt(error);
  recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0f);

  for (unsigned int n = 0; n < output.neurons.size() - 1; n++)
    output.neurons[n].calcOutputGradients(target[n]);

  for (unsigned int layernum = layers.size() - 2; layernum > 0; layernum--)
  {
    Layer & hiddenLayer = layers[layernum];
    Layer & nextLayer = layers[layernum + 1];
    for (unsigned int n = 0; n < hiddenLayer.neurons.size(); n++)
      hiddenLayer.neurons[n].calcHiddenGradients(nextLayer.neurons);
  }

  for (unsigned int layernum = layers.size() - 1; layernum > 0; layernum--)
  {
    Layer & layer = layers[layernum];
    Layer & prevLayer = layers[layernum - 1];
    for (unsigned int n = 0; n < layer.neurons.size(); n++)
      layer.neurons[n].updateInputWeights(prevLayer.neurons);
  }
}

std::vector<float> Network::getResults()
{
  std::vector<float> results;
  results.reserve(layers[layers.size() - 1].neurons.size() - 2);
  for (unsigned int n = 0; n < layers[layers.size() - 1].neurons.size() - 1; n++)
    results.push_back(layers[layers.size() -1].neurons[n].getOutput());
}

std::string Network::vecToString(std::vector<int> top)
{
  std::string result = "[";
  for (unsigned int i = 0; i < top.size(); i++)
    result += (top[i]) + ", ";
  result += "]";
  return result;
}

// void Network::save(std::string filename)
// {
  // std::fstream file (filename, std::ios::in | std::ios::out | std::ios::trunc);
  // if (!file.is_open());
  //   return;
  // file << vecToString(topology) << "\r\n";
  // for (unsigned int layernum = 0; layernum < layers.size(); layernum++)
  // {
  //   file << "layer: " << layernum << "\r\n";
  //   for (unsigned int n = 0; n < layers[layernum].neurons.size(); n++)
  //   {
  //     Neuron & neuron = layers[layernum].neurons[n];
  //     file << "index: " << neuron.getIndex() << "\r\n";
  //     file << "gradient: " << neuron.getGradient() << "\r\n";
  //     file << "output: " << neuron.getOutput() << "\r\n";
  //     for (unsigned int i = 0; i < neuron.outputWeights.size(); i++)
  //       file << "weight: " << neuron.outputWeights[i] << "\r\n";
  //     for (unsigned int i = 0; i < neurons.deltaWeights.size(); i++)
  //       file << "deltaWeights: " << neuron.deltaWeights[i] << "\r\n";
  //   }
  // }
  // file.close();
// }

// std::vector<std::string> Network::split(std::string str,std::string sep){
    // char* cstr=const_cast<char*>(str.c_str());
    // char* current;
    // std::vector<std::string> arr;
    // current=strtok(cstr,sep.c_str());
    // while(current!=NULL){
    //     arr.push_back(current);
    //     current=strtok(NULL,sep.c_str());
    // }
    // return arr;
    // return std::vector<std::string>();
// }

// void load(std::string filename, int & percentage)
// {
  // std::fstream file(filename, std::ios::in | std::ios::out | std::ios::app);
  // std::string line = "";
  // std::getline(file, line);
  // line = line.substr(1, line.size() - 2);
  // std::vector<std::string> vals = split(line, ", ");
  // int totalNeurons = 0;
  // std::vector<int> top;
  // for (unsigned int i = 0; i < vals.size(); i++)
  // {
  //   top.push_back(atoi(vals[i]));
  //   totalNeurons += top[i];
  // }
  // layers.reserve(topology.length);
  // for (unsigned int i = 0; i < topology.size(); i ++)
  //   layers.push_back(topology[i], (i == topology.size() - 1) ? 0 : topology[i + 1]);
  // int layer = -1;
  // int index = -1;
  // int outputIndex = 0;
  // int deltaIndex = 0;
  // int processedNeurons = 0;
  // double f;
  //
  // while (std::getline(file, line))
  // {
  //   percentage = processedNeurons / maxNeurons;
  //   if (line.substr(0 , 1) == "l")
  //     sscanf(line, "%d", &layer);
  //   if (line.substr(0, 1) == "i")
  //   {
  //     processedNeurons++;
  //     sscanf(line, "%d", &index);
  //     outputIndex = 0;
  //     deltaIndex = 0;
  //   }
  //   if (line.substr(0, 1) == "g")
  //   {
  //     sscanf(line, "%f", &f);
  //     layers[layer].neurons[index].setGradient((float)f);
  //   }
  //   if (line.substr(0, 1) == "o")
  //   {
  //     sscanf(line, "%f" &f)
  //     layers[layer].neurons[index].setOutput((float)f);
  //   }
  //   if (line.substr(0, 1) == "w")
  //   {
  //     sscanf(line, "%f", &f);
  //     layers[layer].neurons[index].outputWeights[outputIndex] = f;
  //     outputIndex ++;
  //   }
  //   if (line.substr(0, 1) == "d")
  //   {
  //     sscanf(line, "%f", &f);
  //     layers[layer].neurons[index].deltaWeights[outputIndex] = f;
  //     outputIndex ++;
  //   }
  // }

// }
