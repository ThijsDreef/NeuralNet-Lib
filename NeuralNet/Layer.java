package game.NeuralNet;

/**
 * Created by Thijs Dreef on 12/10/2017.
 */
public class Layer
{
  public Neuron[] neurons;
  public Layer(int size, int numOutPuts)
  {
    neurons = new Neuron[size + 1];
    for (int i = 0; i <= size; i++)
      neurons[i] = new Neuron(numOutPuts, i);
    neurons[size].setOutput(1.0f);
  }
}
