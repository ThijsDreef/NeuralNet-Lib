package game.NeuralNet;

public class Neuron
{
  final float learnRate = 0.15f;
  final float momentum = 0.6f;
  private float output;
  private int index;
  private float [] outputWeights;
  private float [] deltaWeights;
  private float gradient;
  public Neuron(int numOutPuts, int index)
  {
    this.index = index;
    outputWeights = new float[numOutPuts];
    deltaWeights = new float[numOutPuts];
    for (int i = 0; i < outputWeights.length; i++)
      outputWeights[i] = (float)Math.random() * 2 - 1;
  }
  public void feedForward(Layer prevLayer)
  {
    float sum = 0;
    for (int n = 0; n < prevLayer.neurons.length; n++)
    {
      sum += prevLayer.neurons[n].getOutput() * prevLayer.neurons[n].outputWeights[index];
    }
    output = transferFunction(sum);
  }
  public void calcHiddenGradients(Layer nextLayer)
  {
    float dow = sumDOW(nextLayer);
    gradient = dow * derevativeTransferFunction(output);
  }
  public void calcOutputGradients(float target)
  {
    float delta = target - output;
    gradient = delta * derevativeTransferFunction(output);
  }
  public void setOutput(float output)
  {
    this.output = output;
  }
  public float getOutput()
  {
    return output;
  }
  private float transferFunction(float x)
  {
    return (float)Math.tanh(x);
  }
  private float sumDOW(Layer nextLayer)
  {
    float sum = 0.0f;
    for (int n = 0; n < nextLayer.neurons.length -1; n++)
      sum += outputWeights[n] * nextLayer.neurons[n].gradient;
    return sum;

  }
  private float derevativeTransferFunction(float x)
  {
    return 1.0f -  x* x;
  }
  public void updateInputWeights(Layer prevLayer)
  {
    for (int n = 0; n < prevLayer.neurons.length; n++)
    {
      Neuron neuron = prevLayer.neurons[n];
      float oldDeltaWeight = neuron.deltaWeights[index];
      float newDeltaWeight = learnRate * neuron.getOutput() * gradient + momentum * oldDeltaWeight;
      // individual input, magnified by the gradiant and train rate
      neuron.deltaWeights[index] = newDeltaWeight;
      neuron.outputWeights[index] += newDeltaWeight;
    }
  }
  public final float[] getOutputWeights()
  {
    return outputWeights;
  }
  public final float[] getDeltaWeights() {return deltaWeights;}

  public int getIndex()
  {
    return index;
  }

  public void setIndex(int index)
  {
    this.index = index;
  }

  public float getGradient()
  {
    return gradient;
  }

  public void setGradient(float gradient)
  {
    this.gradient = gradient;
  }
  public float [] writableOutputWeights() {return outputWeights; }
  public float [] writableDeltaWeights() {return  deltaWeights;}
}
