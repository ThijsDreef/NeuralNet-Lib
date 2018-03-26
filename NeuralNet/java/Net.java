package game.NeuralNet;

import java.io.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

public class Net
{
  public float visualEpoch = 1;
  float error;
  float recentAverageError;
  float recentAverageSmoothingFactor;
  public int runtime;
  public int [] topology;
  public Layer [] layers;
  public int loadPercentage = 0;
  public Net(String file)
  {
    load(file);
  }
  public Net(int[] topology)
  {
    this.topology =topology;
    layers = new Layer[topology.length];
    for (int i = 0; i < topology.length; i++)
      layers[i] = new Layer(topology[i], (i == topology.length - 1) ? 0 : topology[i + 1]);

  }
  public void feedForward(float[] input)
  {
    if (input.length != layers[0].neurons.length - 1)
    {
      System.out.println("input length: " + input.length + " neuron length: " + layers[0].neurons.length + " mismatch input");
      return;
    }
    //assigning input neurons what to feed forward
    for (int i = 0; i < input.length; i++)
      layers[0].neurons[i].setOutput(input[i]);
    // forward prop
    for (int layernum = 1; layernum < layers.length; layernum++)
    {
      for (int n = 0; n < layers[layernum].neurons.length - 1; n++)
      {
        layers[layernum].neurons[n].feedForward(layers[layernum - 1]);
      }
    }

  }
  public void backProp(float[] target)
  {
    // calculate overall net error rms of output neuron errors
    Layer output = layers[layers.length - 1];

    error = 0;

    for (int n = 0; n < output.neurons.length - 1; n++)
    {
      float delta = target[n] - output.neurons[n].getOutput();
      error += delta * delta;
    }
    error /= output.neurons.length -1;
    error = (float)Math.sqrt(error);

    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0f);
    // calculate output layer gradients

    for (int n = 0; n < output.neurons.length - 1; n++)
    {
      output.neurons[n].calcOutputGradients(target[n]);
    }
    // calculate gradients on hidden layers
    for (int layernum = layers.length - 2; layernum > 0; layernum--)
    {
      Layer hiddenLayer = layers[layernum];
      Layer nextLayer = layers[layernum + 1];
      for (int n = 0; n < hiddenLayer.neurons.length; n++)
        hiddenLayer.neurons[n].calcHiddenGradients(nextLayer);
    }

    // for all layers from outputs to first hidden layer

    //update weights
    for (int layernum = layers.length - 1; layernum > 0; layernum--)
    {
      Layer layer = layers[layernum];
      Layer prevLayer = layers[layernum - 1];
      for (int n = 0; n < layer.neurons.length -1; n++)
        layer.neurons[n].updateInputWeights(prevLayer);
    }
  }
  public float[] getResults()
  {
    float[] result = new float[layers[layers.length - 1].neurons.length - 1];
    for (int n = 0; n < layers[layers.length  -1].neurons.length - 1; n++)
      result[n] = layers[layers.length - 1].neurons[n].getOutput();
    return result;
  }
  public void train(float[][] input, float[][] target, float minEpochError)
  {
    float epochError = 1;
    double timeSinceStarted = System.nanoTime() - (runtime * 1000000000.0);
    while (epochError > minEpochError)
    {
      if (Thread.currentThread().isInterrupted())
        break;
      epochError = 0;
      for (int i = 0; i < input.length; i++)
      {

        feedForward(input[i]);
        backProp(target[i]);
        epochError += recentAverageError;
        runtime = (int)((System.nanoTime() - timeSinceStarted) / 1000000000.0);

      }
      visualEpoch = epochError / input.length;
    }
    visualEpoch = epochError / input.length;
  }
  public void save()
  {
    // format not the best but im able to live with it
    Date date = new Date();
    DateFormat format = new SimpleDateFormat("dd-MM-yyyy-HH-mm-ss");
    System.out.println();
    try
    {
      File file = new File(Arrays.toString(topology) + "-" + format.format(date) + ".nn");
      file.createNewFile();
      FileWriter output = new FileWriter(file);
      PrintWriter writer = new PrintWriter(output);
      //write out the topology for loading purposes
      writer.println(Arrays.toString(topology));
      for (int layernum = 0; layernum < layers.length; layernum++)
      {
        writer.println("layer: " + layernum);
        for (int n = 0; n < layers[layernum].neurons.length; n++)
        {
          writer.println("index: " + layers[layernum].neurons[n].getIndex());
          writer.println("gradient: " + layers[layernum].neurons[n].getGradient());
          writer.println("output: " + layers[layernum].neurons[n].getOutput());
          final float[] outputWeights = layers[layernum].neurons[n].getOutputWeights();
          for (int i = 0; i < outputWeights.length; i++)
            writer.println("weight: " + String.format("%f",outputWeights[i]));
          final float [] deltaWeights = layers[layernum].neurons[n].getDeltaWeights();
          for (int i = 0; i < deltaWeights.length; i++)
            writer.println("deltaWeights: " + String.format("%f", deltaWeights[i]));
        }
      }


      //cleaning up
      writer.close();
      output.close();
    } catch (IOException e)
    {
      e.printStackTrace();
    }
  }
  public void load(String file)
  {
    try
    {
      //please lord fix this parser its hell
      FileInputStream input = new FileInputStream(file);
      Scanner scanner = new Scanner(input);
      String line = scanner.nextLine();
      line = line.replace ("[", "");
      line = line.replace ("]", "");
      int maxNeurons = 0;
      String vals [] = line.split (", ");
      topology = new int[vals.length];
      for (int i = 0; i < vals.length; i ++)
      {
        topology[i] = Integer.parseInt(vals[i]);
        maxNeurons += topology[i];
      }
      layers = new Layer[topology.length];
      for (int i = 0; i < topology.length; i++)
        layers[i] = new Layer(topology[i], (i == topology.length - 1) ? 0 : topology[i + 1]);
      int layer = -1;
      int index = -1;
      int outputIndex = 0;
      int deltaIndex = 0;
      int processedNeurons = 0;
      while (scanner.hasNextLine())
      {
        loadPercentage = (int)(((float)processedNeurons / maxNeurons) * 100);
        line = scanner.nextLine();
        Scanner lineScanner = new Scanner(line);
        lineScanner.next();
        if (line.startsWith("l"))
          layer = lineScanner.nextInt();
        if (line.startsWith("i"))
        {
          processedNeurons++;
          index = lineScanner.nextInt();
          outputIndex = 0;
          deltaIndex = 0;
        }
        if (line.startsWith("g"))
          layers[layer].neurons[index].setGradient((float)lineScanner.nextDouble());
        if (line.startsWith("o"))
          layers[layer].neurons[index].setOutput((float)lineScanner.nextDouble());
        if (line.startsWith("w"))
        {
          layers[layer].neurons[index].writableOutputWeights()[outputIndex] = (float) lineScanner.nextDouble();
          outputIndex++;

        }
        if (line.startsWith("d"))
        {
          layers[layer].neurons[index].writableDeltaWeights()[deltaIndex] = (float) lineScanner.nextDouble();
          deltaIndex++;
        }

      }
    } catch (FileNotFoundException e)
    {
      e.printStackTrace();
    }
  }
}
