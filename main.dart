class Neuron {
  final List<double> weights;
  final double biasWeight;

  late double z;

  @override
  String toString() => '$biasWeight / $weights';

  Neuron(this.weights, this.biasWeight);

  double eval(List<double> inputs, Network network, Layer layer,
      {bool debug = false}) {
    // Take the network's bias, add each of the weighted inputs, and use the activation function to get the result.
    if (inputs.length != weights.length) {
      throw StateError(
          'Neuron has ${weights.length} weights but got ${inputs.length} inputs.');
    }
    double result = layer.bias * biasWeight;
    int i = 0;
    while (i < inputs.length) {
      result += weights[i] * inputs[i];
      i++;
    }
    z = network.activate(result);
    return z;
  }

  void updateWeights(double delta, List<double> prevZs, double learningRate) {
    int i = 0;
    assert(weights.length == prevZs.length);
    while (i < weights.length) {
      weights[i] = weights[i] - learningRate * prevZs[i] * delta;
      i++;
    }
  }
}

class Layer {
  // a non-input layer
  final List<Neuron> neurons;
  final double bias;

  @override
  String toString() => '($bias+$neurons)';

  Layer(this.neurons, this.bias);

  List<double> run(List<double> inputs, Network network, {bool debug = false}) {
    // For each neuron, compute its result, and return their results in a list.
    List<double> outputs = [];
    for (Neuron neuron in neurons) {
      outputs.add(neuron.eval(inputs, network, this, debug: debug));
    }
    return outputs;
  }

  void updateWeights(
      List<double> prevZs, List<double> deltas, double learningRate) {
    int i = 0;
    assert(deltas.length == neurons.length);
    for (Neuron neuron in neurons) {
      neuron.updateWeights(deltas[i], prevZs, learningRate);
      i++;
    }
  }
}

class Network {
  final List<Layer> layers;

  Network(this.layers);

  double activate(double value) {
    // This is the activation function, which converts a number to a more reasonable one.
    // This function is just the identity function.

    return value;
  }

  double activateDerivative(double value) {
    // The derivative of activate
    // This is the derivative of x, according to WolframAlpha.
    return 1;
  }

  List<double> run(List<double> inputs, {bool debug = false}) {
    // For each layer, compute its result and pass it on to the next.
    List<double> results = inputs;
    for (Layer layer in layers) {
      results = layer.run(results, this, debug: debug);
    }
    return results;
  }

  void backPropagate(TrainingSample sample, double learningRate,
      {bool debug = false}) {
    List<double> predictedOutput = run(sample.inputs);
    List<double> losses = [];
    int i = 0;
    assert(predictedOutput.length == sample.outputs.length);
    while (i < sample.outputs.length) {
      losses.add(-(sample.outputs[i] - predictedOutput[i]));
      i++;
    }
    i = layers.length - 1;
    List<double> deltas = losses;
    for (Layer layer in layers.reversed) {
      List<double> prevZs;
      if (i == 0) {
        prevZs = sample.inputs;
      } else {
        prevZs = layers[i - 1].neurons.map((e) => e.z).toList();
      }
      List<double> newDeltas = [];
      for (Neuron neuron in layer.neurons) {
        int j = 0;
        List<double> weightedDeltas = [];
        while (j < neuron.weights.length) {
          weightedDeltas.add(neuron.weights[j] *
              (deltas.reduce(
                      (previousValue, element) => previousValue + element) /
                  deltas.length) *
              activateDerivative(neuron.z));
          j++;
        }
        newDeltas.add(weightedDeltas
                .reduce((previousValue, element) => previousValue + element) /
            weightedDeltas.length);
      }
      deltas = newDeltas;
      layer.updateWeights(prevZs, deltas, learningRate);
      i--;
    }
  }

  void learn(List<TrainingSample> samples, double learningRate) {
    for (TrainingSample sample in samples) {
      backPropagate(sample, learningRate);
    }
  }
}

class TrainingSample {
  final List<double> inputs;
  final List<double> outputs;

  const TrainingSample(this.inputs, this.outputs);
}

void main() {
  Network network = Network(
    [
      Layer(
        [
          Neuron([.01], .01),
          Neuron([.01], .01),
          Neuron([.01], .01),
          Neuron([.01], .01),
        ],
        .5,
      ),
      Layer(
        [
          Neuron([.01, .01, -.01, -.01], .01),
          Neuron([.01, .01, .01, .01], .01),
          Neuron([.01, .01, .01, .01], .01),
          Neuron([.01, .01, .01, .01], .01),
        ],
        .5,
      ),
      Layer(
        [
          Neuron([.01, .01, -.01, -.01], .01),
          Neuron([.01, .01, .01, .01], .01),
          Neuron([.01, .01, .01, .01], .01),
          Neuron([.01, .01, .01, .01], .01),
        ],
        .5,
      ),
      Layer(
        [
          Neuron([.01, .01, .01, .01], .01),
        ],
        0,
      )
    ],
  );
  List<TrainingSample> samples = List.generate(
      100, (index) => TrainingSample([index / 100], [1 - index / 100]));
  for (int _ in List.generate(1000, (index) => 1)) {
    network.learn(
      samples,
      100,
    );
  }
  print('after training');
  for (TrainingSample sample in samples) {
    print('  inputs: ${sample.inputs}');
    print('  predicted: ${network.run(sample.inputs)}');
    print('  actual: ${sample.outputs}');
    print('');
  }
}
