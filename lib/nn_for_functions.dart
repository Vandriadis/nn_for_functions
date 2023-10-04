import 'dart:math';
import 'package:dart_numerics/dart_numerics.dart';

void main() {
  final xs = List.generate(100, (i) => [i * 0.1, sin(i * 0.1)]); 
  final ys = xs.map((x) => 2 * x[1]).toList(); 
  final model = LinearRegressionModel(2); 

  final epochs = 10000;
  final learningRate = 0.01;

  for (int epoch = 0; epoch < epochs; epoch++) {
    model.fit(xs, ys, learningRate: learningRate);

    if (epoch % 1000 == 0) {
      final mse = LinearRegressionModel.meanSquaredError(ys, model.predict(xs));
      final mae = LinearRegressionModel.meanAbsoluteError(ys, model.predict(xs));
      print("Epoch: $epoch, Mean squared error: $mse, Mean absolute error: $mae");
    }
  }

  // Train the model
  model.fit(xs, ys, epochs: 10000, learningRate: 0.01);

  // Evaluate the model on the training data
  final mse = LinearRegressionModel.meanSquaredError(ys, model.predict(xs));
  final mae = LinearRegressionModel.meanAbsoluteError(ys, model.predict(xs));

  print("Mean squared error: $mse");
  print("Mean absolute error: $mae");
}

class LinearRegressionModel {
  final int inputSize;
  late List<double> weights;
  late double bias;

  LinearRegressionModel(this.inputSize) {
    weights = List<double>.generate(inputSize, (_) => Random().nextDouble());
    bias = Random().nextDouble();
  }

  void fit(List<List<double>> xs, List<double> ys,
      {int epochs = 1000, double learningRate = 0.01}) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      // Calculate the predictions
      final ysPred = predict(xs);

      // Update the weights and bias using gradient descent
      final gradient = gradientMeanSquaredError(xs, ys, ysPred, inputSize);
      for (int i = 0; i < inputSize; i++) {
        weights[i] -= learningRate * gradient[i];
      }
      bias -= learningRate * gradient[inputSize];
    }
  }

  List<double> predict(List<List<double>> xs) {
    final predictions = <double>[];
    for (int i = 0; i < xs.length; i++) {
      double prediction = 0;
      for (int j = 0; j < inputSize; j++) {
        prediction += weights[j] * xs[i][j];
      }
      prediction += bias;
      predictions.add(prediction);
    }
    return predictions;
  }

  static double meanSquaredError(List<double> actual, List<double> predicted) {
    double sum = 0;
    for (int i = 0; i < actual.length; i++) {
      final error = actual[i] - predicted[i];
      sum += error * error;
    }
    return sum / actual.length;
  }

  static List<double> gradientMeanSquaredError(
      List<List<double>> xs, List<double> ys, List<double> ysPred, int inputSize) {
    final gradient = List<double>.filled(inputSize + 1, 0.0);
    for (int i = 0; i < xs.length; i++) {
      final error = ysPred[i] - ys[i];
      for (int j = 0; j < inputSize; j++) {
        gradient[j] += 2 * error * xs[i][j];
      }
      gradient[inputSize] += 2 * error;
    }
    for (int j = 0; j < inputSize; j++) {
      gradient[j] /= xs.length;
    }
    gradient[inputSize] /= xs.length;
    return gradient;
  }

  static double meanAbsoluteError(List<double> actual, List<double> predicted) {
    double sum = 0;
    for (int i = 0; i < actual.length; i++) {
      final error = actual[i] - predicted[i];
      sum += error.abs();
    }
    return sum / actual.length;
  }
}
