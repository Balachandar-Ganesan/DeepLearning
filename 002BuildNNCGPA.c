#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.0001
#define EPOCHS 8000

typedef struct {
    double weights_ih[HIDDEN_SIZE][INPUT_SIZE];
    double bias_h[HIDDEN_SIZE];
    double weights_ho[OUTPUT_SIZE][HIDDEN_SIZE];
    double bias_o[OUTPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    double z_hidden[HIDDEN_SIZE];
    double z_output[OUTPUT_SIZE];
} NeuralNetwork;

double relu(double x) {
    return (x > 0) ? x : 0;
}

double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

void init_network(NeuralNetwork *nn) {
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->weights_ih[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
        nn->bias_h[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->weights_ho[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
        nn->bias_o[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
}

void forward_propagation(NeuralNetwork *nn, double input[INPUT_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->z_hidden[i] = nn->bias_h[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->z_hidden[i] += nn->weights_ih[i][j] * input[j];
        }
        nn->hidden[i] = relu(nn->z_hidden[i]);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->z_output[i] = nn->bias_o[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->z_output[i] += nn->weights_ho[i][j] * nn->hidden[j];
        }
        nn->output[i] = nn->z_output[i];
    }
}

void backward_propagation(NeuralNetwork *nn, double input[INPUT_SIZE],
                         double target[OUTPUT_SIZE]) {
    double output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = nn->output[i] - target[i];
    }
    
    double hidden_error[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_error[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error[i] += output_error[j] * nn->weights_ho[j][i];
        }
        hidden_error[i] *= relu_derivative(nn->z_hidden[i]);
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->weights_ho[i][j] -= LEARNING_RATE * output_error[i] * nn->hidden[j];
        }
        nn->bias_o[i] -= LEARNING_RATE * output_error[i];
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->weights_ih[i][j] -= LEARNING_RATE * hidden_error[i] * input[j];
        }
        nn->bias_h[i] -= LEARNING_RATE * hidden_error[i];
    }
}

void normalize_data(double data[][INPUT_SIZE], int num_samples,
                   double min[INPUT_SIZE], double max[INPUT_SIZE]) {
    for (int j = 0; j < INPUT_SIZE; j++) {
        min[j] = data[0][j];
        max[j] = data[0][j];
        for (int i = 1; i < num_samples; i++) {
            if (data[i][j] < min[j]) min[j] = data[i][j];
            if (data[i][j] > max[j]) max[j] = data[i][j];
        }
    }
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
        }
    }
}

void train(NeuralNetwork *nn, double inputs[][INPUT_SIZE],
          double targets[][OUTPUT_SIZE], int num_samples) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < num_samples; i++) {
            forward_propagation(nn, inputs[i]);
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                double error = nn->output[j] - targets[i][j];
                total_loss += error * error;
            }
            backward_propagation(nn, inputs[i], targets[i]);
        }
        total_loss /= num_samples;
        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss);
        }
    }
}

int main() {
    double training_inputs[][INPUT_SIZE] = {
        {1, 2, 3}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2},
        {3, 2, 1}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}
    };
    double training_targets[][OUTPUT_SIZE] = {
        {1.77}, {1.88}, {2.11}, {2.11}, {2.22}, {1}, {2}, {3}
    };
    int num_samples = 8;
    
    double min[INPUT_SIZE], max[INPUT_SIZE];
    normalize_data(training_inputs, num_samples, min, max);
    
    double target_min = training_targets[0][0];
    double target_max = training_targets[0][0];
    for (int i = 1; i < num_samples; i++) {
        if (training_targets[i][0] < target_min) target_min = training_targets[i][0];
        if (training_targets[i][0] > target_max) target_max = training_targets[i][0];
    }
    for (int i = 0; i < num_samples; i++) {
        training_targets[i][0] = (training_targets[i][0] - target_min) / 
                                 (target_max - target_min);
    }
    
    NeuralNetwork nn;
    init_network(&nn);
    
    printf("Training Neural Network for CGPA Prediction\n");
    printf("Architecture: %d    %d    %d\n\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    train(&nn, training_inputs, training_targets, num_samples);
    
    printf("\n=== Testing Predictions ===\n");
    for (int i = 0; i < num_samples; i++) {
        forward_propagation(&nn, training_inputs[i]);
        double predicted = nn.output[0] * (target_max - target_min) + target_min;
        double actual = training_targets[i][0] * (target_max - target_min) + target_min;
        printf("Sample %d: Predicted %.2f, Actual %.2f\n", i+1, predicted, actual);
    }
    
    printf("\n=== New Prediction ===\n");
    double new_cgpa[INPUT_SIZE] = {3, 2, 2};
    for (int j = 0; j < INPUT_SIZE; j++) {
        new_cgpa[j] = (new_cgpa[j] - min[j]) / (max[j] - min[j]);
    }
    forward_propagation(&nn, new_cgpa);
    double prediction = nn.output[0] * (target_max - target_min) + target_min;
    printf("CGPA: 3 , 2 , 2 \n");
    printf("Predicted CGPA: %.2f\n", prediction);
    
    return 0;
}
