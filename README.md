# Tree Survival Prediction

Author: Amir Abu Hani  


## Overview
This project preprocesses the Tree Survival Prediction dataset, builds an Artificial Neural Network (ANN) model using TensorFlow, trains the model, and saves the trained model.

## Required Libraries
- numpy
- pandas
- tensorflow
- sklearn

## Functionality
This project code includes five main parts:

### Part 1: Data Preprocessing
1. **Feature Selection**: 
    - Features (X): All columns except for `No`, `PlantDate`, `Harvest`, `Alive`, and `Event`.
    - Exclusions:
        - `Harvest`: Contains a majority of NaN values.
        - `Alive`: Directly connected to the target variable, causing overfitting.
    - Target Variable (Y): `Event` (0: harvested, 1: dead).
2. **Data Cleaning**: Handle NaN values in relevant columns.
3. **Encoding**: Use One Hot Encoder and categorical encoding.
4. **Splitting the Dataset**: Divide the dataset into training and testing sets.
5. **Feature Scaling**: Apply standardization.

### Part 2: Building the ANN
- **Architecture**:
    - Input Layer
    - Two Hidden Layers: Activation function - ReLU
    - Output Layer: Activation function - Sigmoid

### Part 3: Training the ANN
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Parameters**: 
    - Batch Size: 64
    - Epochs: 30
- **Accuracy**: Between 85% and 92%.
- **Notes**: After many running attempts, I discovered that batch size = 64 and epochs = 30 are suitable for the model to get the optimal result and avoid overfitting.

### Part 4: Saving the Model
- Save the model to `trained_model.h5`.

### Part 5: Making Predictions
- Run the test set and compare `y_test` with `y_pred`.

## How to Run the Code
1. Execute Parts 1, 2, and 3 together to preprocess data, build, and train the model.
2. Run Part 4 to save the trained model.
3. Run Part 5 to predict the test set.

Alternatively:
1. Load the saved model: 
    ```python
    model = tf.keras.models.load_model('trained_model.h5')
    ```
2. Make predictions on the test set:
    ```python
    y_pred = model.predict(x_test)
    ```
