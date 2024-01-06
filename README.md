### Heart Disease Prediction Neural Network 
### Overview
This study focuses on building a neural network using PyTorch to predict the presence of heart disease based on various health-related features. The dataset used for training and evaluation is loaded from the "heart.csv" file. The target variable is binary, indicating whether a patient has heart disease or not.

### Dataset
The heart disease dataset undergoes the following preprocessing steps:

Removal of duplicate records.
Exploratory data analysis with pair plots and a correlation heatmap.
Handling categorical variables (Chest main type, thalassemia) using one-hot encoding.
Splitting the dataset into input features (X) and the target variable (y).
Performing oversampling using SMOTE to address class imbalance.
Standardizing numerical features using StandardScaler.
### Neural Network Architecture
The neural network architecture consists of the following layers:

Input Layer: Neurons equal to the number of input features.
Fully Connected (Linear) Layer with ReLU Activation and Batch Normalization.
Dropout Layer to prevent overfitting.
Output Layer with a single neuron and no activation function.
### Training
The neural network is trained with the following hyperparameters:

Learning Rate: 0.01
Epochs: 30
Batch Size: Entire dataset (no mini-batch training in this example)
The training process involves forward and backward passes, updating model parameters using the sigmoid activation function, and minimizing the cross-entropy loss.

### Evaluation
The trained model is evaluated on the test set. The evaluation metrics include accuracy, F1 score, and a classification report providing detailed information on precision, recall, and F1 score for each class.

### Results
The neural network demonstrates its effectiveness in predicting heart disease based on the provided health features. The chosen hyperparameters lead to satisfactory model performance, as indicated by the evaluation metrics.

### Next Steps
Potential next steps for further improvements could include experimenting with different architectures, conducting additional hyperparameter tuning, and exploring more advanced feature engineering techniques.

Author
Ilaha Musayeva
10/25/23
