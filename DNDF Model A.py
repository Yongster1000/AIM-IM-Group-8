import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import classification_report

# Load data
train_data = pd.read_csv("train_standard.csv")
test_data = pd.read_csv("test_standard.csv")

# Separate features and labels
X_train = train_data.drop(columns=["target"])  # Replace 'target' with actual label column name
y_train = train_data["target"]
X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

# Normalize feature data (recommended for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the data into TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(32)

# Build the DNDF-like model
class DNDF(tf.keras.Model):
    def __init__(self):
        super(DNDF, self).__init__()
        
        # Neural Network layers for feature transformation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        
        # Decision Forest layer using TensorFlow Decision Forests
        self.df_layer = tfdf.keras.GradientBoostedTreesModel(
            task=tfdf.keras.Task.CLASSIFICATION,
            num_trees=50,  # Can be tuned
            max_depth=6    # Can be tuned for interpretability vs. complexity
        )

    def call(self, inputs, training=False):
        # Pass inputs through neural layers first
        x = self.dense1(inputs)
        x = self.dense2(x)
        
        # Pass neural network output to the decision forest layer
        return self.df_layer(x)

# Instantiate and compile the DNDF model
dndf_model = DNDF()
dndf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
dndf_model.fit(train_ds, validation_data=test_ds, epochs=10)

# Evaluate the model
dndf_model.evaluate(test_ds)
y_pred = tf.argmax(dndf_model.predict(X_test_scaled), axis=1)
print(classification_report(y_test, y_pred))

