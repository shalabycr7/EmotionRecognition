import numpy as np
import pandas as pd
from keras.src.models.model import model_from_json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Load the model architecture from JSON file
with open('./model/model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load the model weights from H5 file
model.load_weights('./model/model.h5')

# Load test data
df = pd.read_csv('./fer2013.csv')
X_train = []
y_train = []
X_test = []
y_test = []
for index, row in df.iterrows():
    k = row['pixels'].split(" ")
    if row['Usage'] == 'Training':
        X_train.append(np.array(k))
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        X_test.append(np.array(k))
        y_test.append(row['emotion'])

# Convert lists to numpy arrays and reshape if necessary
X_train = np.array(X_train, dtype='float32')
X_test = np.array(X_test, dtype='float32')
y_train = np.array(y_train, dtype='int')
y_test = np.array(y_test, dtype='int')

# Normalize pixel values (assuming they are 0-255)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data if necessary (assuming 48x48 pixel images)
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=7)

# Predicting the labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Accuracy: {accuracy}')

# Generate classification report
class_report = classification_report(y_true, y_pred_classes,
                                     target_names=['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
print('Classification Report:')
print(class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
