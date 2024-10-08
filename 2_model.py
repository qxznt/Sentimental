
# Import required libraries
import numpy as np
import os
from json_tricks import dump, load
from pydub import AudioSegment, effects
import librosa
import noisereduce as nr
import tensorflow as tf
from keras.models import Sequential
from keras import layers, callbacks
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Function to identify emotion from TESS dataset filenames
def find_emotion_T(name):
    emotions = {
        'neutral': "01",
        'calm': "02",
        'happy': "03",
        'sad': "04",
        'angry': "05",
        'fear': "06",
        'disgust': "07",
        
    }
    for key, value in emotions.items():
        if key in name:
            return value
    return "-1"

# Function to map emotions to a numerical scale for classification
def emotionfix(e_num):
    emotion_map = {
        "01": 0,  # neutral
        "02": 1,  # calm
        "03": 2,  # happy
        "04": 3,  # sad
        "05": 4,  # angry
        "06": 5,  # fear
        "07": 6,  # disgust
       
    }
    return emotion_map.get(e_num, 7)

# Load audio data and process
folder_path = r'C:\Users\sahil\Desktop\Project\Extacy\AudioFiles'
sample_lengths = []

for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        x, sr = librosa.load(os.path.join(subdir, file), sr=None)
        xt, _ = librosa.effects.trim(x, top_db=30)
        sample_lengths.append(len(xt))

print('Maximum sample length:', np.max(sample_lengths))

import time
tic = time.perf_counter()

# Initialize data lists
rms, zcr, mfcc, emotions = [], [], [], []

# Constants for feature extraction
total_length = 104448  # desired frame length
frame_length = 2048
hop_length = 512

# Iterate over files, extract features, and process emotions
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        _, sr = librosa.load(os.path.join(subdir, file), sr=None)
        rawsound = AudioSegment.from_file(os.path.join(subdir, file))
        normalizedsound = effects.normalize(rawsound, headroom=0)
        normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
        xt, _ = librosa.effects.trim(normal_x, top_db=30)
        padded_x = np.pad(xt, (0, total_length - len(xt)), 'constant')
        final_x = nr.reduce_noise(padded_x, sr=sr)

        # Feature extraction
        f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length)
        f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=frame_length, hop_length=hop_length)
        f3 = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=13, hop_length=hop_length)

        # Extract emotion
        name = find_emotion_T(file) if find_emotion_T(file) != "-1" else file[6:8]

        # Append features and emotions to lists
        rms.append(f1)
        zcr.append(f2)
        mfcc.append(f3)
        emotions.append(emotionfix(name))

toc = time.perf_counter()
print(f"Running time: {(toc - tic) / 60:0.4f} minutes")

# Adjusting features shape to the 3D format
f_rms = np.asarray(rms).astype('float32').swapaxes(1, 2)
f_zcr = np.asarray(zcr).astype('float32').swapaxes(1, 2)
f_mfccs = np.asarray(mfcc).astype('float32').swapaxes(1, 2)

# Concatenate features to form X and prepare Y
X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)
Y = np.asarray(emotions).astype('int8').reshape(-1, 1)

# Save X, Y arrays to JSON files
dump(obj=X.tolist(), fp=r'C:\Users\sahil\Desktop\Project\Extacy/X_datanew.json')
dump(obj=Y.tolist(), fp=r'C:\Users\sahil\Desktop\Project\Extacy/Y_datanew.json')

# Load X, Y from JSON files
X = np.asarray(load(r'C:\Users\sahil\Desktop\Project\Extacy/X_datanew.json'), dtype='float32')
Y = np.asarray(load( r'C:\Users\sahil\Desktop\Project\Extacy/Y_datanew.json'), dtype='int8')

# Split data into train, validation, and test sets
x_train, x_tosplit, y_train, y_tosplit = train_test_split(X, Y, test_size=0.125, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_tosplit, y_tosplit, test_size=0.304, random_state=1)

# One-hot encoding for emotion classification
y_train_class = tf.keras.utils.to_categorical(y_train, num_classes=8)
y_val_class = tf.keras.utils.to_categorical(y_val, num_classes=8)

# Check shapes
print(np.shape(x_train), np.shape(x_val), np.shape(x_test))

# Save x_test, y_test
dump(obj=x_test.tolist(), fp='x_test_data.json')
dump(obj=y_test.tolist(), fp='y_test_data.json')

# Initialize and build the LSTM model
model = Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    layers.LSTM(64),
    layers.Dense(8, activation='softmax')
])
print(model.summary())

batch_size = 23

# Callbacks: saving best model and reducing learning rate on plateau
checkpoint_path = r'C:\Users\sahil\Desktop\Project\Extacy/best_weights.weights.h5'
mcp_save = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, monitor='val_categorical_accuracy', mode='max')
rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=100)

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['categorical_accuracy'])
history = model.fit(x_train, y_train_class, epochs=340, batch_size=batch_size, validation_data=(x_val, y_val_class), callbacks=[mcp_save, rlrop])

# Load best weights
model.load_weights(checkpoint_path)

# Visualize training history
plt.figure()
plt.plot(history.history['loss'], label='Loss (training data)')
plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.title('Loss for train and validation')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.plot(history.history['categorical_accuracy'], label='Acc (training data)')
plt.plot(history.history['val_categorical_accuracy'], label='Acc (validation data)')
plt.title('Model accuracy')
plt.ylabel('Acc %')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# Evaluate model on validation set
loss, acc = model.evaluate(x_val, y_val_class, verbose=2)

# Confusion matrix for validation set
y_val_class = np.argmax(y_val_class, axis=1)
predictions = model.predict(x_val)
y_pred_class = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_val_class, y_pred_class)
index = ['neutral', 'happy', 'sad', 'angry','calm']
cm_df = pd.DataFrame(cm, index=index, columns=index)

plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, cmap='PuBu', fmt="d", annot=True)
plt.ylabel('True emotion')
plt.xlabel('Predicted emotion')
plt.show()

# Model and weights saving
saved_model_path = r'C:\Users\sahil\Desktop\Project\Extacy/model8723.json'
saved_weights_path = r'C:\Users\sahil\Desktop\Project\Extacy/model8723_weights.weights.h5'  # Change to .weights.h5

model_json = model.to_json()
with open(saved_model_path, "w") as json_file:
    json_file.write(model_json)

# Save weights
model.save_weights(saved_weights_path)
print("Saved model to disk")


# Define the paths again to ensure they're available
saved_model_path = r'C:\Users\sahil\Desktop\Project\Extacy/model8723.json'
saved_weights_path = r'C:\Users\sahil\Desktop\Project\Extacy/model8723_weights.weights.h5'  # Ensure this is correct

# Reading the model from JSON file
with open(saved_model_path, 'r') as json_file:
    json_savedModel = json_file.read()

# Loading the model architecture, weights
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(saved_weights_path)  # Correct variable name

