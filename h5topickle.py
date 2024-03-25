import tensorflow.keras as keras
import pickle

# Load the Keras model from the HDF5 file
model = keras.models.load_model('action.h5')

# Save the loaded model to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
