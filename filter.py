from keras.models import model_from_json
from keras.models import Model
from keras.datasets import cifar10
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x = np.array([x_train[0]])

layer_outputs = [layer.output for layer in model.layers]
viz_model = Model(input=model.input, output=layer_outputs)
features = viz_model.predict(x)
for feature_map in features:
    print(feature_map.shape)

print(features[9])
