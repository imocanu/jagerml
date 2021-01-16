```bash
    _                       __  __ _     
   (_) __ _  __ _  ___ _ __|  \/  | |    
   | |/ _` |/ _` |/ _ | '__| |\/| | |    
   | | (_| | (_| |  __| |  | |  | | |___ 
  _/ |\__,_|\__, |\___|_|  |_|  |_|_____|
 |__/       |___/                        
                        
```                    

Jager Machine Learning - simple library with a shot of machine learning optimizations


###  Steps to start :

#### Install python3-env :
```bash
sudo apt-get install python3-venv
```
#### Create virtual environment :
```bash
python3 -m venv env
```
#### Activate virtual environment : 
```bash
source env/bin/activate
```
#### Upgrade pip :
```bash
python3 -m pip install --upgrade pip
```
#### Install requirements :
```bash
pip3 install -r requirements.txt
```
#### List dependencies :
```bash
pip3 freeze
```
#### Install setup.py for development :
```bash
python3 setup.py develop
```
#### Run tests :
```bash
python3 -m pytest
```
#### Leave virtual environment : 
```bash
deactivate
```

#### Example for Model
```python
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
from jagerml.model import Model
from jagerml.layers import Dense, Dropout
from jagerml.activations import ReLU, Softmax, SoftmaxLossCrossentropy, Sigmoid, Linear
from jagerml.evaluate import LossCategoricalCrossentropy, \
    LossBinaryCrossentropy, \
    MeanSquaredError, \
    MeanAbsoluteError, \
    AccuracyRegression, \
    AccuracyCategorical
from jagerml.optimizers import SGD, AdaGrad, RMSprop, Adam
from jagerml.helper import *

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)
X_train = X_train[keys]
y_train = y_train[keys]

X_train = (X_train.reshape(X_train.shape[0], 
                           -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], 
                         -1).astype(np.float32) - 127.5) / 127.5

model = Model()
model.add(Dense(X_train.shape[1], 128))
model.add(ReLU())
model.add(Dense(128, 128))
model.add(ReLU())
model.add(Dense(128, 128))
model.add(Softmax())

model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=Adam(decay=1e-4),
    accuracy=AccuracyCategorical()
)

model.fit()
model.train(X_train, y_train, 
            epochs=20, 
            verbose=50, 
            validationData=(X_test, y_test), 
            batchSize=128)
model.evaluate(X_test, y_test)
```
```bash
[Evaluate] :
* acc 0.8364 loss 0.9994725647001709
[*] check first 20 inputs
>test : [9 2 1 1 6 1 4 6 5 7 4 5 7 3 4 1 2 4 8 0]
>pred : [9 2 1 1 6 1 4 6 5 7 4 5 5 3 4 1 2 2 8 0]
```

### TODO

* Improve accuracy
* Implement Convolutional layers
* Integrate pyopencl
