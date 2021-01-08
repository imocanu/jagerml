# jagerml
Jager Machine Learning - tool with a shot of machine learning optimizations

###  Steps to start :

#### Upgrade pip :
```bash
python3 -m pip install --user --upgrade pip
```
#### Install virtualenv : 
```bash
python3 -m pip install --user virtualenv
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
#### Python code for GPU optimization:
```python
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
```

#### Example for Model
```python
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
                epochs=40, 
                verbose=50, 
                validationData=(X_test, y_test), 
                batchSize=128)
    model.evaluate(X_test, y_test)
```
```bash
[Evaluate] :
> acc 0.7498 loss 1.2230935114089874
```

