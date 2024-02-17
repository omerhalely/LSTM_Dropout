# LSTM_Dropout
Word completion with RNN_dropout models.

We implemented this task using LSTM and GRU.

We compare the 2 networks and their versions with dropout.

## Dataset
We use the Penn Tree Bank dataset arranged according to the following:
```
data:
    |   ptb.test.txt
    |   ptb.train.txt
    |   ptb.valid.txt
```
Where the name of the data folder does not matter as long as you set the right value for the data flag.

## Training
For training a model:
```bash
python main.py --model "model" --model-name "model-name" --data "path to data folder" dropout "dropout" --epochs "epochs" lr "lr"
```
The model type given in the --model flag (LSTM / GRU) will be saved to ./models/model_name which will be created automatically.

Example for training a LSTM model with a dropout value of 0.2:
```bash
python main.py --model "LSTM" --model-name "LSTM_Dropout_02" --data "./data" dropout 0.2 --epochs 40 lr 20
```

Example for training a GRU model with a dropout value of 0.2:
```bash
python main.py --model "GRU" --model-name "GRU_Dropout_02" --data "./data" dropout 0.2 --epochs 40 lr 20
```

## Testing
For testing a model:
Example for training an LSTM model with a dropout value of 0.2:
```bash
python main.py --model "model" --model-name "model-name" --data "path to data folder" --test-mode True
```
Now we use a --test-mode flag which is set to True.

If the model name given in the --model-name flag does not exist, an assertion error will occur.

Example for testing a LSTM model:
```bash
python main.py --model "LSTM" --model-name "LSTM_Dropout_02" --data "./data" --test-mode True
```

Example for testing a GRU model:
```bash
python main.py --model "GRU" --model-name "GRU_Dropout_02" --data "./data" --test-mode True
```
