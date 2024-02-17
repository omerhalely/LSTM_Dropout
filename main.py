import argparse
import torch.nn as nn
from model import LSTM_Model, GRU_Model
from Handler import Handler
import torch

parser = argparse.ArgumentParser(
    description="A program to build and train the RNN Dropout model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--model",
    type=str,
    help="Model which will be trained (LSTM / GRU).",
    default="LSTM"
)

parser.add_argument(
    "--model-name",
    type=str,
    help="Name of the model in which the model will be saved.",
    default="LSTM_Dropout_02"
)

parser.add_argument(
    "--dropout",
    type=int,
    help="Dropout value which will be used.",
    default=0.2
)

parser.add_argument(
    "--epochs",
    type=int,
    help="Number of training epochs.",
    default=40
)

parser.add_argument(
    "--lr",
    type=int,
    help="Initial learning rate for training.",
    default=20
)

parser.add_argument(
    "--test",
    type=bool,
    help="If true, loads the model according to the given flags of --model-name, --model and tests the model. ",
    default=True
)


if __name__ == "__main__":
    args = parser.parse_args()
    dropout = args.dropout
    input_size = 1
    hidden_size = 200
    num_layers = 2
    num_tokens = 10000
    num_embeddings = 200
    sequence_length = 35
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs
    train_batch_size = 20
    eval_batch_size = 10
    lr = args.lr
    model_name = args.model_name
    data_path = "./data"
    test_mode = args.test

    print(f"Building {args.model} Model...")
    model = LSTM_Model(hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=dropout,
                       num_tokens=num_tokens,
                       num_embeddings=num_embeddings)
    if args.model == "GRU":
        model = GRU_Model(hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          num_tokens=num_tokens,
                          num_embeddings=num_embeddings)
    print(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on {device}")

    handler = Handler(model=model,
                      model_name=model_name,
                      data_path=data_path,
                      train_batch_size=train_batch_size,
                      eval_batch_size=eval_batch_size,
                      sequence_length=sequence_length,
                      criterion=criterion,
                      epochs=epochs,
                      lr=lr,
                      dropout=dropout,
                      device=device)

    if not test_mode:
        handler.run()
    else:
        handler.load_model()
        handler.test()
