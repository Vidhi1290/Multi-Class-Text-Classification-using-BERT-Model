import torch
import config
from Source.data import TextDataset
from Source.utils import load_file, save_file
from sklearn.model_selection import train_test_split
from Source.model import BertClassifier, train, test


def main():

    print("Loading the files...")
    tokens = load_file(config.tokens_path)
    labels = load_file(config.labels_path)
    labels = labels[:1000]
    label_encoder = load_file(config.label_encoder_path)
    num_classes = len(label_encoder.classes_)

    print("Splitting data into train, valid and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(tokens, labels,
                                                        test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.25)

    print("Creating PyTorch datasets...")
    train_dataset = TextDataset(X_train, y_train)
    valid_dataset = TextDataset(X_valid, y_valid)
    test_dataset = TextDataset(X_test, y_test)

    print("Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                               shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Creating model object...")
    model = BertClassifier(config.dropout, num_classes)
    model_path = config.model_path

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Move model to GPU if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print("Training the model...")
    train(train_loader, valid_loader, model, criterion, optimizer,
          device, config.num_epochs, model_path)

    print("Testing the model...")
    test(test_loader, model, criterion, device)


if __name__ == "__main__":
    main()
