import re
import torch
import config
import argparse
from Source.utils import load_file
from transformers import BertTokenizer
from Source.model import BertClassifier


def main(args_):
    # Process input text
    input_text = args_.test_complaint
    input_text = input_text.lower()
    input_text = re.sub(r"[^\w\d'\s]+", " ", input_text)
    input_text = re.sub("\d+", "", input_text)
    input_text = re.sub(r'[x]{2,}', "", input_text)
    input_text = re.sub(' +', ' ', input_text)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tokenize the input text
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    tokens = tokenizer(input_text, padding="max_length",
                       max_length=config.seq_len, truncation=True,
                       return_tensors="pt")

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    input_ids = torch.squeeze(input_ids, 1)

    # Load label encoder
    label_encoder = load_file(config.label_encoder_path)
    num_classes = len(label_encoder.classes_)

    # Load the model
    model = BertClassifier(config.dropout, num_classes)
    model_path = config.model_path
    model.load_state_dict(torch.load(model_path))
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Forward pass
    out = torch.squeeze(model(input_ids, attention_mask))
    # Find predicted class
    prediction = label_encoder.classes_[torch.argmax(out)]
    print(f"Predicted  Class: {prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_complaint", type=str, help="Test complaint")
    args = parser.parse_args()
    main(args)
