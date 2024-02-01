import re
import config
import pandas as pd
from tqdm import tqdm
from Source.utils import save_file
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder


def main():

    # Read the data file
    print("Processing data file...")
    data = pd.read_csv(config.data_path)
    # Drop rows where the text column is empty
    data.dropna(subset=[config.text_col_name], inplace=True)
    # Replace duplicate labels
    data.replace({config.label_col: config.product_map}, inplace=True)
    # Encode the label column
    label_encoder = LabelEncoder()
    label_encoder.fit(data[config.label_col])
    labels = label_encoder.transform(data[config.label_col])
    # Save the encoded labels
    save_file(config.labels_path, labels)
    save_file(config.label_encoder_path, label_encoder)

    # Process the text column
    input_text = data[config.text_col_name]
    # Convert text to lower case
    print("Converting text to lower case...")
    input_text = [i.lower() for i in tqdm(input_text)]
    # Remove punctuations except apostrophe
    print("Removing punctuations in text...")
    input_text = [re.sub(r"[^\w\d'\s]+", " ", i) for i in tqdm(input_text)]
    # Remove digits
    print("Removing digits in text...")
    input_text = [re.sub("\d+", "", i) for i in tqdm(input_text)]
    # Remove more than one consecutive instance of 'x'
    print("Removing 'xxxx...' in text")
    input_text = [re.sub(r'[x]{2,}', "", i) for i in tqdm(input_text)]
    # Replace multiple spaces with single space
    print("Removing additional spaces in text...")
    input_text = [re.sub(' +', ' ', i) for i in tqdm(input_text)]
    # Tokenize the text
    print("Tokenizing the text...")
    input_text = input_text[:1000]
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    tokens = [tokenizer(i, padding="max_length", max_length=config.seq_len,
                        truncation=True, return_tensors="pt")
              for i in tqdm(input_text)]
    # Save the tokens
    save_file(config.tokens_path, tokens)


if __name__ == "__main__":
    main()
