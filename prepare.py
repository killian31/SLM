import json
import os
import pickle

import numpy as np
import requests

from preprocess_chat import preprocessing_pipeline


def format_data(textfile):
    input_file_path = os.path.join(os.path.dirname(__file__), textfile)

    with open("names.json") as json_file:
        change_names_dic = json.load(json_file)

    data = preprocessing_pipeline(infile=input_file_path, name_dic=change_names_dic)
    raw_data = ""
    raw_data += data["Name"][0] + " :" + "\n" + data["Message"][0]
    for line in range(1, data.shape[0]):
        current_name, current_msg = data["Name"][line], data["Message"][line]
        previous_name, previous_msg = data["Name"][line - 1], data["Message"][line - 1]

        if current_name == previous_name:
            raw_data += "\n" + current_msg
        else:
            raw_data += "\n" * 2 + current_name + "\n" + current_msg

    return raw_data


def encode(stoi, s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


"""
def decode(itos,l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string
"""


def prepare_data(textfile):
    data = format_data(textfile)
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # create the train and test splits
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # encode both to integers
    train_ids = encode(stoi, train_data)
    val_ids = encode(stoi, val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    prepare_data("_chat.txt")
