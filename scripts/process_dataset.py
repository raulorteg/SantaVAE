import pathlib
import random

import pandas as pd


def process_dataset(entry_path: str) -> None:
    """A utility function to be used in extracting two files: training.txt, testing.txt
    contatining the filepaths of the images in training and test set, together."""

    # check that the path given exists and its a directory
    entry_path = pathlib.Path(entry_path)
    if (not entry_path.is_dir()) or (not entry_path.exists()):
        print(Exception(f"Entry path {entry_path} is not a directory. Does it exist?"))
        exit()

    # create a data directory
    pathlib.Path("./data").mkdir(parents=True, exist_ok=True)

    # loop over the training data
    fout = "training.txt"
    buffer = []
    train_path = entry_path / "train" / "not-a-santa"
    for item in train_path.iterdir():
        buffer.append(str(item))

    train_path = entry_path / "train" / "santa"
    for item in train_path.iterdir():
        buffer.append(str(item))

    # Shuffle the rows
    random.shuffle(buffer)

    # create a dataframe and save the data in a csv file
    data = pd.DataFrame()
    data["filename"] = buffer
    data["is_santa"] = data["filename"].apply(lambda x: int("not-a-santa" in str(x)))
    data.to_csv(fout, sep=",")

    # loop over the testing data
    fout = "testing.txt"
    buffer = []
    with open(fout, "a+") as f:
        train_path = entry_path / "test" / "not-a-santa"
        for item in train_path.iterdir():
            buffer.append(str(item))

        train_path = entry_path / "test" / "santa"
        for item in train_path.iterdir():
            buffer.append(str(item))

    # Shuffle the rows
    random.shuffle(buffer)

    # create a dataframe and save the data in a csv file
    data = pd.DataFrame()
    data["filename"] = buffer
    data["is_santa"] = data["filename"].apply(lambda x: int("not-a-santa" in str(x)))
    data.to_csv(fout, sep=",")

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the decompressed directory where the data is.",
    )
    args = parser.parse_args()

    process_dataset(entry_path=args.path)
