from pathlib import Path
import pandas as pd

# Constants for train-test split
TRAIN_SIZE = 0.8  # Fraction of data to be used for training
SEED = 420        # Seed for random operations to ensure reproducibility

def get_paths():
    """
    Retrieves a list of paths for all synthetic text files in the 'data' directory.

    Returns:
        List[Path]: A list of paths for files matching the pattern 'synthetic_*.txt'.
    """
    return list(Path("./data/").glob(pattern="synthetic_*.txt"))

def get_content(path: Path, label_name: str):
    """
    Reads the content of a file and creates a DataFrame with text and associated labels.

    Args:
        path (Path): The path of the file to be read.
        label_name (str): The label name to be assigned to all texts in the file.

    Returns:
        pd.DataFrame: A DataFrame containing the text and its corresponding label.
    """
    with open(path, encoding="utf-8") as f:
        text_list = f.readlines()

    df = pd.DataFrame({"text": text_list, "label": [label_name] * len(text_list)})

    # Displaying duplicates for the specific label, if any
    print(f"duplicates of label {label_name}:")
    print(df[df.text.duplicated()])

    return df

def construct_dataset():
    """
    Constructs a dataset by aggregating contents from multiple files.

    Returns:
        pd.DataFrame: A concatenated and shuffled DataFrame containing text and labels.
    """
    contents = []

    # Looping over each file path and aggregating their contents
    for path in get_paths():
        contents.append(get_content(path, path.stem.split("_")[1]))

    # Concatenating and shuffling the DataFrame
    return pd.concat(contents).reset_index(drop=True).sample(frac=1, random_state=SEED)

def train_test_split(df: pd.DataFrame):
    """
    Splits a DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to be split.

    Raises:
        ValueError: If the DataFrame does not contain a 'label' column.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    # Ensuring the DataFrame contains a 'label' column
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column")

    # Initializing training and testing DataFrames
    train = pd.DataFrame(columns=df.columns)
    test = pd.DataFrame(columns=df.columns)

    # Grouping the DataFrame by the 'label' column
    grouped = df.groupby('label')

    for _, group in grouped:
        # Randomly selecting training examples
        train_examples = group.sample(frac=TRAIN_SIZE, random_state=SEED)

        # Adding selected examples to the training dataset
        train = pd.concat([train, train_examples])

        # Adding remaining examples to the testing dataset
        test = pd.concat([test, group.drop(train_examples.index)])

    # Shuffling datasets and resetting indexes
    train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = test.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train, test

# Constructing the dataset and splitting it into training and testing sets
df = construct_dataset()
train, test = train_test_split(df)

# Storing the datasets in parquet format
train.to_parquet("./data/train_synthetic.parquet")
test.to_parquet("./data/test_synthetic.parquet")
