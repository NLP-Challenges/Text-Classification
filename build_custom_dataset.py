import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.8
RANDOM_STATE = 42

def load_custom_data(file_path, label):
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    return pd.DataFrame({'text': data, 'label': label})

if __name__ == "__main__":
    # load data
    concern = load_custom_data('./data/custom_concern.txt', 'concern')
    harm = load_custom_data('./data/custom_harm.txt', 'harm')
    question = load_custom_data('./data/custom_question.txt', 'question')

    df_concatenated = pd.concat([concern, harm, question], ignore_index=True)

    # split data
    train, test = train_test_split(df_concatenated, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=df_concatenated.label)

    print("train dataset: ")
    print(train.label.value_counts())

    print("test dataset: ")
    print(test.label.value_counts())

    # store
    train.to_parquet("./data/train_custom.parquet")
    test.to_parquet("./data/test_custom.parquet")