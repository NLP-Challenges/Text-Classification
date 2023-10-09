"""
Execute this script to get train and test split (located in data folder).
The split is done with respect to the number of words in the 'text' column and class (all classes have the same presence in the split)
"""

TRAIN_SIZE=0.8 #size of training set
SEED=None #seed for train test split
MIN_WORD_COUNT=3 #how many words at least must be present in a text to be included
MAX_WORD_COUNT=13 #how many words at max are allowed to be in a text to be included
BINS=5 #number of bins for alignment (the more bins the more accurately are the individual dataframes sampled according to word count) (BINS is not allowed to be bigger than [MAX_WORD_COUNT - MIN_WORD_COUNT], in this case each bin gets a single word length)

from datasets import load_dataset
import pandas as pd
import re

def load_questions_dataset():
    """Loads questions from germanquad dataset (https://huggingface.co/datasets/deepset/germanquad)
    """
    data = load_dataset("deepset/germanquad", split="train")

    #TODO remove 50% of the question marks ? (maybe BERT focuses on questionmarks)
    questions = pd.DataFrame(data)[["question"]]
    questions.columns = ["text"] #rename

    return questions

def load_offensive_dataset():
    """Loads offensive text from germeval18 dataset (https://huggingface.co/datasets/philschmid/germeval18)
    Also includes some preprocessing steps like Username removal, removal of emojis and replacement of 'ß' with 'ss'
    """
    
    data = load_dataset("philschmid/germeval18", split="train")

    #convert to pandas dataframe
    germeval18 = pd.DataFrame(data)

    #regex for preprocessing
    match = re.compile(r'[^a-zA-Z\säßüö\"\'.!?]', flags=re.UNICODE)
    def preprocessing(text:str):
        """Removes username calling (@Username), only allows text, punktucation, double quote, single quote and replaces 'ß' with 'ss'
        """

        return match.sub(u'', str.join(" ", [word.replace("ß", "ss") for word in text.split(" ") if not str.startswith(word, "@")]))

    #only select offensive overvations and also only text column
    offensive = germeval18[germeval18.binary == "OFFENSE"][["text"]]
    offensive.columns = ["text"] #rename

    #apply preprocessing
    offensive["text"] = offensive.text.apply(preprocessing)

    return  offensive

def sampler(df1:pd.DataFrame, df2:pd.DataFrame, min_max_bins:tuple, train_split:float=0.8, seed=None):
    """Align distribution of word count per sample between df1 and df2 by random sampling. Additionally do a train test split by intervall.
    Args:
        df1 (pd.DataFrame): first dataframe
        df2 (pd.DataFrame): second dataframe
        min_max_bins (tuple): min token count, max token count, bin_size
        train_amount (float): train size
        seed (int): seed for sampling
    
    Returns:
        (resampled train split from df1, resampled train split from df2, resampled test split from df1, resampled test split from df2) 
    """

    bins = pd.interval_range(start=min_max_bins[0], end=min_max_bins[1], freq=None, periods=min_max_bins[2], closed='right')

    df1_interals = pd.cut(df1.text.apply(lambda x: len(x.split())), bins)
    df2_interals = pd.cut(df2.text.apply(lambda x: len(x.split())), bins)

    # Align counts Series
    aligned1, aligned2 = df1_interals.value_counts().align(df2_interals.value_counts(), fill_value=0)

    # Combine and choose smallest value
    result = aligned1.combine(aligned2, func=lambda x, y: min(x, y))

    # Remove intervals which only appear in one series
    result = result[result != 0]

    #create groups
    df1_grouped = df1.groupby(df1_interals, observed=False)
    df2_grouped = df2.groupby(df2_interals, observed=False)

    #sampled dataframes stored here
    df1_sampled_train = []
    df2_sampled_train = []
    df1_sampled_test = []
    df2_sampled_test = []

    #iterate over intervals and random sample text with certain length
    for group, count in result.items():
        
        #random sample amount of 'count' from original
        df1_sampled = df1_grouped.get_group(group).sample(count, ignore_index=True, random_state=seed)
        df2_sampled = df2_grouped.get_group(group).sample(count, ignore_index=True, random_state=seed)

        #get training split
        df1_train = df1_sampled.sample(frac=train_split, random_state=seed)
        df2_train = df2_sampled.sample(frac=train_split, random_state=seed)

        #get test split
        df1_test = df1_sampled.drop(df1_train.index)
        df2_test = df2_sampled.drop(df2_train.index)

        df1_sampled_train.append(df1_train)
        df2_sampled_train.append(df2_train)
        df1_sampled_test.append(df1_test)
        df2_sampled_test.append(df2_test)
        
        
    #return concatted
    return pd.concat(df1_sampled_train, ignore_index=True), pd.concat(df2_sampled_train, ignore_index=True), pd.concat(df1_sampled_test, ignore_index=True), pd.concat(df2_sampled_test, ignore_index=True)

if __name__ == "__main__":
    sampled_questions_train, sampled_offense_train, sampled_questions_test, sampled_offense_test = sampler(load_questions_dataset(), load_offensive_dataset(), (MIN_WORD_COUNT, MAX_WORD_COUNT, BINS), TRAIN_SIZE, SEED)

    #assign label to questions
    sampled_questions_train["label"] = "question"
    sampled_questions_test["label"] = "question"

    #assign label to other
    sampled_offense_train["label"] = "other"
    sampled_offense_test["label"] = "other"         

    #combine
    train = pd.concat([sampled_questions_train, sampled_offense_train], ignore_index=True)
    test = pd.concat([sampled_questions_test, sampled_offense_test], ignore_index=True)

    print("train dataset: ")
    print(train.label.value_counts())

    print("test dataset: ")
    print(test.label.value_counts())

    #store
    train.to_parquet("./data/train.parquet")
    test.to_parquet("./data/test.parquet")

