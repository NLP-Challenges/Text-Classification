"""
Execute this script to get train and test split (located in data folder).
The split is done with respect to the number of words in the 'text' column and class (all classes have the same presence in the split).
The concern part of the dataset is derived from a english dataset by translation. Thats why the execution of the script can take a couple of minutes.
"""

DEVICE='mps'

TRAIN_SIZE=0.8 #size of training set
SEED=None #seed for train test split
MIN_WORD_COUNT=3 #how many words at least must be present in a text to be included
MAX_WORD_COUNT=13 #how many words at max are allowed to be in a text to be included
BINS=5 #number of bins for alignment (the more bins the more accurately are the individual dataframes sampled according to word count) (BINS is not allowed to be bigger than [MAX_WORD_COUNT - MIN_WORD_COUNT], in this case each bin gets a single word length)

from datasets import load_dataset
import pandas as pd
import re
import requests, zipfile, io
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100ForConditionalGeneration

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

def load_concern_dataset():
    """ Loads SAD dataset from(https://github.com/PervasiveWellbeingTech/Stress-Annotated-Dataset-SAD)
    Only take school related concern text.
    """

    #load 
    r = requests.get("https://raw.githubusercontent.com/PervasiveWellbeingTech/Stress-Annotated-Dataset-SAD/main/SAD_v1.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))

    #extract excel file from zip
    filename = z.extract("SAD_v1.xlsx")

    #read excel file
    data = pd.read_excel(filename)

    #delete excel file
    os.remove(filename)

    #filter for "School"
    concern = data[data.top_label == "School"][["sentence"]].reset_index(drop=True, inplace=False)
    concern.columns = ["text"] #rename

    return concern

def sampler(dfs:list, min_max_bins:tuple, train_split:float=0.8, seed=None):
    bins = pd.interval_range(start=min_max_bins[0], end=min_max_bins[1], freq=None, periods=min_max_bins[2], closed='right')
    intervals_list = [pd.cut(df.text.apply(lambda x: len(x.split())), bins) for df in dfs]

    # Calculate the minimum counts for each interval across all dataframes
    min_counts = intervals_list[0].value_counts()
    for intervals in intervals_list[1:]:
        current_counts = intervals.value_counts()
        min_counts = pd.concat([min_counts, current_counts], axis=1).min(axis=1)

    # Group each dataframe by their intervals
    grouped_list = [df.groupby(intervals, observed=True) for df, intervals in zip(dfs, intervals_list)]
    
    results = []
    for df, df_grouped in zip(dfs, grouped_list):
        df_sampled_train = []
        df_sampled_test = []

        for interval, min_count in min_counts.items():
            if interval in df_grouped.groups:
                df_sampled = df_grouped.get_group(interval).sample(n=min_count, random_state=seed)

                df_train = df_sampled.sample(frac=train_split, random_state=seed)
                df_test = df_sampled.drop(df_train.index)

                df_sampled_train.append(df_train)
                df_sampled_test.append(df_test)

        results.append((pd.concat(df_sampled_train, ignore_index=True), pd.concat(df_sampled_test, ignore_index=True)))

    return results

class Translate_EN_DE():
    def __init__(self, device:str="cuda:0", max_length:int=50) -> None:
        """Translates a list of text from english to german. Uses the (nllb-200-distilled-1.3B) model
        Args:
            device (str): device to run the model on
            max_length (int): generate text until max length (not sure?)
        """
        # Model: https://huggingface.co/docs/transformers/model_doc/nllb & https://huggingface.co/facebook/nllb-200-distilled-1.3B
        # This model performed best in comparison with (https://huggingface.co/facebook/nllb-200-distilled-600M, https://huggingface.co/t5-base, https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) in a qualitative view
        # Model Size is also restrited due to limiations of gpu memory
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", fast=True)
        self.model:M2M100ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", device_map=device)

    def translate(self, data_en:list):
        """Translates a list of texts from english to german
        Args:
            data_en (list): list of english texts

        Returns:
            list of translated texts
        """

        inputs = self.tokenizer(data_en, padding=True, return_tensors="pt").to(self.device)

        translated_tokens = self.model.generate(
            **inputs, 
            forced_bos_token_id=self.tokenizer.lang_code_to_id["deu_Latn"], 
            max_length=self.max_length
        )

        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    #load question data
    questions = load_questions_dataset()

    #load offensive data
    offensive = load_offensive_dataset()

    #load concern and translate to german
    concern = load_concern_dataset()
    concern["text"] = Translate_EN_DE(DEVICE).translate(concern.text.to_list())

    #sample datasets
    (sampled_questions_train, sampled_questions_test), (sampled_offense_train, sampled_offense_test), (sampled_concern_train, sampled_concern_test) = sampler([questions, offensive, concern], (MIN_WORD_COUNT, MAX_WORD_COUNT, BINS), TRAIN_SIZE, SEED)

    #assign label to questions
    sampled_questions_train["label"] = "question"
    sampled_questions_test["label"] = "question"

    #assign label to other
    sampled_offense_train["label"] = "other"
    sampled_offense_test["label"] = "other"         

    #assign label to concern
    sampled_concern_train["label"] = "concern"
    sampled_concern_test["label"] = "concern"       

    #combine
    train = pd.concat([sampled_questions_train, sampled_offense_train, sampled_concern_train], ignore_index=True)
    test = pd.concat([sampled_questions_test, sampled_offense_test, sampled_concern_test], ignore_index=True)

    print("train dataset: ")
    print(train.label.value_counts())

    print("test dataset: ")
    print(test.label.value_counts())

    #store
    train.to_parquet("./data/train.parquet")
    test.to_parquet("./data/test.parquet")

