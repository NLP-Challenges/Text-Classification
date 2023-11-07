"""
Execute this script to get train and test split (located in data folder).
The split is done with respect to the number of words in the 'text' column and class (all classes have the same presence in the split).
The concern part of the dataset is derived from a english dataset by translation. Thats why the execution of the script can take a couple of minutes.
"""

DEVICE='auto'

TRAIN_SIZE=0.8 #size of training set
SEED=420 #seed for train test split
MIN_WORD_COUNT=3 #how many words at least must be present in a text to be included
MAX_WORD_COUNT=13 #how many words at max are allowed to be in a text to be included
BINS=5 #number of bins for alignment (the more bins the more accurately are the individual dataframes sampled according to word count) (BINS is not allowed to be bigger than [MAX_WORD_COUNT - MIN_WORD_COUNT], in this case each bin gets a single word length)
WORD_DIST_ADJUST=False # true -> tries to keep the word count distribution same for all classes

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
    concern = data[data.top_label.isin(["School", "Work", "Family Issues"])][["sentence"]].reset_index(drop=True, inplace=False)
    concern.columns = ["text"] #rename

    return concern

def sampler(dfs:list, train_split:float=0.8, seed=None, use_bins:bool=True, min_max_bins:tuple=None):
    # Calculate the minimum count across all dataframes
    min_count = min(df.shape[0] for df in dfs)

    results = []
    for df in dfs:
        df_sampled_train = []
        df_sampled_test = []

        if use_bins and min_max_bins is not None:
            # Create bins and bin the data
            bins = pd.interval_range(start=min_max_bins[0], end=min_max_bins[1], periods=min_max_bins[2], closed='right')
            intervals = pd.cut(df.text.apply(lambda x: len(x.split())), bins)

            # Group dataframe by intervals
            df_grouped = df.groupby(intervals, observed=True)

            for interval in bins:
                if interval in df_grouped.groups:
                    df_interval_group = df_grouped.get_group(interval)
                    # Ensure we don't try to sample more items than exist in the smallest interval group
                    interval_min_count = min(min_count, df_interval_group.shape[0])
                    df_sampled = df_interval_group.sample(n=interval_min_count, random_state=seed)

                    df_train = df_sampled.sample(frac=train_split, random_state=seed)
                    df_test = df_sampled.drop(df_train.index)

                    df_sampled_train.append(df_train)
                    df_sampled_test.append(df_test)
        else:
            # Sample without binning
            df_sampled = df.sample(n=min_count, random_state=seed)
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
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", fast=True)
        self.model:M2M100ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", device_map=device)

    def translate(self, data_en:list, batch_size:int=32):
        """Translates a list of texts from english to german
        Args:
            data_en (list): list of english texts
            batch_size (int): batch size for processing (to reduce ram useage on GPU)

        Returns:
            list of translated texts
        """

        def translate_batch(data:list):
            inputs = self.tokenizer(data, padding=True, return_tensors="pt").to(self.model.device)

            translated_tokens = self.model.generate(
                **inputs, 
                forced_bos_token_id=self.tokenizer.lang_code_to_id["deu_Latn"], 
                max_length=self.max_length
            )

            return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        
        return sum([translate_batch(data_en[i:i + batch_size]) for i in range(0, len(data_en), batch_size)], [])

if __name__ == "__main__":
    #load question data
    questions = load_questions_dataset()

    #load offensive data
    offensive = load_offensive_dataset()

    #load concern and translate to german
    concern = load_concern_dataset()
    concern["text"] = Translate_EN_DE(DEVICE).translate(concern.text.to_list())

    #sample datasets
    (sampled_questions_train, sampled_questions_test), (sampled_offense_train, sampled_offense_test), (sampled_concern_train, sampled_concern_test) = sampler(dfs=[questions, offensive, concern], min_max_bins=(MIN_WORD_COUNT, MAX_WORD_COUNT, BINS), train_split=TRAIN_SIZE, seed=SEED, use_bins=WORD_DIST_ADJUST)

    #assign label to questions
    sampled_questions_train["label"] = "question"
    sampled_questions_test["label"] = "question"

    #assign label to harm
    sampled_offense_train["label"] = "harm"
    sampled_offense_test["label"] = "harm"         

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

