# Text Classification 🤖📥

This repository hosts the code for the Text Classifier used in our [Study Bot Implementation](https://github.com/NLP-Challenges/Study-Bot).

The BERT-based classifiers are published under the [nlpchallenges organization on Hugging Face](https://huggingface.co/nlpchallenges). You can find the models specific to this project at [nlpchallenges/Text-Classification](https://huggingface.co/nlpchallenges/Text-Classification) and [nlpchallenges/Text-Classification-Synthetic-Dataset](https://huggingface.co/nlpchallenges/Text-Classification-Synthethic-Dataset).

## Repository Structure

The structure of this repository is organized as follows:

```markdown
└── 📁Text-Classification
    └── README.md
    └── requirements.txt
    └── 📁assets [ℹ️ Assets used in README.md/Notebooks]
    └── 📁classifier-for-bot [ℹ️ The Classifier built for the Data Chatbot]
        └── 📁data
        └── 📁src
            └── BERT-classification-synthetic.ipynb [ℹ️ Notebook for training BERT-based classifier on the synthetic dataset]
            └── build_synthetic_dataset.py [ℹ️ Script to build the synthetic dataset for the classifier]
    └── 📁data
    └── 📁src
        └── 📁poc [Proof of Concept for each of the 3 classifier variants]
        └── build_custom_dataset.py 
        └── build_dataset.py [ℹ️ Script to build the dataset for the classifiers]
        └── classification.ipynb [ℹ️ Notebook for 3 classifier experiments (NPR MC1)]
```

## Setup

### Prerequisites

1. **Python Environment**: Create a new virtual Python environment to ensure isolated package management (we used Python 3.11.6).
2. **Installation**: Navigate to the repository's root directory and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Clone the Repository**: Clone this repository to your local machine.

## Testing Different Classifiers

First, we implemented three different classifier approaches and tested their performance on our constructed dataset. The three approaches are:

- Linear-SVC
- LSTM-CNN
- BERT

For details on the implementation and the results, please refer to the [classification.ipynb](src/classification.ipynb) notebook.

## Building the Classifier for the Data Chatbot

Based on the results obtained comparing the three approaches, we decided to use BERT as the classifier for our Data Chatbot. We trained the classifier on a synthetic dataset, which we constructed using the [build_synthetic_dataset.py](classifier-for-bot/src/build_synthetic_dataset.py) script. The notebook [BERT-classification-synthetic.ipynb](classifier-for-bot/src/BERT-classification-synthetic.ipynb) contains the code for training the classifier on the synthetic dataset.

The files `hf_concern.txt`, `hf_question.txt`, and `hf_harm.txt` can be used to add further examples that failed during production use to the training data. The files are part of the training data additionally to the synthetic dataset, when the classifier is trained.
