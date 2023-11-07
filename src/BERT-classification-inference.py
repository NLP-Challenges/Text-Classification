import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def classify_text(user_input, model, tokenizer, max_len=100):
    # Tokenize the user input
    inputs = tokenizer(
        user_input,
        None,
        add_special_tokens=True, # Add '[CLS]' and '[SEP]', default True
        max_length=max_len, # Maximum length to use by one of the truncation/padding parameters
        padding='max_length', # Pad to a maximum length specified with the argument max_length
        truncation=True, # Truncate to a maximum length specified with the argument max_length
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0) # Indices of input sequence tokens in the vocabulary
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0) # Mask to avoid performing attention on padding token indices
    
    
    # Get model output
    model.eval()
    with torch.no_grad():
        output = model(ids, attention_mask=mask)
    
    # Get predicted label index
    _, predicted_idx = torch.max(output.logits, 1)
    
    # Map index to label
    label_mapping = {0: 'other', 1: 'question', 2: 'concern'}
    return label_mapping[predicted_idx.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify text using a fine-tuned BERT model.')
    parser.add_argument('text', type=str, help='The text to classify')
    args = parser.parse_args()

    # Load fine-tuned model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('nlpchallenges/Text-Classification')
    model = BertForSequenceClassification.from_pretrained("nlpchallenges/Text-Classification")

    label = classify_text(args.text, model, tokenizer, None)
    print(f"The text is classified as: {label}")
