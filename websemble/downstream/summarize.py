#!/usr/bin/env python3

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from websemble.web_trainer import DEVICE


def build_model(model_name):
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    return model, tokenizer


def summarize(texts, summarizer, tokenizer):
    inputs = tokenizer(
        texts, 
        truncation=True,
        padding='longest', # True
        add_special_tokens=True,
        return_tensors='pt'
    ).to(DEVICE)
    with torch.no_grad(): output = summarizer.generate(max_new_tokens=64, **inputs)
    summarized = tokenizer.batch_decode(output, skip_special_tokens=True)
    return {'summarized': summarized} # datasets.Dataset.map() requires the function to return a dict