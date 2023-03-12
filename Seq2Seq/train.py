import torch
from torch import optim, nn
import spacy
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<sos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<sos>')

train_data,validation_data,test_data=Multi30k.splits(exts=('.de','en'),fields=(german,english))

german.build_vocab(train_data,max_size=10000,min_freq=2)
english.build_vocab(train_data,max_size=10000,min_freq=2)

