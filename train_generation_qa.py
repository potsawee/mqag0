import os
import sys
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_loader import RaceQuestionAnswerGeneration

t5_model = 't5-large'
save_dir = "model_weights/"
model_name = f"{t5_model}-Race-QA-Generation-version0"

lr0          = 5e-5
batch_size   = 8
num_workers  = 0
num_epochs   = 10
max_length   = 512
valid_step   = 5000

t5_tokenizer = AutoTokenizer.from_pretrained(t5_model, model_max_length=max_length)
t5_tokenizer.add_special_tokens({"sep_token": "<sep>"})

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("torch_device:", torch_device)

print("model_name:", model_name)
print("t5_model:", t5_model)
print("lr0:", lr0)
print("batch_size:", batch_size)
print("num_workers:", num_workers)
print("num_epochs:", num_epochs)
print("valid_step:", valid_step)
print("max_length:", max_length)

def experiment():
    # ---------------------------- Data ---------------------------- #
    train_data = RaceQuestionAnswerGeneration(
        tokenizer = t5_tokenizer,
        data_split = "train",
        separator = t5_tokenizer.sep_token,
    )
    print("len_train_data:", len(train_data))
    train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    collate_fn=collate_fn)

    valid_data = RaceQuestionAnswerGeneration(
        tokenizer = t5_tokenizer,
        data_split = "validation",
        separator = t5_tokenizer.sep_token,
    )
    print("len_valid_data:", len(valid_data))
    valid_loader = torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    collate_fn=collate_fn)

    # ---------------------------- Model ---------------------------- #
    model = AutoModelForSeq2SeqLM.from_pretrained(t5_model)
    if torch_device == "cuda":
        model.cuda()
    print("#parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # by default, it's not training!!!
    model.train()

    # ----------------- Optimizer and Loss Function ----------------- #
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()
    training_step = 0
    stop_counter = 0
    best_val_loss = 99999999
    for epoch_i in range(num_epochs):

        for iter_, sample in enumerate(train_loader):
            if sample is None:
                continue

            input_ids, attention_mask = sample['input_ids'], sample['attention_mask']
            labels = sample['labels']

            if torch_device == 'cuda':
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if training_step % 1 == 0:
                print("{}, step = {}, loss = {:.8f}".format(str(datetime.now()), training_step, loss))
                sys.stdout.flush()

            if training_step % valid_step == 0:
                state = {
                    'training_step': training_step,
                    'model': model.state_dict(),
                }
                savepath = "{}/{}-step{}.pt".format(save_dir, model_name, training_step)
                torch.save(state, savepath)
                print("Saved at {}".format(savepath))

                model.eval()
                with torch.no_grad():
                    valid_loss = validation(model, valid_loader)
                print("Valid Loss = {:.6f}".format(valid_loss))
                model.train()

                if valid_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = valid_loss
                    print("Model improved".format(stop_counter))
                else:
                    stop_counter += 1
                    print("Model not improved #{}".format(stop_counter))
                    if stop_counter == 3:
                        print("Stop training!")
                        return

            training_step += 1

        print("finish epoch: {}".format(epoch_i+1))

    print("Finish Training")

def validation(model, valid_loader):
    valid_loss = 0
    counter = 0
    for sample in valid_loader:
        input_ids, attention_mask = sample['input_ids'], sample['attention_mask']
        labels = sample['labels']

        if torch_device == 'cuda':
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        valid_loss += loss.item()
        counter += 1
        if counter % 50 == 0:
            print("#", end="")
            sys.stdout.flush()
    print()
    return valid_loss / counter
def collate_fn(list_of_items):
    """
    each item is a dictionary:
    """
    list_of_items = [x for x in list_of_items if x is not None]
    batch_size = len(list_of_items)
    if batch_size == 0: return None

    input_sequences, output_sequences = [], []
    for item in list_of_items:
        input_sequences.append(item['input'])
        output_sequences.append(item['output'])

    encoding = t5_tokenizer(
        input_sequences,
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    target_encoding = t5_tokenizer(
        output_sequences,
        padding="longest",
        max_length=max_length,
        truncation=True,
    )

    # the forward function automatically creates the correct decoder_input_ids
    labels = target_encoding.input_ids
    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels = torch.tensor(labels)
    labels[labels == t5_tokenizer.pad_token_id] = -100

    return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
    }

if __name__ == "__main__":
    experiment()
