import os
import sys
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import LongformerTokenizer, LongformerForMultipleChoice
from data_loader import RaceAnsweringModel

longformer_model = 'allenai/longformer-large-4096'
save_dir = "model_weights/"
model_name = f"longformer-large-4096-Race-Ansewring-version0"

lr0          = 1e-6
batch_size   = 2
num_workers  = 3
num_epochs   = 10
max_length   = 4096
valid_step   = 5000 * 4
num_options  = 4

longformer_tokenizer = LongformerTokenizer.from_pretrained(longformer_model)

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("torch_device:", torch_device)

print("model_name:", model_name)
print("longformer_model:", longformer_model)
print("num_options:", num_options)
print("lr0:", lr0)
print("batch_size:", batch_size)
print("num_workers:", num_workers)
print("num_epochs:", num_epochs)
print("valid_step:", valid_step)
print("max_length:", max_length)



def experiment():
    # ---------------------------- Data ---------------------------- #
    train_data = RaceAnsweringModel(
        data_split = "train",
    )
    print("len_train_data:", len(train_data))
    train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    collate_fn=collate_fn)

    valid_data = RaceAnsweringModel(
        data_split = "validation",
    )
    print("len_valid_data:", len(valid_data))
    valid_loader = torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    collate_fn=collate_fn)

    # ---------------------------- Model ---------------------------- #
    model = LongformerForMultipleChoice.from_pretrained(longformer_model)
    if torch_device == "cuda":
        model.cuda()
    print("#parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

    # by default, it's not training!!!
    model.train()

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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if training_step % 1 == 0:
                print("{}, step = {}, loss = {:.8f}".format(str(datetime.now()), training_step, loss))
                sys.stdout.flush()

            if training_step % valid_step == 0 and training_step > 0:
                state = {
                    'training_step': training_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
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
    c_plus_q_array = []
    option_array = []
    answer_array = []

    for item in list_of_items:
        c_plus_q = item['context'] + ' ' + longformer_tokenizer.bos_token + ' ' + item['question']
        c_plus_q_array.append([c_plus_q] * num_options)
        option_array.append(item['options'])
        answer_array.append(item['answer_i'])

    # flatten two lists so you can tokenize them
    c_plus_q_array = sum(c_plus_q_array, [])
    option_array = sum(option_array, [])

    tokenized_examples = longformer_tokenizer(
        c_plus_q_array, option_array,
        max_length=max_length,
        padding="longest",
        truncation=True,
        # return_tensors="pt",
    )

    # unflatten
    encoding = {k: [v[i : i + num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()}
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    answer_array = torch.tensor(answer_array)
    return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': answer_array,
    }

if __name__ == "__main__":
    experiment()
