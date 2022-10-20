import random
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset

class RaceQuestionAnswerGeneration(Dataset):
    def __init__(self, tokenizer, data_split, separator='<sep>'):
        """
        task:
            - input: article (i.e. context)
            - output: question <sep> answer
        args:
            tokenizer: tokenizer
            data_split: train, validation, test
        """
        data = load_dataset("race", "all", split=data_split)
        self.data = data
        self.tokenizer = tokenizer
        self.separator = separator
        self.label_mapping = {label: i for i, label in enumerate(["A", "B", "C", "D"])}
        print("RaceQuestionAnswerGeneration Initialized")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        # example_id = example["example_id"]
        question = example["question"]
        context = example["article"]
        options = example["options"]
        label_example = example["answer"]
        answer = options[self.label_mapping[label_example]]

        # input & output
        input = context
        output = question + ' ' + self.separator + ' ' + answer
        return {'input': input, 'output': output}

class RaceDistractorGeneration(Dataset):
    def __init__(self, tokenizer, data_split, shuffle_distractors=False, separator='<sep>'):
        """
        task:
            - input: question <sep> answer <sep> article
            - output: distractor1 <sep> distractor2 <sep> distractor3
        args:
            tokenizer: tokenizer
            data_split: train, validation, test
        """
        data = load_dataset("race", "all", split=data_split)
        self.data = data
        self.tokenizer = tokenizer
        self.separator = separator
        self.label_mapping = {label: i for i, label in enumerate(["A", "B", "C", "D"])}
        self.all_labels = [0, 1, 2, 3]
        self.shuffle_distractors = shuffle_distractors
        print("RaceQuestionAnswerGeneration Initialized")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        # example_id = example["example_id"]
        question = example["question"]
        context = example["article"]
        options = example["options"]
        label_example = example["answer"]
        answer_i = self.label_mapping[label_example]
        answer = options[answer_i]
        distractor_ids = [i for i in self.all_labels if i != answer_i]
        if self.shuffle_distractors:
            random.shuffle(distractor_ids)
        distractors = [options[i] for i in distractor_ids]

        # input & output
        input = question + ' ' + self.separator + ' ' + answer + ' ' + self.separator + ' ' + context
        output = distractors[0] + ' ' + self.separator + ' ' + distractors[1] + ' ' + self.separator + ' ' + distractors[2]
        return {'input': input, 'output': output}

class RaceAnsweringModel(Dataset):
    def __init__(self,
            data_split,
        ):
        """
        """
        data = load_dataset("race", "all", split=data_split)
        self.data = data
        self.label_mapping = {label: i for i, label in enumerate(["A", "B", "C", "D"])}
        self.all_labels = [0, 1, 2, 3]
        print("RaceAnsweringModel Initialized")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        context = example["article"]
        options = example["options"]
        label_example = example["answer"]
        answer_i = self.label_mapping[label_example]

        return {'context': context, 'question': question, 'options': options, 'answer_i': answer_i}
