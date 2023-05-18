def prepare_qa_input(
    t5_tokenizer,
    context,
    max_length=512,
    torch_device='cpu',
):
    """
    input: context
    output: question <sep> answer
    """
    encoding = t5_tokenizer(
        [context],
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding.input_ids
    if torch_device == 'cuda':
        input_ids = input_ids.cuda()
    return input_ids

def prepare_distractor_input(
    t5_tokenizer,
    context,
    question,
    answer,
    separator='<sep>',
    max_length=512,
    torch_device='cpu',
):
    """
    input: question <sep> answer <sep> article
    output: distractor1 <sep> distractor2 <sep> distractor3
    """
    input_text = question + ' ' + separator + ' ' + answer + ' ' + separator + ' ' + context
    encoding = t5_tokenizer(
        [input_text],
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding.input_ids
    if torch_device == 'cuda':
        input_ids = input_ids.cuda()
    return input_ids

def prepare_answering_input(
    tokenizer, # longformer_tokenizer
    question,
    options,
    context,
    max_seq_length=4096,
    torch_device='cpu',
):
    """
    this currently only supports longformer
    """
    c_plus_q = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)

    tokenized_examples = tokenizer(
        c_plus_q_4, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)

    if torch_device == 'cuda':
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    example_encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    return example_encoded
