
def get_data_loader_pairs(
    tokenizer,
    data_path,
    labels_task_path,
    labels_prot_path,
    batch_size=16,
    max_length=200,
    raw=False,
    shuffle=True,
    debug=False
):

    def batch_fn_pairs(batch):
        b = [torch.stack(l) for l in zip(*batch)]
        x1 = {
            "input_ids": b[0],
            "token_type_ids": b[1],
            "attention_mask": b[2]
        }
        x2 = {
            "input_ids": b[5],
            "token_type_ids": b[6],
            "attention_mask": b[7]
        }
        return x1, b[3], b[4], x2, b[8], b[9]

    if raw:
        text_fn = lambda x: x['raw'][x['start_pos']:]
    else:
        text_fn = lambda x: x["bio"]

    with open(data_path, 'rb') as file:
        data, id_pairs = pickle.load(file)

    if debug:
        cutoff = min(int(batch_size*10), len(id_pairs))
        id_pairs = id_pairs[:cutoff]

    keys = ["gender", "title"]
    x = [[d[k] for k in keys] + [text_fn(d)] for d in data]
    keys.append("text")

    data_dict = dict(zip(keys, zip(*x)))

    input_ids, token_type_ids, attention_masks = multiprocess_tokenization(list(data_dict["text"]), tokenizer, max_length)

    labels_task = read_label_file(labels_task_path)
    labels_task = torch.tensor([labels_task[t] for t in data_dict["title"]], dtype=torch.long)
    labels_prot = read_label_file(labels_prot_path)
    labels_prot = torch.tensor([labels_prot[t] for t in data["gender"]], dtype=torch.long)

    tokenized_samples = []
    for id1, id2 in id_pairs:
        tokenized_samples.append((
            input_ids[id1],
            token_type_ids[id1],
            attention_masks[id1],
            labels_task[id1],
            labels_prot[id1],
            input_ids[id2],
            token_type_ids[id2],
            attention_masks[id2],
            labels_task[id2],
            labels_prot[id2],
        ))
    tds = [torch.stack(t) for t in zip(*tokenized_samples)]

    _dataset = TensorDataset(*tds)

    _loader = DataLoader(_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=batch_fn_pairs)

    return _loader