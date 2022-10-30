import sys
sys.path.insert(0,'..')

import ruamel.yaml as yaml
import pickle
from itertools import product
import random
import argparse


random.seed(0)


def sample_equal(base_l, sample_l):
    q, mod = divmod(len(base_l), len(sample_l))
    n_samples = q * [len(sample_l)] + [mod]
    samples = []
    for n in n_samples:
        samples.extend(random.sample(sample_l, n))
    return samples


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="dataset")
    parser.add_argument("--pk", type=str, help="protected key to create dataset for")
    args = parser.parse_args()

    with open("../cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{args.ds}"
    args = argparse.Namespace(**cfg[data_cfg], **vars(args))

    with open(args.train_pkl, 'rb') as file:
        data_dicts = pickle.load(file)

    protected_key = args.protected_key
    if isinstance(protected_key, str):
        protected_key = [protected_key]

    keys = [args.task_key, *protected_key, args.text_key]
    x = [[d[k] for k in keys] for d in data_dicts]

    pk = protected_key if args.pk is None else [args.pk]
    pk_id = [i for i,p in enumerate(keys) if p in pk]

    data = dict(zip(keys, zip(*x)))

    protected_key_combs = dict(enumerate(product(set(data[args.task_key]), *[set(data[k]) for k in pk])))
    comb_data_dict = {i:[t for t in x if t[0]==v[0] and all([t[id]==v[j+1] for j, id in enumerate(pk_id)])] for i, v in protected_key_combs.items()}

    del data
    # import IPython; IPython.embed(); exit(1)
    

    triplet_list = []
    for i, data in comb_data_dict.items():
        data_dict = dict(zip(keys, zip(*data)))
        v = protected_key_combs[i]
        tv, pvs = v[0], v[1:]

        other_pv = {0: [k for k, v in protected_key_combs.items() if v[0]==tv and all([a!=b for a,b in zip(v[1:],pvs)])]}

        if len(pvs)>1:
            for j, pv in enumerate(pvs):
                pv_other = list(pvs[:j] + pvs[j+1:])
                for k, v in protected_key_combs.items():
                    v_other = v[1:j] + v[j+2:]
                    if v[0]==tv and v[j+1]!=pv and all([a==b for a,b in zip(v_other, pv_other)]):
                        try:
                            other_pv[j+1].append(k)
                        except KeyError:
                            other_pv[j+1] = [k]

        a_list = []
        for j, ids in other_pv.items():
            other_pv_texts = [t[-1] for id in ids for t in comb_data_dict[id]]
            samples = sample_equal(data_dict[args.text_key], other_pv_texts)
            a_list.append(samples)

        other_tv_texts = [t[-1] for t in x if t[0]!=tv]
        negative = sample_equal(data_dict[args.text_key], other_tv_texts)

        triplet_list.append((data_dict[args.text_key], data_dict[args.task_key], negative, *a_list, *[data_dict[pk] for pk in protected_key]))

    if len(pk)>1:
        other_pv_names = ["input_other_pv_all"] + [f"input_other_pv_{k}" for k in pk]
        weights = [len(pk)] + [1] * len(pk)
    else:
        other_pv_names = ["input_other_pv"]
        weights = [1]

    triplet_list = [[v for sub_l in l for v in sub_l] for l in zip(*triplet_list)]

    idx_list = list(range(len(triplet_list[0])))
    random.shuffle(idx_list)
    triplet_list = [[sub_l[i] for i in idx_list] for sub_l in triplet_list]


    weights = [1] if len(pk)==1 else [len(pk)] + [1] * len(pk)
    other_pv_names = [f"input_other_pv_{i}" for i in range(len(weights))]
    new_keys = [args.text_key, args.task_key, "input_other_tv", *protected_key]
    triplet_ds = []
    for x in zip(*triplet_list):
        a = list(x[:3] + x[-len(protected_key):])
        b = list(zip(x[3:-len(protected_key)], weights))
        triplet_ds.append(dict(zip(new_keys+other_pv_names, a+b)))

    with open(f"../train_triplet_{args.ds}_{'_'.join(pk)}.pkl","wb") as f:
        pickle.dump(triplet_ds, f)


if __name__ == "__main__":

    main()

