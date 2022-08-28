import sys
sys.path.insert(0,'..')

import os
import argparse
import math
import ruamel.yaml as yaml
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from src.utils import set_optional_args, get_num_labels
from src.training_logger import TrainLogger


def normal_equation(X, Y):
    return torch.inverse(X.T@X)@X.T@Y


def map_analytical(train_ds, val_ds, loss_fn, train_logger):

    X_train, Y_train, _, _ = train_ds.tensors
    X_val, Y_val, _, _ = val_ds.tensors

    W = normal_equation(X_train, Y_train)

    Y_hat_train = X_train@W
    Y_hat_val = X_val@W

    loss_train_analytical = loss_fn(Y_hat_train, Y_train).mean().item()
    loss_val_analytical = loss_fn(Y_hat_val, Y_val).mean().item()

    train_logger.reset()
    train_logger.steps = train_logger.logging_step
    train_logger.step_loss(
        step = 0,
        loss = loss_train_analytical,
        increment_steps = False,
        suffix = "analytical"
    )
    train_logger.validation_loss(
        eval_step = 0,
        result = {"loss": loss_val_analytical},
        suffix = "analytical"
    )

    print(f"train loss analytical: {loss_train_analytical:.3f}")
    print(f"val loss analytical: {loss_val_analytical:.3f}")

    # Version with bias (doesnt work as well)
    X_train_ = torch.cat([X_train, torch.ones((X_train.shape[0],1))], dim=1)
    X_val_ = torch.cat([X_val, torch.ones((X_val.shape[0],1))], dim=1)

    W_ = normal_equation(X_train_, Y_train)

    Y_hat_train_ = X_train_@W_
    Y_hat_val_ = X_val_@W_

    loss_train_ = loss_fn(Y_hat_train_, Y_train).mean().item()
    loss_val_ = loss_fn(Y_hat_val_, Y_val).mean().item()
    print(f"train loss analytical (with bias): {loss_train_:.3f}")
    print(f"val loss analytical (with bias): {loss_val_:.3f}")

    # import IPython; IPython.embed(); exit(1)

    return W, torch.zeros((W.shape[0],)) # placeholder for bias


def map_gradient_based(train_ds, val_ds, loss_fn, train_logger):

    hparams = {
        "batch_size": 128,
        "lr": 1e-4,
        "num_epochs": 10,
        "seed": 0
    }
    hparams = argparse.Namespace(**hparams)

    torch.manual_seed(hparams.seed)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=hparams.batch_size, drop_last=False)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=hparams.batch_size, drop_last=False)

    emb_size = train_ds.tensors[0].shape[1]

    llayer = torch.nn.Linear(emb_size, emb_size)
    optimizer = SGD(llayer.parameters(), lr=hparams.lr)

    train_logger.reset()

    train_str = "Epoch {}, val loss: {:7.5f}"
    train_iterator = trange(hparams.num_epochs, desc=train_str.format(0, math.nan), leave=False, position=0)
    for epoch in train_iterator:

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, (X, Y, _, _) in enumerate(epoch_iterator):

            llayer.train()

            outputs = llayer(X)
            loss = loss_fn(outputs, Y)
            loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            llayer.zero_grad()

            train_logger.step_loss(
                step = epoch * len(train_loader) + step,
                loss = loss.item(),
                suffix = "gradient_based"
            )

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

        val_iterator = tqdm(val_loader, desc=f"evaluating", leave=False, position=1)
        llayer.eval()
        for step, (X, Y, _, _) in enumerate(val_iterator):
            loss_val = []
            with torch.no_grad():
                outputs = llayer(X)
                loss_val.append(loss_fn(outputs, Y))
            loss_val = torch.cat(loss_val).mean()
        
        train_logger.validation_loss(
            eval_step = epoch,
            result = {"loss": loss_val.item()},
            suffix = "gradient_based"
        )

        train_iterator.set_description(train_str.format(epoch, loss_val.item()), refresh=True)

    with torch.no_grad():
        Y_hat_train = llayer(train_ds.tensors[0])
        Y_hat_val = llayer(val_ds.tensors[0])

    loss_train_sgd = loss_fn(Y_hat_train, train_ds.tensors[1])
    loss_val_sgd = loss_fn(Y_hat_val, val_ds.tensors[1])

    print(f"train loss sgd: {loss_train_sgd.mean().item():.3f}")
    print(f"val loss sgd: {loss_val_sgd.mean().item():.3f}")

    return llayer.weight.data, llayer.bias.data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bertl4", help="bertbase or bertl4")
    parser.add_argument("--emb_type_id", type=int, help="embedding type id")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    args = parser.parse_args()


    log_dir = "../logs_linear_map"
    emb_dir = f"../embeddings/{args.ds}/{args.model_type}"
    emb_types = [
        "modularFalse_baseline",
        "modularFalse_fixmask0.1",
        "modularFalse_fixmask0.05",
        "modularTrue_baseline",
        "modularTrue_fixmask0.1",
        "modularTrue_fixmask0.05"
    ]
    emb_type = emb_types[args.emb_type_id]


    train_logger = TrainLogger(
        log_dir = log_dir,
        logger_name = "_".join([x for x in emb_dir.split("/") if x!=".."]) + "_" + emb_type,
        logging_step = 5
    )

    train_embeddings_ds = torch.load(os.path.join(emb_dir, f"train_embeddings_ds_{emb_type}.pth"))
    val_embeddings_ds = torch.load(os.path.join(emb_dir, f"val_embeddings_ds_{emb_type}.pth"))

    loss_fn = torch.nn.MSELoss(reduction="none")

    # analytical mapping
    W_ana, b_ana = map_analytical(train_embeddings_ds, val_embeddings_ds, loss_fn, train_logger)

    # gradient based mapping
    # W_grad, b_grad = map_gradient_based(train_embeddings_ds, val_embeddings_ds, loss_fn, train_logger)

    modified_train_embeddings_ds = TensorDataset(
        train_embeddings_ds.tensors[0],
        train_embeddings_ds.tensors[0]@W_ana + b_ana,
        train_embeddings_ds.tensors[2],
        train_embeddings_ds.tensors[3]
    )
    torch.save(modified_train_embeddings_ds, os.path.join(emb_dir, f"train_embeddings_ds_{emb_type}_modified.pth"))
    modified_val_embeddings_ds = TensorDataset(
        val_embeddings_ds.tensors[0],
        val_embeddings_ds.tensors[0]@W_ana + b_ana,
        val_embeddings_ds.tensors[2],
        val_embeddings_ds.tensors[3]
    )
    torch.save(modified_val_embeddings_ds, os.path.join(emb_dir, f"val_embeddings_ds_{emb_type}_modified.pth"))


if __name__ == "__main__":

    main()