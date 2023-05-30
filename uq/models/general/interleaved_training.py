import torch


def compute_loss_model(model, *eval_args):
    # `eval_args` contains objects necessary for the evaluation such as `dist`, `quantiles` and `y`
    base_loss = torch.full([], torch.nan)
    regul = torch.full([], torch.nan)

    if model.hparams.interleaved:
        if model.current_epoch % 2 == 1:   # Training dataset
            base_loss = model.compute_base_loss(*eval_args)
            loss = base_loss
        else:   # Interleaved dataset
            regul = model.compute_regul(*eval_args)
            loss = model.hparams.lambda_ * regul
    else:
        base_loss = model.compute_base_loss(*eval_args)
        loss = base_loss
        if model.hparams.lambda_ != 0:
            regul = model.compute_regul(*eval_args)
            loss = loss + model.hparams.lambda_ * regul

    return loss, base_loss, regul
