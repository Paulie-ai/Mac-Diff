import torch


def load_pretrain(model, pt, device):
    pretrained_ckpt = torch.load(pt, map_location=device)['state_dict']
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_ckpt)
    model.load_state_dict(model_state_dict)
    return model


def geometry_loss(pred_geom, native_geom, device):
    loss = 0
    count = 0
    for k in pred_geom:
        native = native_geom[k].to(device)
        loss += -(torch.log(pred_geom[k].squeeze(0) + 1e-6) * native).sum(dim=-1).mean()
        count += 1
    return loss / count

def distill_loss(pred_geom, soft_labels, device):
    loss = 0
    count = 0
    for k in pred_geom:
        soft_label = soft_labels[k].to(device)
        loss += KLloss(pred_geom[k].squeeze(0),soft_label)
        count += 1
    return loss / count


def KLloss(pred_prob, real_prob):
    """
    :param pred_prob: *,d
    :param real_prob: *,d
    :return:
    """
    p, q = real_prob, pred_prob
    return (p * (torch.log(p + 1e-6) - torch.log(q + 1e-6))).sum(dim=-1).mean()

