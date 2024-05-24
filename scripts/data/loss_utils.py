import torch
import torch.nn.functional as F


def lx(x: torch.Tensor, x_pred: torch.Tensor, m: torch.Tensor):
    x_masked = x * m
    x_pred_masked = x_pred * m

    l1_loss = F.l1_loss(x_masked, x_pred_masked, reduction="sum")

    mask_count = m.sum()

    if mask_count == 0:
        return torch.tensor(0.0)
    normalized_loss = l1_loss / mask_count
    return normalized_loss


def lm(m_pred: torch.Tensor, m_target: torch.Tensor):
    m_pred = m_pred.float()
    m_target = m_target.float()
    # print(f"m_pred.size(): {m_pred.size()}")
    # print(f"m_target.size(): {m_target.size()}")
    # print(m_pred)
    # print(m_target)
    return F.binary_cross_entropy(m_pred, m_target)


def custom_loss(x, x_pred, m, m_pred):
    return 2 * lx(x, x_pred, m) + lm(m_pred, m)
