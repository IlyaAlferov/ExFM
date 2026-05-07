import torch
from torch.func import jacrev, vmap


def compute_velocity_and_normal_acceleration(model, x, t, eps: float = 1e-8):
    """
    Computes:
    v(x, t) normal component of material acceleration: a = d_t v + (Dv) v

    Args:
        model: callable with signature model(x, t) -> [B, D]
        x: [B, D]
        t: [B]

    Returns:
        v: [B, D]
        acceleration_normal: [B, D]
    """
    if not x.requires_grad:
        x = x.requires_grad_(True)
    if not t.requires_grad:
        t = t.requires_grad_(True)

    def f_single(x_i, t_i):
        out = model(x_i.unsqueeze(0), t_i.unsqueeze(0))   # [1, D]
        return out.squeeze(0)                             # [D]

    v = model(x, t)  # [B, D]

    dv_dx, dv_dt = vmap(jacrev(f_single, argnums=(0, 1)))(x, t)
    # dv_dx: [B, D, D]
    # dv_dt: [B, D]

    acceleration = dv_dt + torch.einsum("bij,bj->bi", dv_dx, v)

    v_unit = v / (v.norm(dim=-1, keepdim=True) + eps)
    acceleration_tang = (acceleration * v_unit).sum(dim=-1, keepdim=True) * v_unit
    acceleration_normal = acceleration - acceleration_tang

    return v, acceleration_normal


def acceleration_penalty_loss(model, x, t, eps: float = 1e-8):
    """
    Penalizes normal component of material acceleration.
    Encourages straighter trajectories.
    """
    _, acceleration_normal = compute_velocity_and_normal_acceleration(
        model=model,
        x=x,
        t=t,
        eps=eps,
    )
    return (acceleration_normal ** 2).mean()


def consistency_loss(model, x, t, epsilon: float = 0.01):
    """
    One-step local consistency regularizer.

    Euler step:
        x_next = x + eps * v(x, t)
    Compare with:
        x_pred = x + eps * v(x_next, t + eps)

    Penalizes local variation of the field.
    """
    v_t = model(x, t)
    x_next = x + epsilon * v_t
    t_next = t + epsilon
    v_t_next = model(x_next, t_next)
    # x_pred = x + epsilon * v_t_next
    return ((v_t_next - v_t) ** 2).mean()
