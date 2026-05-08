import torch
from torch.func import jacrev, vmap


def compute_velocity_and_acceleration_components(model, x, t, eps: float = 1e-8):
    """
    Computes velocity and acceleration components for a velocity field model.
    
    Material acceleration: a = d_t v + (Dv) v
    Normal acceleration: a_normal = a - a_tangent
    Curvature: kappa = ||a_normal|| / ||v||^2

    Args:
        model: callable with signature model(x, t) -> [B, D]
        x: [B, D]
        t: [B]
        eps: small constant for numerical stability

    Returns:
        v: [B, D] - velocity field
        acceleration: [B, D] - full material acceleration
        acceleration_normal: [B, D] - normal component of acceleration
        curvature: [B] - trajectory curvature
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
    curvature = acceleration_normal.norm(dim=-1) / (
        v.norm(dim=-1) ** 2 + eps
    )

    return v, acceleration, acceleration_normal, curvature


def normal_acceleration_penalty_loss(model, x, t, eps: float = 1e-8):
    """
    Penalizes normal component of material acceleration.
    Encourages straighter trajectories.
    """
    _, _, acceleration_normal, _ = compute_velocity_and_acceleration_components(
        model=model,
        x=x,
        t=t,
        eps=eps,
    )
    return (acceleration_normal.norm(dim=-1) ** 2).mean()


def full_acceleration_penalty_loss(model, x, t, eps: float = 1e-8):
    """
    Penalizes full material acceleration.
    Encourages smoother velocity fields.
    """
    _, acceleration, _, _ = compute_velocity_and_acceleration_components(
        model=model,
        x=x,
        t=t,
        eps=eps,
    )
    return (acceleration.norm(dim=-1) ** 2).mean()


def curvature_penalty_loss(model, x, t, eps: float = 1e-8):
    """
    Penalizes trajectory curvature.
    Encourages straighter trajectories.
    """
    _, _, _, curvature = compute_velocity_and_acceleration_components(
        model=model,
        x=x,
        t=t,
        eps=eps,
    )
    return (curvature ** 2).mean()


def velocity_consistency_loss(model, x, t, epsilon: float = 0.01):
    """
    Penalizes local variation of the velocity field over small time steps.
    Encourages smooth changes in velocity magnitude and direction.
    """
    v_t = model(x, t)
    x_next = x + epsilon * v_t
    t_next = t + epsilon
    v_t_next = model(x_next, t_next)
    # x_pred = x + epsilon * v_t_next
    return ((v_t_next - v_t) ** 2).mean()


def direction_consistency_loss(model, x, t, epsilon: float = 0.01, eps: float = 1e-8):
    """
    Penalizes changes in velocity direction over small time steps.
    Encourages consistent flow direction locally.
    """
    v_t = model(x, t)
    x_next = x + epsilon * v_t
    t_next = t + epsilon
    v_t_next = model(x_next, t_next)
    
    v_t_unit = v_t / (v_t.norm(dim=-1, keepdim=True) + eps)
    v_t_next_unit = v_t_next / (v_t_next.norm(dim=-1, keepdim=True) + eps)

    return (1 - (v_t_unit * v_t_next_unit).sum(-1)).mean()
