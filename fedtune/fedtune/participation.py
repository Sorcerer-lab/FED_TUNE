"""Who takes part in a round. Step 3: everyone. Step 4 replaces this with simulated dropout."""


def sample_participation(cfg: dict, n_clients: int, round_idx: int):
    """Return (participating client ids, dropped client ids)."""
    return list(range(n_clients)), []
