"""Opt-in, weights-only-compatible RNG snapshots for interrupted new runs."""
import random

import numpy as np
import torch


def capture_rng_state():
    numpy = np.random.get_state()
    return dict(python=random.getstate(), numpy=[numpy[0], numpy[1].tolist(), *numpy[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng_state(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
