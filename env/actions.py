import numpy as np

def action_set(name: str):
    """ actiunile pe care le face masina"""
    if name == "dqn5_simple":
        return [
            (-1.0, 0.6, 0.0),
            ( 1.0, 0.6, 0.0),
            ( 0.0, 0.8, 0.0),
            ( 0.0, 0.0, 0.8),
            ( 0.0, 0.0, 0.0),
        ]

