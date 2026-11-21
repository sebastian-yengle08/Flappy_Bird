"""Entrena el agente (seed 42) y guarda la Q-Table en models/. Uso: python train.py"""
import os
import numpy as np
from src.flappy.config import CONFIG, TEST_SEED0
from src.flappy.discretizer import StateDiscretizer
from src.flappy.training import train, evaluate

if __name__ == "__main__":
    agent, log, val_log, final_Q = train(CONFIG, CONFIG["n_train"], CONFIG["eval_every"], CONFIG["eval_episodes"])
    os.makedirs("models", exist_ok=True)
    agent.save("models/qtable_flappy.pkl")
    np.save("models/qtable_flappy.npy", agent.Q)
    test = evaluate(agent, StateDiscretizer(CONFIG), range(TEST_SEED0, TEST_SEED0 + CONFIG["n_test"]), CONFIG["max_steps_test"])
    print(f"Prueba ({CONFIG['n_test']} ep.): media={test.score.mean():.1f} max={test.score.max()} "
          f"supervivencia={1 - test.choque.mean():.1%}")
