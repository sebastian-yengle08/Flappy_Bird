import time

import numpy as np
import pandas as pd

from .config import VAL_SEED0
from .env import FlappyBirdEnv
from .discretizer import StateDiscretizer
from .agent import QLearningAgent


# Celda 7 – 4.2 Módulo 4/5: bucle de episodio, evaluación y entrenamiento con validación periódica
def run_episode(env, disc, agent, train=False, seed=None):
    s = disc.discretize(env.reset(seed)); total, n = 0.0, 0
    while True:
        a = agent.select_action(s, greedy=not train)
        obs, r, done, info = env.step(a); s2 = disc.discretize(obs)
        if train: agent.update(s, a, r, s2, info["crashed"])
        s, total, n = s2, total + r, n + 1
        if done: return info["score"], total, n, info["crashed"]

def evaluate(agent, disc, seeds, max_steps):
    env = FlappyBirdEnv(max_steps=max_steps)
    res = [run_episode(env, disc, agent, train=False, seed=sd) for sd in seeds]
    return pd.DataFrame(res, columns=["score", "recompensa", "pasos", "choque"])

def train(cfg, n_episodes, eval_every, eval_episodes, verbose=True):
    env = FlappyBirdEnv(seed=cfg["seed"], max_steps=cfg["max_steps_train"])
    disc, agent = StateDiscretizer(cfg), QLearningAgent(cfg, seed=cfg["seed"])
    log, val_log, best_score, best_Q = [], [], -1, agent.Q.copy()
    val_seeds = range(VAL_SEED0, VAL_SEED0 + eval_episodes); t0 = time.time()
    for ep in range(1, n_episodes + 1):
        score, ret, n, _ = run_episode(env, disc, agent, train=True)
        agent.end_episode(ep)
        seen = agent.visits > 0
        log.append(dict(episodio=ep, score=score, recompensa=ret, pasos=n, epsilon=agent.eps,
                        q_medio=float(agent.Q[seen].mean())))
        if ep % eval_every == 0:                                  # validación greedy (Q congelada)
            v = evaluate(agent, disc, val_seeds, cfg["max_steps_val"])
            val_log.append(dict(episodio=ep, media=v.score.mean(), std=v.score.std(), max=v.score.max()))
            if v.score.mean() > best_score:                       # checkpoint de la mejor Q-Table
                best_score, best_Q = v.score.mean(), agent.Q.copy()
            if verbose and ep % (eval_every * 2) == 0:
                print(f"ep {ep:>6} | ε={agent.eps:.3f} | score train(últ.500)={np.mean([l['score'] for l in log[-500:]]):6.1f}"
                      f" | validación={v.score.mean():6.1f} (máx {v.score.max()}) | {time.time()-t0:5.0f}s")
    final_Q = agent.Q.copy()
    agent.Q = best_Q                                              # el agente final usa el mejor checkpoint
    return agent, pd.DataFrame(log), pd.DataFrame(val_log), final_Q
