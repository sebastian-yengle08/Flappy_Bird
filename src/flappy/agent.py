import pickle
import random

import numpy as np

# Celda 6 – 4.2 Módulo 3: QLearningAgent (RF-03, RF-07)
class QLearningAgent:
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.Q = np.full(tuple(cfg["bins"]) + (2,), cfg["q_init"], dtype=np.float64)
        self.visits = np.zeros(tuple(cfg["bins"]), dtype=np.int64)
        self.alpha, self.gamma = cfg["alpha"], cfg["gamma"]
        self.eps0, self.eps_min, self.decay, self.eps = cfg["eps0"], cfg["eps_min"], cfg["decay"], cfg["eps0"]
        self.rng = random.Random(seed)

    def select_action(self, s, greedy=False):           # política ε-greedy
        if not greedy and self.rng.random() < self.eps:
            return self.rng.randint(0, 1)               # exploración
        q = self.Q[s]
        return int(q[1] > q[0])                         # explotación (empate → no saltar)

    def update(self, s, a, r, s_next, terminal):        # Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
        target = r if terminal else r + self.gamma * self.Q[s_next].max()
        self.Q[s + (a,)] += self.alpha * (target - self.Q[s + (a,)])
        self.visits[s] += 1

    def end_episode(self, episode):                     # ε = max(ε_min, ε0·decay^episodio)
        self.eps = max(self.eps_min, self.eps0 * self.decay ** episode)

    def save(self, path="qtable_flappy.pkl"):
        with open(path, "wb") as f: pickle.dump(dict(Q=self.Q, visits=self.visits, cfg=self.cfg), f)

    @classmethod
    def load(cls, path="qtable_flappy.pkl"):
        with open(path, "rb") as f: d = pickle.load(f)
        ag = cls(d["cfg"]); ag.Q, ag.visits, ag.eps = d["Q"], d["visits"], 0.0
        return ag

class RandomAgent:                                       # línea base: acciones aleatorias
    def __init__(self, seed=0): self.rng = random.Random(seed)
    def select_action(self, s, greedy=False): return self.rng.randint(0, 1)
