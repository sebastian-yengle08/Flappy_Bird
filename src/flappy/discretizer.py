from .env import FlappyBirdEnv

# Celda 4 – 3.2 Normalización y discretización (Módulo 2: StateDiscretizer)
class StateDiscretizer:
    def __init__(self, cfg):
        self.bins = cfg["bins"]; self.dy_range = cfg["dy_range"]
        self.W = FlappyBirdEnv.W; self.VT = FlappyBirdEnv.V_MAX

    def normalize(self, obs):
        dx, dy, vel = obs
        ndx = min(max(dx / self.W, 0.0), 1.0)                  # [0, 1]  (clipping)
        ndy = min(max(dy / self.dy_range, -1.0), 1.0)          # [-1, 1]
        nv  = min(max(vel / self.VT, -1.0), 1.0)               # [-1, 1]
        return ndx, ndy, nv

    def discretize(self, obs):
        ndx, ndy, nv = self.normalize(obs); b = self.bins
        return (min(int(ndx * b[0]), b[0] - 1),
                min(int((ndy + 1) / 2 * b[1]), b[1] - 1),
                min(int((nv + 1) / 2 * b[2]), b[2] - 1))
