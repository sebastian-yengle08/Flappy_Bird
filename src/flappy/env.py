import random

# Módulo 1: FlappyBirdEnv (RF-01). Simulación headless de la física de Flappy Bird.
class FlappyBirdEnv:
    W, H, GROUND = 288, 512, 400            # pantalla y altura del suelo (px)
    BIRD_X, BIRD_R = 60, 12                 # posición horizontal fija y radio del pájaro
    GRAVITY, FLAP_V, V_MAX = 1.0, -9.0, 10.0
    PIPE_W, PIPE_SPEED, GAP, SPACING = 52, 4, 100, 180

    def __init__(self, seed=None, max_steps=5000):
        self.rng = random.Random(seed)
        self.max_steps = max_steps

    def _new_pipe(self, x):                 # [x, centro de la abertura, ya_cruzado]
        lo, hi = self.GAP // 2 + 40, self.GROUND - self.GAP // 2 - 40
        return [x, self.rng.randint(lo, hi), False]

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.y, self.vel, self.score, self.steps = 256.0, 0.0, 0, 0
        self.pipes = [self._new_pipe(self.W + 40)]
        while len(self.pipes) < 4:
            self.pipes.append(self._new_pipe(self.pipes[-1][0] + self.SPACING))
        return self._obs()

    def _next_pipe(self):
        for p in self.pipes:
            if p[0] + self.PIPE_W >= self.BIRD_X - self.BIRD_R:
                return p
        return self.pipes[-1]

    def _obs(self):
        """Observación continua: (dx, dy, vel). dy>0: pájaro por encima del centro; vel>0: ascenso."""
        p = self._next_pipe()
        return (max(0.0, p[0] - self.BIRD_X), p[1] - self.y, -self.vel)

    def step(self, action):                 # action: 0 = no hacer nada, 1 = saltar
        self.steps += 1
        self.vel = self.FLAP_V if action == 1 else min(self.vel + self.GRAVITY, self.V_MAX)
        self.y += self.vel
        for p in self.pipes:
            p[0] -= self.PIPE_SPEED
        if self.pipes[0][0] + self.PIPE_W < 0:
            self.pipes.pop(0)
            self.pipes.append(self._new_pipe(self.pipes[-1][0] + self.SPACING))
        r, crashed = self.BIRD_R, False
        if self.y - r < 0 or self.y + r > self.GROUND:
            crashed = True
        else:
            for p in self.pipes:
                if p[0] < self.BIRD_X + r and p[0] + self.PIPE_W > self.BIRD_X - r:
                    if self.y - r < p[1] - self.GAP / 2 or self.y + r > p[1] + self.GAP / 2:
                        crashed = True
                    break
        reward = -1.0 if crashed else 0.1                      # RF-04: -1 colisión, +0.1 por sobrevivir
        if not crashed:
            for p in self.pipes:
                if not p[2] and p[0] + self.PIPE_W < self.BIRD_X - r:
                    p[2] = True; self.score += 1; reward += 1.0  # +1 por cruzar un par de tubos
        done = crashed or self.steps >= self.max_steps         # límite de pasos = episodio superado
        return self._obs(), reward, done, {"score": self.score, "crashed": crashed}

    def snapshot(self):
        return dict(y=self.y, pipes=[list(p) for p in self.pipes], score=self.score)
