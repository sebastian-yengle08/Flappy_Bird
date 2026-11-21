import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.flappy.config import CONFIG
from src.flappy.env import FlappyBirdEnv
from src.flappy.discretizer import StateDiscretizer
from src.flappy.agent import QLearningAgent, RandomAgent
from src.flappy.training import run_episode


def test_env_reset_and_step():
    env = FlappyBirdEnv(seed=0)
    obs = env.reset()
    assert len(obs) == 3
    nxt, r, done, info = env.step(0)
    assert len(nxt) == 3 and isinstance(done, bool) and "score" in info


def test_env_reproducible():
    a, b = FlappyBirdEnv(), FlappyBirdEnv()
    assert a.reset(7) == b.reset(7)


def test_discretizer_bounds():
    d = StateDiscretizer(CONFIG)
    for obs in [(0, -1e6, -50), (500, 1e6, 50), (100, 0, 0)]:
        s = d.discretize(obs)
        assert all(0 <= v < n for v, n in zip(s, CONFIG["bins"]))


def test_q_update_bellman():
    ag = QLearningAgent(CONFIG)
    s, s2 = (0, 0, 0), (1, 1, 1)
    ag.update(s, 1, 1.1, s2, False)
    expected = 2.0 + 0.1 * (1.1 + 0.95 * 2.0 - 2.0)  # = 2.1
    assert abs(ag.Q[s + (1,)] - expected) < 1e-9


def test_epsilon_decay():
    ag = QLearningAgent(CONFIG)
    ag.end_episode(10_000)
    assert ag.eps == CONFIG["eps_min"]


def test_qtable_shape_and_persistence(tmp_path):
    ag = QLearningAgent(CONFIG)
    assert ag.Q.shape == (15, 20, 12, 2)
    p = tmp_path / "q.pkl"; ag.save(str(p))
    assert np.array_equal(QLearningAgent.load(str(p)).Q, ag.Q)


def test_trained_agent_beats_random():
    q = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qtable_flappy.npy")
    ag = QLearningAgent(CONFIG); ag.Q = np.load(q); ag.eps = 0.0
    d = StateDiscretizer(CONFIG)
    env = FlappyBirdEnv(max_steps=2000)
    trained = np.mean([run_episode(env, d, ag, seed=s)[0] for s in range(10_000, 10_020)])
    rand = np.mean([run_episode(env, d, RandomAgent(0), seed=s)[0] for s in range(10_000, 10_020)])
    assert trained > rand
