"""Hiperparámetros y bloques de semillas (sección 4.2 del informe)."""
SEED = 42
CONFIG = dict(
    alpha=0.1, gamma=0.95,
    eps0=1.0, eps_min=0.01, decay=0.995,
    bins=(15, 20, 12),
    q_init=2.0,
    dy_range=128.0,
    n_train=10_000, n_test=200,
    eval_every=500, eval_episodes=50,
    max_steps_train=5_000, max_steps_val=5_000, max_steps_test=10_000,
    seed=SEED,
)
VAL_SEED0, TEST_SEED0, CROSS_SEED0 = 20_000, 10_000, 50_000
