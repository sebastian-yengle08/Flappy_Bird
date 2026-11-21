"""App web (Streamlit): el agente Q-Learning juega Flappy Bird con la Q-Table entrenada."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import streamlit as st

from src.flappy.config import CONFIG
from src.flappy.env import FlappyBirdEnv
from src.flappy.discretizer import StateDiscretizer
from src.flappy.agent import QLearningAgent

st.set_page_config(page_title="Agente Q-Learning · Flappy Bird", page_icon="🐤", layout="centered")


@st.cache_resource
def load_agent():
    ag = QLearningAgent(CONFIG)
    ag.Q = np.load("models/qtable_flappy.npy")
    ag.eps = 0.0
    return ag


@st.cache_data(show_spinner=False)
def play(seed: int, max_steps: int):
    agent, disc = load_agent(), StateDiscretizer(CONFIG)
    env = FlappyBirdEnv(max_steps=max_steps)
    s = disc.discretize(env.reset(seed))
    frames, actions = [env.snapshot()], []
    while True:
        a = agent.select_action(s, greedy=True)
        obs, r, done, info = env.step(a)
        s = disc.discretize(obs)
        frames.append(env.snapshot()); actions.append(a)
        if done:
            return frames, actions, info["score"], info["crashed"]


def draw(s):
    E = FlappyBirdEnv
    fig, ax = plt.subplots(figsize=(2.9, 5.1))
    ax.set_xlim(0, E.W); ax.set_ylim(E.H, 0); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), E.W, E.GROUND, color="#70c5ce"))
    ax.add_patch(Rectangle((0, E.GROUND), E.W, E.H - E.GROUND, color="#ded895"))
    for x, cy, _ in s["pipes"]:
        ax.add_patch(Rectangle((x, 0), E.PIPE_W, cy - E.GAP / 2, color="#5aa02c", ec="k"))
        ax.add_patch(Rectangle((x, cy + E.GAP / 2), E.PIPE_W, E.GROUND - cy - E.GAP / 2, color="#5aa02c", ec="k"))
    ax.add_patch(Circle((E.BIRD_X, s["y"]), E.BIRD_R, color="#f7d51d", ec="k", zorder=5))
    ax.text(E.W / 2, 50, str(s["score"]), ha="center", fontsize=22, weight="bold", color="white")
    fig.tight_layout(pad=0)
    return fig


st.title("🐤 Agente Q-Learning para Flappy Bird")
st.caption("Proyecto de IA · UPAO · IA Principios y Técnicas (NRC 5028). Política greedy (ε = 0) sobre la Q-Table entrenada.")

with st.sidebar:
    st.header("Episodio")
    seed = st.number_input("Semilla del entorno", min_value=0, value=10_000, step=1,
                           help="10 000–10 199: semillas de prueba del informe.")
    max_steps = st.select_slider("Límite de pasos", options=[500, 1000, 2000, 5000, 10000], value=2000)
    st.header("Hiperparámetros")
    st.json({k: CONFIG[k] for k in ("alpha", "gamma", "eps0", "eps_min", "decay", "bins", "q_init", "dy_range")})

frames, actions, score, crashed = play(int(seed), int(max_steps))
c1, c2, c3 = st.columns(3)
c1.metric("Puntaje", score)
c2.metric("Pasos", len(actions))
c3.metric("Resultado", "Choque" if crashed else "Superó el límite")

k = st.slider("Fotograma (paso)", 0, len(frames) - 1, min(len(frames) - 1, 0))
fig = draw(frames[k])
st.pyplot(fig, width="content")
plt.close(fig)
if k > 0:
    st.write("Acción en este paso:", "🔼 saltar" if actions[k - 1] == 1 else "⏬ no hacer nada")

with st.expander("¿Cómo decide el agente?"):
    st.markdown(
        "El estado (dx, dy, vel) se normaliza y discretiza en 15×20×12 = 3 600 estados. "
        "En cada paso elige la acción con mayor valor Q: "
        "`Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]`."
    )
