"""App web (Flask) desplegable en Vercel: el agente Q-Learning juega Flappy Bird con la Q-Table entrenada.

Local:  pip install -r requirements.txt && flask --app app run
"""
import os

import numpy as np
from flask import Flask, jsonify, request, Response

from src.flappy.config import CONFIG
from src.flappy.env import FlappyBirdEnv
from src.flappy.discretizer import StateDiscretizer
from src.flappy.agent import QLearningAgent

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))

_agent = QLearningAgent(CONFIG)
_agent.Q = np.load(os.path.join(HERE, "models", "qtable_flappy.npy"))
_agent.eps = 0.0
_disc = StateDiscretizer(CONFIG)


def play(seed: int, max_steps: int):
    """Un episodio con política greedy (ε = 0). Devuelve fotogramas compactos."""
    env = FlappyBirdEnv(max_steps=max_steps)
    s = _disc.discretize(env.reset(seed))
    snap = env.snapshot()
    frames = [[round(snap["y"], 1), [[p[0], p[1]] for p in snap["pipes"]], 0, 0]]
    while True:
        a = _agent.select_action(s, greedy=True)
        obs, _, done, info = env.step(a)
        s = _disc.discretize(obs)
        snap = env.snapshot()
        frames.append([round(snap["y"], 1), [[p[0], p[1]] for p in snap["pipes"]], snap["score"], a])
        if done:
            return frames, info["score"], info["crashed"]


@app.get("/api/play")
def api_play():
    seed = request.args.get("seed", 10000, type=int)
    max_steps = min(max(request.args.get("max_steps", 1500, type=int), 50), 3000)
    frames, score, crashed = play(seed, max_steps)
    E = FlappyBirdEnv
    return jsonify(frames=frames, score=score, crashed=crashed, steps=len(frames) - 1,
                   env=dict(W=E.W, H=E.H, GROUND=E.GROUND, BIRD_X=E.BIRD_X, BIRD_R=E.BIRD_R,
                            PIPE_W=E.PIPE_W, GAP=E.GAP))


@app.get("/api/info")
def api_info():
    return jsonify({k: CONFIG[k] for k in ("alpha", "gamma", "eps0", "eps_min", "decay", "bins", "q_init", "dy_range")})


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agente Q-Learning · Flappy Bird</title>
<style>
body{font-family:Georgia,'Times New Roman',serif;background:#f4f6fa;color:#1a1a1a;margin:0}
main{max-width:760px;margin:0 auto;padding:24px 16px}
h1{color:#1f3864;font-size:1.6rem;margin:0 0 4px}
p.sub{color:#555;margin:0 0 18px}
.row{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}
canvas{background:#70c5ce;border:1px solid #1f3864;border-radius:4px;max-width:100%}
.panel{flex:1;min-width:240px}
label{display:block;font-size:.9rem;margin:10px 0 4px}
input,select{padding:6px 8px;border:1px solid #9db0cf;border-radius:4px;width:100%;box-sizing:border-box}
button{margin-top:14px;padding:8px 16px;background:#1f3864;color:#fff;border:0;border-radius:4px;cursor:pointer}
.m{display:flex;gap:12px;margin-top:18px}.m div{flex:1;background:#fff;border:1px solid #d5ddec;border-radius:4px;padding:8px}
.m b{display:block;font-size:1.3rem;color:#1f3864}
small{color:#555}
pre{background:#eef2f9;padding:10px;border-radius:4px;font-size:.8rem;overflow:auto}
</style></head><body><main>
<h1>Agente Q-Learning para Flappy Bird</h1>
<p class="sub">IA Principios y Técnicas · UPAO (NRC 5028). Política greedy (ε = 0) sobre la Q-Table entrenada.</p>
<div class="row">
<canvas id="c" width="288" height="512"></canvas>
<div class="panel">
<label for="seed">Semilla del entorno <small>(10000–10199: semillas de prueba del informe)</small></label>
<input id="seed" type="number" value="10000" min="0">
<label for="ms">Límite de pasos</label>
<select id="ms"><option>500</option><option>1000</option><option selected>1500</option><option>3000</option></select>
<button id="go">Jugar episodio</button>
<div class="m"><div><small>Puntaje</small><b id="sc">–</b></div><div><small>Pasos</small><b id="st">–</b></div><div><small>Resultado</small><b id="rs">–</b></div></div>
<p><small id="act"></small></p>
<pre>Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
estado (dx, dy, vel) → 15×20×12 = 3 600 estados</pre>
</div></div></main>
<script>
const cv=document.getElementById('c'),g=cv.getContext('2d');let timer=null;
function draw(f,E){const[y,pipes,score]=f;g.fillStyle='#70c5ce';g.fillRect(0,0,E.W,E.GROUND);g.fillStyle='#ded895';g.fillRect(0,E.GROUND,E.W,E.H-E.GROUND);
g.fillStyle='#5aa02c';g.strokeStyle='#000';for(const[x,cy]of pipes){g.fillRect(x,0,E.PIPE_W,cy-E.GAP/2);g.strokeRect(x,0,E.PIPE_W,cy-E.GAP/2);
g.fillRect(x,cy+E.GAP/2,E.PIPE_W,E.GROUND-cy-E.GAP/2);g.strokeRect(x,cy+E.GAP/2,E.PIPE_W,E.GROUND-cy-E.GAP/2);}
g.fillStyle='#f7d51d';g.beginPath();g.arc(E.BIRD_X,y,E.BIRD_R,0,7);g.fill();g.stroke();
g.fillStyle='#fff';g.font='bold 30px Georgia';g.textAlign='center';g.fillText(score,E.W/2,60);}
async function run(){clearInterval(timer);const r=await fetch('/api/play?seed='+seed.value+'&max_steps='+ms.value);const d=await r.json();
sc.textContent=d.score;st.textContent=d.steps;rs.textContent=d.crashed?'Choque':'Superó el límite';let k=0;
timer=setInterval(()=>{draw(d.frames[k],d.env);act.textContent=k?('Acción: '+(d.frames[k][3]?'saltar':'no hacer nada')):'';if(++k>=d.frames.length)clearInterval(timer);},25);}
go.onclick=run;run();
</script></body></html>"""
