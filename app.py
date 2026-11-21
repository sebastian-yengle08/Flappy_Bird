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
:root{--navy:#1f3864;--line:#d5ddec;--bg:#f1f4fa}
*{box-sizing:border-box}
body{font-family:Georgia,'Times New Roman',serif;background:var(--bg);color:#111;margin:0}
header{background:var(--navy);color:#fff;padding:22px 16px}
header div{max-width:960px;margin:0 auto}
header h1{margin:0;font-size:1.7rem;letter-spacing:.2px}
header p{margin:6px 0 0;color:#d6e0f3;font-size:.95rem}
main{max-width:960px;margin:0 auto;padding:22px 16px 40px;display:grid;grid-template-columns:320px 1fr;gap:22px}
@media(max-width:760px){main{grid-template-columns:1fr}}
.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(31,56,100,.08)}
.card h2{font-size:1rem;margin:0 0 12px;color:var(--navy);border-bottom:2px solid var(--navy);padding-bottom:6px;text-transform:uppercase;letter-spacing:.5px}
canvas{display:block;margin:0 auto;border:2px solid var(--navy);border-radius:6px;max-width:100%;height:auto}
label{display:block;font-size:.88rem;margin:12px 0 4px;font-weight:bold}
input,select{padding:8px;border:1px solid #9db0cf;border-radius:4px;width:100%;font:inherit}
.btns{display:flex;gap:8px;margin-top:14px}
button{flex:1;padding:9px 12px;background:var(--navy);color:#fff;border:0;border-radius:4px;cursor:pointer;font:inherit}
button.alt{background:#fff;color:var(--navy);border:1px solid var(--navy)}
.m{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}
.m div{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;text-align:center}
.m small{display:block;color:#333;font-size:.78rem}.m b{font-size:1.5rem;color:var(--navy)}
table{width:100%;border-collapse:collapse;font-size:.88rem;margin-top:4px}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}th{background:#d9e2f3}
pre{background:var(--bg);padding:10px;border-radius:4px;font-size:.78rem;overflow:auto;margin:10px 0 0}
small.h{color:#333;font-weight:normal}
footer{text-align:center;font-size:.85rem;padding:0 16px 28px}footer a{color:var(--navy)}
</style></head><body>
<header><div><h1>Agente Q-Learning para Flappy Bird</h1>
<p>IA Principios y Técnicas · UPAO (NRC 5028) — política greedy (ε = 0) sobre la Q-Table entrenada</p></div></header>
<main>
<section class="card"><h2>Partida</h2><canvas id="c" width="288" height="512"></canvas></section>
<div>
<section class="card"><h2>Control</h2>
<div class="m"><div><small>Puntaje</small><b id="sc">–</b></div><div><small>Pasos</small><b id="st">–</b></div><div><small>Resultado</small><b id="rs" style="font-size:1.05rem">–</b></div></div>
<label for="seed">Semilla del entorno <small class="h">(10000–10199: semillas de prueba del informe)</small></label>
<input id="seed" type="number" value="10000" min="0">
<label for="ms">Límite de pasos</label>
<select id="ms"><option>500</option><option>1000</option><option selected>1500</option><option>3000</option></select>
<label for="sp">Velocidad</label>
<select id="sp"><option value="45">Lenta</option><option value="25" selected>Normal</option><option value="10">Rápida</option></select>
<div class="btns"><button id="go">Jugar episodio</button><button id="rnd" class="alt">Semilla al azar</button></div>
<p id="act" style="margin:12px 0 0;font-size:.9rem"></p></section>
<section class="card" style="margin-top:18px"><h2>Resultados del informe (semilla 42)</h2>
<table><tr><th>Métrica (200 ep. de prueba)</th><th>Valor</th></tr>
<tr><td>Puntaje medio</td><td>98.7</td></tr><tr><td>Puntaje máximo</td><td>221</td></tr>
<tr><td>Supervivencia</td><td>11.5 %</td></tr><tr><td>Agente aleatorio</td><td>0.0</td></tr></table>
<pre>Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
estado (dx, dy, vel) → 15×20×12 = 3 600 estados</pre></section>
</div></main>
<footer><a href="https://github.com/sebastian-yengle08/Flappy_Bird" target="_blank" rel="noopener">Código en GitHub</a> ·
<a href="https://colab.research.google.com/drive/1pZurb6Atj5QT_7hb1Ha10muqQY7Yf_aH" target="_blank" rel="noopener">Cuaderno de Colab</a></footer>
<script>
const cv=document.getElementById('c'),g=cv.getContext('2d');let timer=null;
function pipe(x,y,h,E,cap){g.fillStyle='#5aa02c';g.fillRect(x,y,E.PIPE_W,h);g.strokeStyle='#2f5d16';g.lineWidth=2;g.strokeRect(x,y,E.PIPE_W,h);
g.fillStyle='#74c043';g.fillRect(x-3,cap,E.PIPE_W+6,14);g.strokeRect(x-3,cap,E.PIPE_W+6,14);}
function draw(f,E){const[y,pipes,score]=f;const sky=g.createLinearGradient(0,0,0,E.GROUND);sky.addColorStop(0,'#4ec0ca');sky.addColorStop(1,'#bfe9ee');
g.fillStyle=sky;g.fillRect(0,0,E.W,E.GROUND);
for(const[x,cy]of pipes){const t=cy-E.GAP/2,b=cy+E.GAP/2;pipe(x,0,t,E,t-14);pipe(x,b,E.GROUND-b,E,b);}
g.fillStyle='#ded895';g.fillRect(0,E.GROUND,E.W,E.H-E.GROUND);g.fillStyle='#8fcf4a';g.fillRect(0,E.GROUND,E.W,10);
g.fillStyle='#f7d51d';g.strokeStyle='#000';g.lineWidth=1.5;g.beginPath();g.arc(E.BIRD_X,y,E.BIRD_R,0,7);g.fill();g.stroke();
g.fillStyle='#fff';g.beginPath();g.arc(E.BIRD_X+4,y-3,3.5,0,7);g.fill();g.fillStyle='#000';g.beginPath();g.arc(E.BIRD_X+5,y-3,1.5,0,7);g.fill();
g.fillStyle='#f08a24';g.fillRect(E.BIRD_X+9,y+1,7,4);
g.fillStyle='#fff';g.strokeStyle='#000';g.lineWidth=3;g.font='bold 34px Georgia';g.textAlign='center';g.strokeText(score,E.W/2,60);g.fillText(score,E.W/2,60);}
async function run(){clearInterval(timer);go.disabled=true;const r=await fetch('/api/play?seed='+seed.value+'&max_steps='+ms.value);const d=await r.json();go.disabled=false;
sc.textContent=d.score;st.textContent=d.steps;rs.textContent=d.crashed?'Choque':'Superó el límite';let k=0;
timer=setInterval(()=>{draw(d.frames[k],d.env);act.textContent=k?('Acción: '+(d.frames[k][3]?'🔼 saltar':'⏬ no hacer nada')+' · paso '+k):'';if(++k>=d.frames.length)clearInterval(timer);},+sp.value);}
go.onclick=run;rnd.onclick=()=>{seed.value=Math.floor(Math.random()*90000);run();};run();
</script></body></html>"""
