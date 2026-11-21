# Agente Q-Learning para Flappy Bird

Proyecto de IA — UPAO, curso *IA Principios y Técnicas* (NRC 5028). Agente de aprendizaje por refuerzo (Q-Learning tabular) que aprende a jugar Flappy Bird en un entorno simulado.

## Resultados (semilla 42)
| Métrica (200 ep. de prueba, ε = 0) | Valor |
|---|---|
| Puntaje medio | 98.7 |
| Puntaje máximo | 221 |
| Tasa de supervivencia | 11.5 % |
| Agente aleatorio | 0.0 |

## Estructura
- `src/flappy/` entorno, discretizador, agente y entrenamiento
- `train.py` entrena y guarda `models/qtable_flappy.npy` / `.pkl`
- `app.py` aplicación web (Streamlit)
- `tests/` 7 pruebas con pytest
- `docs/` cuaderno de Colab

## Ejecución
```bash
pip install -r requirements.txt
python train.py          # opcional: la Q-Table ya viene en models/
python -m pytest -q
streamlit run app.py
```
Cuaderno: https://colab.research.google.com/drive/1pZurb6Atj5QT_7hb1Ha10muqQY7Yf_aH
