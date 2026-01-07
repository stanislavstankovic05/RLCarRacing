# RL CarRacing Structura 

Momentan repo-ul are numai structura. Trebuie doar sa adaugam agentii in `./agents/<agent.py>` (explicat totul mai jos)

## Instalare

```bash
conda create -n rlcarracing python=3.11
conda activate rlcarracing
pip install -r requirements.txt
(sunt sanse sa fie nevoie si de un pip install torch)
```

## Verificare rapida (fara agenti)

Deschide o fereastra CarRacing si ruleaza actiuni random:

```bash
python debug_env.py
```

## Structura

- `config/`
  - `env_base.yaml` setari de baza pentru env + reward shaping
  - `template_agent.yaml` sablon pentru un agent
- `env/` wrappers pentru CarRacing (reward, done, metrici, actiuni discrete)
- `runner/` bucle comune (env factory, render)
- `agents/` doar API + loader de plugin (fara agenti inclusi)

## Cum adaugi un agent

1) Creezi fisierul `agents/<agent>.py`
2) Creezi fisierul `config/<agent>.yaml`
3) Rulezi `train.py` / `eval.py` / `render_agent.py` cu `--agent <agent>`

### Interfata al agentului

Fisierul `agents/<agent>.py` trebuie sa defineasca:

- `train(cfg: dict, seed: int, episodes: int, timesteps: int) -> None`
- `load(model_path: str, cfg: dict) -> agents.api.LoadedAgent`
(acestea sunt headerele, le lasati asa)

`LoadedAgent` contine:
- `spec`: spune ce wrappers trebuie aplicate (discrete/pixels/frame stack)
- `policy`: are metoda `predict(obs, env=None)` si intoarce o actiune

### Exemplu de `agents/<agent>.py` 

```python
from agents.api import LoadedAgent, AgentSpec

class Policy:
    def predict(self, obs, env=None):
        return 0

def train(cfg: dict, seed: int, episodes: int, timesteps: int) -> None:
    pass

def load(model_path: str, cfg: dict) -> LoadedAgent:
    spec = AgentSpec(
        needs_pixels=cfg["agent"]["needs_pixels"],
        discrete_wrapper=cfg["agent"]["discrete_wrapper"],
        action_set_name=cfg["agent"].get("action_set"),
        resize=int(cfg["agent"].get("resize", 84)),
        frame_stack=int(cfg["agent"].get("frame_stack", 4)),
    )
    return LoadedAgent(name="agent", spec=spec, policy=Policy())
```

### Config YAML

Pleci de la `config/template_agent.yaml` si il salvezi ca `config/<agent>.yaml`.

Chei importante:
- `reward.use_native: true` pentru reward nativ Gymnasium
- `reward.use_native: false` pentru reward shaping din wrapper

## Rulare training

```bash
python train.py --agent <agent> --seed 0 --episodes 2000
```

sau pentru agenti timestep-based:

```bash
python train.py --agent <agent> --seed 0 --timesteps 800000
```

Output standard:
- `results/models/`
- `results/logs/`
- `results/videos/`

## Evaluare

```bash
python eval.py --agent <agent> --seed 0 --episodes 20 --model_path <path>
```

## Render (human/video)

Human:

```bash
python render_agent.py --agent <agent> --seed 0 --episodes 3 --model_path <path> --render human
```

Video:

```bash
python render_agent.py --agent <agent> --seed 0 --episodes 3 --model_path <path> --render video --video_path results/videos/run.mp4
```
