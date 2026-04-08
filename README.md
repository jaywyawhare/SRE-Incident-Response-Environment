---
title: SRE Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# SRE Incident Response Environment

## Run

```bash
pip install -r requirements.txt
python -m server.app
```

`curl localhost:7860/health` should say `"healthy"`. Docker: `docker build -t sre-incident-env .` then `docker run -p 7860:7860 sre-incident-env`. On HF Spaces, `PORT` is set for you.

## Using the API

Reset picks the scenario: `POST /reset` with `{"task_id":"easy"}` (or `medium` / `hard`). Each turn: `POST /step` with an action—shape is in `env/models.py` (`SREAction`). There’s also `/state`, `/schema`, `/metadata` if you need them.

## `inference.py`

Hits this server over HTTP and calls an LLM via the OpenAI client. Point it at the env with `SRE_ENV_URL` (default `http://127.0.0.1:7860`). For the model: `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` (no default—unset runs a scripted baseline). Stdout uses structured lines: `START`, one `STEP` per env step, then `END` with the run summary.

## Checks before submit

```bash
python validate_submission.py
```

Add `HF_SPACE_URL` if you want it to ping your deployed Space. `--skip-docker` if Docker isn’t available.

## Code map

`env/` is the sim, `api/server.py` + `server/app.py` serve it, rewards in `env/reward.py`, grader in `graders/grader.py`. Tests: `pytest tests/`.
