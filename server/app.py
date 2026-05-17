from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import subprocess
import threading
import json
import os
import signal
import numpy as np

from src.util.config_dataclasses import TrainConfig, EnvConfig
from src.util.config_util import get_params
from src.config import PROJ_DIR, DATA_DIR
from server.fs_utils import available_histories, available_maps, available_models


app = FastAPI()
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")

current_process = None
logs = []

def stream_logs(pipe):
    global logs

    for line in iter(pipe.readline, ''):
        logs.append(line.rstrip())
        if len(logs) > 1000:
            logs = logs[-1000:]



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "maps": available_maps(),
            "histories": available_histories(),
            "models": available_models()
        }
    )

@app.get("/config-schema")
def config_schema():
    return {
        "train": get_params(TrainConfig),
        "env": get_params(EnvConfig)
    }

@app.get("/logs")
async def get_logs():
    return JSONResponse(logs)


@app.get("/history/{model_name}")
async def get_history(model_name: str):

    path = os.path.join(
        PROJ_DIR,
        DATA_DIR,
        model_name,
        "history",
        f"{model_name}.npz"
    )

    if not os.path.exists(path):
        return JSONResponse({"error": "history not found"})

    data = np.load(path)

    result = {}

    for key in data.files:
        result[key] = data[key].tolist()

    return JSONResponse(result)


@app.post("/launch")
async def launch_training(request: Request):
    global current_process
    body = await request.json()
    config_json = json.dumps(body)

    if current_process is not None:
        return JSONResponse({
            "error": "training already running"
        })

    current_process = subprocess.Popen(
        [
            "python",
            "-u",
            "train_runner.py",
            config_json
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    threading.Thread(
        target=stream_logs,
        args=(current_process.stdout,),
        daemon=True
    ).start()

    return JSONResponse({
        "status": "started"
    })


@app.post("/stop")
async def stop_training():

    global current_process

    if current_process is not None:
        current_process.send_signal(signal.SIGTERM)
        current_process = None

    return JSONResponse({
        "status": "stopped"
    })


@app.post("/launch-gui")
async def launch_gui(request: Request):
    body = await request.json()

    config_json = json.dumps(body)

    subprocess.Popen([
        "python",
        "-u",
        "gui_runner.py",
        config_json
    ])

    return JSONResponse({
        "status": "gui started"
    })