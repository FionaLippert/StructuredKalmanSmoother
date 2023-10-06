import json
import time
import argparse
import wandb
import subprocess
import os, glob
import os.path as osp
import yaml
import argparse
from datetime import datetime
import multiprocessing as mp

from analyse_transitions_spdgmrf import *

def get_runs(project_path):
    api = wandb.Api()
    runs = api.runs(project_path)

    return runs


parser = argparse.ArgumentParser(description='get transition matrices for all runs in a given project')
parser.add_argument('project', help='wandb project path')
args = parser.parse_args()

start_time = datetime.now()
processes = set()
max_processes = 2

runs = get_runs(args.project)
logfile = 'log.txt'

for r in runs:
    user, project, run_id = r.path
    # run_path = osp.join(login, project, f'model-{run}:v0')
    run_path = osp.join(user, project, run_id)
    print(run_path)


    processes.add(subprocess.Popen(['python', 'analyse_transitions_stdgmrf.py', f'+wandb_run={run_path}'],
                            stdout=open(logfile, 'a+'),
                            stderr=open(logfile, 'a+')))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])

#Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()
