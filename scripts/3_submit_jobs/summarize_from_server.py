"""Summarize from Server

This script checks on progress of multiple experiments and aggregates the results json into a csv.

"""
import os
import json
import shutil
import argparse
from datetime import datetime
import pandas as pd

def main():
    DIR = './ActiveTransformers-for-AbusiveLanguage'
    parser = argparse.ArgumentParser(
        description="Summarize completed and failed experiments.")
    parser.add_argument('--move_files',help="bool for whether to move failed /nonstarted experiments")
    parser.add_argument('--which_movers',help="option to move failed experiments or non-starters into cache folder")
    parser.add_argument('-t','--test',action='store_false',help="test script without submitting jobs")
    args = parser.parse_args()
    # Log time
    current_datetime = datetime.now()
    # Load SL baselines
    subfolders = [x for x in os.listdir(f'{DIR}/results') if x.startswith('SL_')]
    frames = []
    failed_experiments = []
    non_starters = []
    still_running = []
    for folder in subfolders:
        results_filepath =[filename for filename in os.listdir(f'{DIR}/results/{folder}') if filename.startswith("END")]
        started_filepath =[filename for filename in os.listdir(f'{DIR}/results/{folder}') if filename.startswith("START")]
        failed_filepath =[filename for filename in os.listdir(f'{DIR}/results/{folder}') if filename.startswith("FAILED")]
        if len(results_filepath) == 1:
            f = open(f'./results/{folder}/{results_filepath[0]}',)
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index')
            frames.append(df)
        # Experiment has not finished
        elif len(results_filepath) == 0:
            # Experiment is running
            if len(started_filepath) == 1:
                continue
            # If folder created but no failed or start files
            elif len(started_filepath) == 0:
                failed_experiments.append(folder)
            # If experiment failed
            elif len(failed_filepath) >= 1:
                failed_experiments.append(folder)
    SL_df = pd.concat(frames, axis = 1, ignore_index = True)
    SL_df = SL_df.transpose()
    print(f'len SL_df {len(SL_df)}')
    # Save df
    SL_df.to_csv(f'{DIR}/summary_results/raw/SL.csv')
    # Load AL baselines
    subfolders = [x for x in os.listdir(f'{DIR}/results') if x.startswith('AL')]
    frames = []
    for folder in subfolders:
        results_filepath =[filename for filename in os.listdir(f'{DIR}/results/{folder}') if filename.startswith("END")]
        started_filepath =[filename for filename in os.listdir(f'{DIR}/results/{folder}') if filename.startswith("START")]
        failed_filepath =[filename for filename in os.listdir(f'{DIR}/results/{folder}') if filename.startswith("FAILED")]
        if len(results_filepath) == 1:
            f = open(f'{DIR}/results/{folder}/{results_filepath[0]}',)
            data = json.load(f)
            data['endtime'] = str(results_filepath[0]).replace('END_', '').replace('.json','')
            data['starttime'] = str(started_filepath[0]).replace('START_', '').replace('.json','')
            df = pd.DataFrame.from_dict(data, orient='index')
            frames.append(df)
        # Experiment has not finished
        elif len(results_filepath) == 0:
            # Experiment is running
            if (len(started_filepath) == 1) & (len(failed_filepath)==0):
                still_running.append(folder)
            # If folder created but no failed or start files (needs rerunning)
            elif (len(started_filepath) == 0) & (len(failed_filepath)==0):
                non_starters.append(folder)
            # If experiment failed
            elif len(failed_filepath) >= 1:
                failed_experiments.append(folder)
                f = open(f'{DIR}/results/{folder}/{failed_filepath[0]}',)
                data = json.load(f)
                data['endtime'] = 0
                data['starttime'] = 0
                df = pd.DataFrame.from_dict(data, orient='index')
                frames.append(df)
    AL_df = pd.concat(frames, axis = 1, ignore_index = True)
    AL_df = AL_df.transpose()
    print(f'len AL_df {len(AL_df)}')
    # Save df
    AL_df.to_csv(f'{DIR}/summary_results/raw/AL.csv')
    print('\nfailed_experiments', failed_experiments)
    print('\nnon_starters', non_starters)
    print('\nnon_starters len', len(non_starters))
    print('\nstill_running', still_running)
    print('\nstill running len', len(still_running))
    # Check overlap
    combined_lists = failed_experiments + non_starters + still_running
    assert len(combined_lists) == len(set(combined_lists))
    if int(args.move_files) == 1:
        if args.which_movers == 'failed':
            move_folders = failed_experiments
        elif args.which_movers == 'nonstarters':
            move_folders = non_starters
        # Move failed experiments to failed folder
        print(f'moving {args.which_movers} experiments to failed folder')
        for folder in move_folders:
            original = f'./results/{folder}'
            target = f'./results/{args.which_movers}_{current_datetime}/{folder}'
            shutil.move(original,target)

if __name__ == "__main__":
    main()
