"""Submit Jobs

This script submits multiple slurm jobs based on the job csv.

"""
import argparse
import os
import subprocess
import time
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Automatically submit jobs using a csv file")
    parser.add_argument('--jobscript', help="job script to use")
    parser.add_argument('--parameters',help="csv parameter file to use")
    parser.add_argument('--from_experiment',help="index of experiment to run from")
    parser.add_argument('--to_experiment',help="index of experiment to run to, -1 for final idx")
    parser.add_argument('-t','--test',action='store_false',help="test script without submitting jobs")
    args = parser.parse_args()
    param_df = pd.read_csv(args.parameters)
    if int(args.to_experiment) == -1:
        final_idx = len(param_df) + 1
    else:
        final_idx = int(args.to_experiment)
    for job_id in param_df.index[int(args.from_experiment):final_idx]:
        param_dict = {}
        for col in param_df.columns:
            param_dict[col.upper()] = param_df[col].loc[job_id]
        if param_dict['METHOD'] == 'SL':
            print('framework: SL')
            experiment_output_dir = f"{param_dict['OUTDIR']}/{param_dict['METHOD']}_{param_dict['FRAMEWORK']}_{param_dict['DATASET']}_{param_dict['CLASS_IMBALANCE']}_{param_dict['TRAIN_N']}"
        else:
            print('framework: AL')
            experiment_output_dir = f"{param_dict['OUTDIR']}/{param_dict['METHOD']}_{param_dict['FRAMEWORK']}_{param_dict['DATASET']}_{param_dict['CLASS_IMBALANCE']}_{param_dict['INIT_N']}_{param_dict['COLD_STRATEGY']}_{param_dict['QUERY_STRATEGY']}_{param_dict['QUERY_N']}"
        job_name = experiment_output_dir.split('/')[-1]
        job_name = job_name.replace('()','')
        job_name = job_name.replace('(','')
        job_name = job_name.replace(')','')
        job_name = job_name.replace('=True','')
        print(f'\n\n---EXPERIMENT:{job_name}---')
        print(f'outdir: {experiment_output_dir}')
        # Only submit job is dir doesnt exit AND there is no end file:
        if os.path.exists(experiment_output_dir):
            end_filepath =[filename for filename in os.listdir(experiment_output_dir) if filename.startswith("END")]
            start_filepath =[filename for filename in os.listdir(experiment_output_dir) if filename.startswith("START")]
            failed_filepath =[filename for filename in os.listdir(experiment_output_dir) if filename.startswith("FAILED")]
            print(end_filepath, start_filepath, failed_filepath)
            if len(end_filepath) == 1:
                print('Experiment has an END file. NOT SUBMITTING.')
                run = False
            elif len(end_filepath) == 0:
                if (len(start_filepath) == 1) & (len(failed_filepath) == 0):
                    print('Experiment has started, and is still running. NOT SUBMITTING.')
                    run = False
                elif (len(start_filepath) == 1) & (len(failed_filepath) == 1):
                    print('Experiment has started, and has failed. NOT SUBMITTING.')
                    run = False
                else:
                    print('Check Experiment. NOT SUBMITTING')
                    print(end_filepath, start_filepath, failed_filepath)
                    run = False
        else:
            print('Experiment outdir doesnt exist. SUBMITTING.')
            os.makedirs(experiment_output_dir)
            run = True
        if run is True:
            export_string = 'export=' + ','.join([f'{k}=\'{v}\'' for k,v in param_dict.items()])
            submit_command = (
                "sbatch " +
                f"--job-name={job_name} " +
                f"--{export_string} " + args.jobscript)
            print(submit_command)
            time.sleep(1)
            if not args.test:
                print(submit_command)
            else:
                exit_status = subprocess.call(submit_command,shell=True)
                # Check to make sure the job submitted
                if exit_status == 1:
                    print("Job {0} failed to submit".format(submit_command))
    print("Done submitting jobs")

if __name__ == "__main__":
    main()
