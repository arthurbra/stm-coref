import glob
import os
import random
import subprocess
from typing import List

import numpy as np
# import tensorflow as tf


def change_conf_params(configuration, conf_path, params_to_new_values):
    new_file_contents = ''
    changed_params = []
    with open(conf_path, mode='r') as file:
        is_correct_configuration = False
        for line in file:
            is_comment = '#' in line
            is_configuration_start = not is_comment and len(line) > 1 and line[-2] == '{'
            is_configuration_end = not is_comment and line[0] == '}'
            is_setting = not is_comment and '=' in line and not is_configuration_start

            if is_correct_configuration and is_configuration_end:
                is_correct_configuration = False
                params_to_add = set(params_to_new_values.keys()) - set(changed_params)
                print(f'Changed params: {changed_params}')
                print(f'Adding params: {params_to_add}')
                for param in params_to_add:
                    new_file_contents += f'  {param} = {params_to_new_values[param]}\n'

            if is_correct_configuration and is_setting:
               for param in params_to_new_values:
                    if line.split(' = ')[0].strip() == param:
                        new_file_contents += f'  {param} = {params_to_new_values[param]}\n'
                        changed_params.append(param)
                        break
               else:
                    new_file_contents += line
            else:
                new_file_contents += line

            if is_configuration_start and line.split(' ')[0] == configuration:
                is_correct_configuration = True

    with open(conf_path, mode='w') as file:
        file.write(new_file_contents)
        file.flush()


def set_seed_value(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)


def get_files_in_folder(folder_path, pattern='*.*'):
    if folder_path[-1] != '/':
        folder_path += '/'
    return [f for f in glob.glob(f'{folder_path}**/{pattern}', recursive=True)]


def flatten(l):
    if isinstance(l, list) and isinstance(l[0], list):
        return [x for sublist in l for x in sublist]
    else:
        return l


def execute(command: List[str], show_stderr_first: bool = False) -> None:
    # prefix = '!' if BFCRModel.RUN_IN_IPYTHON else ''
    # os.system(prefix + command[0] + ' '.join(command[1:]))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if not show_stderr_first:
        for line in process.stdout:
            print(line, end='')
        for line in process.stderr:
            print(line, end='')
    else:
        for line in process.stderr:
            print(line, end='')
        for line in process.stdout:
            print(line, end='')

    # while process.poll() is None:
    #     out = process.stdout.readline()
    #     if out != '':
    #         print(out, end='')
    #     err = process.stderr.readline()
    #     if err != '':
    #         print(err, end='')

# print(line, end='')
    # for line in iter(lambda: process.stdout.readline() + '\n' + process.stderr.readline(), ''):
    #     print(line, end='')
    # for line in process.stderr:
    #     print(line, end='')
