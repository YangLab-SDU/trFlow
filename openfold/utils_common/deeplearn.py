# Helper utilities: GPU probing and affinity setup.

import re, os


def find_empty_gpu(cpu_ok=False, mem_cutoff=30000, count=1):
    """find empty gpu"""
    memory_gpu = [int(x.split()[2]) for x in os.popen('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free').readlines()]
    mem_dict = {}
    for i, g in enumerate(memory_gpu):
        mem_dict[i] = g
    candidates = list(filter(lambda i: mem_dict[i] > mem_cutoff, mem_dict))
    if len(candidates) > 0:
        to_use = sorted(candidates, key=lambda i: mem_dict[i], reverse=True)[:count]
        print(f'use gpu: {to_use}')
        return to_use[0] if count == 1 else ','.join([str(i) for i in to_use])
    if cpu_ok:
        print('No empty gpu remains! Use cpu instead')
        return -1
    raise EnvironmentError('No empty gpu remains!')


def get_UUID(gpu_id):
    gpu_id = int(gpu_id)
    if gpu_id == -1:
        return '-1'
    gpus = os.popen('nvidia-smi -L').readlines()
    for gpu in gpus:
        id_ = int(gpu.split()[1].replace(':', ''))
        if id_ == gpu_id:
            uuid = gpu.split()[-1].replace(')', '')
            return uuid
    raise ValueError(f'cannot find gpu {gpu_id}!')
