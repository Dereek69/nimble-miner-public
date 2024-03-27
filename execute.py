"""This module contains the code to execute the task."""

#TODO: Add the checkpoint loading and logging

import sys, time
import torch.multiprocessing as mp
from training.core import Task
from training.utils import ArgumentHandler, identify_devices
from training.benchmark import benchmark

def single_gpu_loop(addr_list):
    #In an infinite loop, try doing a task with the first address in the list, and if it fails, try the next address.
    #When all the addresses are done, start from the beginning
    print("\nStarting mining on the gpu: ", addr_list[0][1].name)
    while True:
        for addr in addr_list:
            try:
                task = Task(addr[0], addr[1])
                task.request()
                task.execute()
                task.complete()
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(30)
                continue

def perform():

    args = ArgumentHandler(sys.argv)

    addr_list = identify_devices(args.addr_list, args.device_index)

    if args.benchmark:
        benchmark(args,addr_list)
        sys.exit()

    #If the devices_index length is 1, use the single gpu loop
    if len(args.device_index) == 1:
        single_gpu_loop(addr_list)
    #If the devices_index length is more than 1, separate the addr_list into multiple lists with the same device
    else:
        mp.set_start_method('spawn')
        for i in range(len(args.device_index)):
            filtered_addr_list = [addr for addr in addr_list if addr[1].info.index == args.device_index[i]]

            #Then launch a separate process for each list
            p = mp.Process(target=single_gpu_loop, args=(filtered_addr_list,))
            p.start()
            p.join()
    
if __name__ == "__main__":
    perform()
