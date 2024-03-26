"""This module contains the code to execute the task."""

#TODO: Add the checkpoint loading and deleting, the multigpu support and logging

import json, os, shutil, sys, time, requests, torch, cpuinfo
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
import torch.multiprocessing as mp

node_url = "https://mainnet.nimble.technology:443"

# Class that defines the device (cpu or gpu) properties and best settings
class Device:
    def __init__(self, index = 0):
        self.info = self.get_type(index)
        self.name = self.get_device_name()
        self.memory = self.get_memory()
        self.tf32 = self.is_tf32_supported()
        self.batch_size = self.get_best_batch_size()

    def get_type(self, index):
        # If the index is not provided, check whether a gpu is available
        if index == 0:
            return torch.device(f"cuda:{index}") if torch.cuda.device_count() >= 1 and torch.cuda.is_available() else torch.device("cpu")
        # If the index is provided, check whether the gpu with that index is available. If not, raise an exception that the gpu provided is not available
        else:
            if torch.cuda.is_available() and torch.cuda.device_count() > index:
                return torch.device(f"cuda:{index}")
            else:
                raise Exception(f"GPU {index} is not available, please only provide available GPU indexes at startup.")

    def get_memory(self):
        # If the device is a gpu, return the vram of the gpu
        if self.info.type == "cuda":
            return torch.cuda.get_device_properties(self.info).total_memory
        # If the device is a cpu, return the ram of the system
        else:
            return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    
    def get_device_name(self):
        # If the device is a gpu, return the name of the gpu
        if self.info.type == "cuda":
            return torch.cuda.get_device_name(self.info)
        # If the device is a cpu, return the name of the cpu
        else:
            return cpuinfo.get_cpu_info()['brand_raw']
        
    def is_tf32_supported(self):
        if self.info.type == "cuda":
            major, minor = torch.cuda.get_device_capability(self.info)
            return (major >= 8)
        else:
            return False
        
    def get_best_batch_size(self):
        # Return the best batch size given the memory size of the device
        # Under 4GB, the batch size is 8
        if self.memory < 4e9:
            return 8
        # Under 8GB, the batch size is 16
        elif self.memory < 8e9:
            return 16
        # Under 16GB, the batch size is 32
        elif self.memory < 16e9:
            return 32
        # Under 32GB, the batch size is 64
        elif self.memory < 32e9:
            return 64
        # Over 32GB, the batch size is 128
        elif self.memory >= 32e9:
            return 128
    

# Class that holds all of the information and methods for a task
class Task:
    def __init__(self, addr, device: Device):
        self.device = device
        self.addr = addr
        self.task = self.request(addr)
        self.task['time_received'] = time.time()
    
    def request(self, addr):
        print("Requesting task from the address: ", addr, " on the device: ", self.device.name + " (" + self.device.info.type + ")")
        url = f"{node_url}/register_particle"

        try:
            response = requests.post(url, timeout=10, json={"address": addr})
        except requests.exceptions.Timeout:
            raise Exception("Server is overloaded, try later")
        if response.status_code == 502:
            raise Exception(f"Server is overloaded, try later")
        elif response.status_code == 500:
            raise Exception(f"Failed to init particle: Try later.")
        elif response.status_code != 200:
            raise Exception(f"Error connecting to the server. Error: {response.text}")
        
        task = response.json()
        return task['args']
    
    def complete(self):
        url = f"{node_url}/complete_task"
        files = {
            "file1": open("my_model/config.json", "rb"),
            "file2": open("my_model/training_args.bin", "rb"),
        }
        json_data = json.dumps({"address": self.addr})
        files["r"] = (None, json_data, "application/json")
        response = requests.post(url, files=files, timeout=60)
        if response.status_code != 200:
            raise Exception(f"Failed to complete task: Try later. Error: {response.text}")
        self.task_duration = time.time() - self.task['time_received']
        print(f"Device {self.device.name} completed the taks in {self.task_duration} seconds")
        return response.json()

    def compute_metrics_gpu(self, eval_pred):
        """This function computes the accuracy of the model."""
        try:
            logits, labels = eval_pred
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean().item()
        except Exception as e:
            print(f"Error when computing the metrics with the gpu: {e}")
            print(f"Using the cpu instead")
            return self.compute_metrics_cpu(eval_pred)
        
        return {
            "accuracy": accuracy
        }
    
    def compute_metrics_cpu(self, eval_pred):
        """This function computes the accuracy of the model."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": (predictions == labels).astype(np.float32).mean().item()
        }

    def execute(self):
        """This function executes the task."""
        print("Starting training...")
        tokenizer = AutoTokenizer.from_pretrained(self.task["model_name"])

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], padding="max_length", truncation=True
            )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.task["model_name"], num_labels=self.task["num_labels"]
        )

        device = self.device.info.type
        model.to(device)

        dataset = load_dataset(self.task["dataset_name"])
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        small_train_dataset = (
            tokenized_datasets["train"].shuffle(seed=self.task["seed"]).select(range(self.task["num_rows"]))
        )
        small_eval_dataset = (
            tokenized_datasets["train"].shuffle(seed=self.task["seed"]).select(range(self.task["num_rows"]))
        )
        training_args = TrainingArguments(
            output_dir="my_model", 
            evaluation_strategy="epoch", 
            tf32=self.device.tf32, 
            per_device_train_batch_size=self.device.batch_size, 
            per_device_eval_batch_size=self.device.batch_size
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=self.compute_metrics_gpu,
        )
        trainer.train()
        trainer.save_model("my_model")


def print_in_color(text, color_code):
    """This function prints the text in the specified color."""
    END_COLOR = "\033[0m"
    print(f"{color_code}{text}{END_COLOR}")

def arguments_handling():
    #The possible arguments are: -a (address), -af (address file) , -g (gpu index), -h (help)
    #The default values are: -a None, -af None, -g 0

    #Help message
    help_message = "This script is used to execute the task given by the nimble node. The possible arguments are: \n\
    -a (address): The address of the miner. \n\
    -af (address file): Location of a file containing a list of addresses used to mine. \n\
                        every line in the file should contain an address \n\
    -g (gpu index): The index of the gpus that will be used to mine. (Leave empty if the system doesn't have a GPU or you are using the first GPU in the system)\n\
                    if multiple gpus are provided, they should be separated by a comma. \n\
                    Example: 0,1,2 \n\
    -h (help): Display this message. \n" 

    #Default values
    addr = None
    addr_file = None
    gpu_index = [0]
    addr_list = []

    #Arguments handling
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-a":
            addr = sys.argv[i+1]
        elif sys.argv[i] == "-af":
            addr_file = sys.argv[i+1]
            #Check if the file exists
            if not os.path.exists(addr_file):
                print_in_color("Error: Address file does not exist.", "\033[31m")
                sys.exit()
        elif sys.argv[i] == "-g":
            gpu_index = sys.argv[i+1]
            #Check if the gpu index is a number or a list of numbers separated by a comma
            if not gpu_index.replace(",", "").isdigit():
                print_in_color("Error: Invalid gpu index. It should be a number or a list of numbers separated by a comma.\n Example: 0,1,2", "\033[31m")
                sys.exit()
            #If the gpu index is a list of numbers, convert it to a list of integers
            if "," in gpu_index:
                gpu_index = [int(i) for i in gpu_index.split(",")]
            else:
                gpu_index = [int(gpu_index)]
        elif sys.argv[i] == "-h":
            print(help_message)
            sys.exit()

    #If the address is not provided and the address file is not provided, print an error message and exit
    if (addr is None) and (addr_file is None):
        print_in_color("Error: Address not provided.", "\033[31m")
        sys.exit()

    #If both the address and the address file are provided, only the address file will be used
    if (addr is not None) and (addr_file is not None):
        addr = None

    #4 cases:
    # 1. Gpu index lenght is 1, address is provided and address file is not provided
    # 2. Gpu index lenght is more than 1 , address is provided and address file is not provided
    # 3. Gpu index lenght is 1 and address file is provided
    # 4. Gpu index lenght is more than 1 and address file is not provided

    #Case 1
    if (len(gpu_index) == 1) and (addr is not None) and (addr_file is None):
        addr_list.append((addr, gpu_index[0]))

    #Case 2
    elif (len(gpu_index) > 1) and (addr is not None) and (addr_file is None):
        #Print an error message because multiple gpus on a single address is not supported yet
        print_in_color("Error: Multiple gpus on a single address is not supported yet.", "\033[31m")
        sys.exit()

    #Case 3
    elif (len(gpu_index) == 1) and (addr_file is not None):
        #Read all the addresses from the file, divide them by the number of gpus and assign them all to the gpu
        with open(addr_file, "r") as f:
            addresses = f.readlines()
        for i in range(len(addresses)):
            addr_list.append((addresses[i].strip(), gpu_index[0]))

    #Case 4
    elif (len(gpu_index) > 1) and (addr_file is not None):
        #Read all the addresses from the file and assign them to the gpus
        with open(addr_file, "r") as f:
            addresses = f.readlines()
        #If the number of addresses is less than the number of gpus, print an error message and exit
        if len(addresses) < len(gpu_index):
            print_in_color("Error: The number of addresses in the file is less than the number of gpus provided", "\033[31m")
            sys.exit()
        for i in range(len(addresses)):
            addr_list.append((addresses[i].strip(), gpu_index[i % len(gpu_index)]))

    return addr_list, gpu_index

def single_gpu_loop(addr_list):
    #In an infinite loop, try doing a task with the first address in the list, and if it fails, try the next address.
    #When all the addresses are done, start from the beginning
    print("\nStarting mining on the gpu: ", addr_list[0][1].name)
    while True:
        for addr in addr_list:
            try:
                task = Task(addr[0], addr[1])
                task.execute()
                task.complete()
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(30)
                continue

def perform():

    addr_list, devices_index = arguments_handling()

    #get all the available devices and their properties
    devices = [Device(i) for i in range(torch.cuda.device_count())]
    print("\nAvailable devices: ")
    for i in range(len(devices)):
        print(f"Device {i}: {devices[i].name}")

    #If there are no gpus, use the cpu
    if len(devices) == 0:
        devices = [Device()]
    
    #If the length of the devices is less than the number of devices provided, print an error message and exit
    if len(devices) < len(devices_index):
        print_in_color("Error: The number of gpus provided is more than the number of available gpus", "\033[31m")
        print(f"\nAvailable Devices:")
        for i in range(len(devices)):
            print(f"Device {i}: {devices[i].name}")
        sys.exit()

    #Replace the device index with the device object in the devices list
    for i in range(len(addr_list)):
        addr_list[i] = (addr_list[i][0], devices[addr_list[i][1]])

    #If the devices_index length is 1, use the single gpu loop
    if len(devices_index) == 1:
        single_gpu_loop(addr_list)
    #If the devices_index length is more than 1, separate the addr_list into multiple lists with the same device
    else:
        mp.set_start_method('spawn')
        for i in range(len(devices_index)):
            addr_list = [addr for addr in addr_list if addr[1].info.index == devices_index[i]]

            #Then launch a separate process for each list
            p = mp.Process(target=single_gpu_loop, args=(addr_list,))
            p.start()
            p.join()
    
if __name__ == "__main__":
    perform()
