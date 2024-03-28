import json, os, time, requests, torch, cpuinfo, sys
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


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
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        if index == 0:
            return torch.device(f"cuda:{index}") if torch.cuda.device_count() >= 1 and torch.cuda.is_available() else torch.device("cpu")
        # If the index is provided, check whether the gpu with that index is available. If not, raise an exception that the gpu provided is not available
        else:
            if torch.cuda.is_available() and torch.cuda.device_count() > index:
                return torch.device(f"cuda:{index}")
            else:
                raise Exception(f"GPU {index} is not available, please only provide available GPU indexes at startup.")

    def get_device_name(self):
        # If the device is a gpu, return the name of the gpu
        if self.info.type == "cuda":
            return torch.cuda.get_device_name(self.info)
        # If the device is a cpu, return the name of the cpu
        else:
            return cpuinfo.get_cpu_info()['brand_raw']
        
    def get_memory(self):
        # If the device is a gpu, return the vram of the gpu
        if self.info.type == "cuda":
            return torch.cuda.get_device_properties(self.info).total_memory
        # If the device is a cpu, return the ram of the system
        else:
            return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    
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
            print("Memory is less than 4GB, using batch size 8")
            return 4
        # Under 8GB, the batch size is 16
        elif self.memory < 8e9:
            print("Memory is less than 8GB, using batch size 16")
            return 8
        # Under 16GB, the batch size is 32
        elif self.memory < 16e9:
            print("Memory is less than 16GB, using batch size 32")
            return 16
        # Under 32GB, the batch size is 64
        elif self.memory < 32e9:
            print("Memory is less than 32GB, using batch size 64")
            return 32
        # Over 32GB, the batch size is 128
        elif self.memory >= 32e9:
            print("Memory is more than 32GB, using batch size 128")
            return 64
    

# Class that holds all of the information and methods for a task
class Task:
    def __init__(self, addr, device: Device):
        self.device = device
        self.addr = addr
    
    def request(self):
        print("Requesting task from the address: ", self.addr, " on the device: ", self.device.name + " (" + self.device.info.type + ")")
        url = f"{node_url}/register_particle"

        try:
            response = requests.post(url, timeout=10, json={"address": self.addr})
            received_time = time.time()
        except requests.exceptions.Timeout:
            raise Exception("Server is overloaded, try later")
        if response.status_code == 502:
            raise Exception(f"Server is overloaded, try later")
        elif response.status_code == 500:
            raise Exception(f"Failed to init particle: Try later.")
        elif response.status_code != 200:
            raise Exception(f"Error connecting to the server. Error: {response.text}")
        
        self.task = response.json()['args']
        self.task["time_received"] = received_time
    
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
        return response.json()

    def compute(self,eval_pred):
        # This function uses both the compute_metrics_gpu and compute_metrics_cpu functions to compute the metrics
        # And it prints the time each of them takes
        start_time = time.time()
        metrics = self.compute_metrics_gpu(eval_pred)
        gpu_time = time.time() - start_time
        start_time = time.time()
        metrics_cpu = self.compute_metrics_cpu(eval_pred)
        cpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time}, CPU time: {cpu_time}")
        return metrics
    
    def compute_metrics_gpu(self, eval_pred):
        """This function computes the accuracy of the model."""
        try:
            logits, labels = eval_pred
            # Convert logits and labels to PyTorch tensors
            logits = torch.tensor(logits)
            labels = torch.tensor(labels)
            # Compute predictions
            predictions = torch.argmax(logits, dim=-1)
            # Compute accuracy
            accuracy = (predictions == labels).float().mean().item()
        except Exception as e:
            print(f"Error when computing the metrics with the gpu: {e}")
            print(f"Using the cpu instead")
            return self.compute_metrics_cpu(eval_pred)
        
        return {
            "accuracy": accuracy
        }
    
    # CPU computing is slighly faster on my system which is an AMD Ryzen Threadripper PRO 3955WX 16-Cores
    # The difference is not significant, but it is consistent
    # But on slower setups, the GPU computing might be faster
    def compute_metrics_cpu(self, eval_pred):
        """This function computes the accuracy of the model."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": (predictions == labels).astype(np.float32).mean().item()
        }

    def execute(self, eval = False):
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

        device = self.device.info
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
            compute_metrics=self.compute_metrics_cpu,
        )

        start_time = time.time()
        self.train_output = trainer.train()
        self.task_execution_length = time.time() - start_time
        print(f"Device {self.device.name} {self.device.info.index} completed the taks in {self.task_execution_length} seconds")
        
        if eval:
            self.metrics = trainer.evaluate()

        trainer.save_model("my_model")

