# This module is used to benchmark the performance of the training process.
# It will use a small training set and a small validation set
# And it will try different training settings to achieve the best performance.
# It will then save the results to a json file and print the results

import time, os
import torch.multiprocessing as mp
from training.core import Task
from training.utils import identify_devices

# Example of a long task that takes approximately 1 hour to complete
long_task = {'num_labels': 5, 'model_name': 'distilbert-base-uncased', 'dataset_name': 'yelp_review_full', 'slices': 2000, 'num_rows': 276884, 'seed': 47}

# Example of a short task that takes approximately 1 minute to complete
short_task = {'num_labels': 5, 'model_name': 'distilbert-base-uncased', 'dataset_name': 'yelp_review_full', 'slices': 2000, 'num_rows': 2768, 'seed': 47}

def single_gpu_loop(addr_list, lock = None):
    # Run the loop 20 times, try doing a task with the first address in the list, and if it fails, try the next address.
    # When all the addresses are done, start from the beginning
    original_index = addr_list[0][1]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(addr_list[0][1])
    addr_list = identify_devices(addr_list,[addr_list[0][1]])

    executions = []
    addr = addr_list[0]
    print("\nStarting mining on the gpu: ", addr[1].name)
    for i in range(1,5):
        #change the batch size to 8
        addr[1].batch_size = i*8
        for _ in range(5):
            print(f"Run {_ + 1} on GPU {addr[1].name} ({original_index})")

            try:
                task = Task(addr[0], addr[1])
                task.task = short_task
                task.execute(eval = True)

                executions.append((task.task_execution_length, task.train_output, task.metrics))
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(30)
                continue
    
        # Save the results to a json file
        # The json file will contain the execution times, the training loss, the metrics and an average of all of those
        # The file should contain the informations of the gpu used for each benchmark
        # To avoid conflicts, a lock is used to prevent multiple processes from writing to the file at the same time

        if lock is not None:
            lock.acquire()
            try:
                with open("benchmark_results.txt", 'a') as f:
                    f.write("--------------------------------------------\n")
                    f.write(f"GPU {addr[1].name}: {original_index}\n")

                    f.write("Average\n")
                    f.write(f"Execution time: {sum([execution[0] for execution in executions]) / len(executions)}\n")
                    f.write(f"Training Loss: {sum([execution[1].training_loss for execution in executions]) / len(executions)}\n")
                    f.write(f"Evaluation Loss: {sum([execution[2]['eval_loss'] for execution in executions]) / len(executions)}\n")
                    f.write(f"Evaluation Accuracy: {sum([execution[2]['eval_accuracy'] for execution in executions]) / len(executions)}\n")
                    f.write(f"Evaluation Runtime: {sum([execution[2]['eval_runtime'] for execution in executions]) / len(executions)}\n")
                    f.write(f"Evaluation Samples Per Second: {sum([execution[2]['eval_samples_per_second'] for execution in executions]) / len(executions)}\n")
                    f.write("\n")

                    f.write("Run #, Execution Time, Training Loss, Evaluation Loss, Evaluation Accuracy, Evaluation Runtime, Evaluation Samples Per Second\n")
                    for execution in executions:
                        f.write(f"{executions.index(execution) + 1}, {execution[0]}, {execution[1].training_loss}, {execution[2]['eval_loss']}, {execution[2]['eval_accuracy']}, {execution[2]['eval_runtime']}, {execution[2]['eval_samples_per_second']}\n")

                    f.write("\n")
            finally:
                lock.release()

def benchmark(args,addr_list):

    #If the devices_index length is 1, use the single gpu loop
    #if len(args.device_index) == 1:
    #    single_gpu_loop(addr_list)
    #If the devices_index length is more than 1, separate the addr_list into multiple lists with the same device
    #else:
    mp.set_start_method('spawn')
    lock = mp.Lock()
    processes = []
    for i in range(len(args.device_index)):
        filtered_addr_list = [addr for addr in addr_list if addr[1] == args.device_index[i]]
        #Then launch a separate process for each list
        p = mp.Process(target=single_gpu_loop, args=(filtered_addr_list,lock))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()