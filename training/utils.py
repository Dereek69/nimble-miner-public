import os, sys, torch
from .core import Device

def print_in_color(text, color_code):
    """This function prints the text in the specified color."""
    END_COLOR = "\033[0m"
    print(f"{color_code}{text}{END_COLOR}")

class ArgumentHandler:
    #The possible arguments are: -a (address), -af (address file) , -g (gpu index), -h (help)
    #The default values are: -a None, -af None, -g 0
    def __init__(self, args):
        self.args = args
        self.help_message = "This script is used to execute the task given by the nimble node. The possible arguments are: \n\
        -a (address): The address of the miner. \n\
        -af (address file): Location of a file containing a list of addresses used to mine. \n\
                            every line in the file should contain an address \n\
        -g (gpu index): The index of the gpus that will be used to mine. (Leave empty if the system doesn't have a GPU or you are using the first GPU in the system)\n\
                        if multiple gpus are provided, they should be separated by a comma. \n\
                        Example: 0,1,2 \n\
        -b (benchmark): If this flag is provided, the script will run in benchmark mode. \n\
        -h (help): Display this message. \n" 
        self.addr = None
        self.addr_file = None
        self.device_index = [0]
        self.addr_list = []
        self.benchmark = False
        self.handle_arguments()

    def handle_arguments(self):
        # Loop through the arguments provided by the user
        for i in range(1, len(self.args)):
            # If the argument is "-a", set the address
            if self.args[i] == "-a":
                self.addr = self.args[i+1]
            # If the argument is "-af", set the address file
            elif self.args[i] == "-af":
                self.addr_file = self.args[i+1]
                # If the address file does not exist, print an error message and exit
                if not os.path.exists(self.addr_file):
                    print_in_color("Error: Address file does not exist.", "\033[31m")
                    sys.exit()
            # If the argument is "-g", set the gpu index
            elif self.args[i] == "-g":
                self.device_index = self.args[i+1]
                # If the gpu index is not a number or a list of numbers separated by a comma, print an error message and exit
                if not self.device_index.replace(",", "").isdigit():
                    print_in_color("Error: Invalid gpu index. It should be a number or a list of numbers separated by a comma.\n Example: 0,1,2", "\033[31m")
                    sys.exit()
                # If the gpu index is a list of numbers, convert it to a list of integers
                if "," in self.device_index:
                    self.device_index = [int(i) for i in self.device_index.split(",")]
                else:
                    self.device_index = [int(self.device_index)]
            # If the argument is "-b", set the benchmark flag to True
            elif self.args[i] == "-b":
                self.benchmark = True
            # If the argument is "-h", print the help message and exit
            elif self.args[i] == "-h":
                print(self.help_message)
                sys.exit()

        # If neither the address nor the address file is provided, print an error message and exit
        if (self.addr is None) and (self.addr_file is None):
            print_in_color("Error: Address not provided.", "\033[31m")
            sys.exit()

        # If both the address and the address file are provided, ignore the address
        if (self.addr is not None) and (self.addr_file is not None):
            self.addr = None

        self.handle_cases()

    def handle_cases(self):
        # If there is only one gpu and an address is provided but no address file, add the address and gpu index to the address list
        if (len(self.device_index) == 1) and (self.addr is not None) and (self.addr_file is None):
            self.addr_list.append((self.addr, self.device_index[0]))

        # If there are multiple gpus and an address is provided but no address file, print an error message and exit
        elif (len(self.device_index) > 1) and (self.addr is not None) and (self.addr_file is None):
            print_in_color("Error: Multiple gpus on a single address is not supported yet.", "\033[31m")
            sys.exit()

        # If there is only one gpu and an address file is provided, add each address in the file with the gpu index to the address list
        elif (len(self.device_index) == 1) and (self.addr_file is not None):
            with open(self.addr_file, "r") as f:
                addresses = f.readlines()
            for i in range(len(addresses)):
                self.addr_list.append((addresses[i].strip(), self.device_index[0]))

        # If there are multiple gpus and an address file is provided, add each address in the file with a gpu index to the address list
        elif (len(self.device_index) > 1) and (self.addr_file is not None):
            with open(self.addr_file, "r") as f:
                addresses = f.readlines()
            # If the number of addresses in the file is less than the number of gpus provided, print an error message and exit
            if len(addresses) < len(self.device_index):
                print_in_color("Error: The number of addresses in the file is less than the number of gpus provided", "\033[31m")
                sys.exit()
            # Loop through the addresses and add each one with a gpu index to the address list
            for i in range(len(addresses)):
                self.addr_list.append((addresses[i].strip(), self.device_index[i % len(self.device_index)]))

def identify_devices(addr_list, devices_index):
    
    #get all the available devices and their properties
    devices = [Device(i) for i in range(torch.cuda.device_count())]
    
    #If there are no gpus, use the cpu
    if len(devices) == 0:
        devices = [Device()]
    
    #If the length of the devices is less than the number of devices provided, print an error message and exit
    if len(devices) < len(devices_index):
        print("Error: The number of gpus provided is more than the number of available gpus")
        print(f"\nAvailable Devices:")
        for i in range(len(devices)):
            print(f"Device {i}: {devices[i].name}")
        sys.exit()
    
    #Replace the device index with the device object in the devices list
    for i in range(len(addr_list)):
        addr_list[i] = (addr_list[i][0], devices[0])  #IT SHOULD HAVE BEEN devices[devices_index[i]] INSTEAD OF devices[0] IF RUN ON DDP

    return addr_list