import requests
import platform
import signal
import subprocess
import math
import torch
import time

from pynvml import *


def empty_cuda_cache():
    print("\n\nEmptying CUDA cache\n\n")
    # check if torch.cuda is available
    if torch.cuda.is_available():
        try:
            print("Attempting to empty cuda cache")
            torch.cuda.empty_cache()
            print("CUDA cache successfully emptied")
        except Exception as e:
            raise Exception(f"Could not empty cuda cache, encountered error: {e}")
    else:
        print("\n\nCUDA is not available, skipping cache-emptying\n\n")
        return True


def safe_empty_cuda_cache():
    try:
        empty_cuda_cache()
    except Exception as e:
        print(f"Could not empty CUDA cache, encountered error: {e}")


def is_local_server_online(server_base_url):
    server_health_url = server_base_url + '/health'
    print(f"\n\nChecking server status at URL: {server_health_url}\n\n")

    try:
        response = requests.get(server_health_url)
        
        if response.status_code == 200:
            data = response.json()  # parse the JSON response to determine the server status
            if data['status'] == 'ok':
                print(f"LLM Server ready and online at URL: {server_health_url}")
                return {"server_available":True, "server_online":True, "loading_model":False, "status_code":200}
            elif data['status'] == 'no slot available':
                print("No slots available. Server is running but cannot handle more requests.")
                return {"server_available":False, "server_online":True, "loading_model":False, "status_code":200}
            
        elif response.status_code == 503:   # model still loading or no slots
            data = response.json()
            if data['status'] == 'loading model':
                print("Server is loading the selected LLM, please wait")
                return {"server_available":False, "server_online":True, "loading_model":True, "status_code":503}
            else:
                print("No slots available. Server is running but cannot handle more requests.")
                return {"server_available":False, "server_online":True, "loading_model":False, "status_code":503}
        
        elif response.status_code == 500:
            print("Server error: Failed to load LLM.")
            return {"server_available":False, "server_online":True, "loading_model":False, "status_code":500}
        
        else:
            return {"server_available":False, "server_online":False, "loading_model":False, "status_code":500}
    
    except requests.exceptions.ConnectionError as e:
        print("\n\nECONNREFUSED event\n\n")
        return {"server_available":False, "server_online":False, "loading_model":True, "status_code":500}
    except Exception as e:
        print(f"\n\nCould not check local LLM Server health, encountered error: {e}\n\n")
        return {"server_available":False, "server_online":False, "loading_model":False, "status_code":500}


def shutdown_waitress_server(base_url = 'http://localhost:9069'):
    try:
        print(f"\n\nShutting down HF-Waitress server at: {base_url}\n\n")

        if is_local_server_online(base_url)['server_online']:
            url = f"{base_url}/shutdown_hf_waitress"
            payload = ""
            headers = {}
            response = requests.post(url, data=payload, headers=headers)
            return {"success":response.json()['success'], "message":response.json()['message'], "was_offline":False}
        else:
            return {"success":True, "message":"HF-Waitress server is already offline at the specified URL.", "was_offline":True}
    except Exception as e:
        return {"success":False, "message":f"Could not shutdown HF-Waitress server, encountered error: {e}", "was_offline":False}
    

def send_ctrl_c_to_process(process):
    try:
        if process.poll() is None:  # check if process is still running via poll(), which returns None if a process is still running 
            if platform.system() == 'Windows':
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            try:
                # Wait a bit for the process to terminate gracefully:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print("\n\nProcess did not terminate within timeout, will be force-killed.\n\n")
                process.kill()  # Sends 'SIGKILL' on Unix-like to force-kill immediately / 'TerminateProcess' on Windows which still allows for graceful termination
                process.wait()
                if process.poll() is not None:
                    print("\n\nProcess has been killed successfully.\n\n")
                else:
                    raise Exception("\n\nProcess still running after force kill attempt.\n\n")
    except Exception as e:
        raise Exception(f"Could not force-kill process, encountered error: {e}")


def terminate_local_llm_server_process(process):
    try:
        # process.terminate() sends 'SIGTERM' on Unix-like systems / 'TerminateProcess' on Windows, allows for graceful termination
        # process.wait()
        send_ctrl_c_to_process(process)
        if process.poll() is not None:  # process has indeed terminated
            print("\n\nProcess terminated gracefully.\n\n")
    except Exception as e:
        raise Exception(f"Failed to terminate local LLM server process, encountered error: {e}")


def get_nvidia_gpu_info():
    """
    Gets memory & other information for all available NVIDIA GPUs.

    Returns:
        tuple: A tuple containing:
            
            1. list: A list of dictionaries, where each dictionary contains:
                  'id': GPU ID (int)
                  'name': GPU name (str)
                  'total_memory_mib': Total memory in MiB (float)
                  'used_memory_mib': Used memory in MiB (float)
              'free_memory_mib': Free memory in MiB (float)
              If NVML fails to initialize, returns an empty list and prints an error.

            2. int: Total free memory in MiB
    """
    print("\n\nGetting NVIDIA GPU info...\n\n")

    gpu_info_list = []
    total_free_memory_mib = 0.0
    try:
        nvmlInit()  # Initialize NVML

        device_count = nvmlDeviceGetCount()
        print(f"Found {device_count} NVIDIA GPU(s) on this machine.")

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)

            # Get GPU name:
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Get memory info:
            mem_info = nvmlDeviceGetMemoryInfo(handle)

            # Convert bytes to MiB
            total_memory_mib = mem_info.total / (1024**2)   # Exponentiation operation equivalent to mem_info.total * 1024 * 1024, required here for converting bytes to megabytes
            used_memory_mib = mem_info.used / (1024**2)
            free_memory_mib = mem_info.free / (1024**2)

            gpu_info = {
                'id': i,
                'name': name,
                'total_memory_mib': total_memory_mib,
                'used_memory_mib': used_memory_mib,
                'free_memory_mib': free_memory_mib
            }

            gpu_info_list.append(gpu_info)
            total_free_memory_mib += free_memory_mib

    except Exception as e:
        print(f"Error initializing NVML or getting GPU info: {e}")
        return []
    finally:
        # Shutdown NVML (important!)
        try:
            nvmlShutdown()
        except Exception:
            # This can happen if nvmlInit() failed
            pass

    return gpu_info_list, math.ceil(total_free_memory_mib)


def ensure_minimum_free_vram(vram_amount_mib=5120, shutdown_server_at_url_to_make_room=['http://localhost:9069']):
    '''
    Receives a request to ensure a certain amount of free VRAM, and a list of servers to shut down if necessary, in order of priority.
    '''
    try:
        _, total_free_memory_mib = get_nvidia_gpu_info()
        if total_free_memory_mib < vram_amount_mib:
            print(f"\nTotal free GPU memory is less than {vram_amount_mib}MB, attempting server shutdowns...\n")
            try:
                for server_url in shutdown_server_at_url_to_make_room:
                    print(f"\nSending shutdown command to server at URL {server_url}\n")
                    response = shutdown_waitress_server(server_url)    # Will either return True or False, either way we check the free VRAM again and continue if necessary
                    if response.get('was_offline'):
                        print(f"\nServer at URL {server_url} was already offline, skipping...\n")
                        continue

                    wait_timeout = 7    # seconds
                    max_attempts = 3    # ~21 seconds to allow for the server the shutdown before proceeding to the next one

                    for _ in range(max_attempts):
                        print(f"\nWaiting for {wait_timeout} seconds before checking free VRAM again...\n")
                        time.sleep(wait_timeout)
                        _, total_free_memory_mib = get_nvidia_gpu_info()
                        if total_free_memory_mib >= vram_amount_mib:
                            print(f"Requested VRAM freed by successfully shutting down server at URL {server_url}.")
                            return True
                
                # Finished iterating through the server list, and still not enough free VRAM as True would have been returned by now...
                _, total_free_memory_mib = get_nvidia_gpu_info()
                if total_free_memory_mib < vram_amount_mib:
                    raise Exception("Could not free the requested amount of GPU VRAM.")
            except Exception as e:
                raise Exception(f"Could not ensure minimum free GPU VRAM, encountered error: {e}")
    except Exception as e:
        raise Exception(f"Could not ensure minimum free GPU VRAM, encountered error: {e}")

