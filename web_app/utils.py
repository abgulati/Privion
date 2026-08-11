import requests
import platform
import signal
import subprocess
import math
import torch
import time
import multiprocessing
import re

from pynvml import *

from privion_config_concierge import read_config

def empty_cuda_cache():
    print("\n\nEmptying CUDA cache (in a separate process)\n\n")
    '''
    - Source: https://docs.pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html

    - Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
    - empty_cache() doesn’t increase the amount of GPU memory available for PyTorch. However, it may help reduce fragmentation of GPU memory in certain cases.
    - Think of it this way: It does not evict the people from the occupied offices!
    - See Memory Management for more details: https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
    '''

    # --- To test the timeout, uncomment the next line ---
    # print("Simulating a long-running operation that releases the GIL when waiting...")
    # time.sleep(10)

    # --- To test the deadlock, uncomment the next line ---
    # print("Simulating a long-running operation that does NOT release the GIL when waiting...")
    # while True:
    #     pass  # pass is a no-op statement that does nothing. It's used to create a "rude hang" that does not release the GIL.

    if torch.cuda.is_available():
        try:
            print("Attempting to empty cuda cache")
            torch.cuda.empty_cache()
            print("CUDA cache successfully emptied")
            return True
        except Exception as e:
            print(f"Could not empty cuda cache, encountered error: {str(e)}")
            return False
    else:
        print("\n\nCUDA is not available, skipping cache-emptying\n\n")
        return True


def safe_empty_cuda_cache(timeout:int=10):
    '''
    - Calls `empty_cuda_cache` in a separate process with a timeout.
    - CANNOT use concurrent.futures's ThreadPoolExecutor nor ProcessPoolExecutor as neither gurantess termination and the former even risks holding the GIL hostage. Details:

    - ThreadPoolExecutor (Unsafe):
        
        - Mechanism: Launches a thread, which shares the main process's Global Interpretor Lock (GIL) and other resources such as memory.
        - Intended Use: I/O bound (background) tasks (network requests/downloads, disk access, etc.) that release the GIL while waiting.
        - Why it fails:
            - Failure Mode 1 -- GIL-Releasing Hang (stuck network calls, etc.): `TimeoutError` will trigger but will result in a zombie thread leading to resource leakage and unpredictable behavior (operations, logs, etc.)
            - Failure Mode 2 -- GIL-Holding Hang (deadlocks): Worker thread holds the GIL hostage. Python threads cannot be killed forcefully, so the main thread can never acquire the GIL. `TimeError` is never raised and the entire application freezes.
        - Key Takeaways:
            - Unrelaible because we cannot predict the failure mode.
            - Python threads cannot be forcefully killed as a fundamental design philosophy: Forcefully killing a thread is extremely dangerous because threads share memory and resources.
            - Testing with `time.sleep()` is misleading as it's a "polite hang" (GIL-releasing). A "rude hang" (`while True:`) is a more accurate way of testing for a deadlock.

    - ProcessPoolExecutor (Deceptively Flawed):

        - Mechanism: Launches a completely separate process with it's own GIL, memory & resources.
        - Why it fails: High-level API that lacks the control needed for guaranteed termination.
            - If using `with ProcessPoolExecutor(max_workers=1) as executor`: The `with` contains an implicit `executor.shutdown(wait=True)`, which aims for graceful shutdown (completion) of the task. If that task hangs, the main process also hangs indefinitely waiting for it.
            - If manually managing `TimeoutError` Exceptions with `executor.shutdown(wait=False)`: Fire's and forgets a SIGTERM, without awaiting termination, allowing the main process to continue. Child process might not terminate at all(hung/deadlocked) or may still have time to complete leading to unpredictable behavior (operations, logs, etc.)
    '''

    try:
        '''
        For CUDA, it's safest to use the 'spawn' start method to create a clean process: this creates a new interpreter, re-imports all modules, serializes and sends data before executing the target function.
        The other option is `fork`, which is a near-instant clone of the parent processbut shares the same memory space as the main process, which is not safe for CUDA. In fact, it's fundamentally incompatible:
            - If the parent process has already touched the GPU and initialized the CUDA context, the child process inherits a "stale" copy of that context. 
            - When the child then tries to use CUDA, the driver sees a new process ID trying to operate on a context owned by the parent process ID. 
            - This leads to a state mismatch that almost always results in a crash, a hard deadlock, or a CUDA_ERROR_NOT_INITIALIZED error. fork and CUDA are thus fundamentally incompatible!
        '''
        ctx = multiprocessing.get_context('spawn')
        p = ctx.Process(target=empty_cuda_cache)

        print(f"\nAttempting empty_cuda_cache with a {timeout}-second timeout\n")
        p.start()
        p.join(timeout=timeout) # wait for the process to complete or timeout

        if p.is_alive():
            print(f"\nCUDA cache could not be emptied within {timeout} seconds. Terminating operation and continuing...\n")
            p.terminate()
            p.join()    # curcial: waits for the process to fully terminate and clean up resources. Prevents unpredicatable behavior (operations, logs, etc.)
            print(f"\nCUDA cache-clearing operation terminated\n")

        else:
            if p.exitcode == 0:
                print(f"\nCUDA cache successfully emptied\n")
            else:
                print(f"\nCUDA cache clearing operation finsihed with an error: (exit code {p.exitcode}).\n")

    except Exception:
        print("\nReturning without emptying CUDA cache\n")


def clean_cuda_devices_list(cuda_devices_list:str | list[str | int]) -> str:
    if isinstance(cuda_devices_list, str):
        cleaned = re.sub(r'\s+', '', cuda_devices_list)
    elif isinstance(cuda_devices_list, list):
        cleaned = ','.join(str(x).strip() for x in cuda_devices_list)
    else:
        raise ValueError(
            "Invalid type for cuda_devices_list:"
            f"{type(cuda_devices_list)}"
        )

    if not cleaned:
        raise ValueError(
            "CUDA device list is empty. Provide device IDs like '0' or '0,1', "
            "or enable llama_cpp_cuda_all_devices."
        )

    return cleaned


def add_column_if_not_exists(cursor, table_name, column_name, column_type):
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        if column_name not in column_names:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    except Exception as e:
        print(f"Error adding column {column_name} to {table_name}: ", e)


def get_container_id_by_container_name(container_name:str) -> str:
    print(f"\nChecking if container {container_name} is running...\n")
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name={container_name}' , '--format', '{{.ID}}'], # get container ID
            capture_output=True,    # captures the command's output and error, while suppressing the print to the terminal
            text=True,  # Get output as string and not bytes
            check=True
        )
        return result.stdout.strip()    # Will return the container ID if the container is running, otherwise will return an empty string!
    except Exception as e:
        raise Exception(f"Could not check if {container_name} Docker container is running, encountered error: {e}")


def check_if_container_is_running(container_name:str) -> bool:
    try:
        container_id = get_container_id_by_container_name(container_name)
        return container_id is not None and container_id != ""
    except Exception as e:
        print(f"Could not check if {container_name} Docker container is running, encountered error: {e}")
        return False


def check_if_container_exists(container_name:str) -> bool:
    '''
    Check if container exists (running or stopped) and return the ID if so.
    Use this to check for already existing containers, because `docker run` will throw a silent conflict error 
    if attempting to start a containe rwith the name of an already existing container, even if it's stopped.
    '''

    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', f'name={container_name}' , '--format', '{{.Names}}'], # get container ID
            capture_output=True,    # captures the command's output and error, while suppressing the print to the terminal
            text=True,  # Get output as string and not bytes
            check=True
        )
        containers = result.stdout.strip().split('\n')
        return container_name in containers
    except Exception as e:
        print(f"Could not check if {container_name} Docker container exists, encountered error: {e}")
        return False


def start_container(container_name:str) -> bool:
    '''Start an existing container by name.'''
    try:
        print(f"\nStarting existing Docker container '{container_name}'...\n")
        result = subprocess.run(
            ['docker', 'start', container_name],
            capture_output=True,    # captures the command's output and error, while suppressing the print to the terminal
            text=True,  # Get output as string and not bytes
            check=True
        )
        print(f"\nDocker container '{container_name}' successfully started\n")
        return True
    except Exception as e:
        raise Exception(f"Could not start {container_name} Docker container, encountered error: {e}")


def get_url_for_server(server_to_check):
    if server_to_check == 'llama-cpp':
        try:
            read_return = read_config(['llama_cpp_access_url', 'llama_cpp_server_port'])
            return f'http://{read_return["llama_cpp_access_url"]}:{read_return["llama_cpp_server_port"]}'
        except Exception as e:
            print("Could not read llama_cpp_access_url and llama_cpp_server_port from config.json, using default localhost:8080 instead. Encountered error: ", e)
            return 'http://localhost:8080'
    elif server_to_check == 'hf-waitress':
        try:
            read_return = read_config(['hf_waitress_access_url', 'hf_waitress_server_port'])
            return f'http://{read_return["hf_waitress_access_url"]}:{read_return["hf_waitress_server_port"]}'
        except Exception as e:
            print("Could not read hf_waitress_access_url and hf_waitress_server_port from config.json, using default localhost:9069 instead. Encountered error: ", e)
            return 'http://localhost:9069'
    elif server_to_check == 'asr-waitress':
        try:
            read_return = read_config(['asr_waitress_access_url', 'asr_waitress_server_port'])
            return f'http://{read_return["asr_waitress_access_url"]}:{read_return["asr_waitress_server_port"]}'
        except Exception as e:
            print("Could not read asr_waitress_access_url and asr_waitress_server_port from config.json, using default localhost:10087 instead. Encountered error: ", e)
            return 'http://localhost:10087'
    else:
        raise Exception(f"Invalid server choice, expected 'llama-cpp', 'hf-waitress', or 'asr-waitress', received: {server_to_check}")


def is_local_server_online(server_base_url:str) -> dict:
    '''
    Checks if a local server is online.

    Args:
        server_base_url: The base URL of the server to check

    Returns:
        dict: A dictionary containing:
            'server_available': True if the server is available, False otherwise
            'server_online': True if the server is online, False otherwise
            'loading_model': True if the server is loading a model, False otherwise
            'status_code': The HTTP status code of the response

    Raises:
        Exception: If the server could not be checked, or if an error occurs during the process
    '''

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


def shutdown_waitress_server(base_url:str='http://localhost:9069') -> dict:
    '''
    Shuts down an HF-Waitress server.

    Args:
        base_url: The base URL of the server to shut down

    Returns:
        dict: A dictionary containing:
            'success': True if the server was shutdown, False otherwise
            'message': A message describing the result
            'was_offline': True if the server was already offline, False otherwise

    Raises:
        Exception: If the server could not be shutdown, or if an error occurs during the process
    '''
    print(f"\n\nShutting down HF-Waitress server at: {base_url}\n\n")
    try:
        if is_local_server_online(base_url)['server_online']:
            url = f"{base_url}/shutdown_hf_waitress"
            payload = ""
            headers = {}
            response = requests.post(url, data=payload, headers=headers)
            print("\nHF-Waitress server successfully shutdown\n")
            return {"success":response.json()['success'], "message":response.json()['message'], "was_offline":False}
        else:
            print("\nHF-Waitress server is already offline at the specified URL.\n")
            return {"success":True, "message":"HF-Waitress server is already offline at the specified URL.", "was_offline":True}
    
    except Exception as e:
        print(f"\n\nCould not shutdown HF-Waitress server, encountered error: {e}\n\n")
        return {"success":False, "message":f"Could not shutdown HF-Waitress server, encountered error: {e}", "was_offline":False}
    

def send_ctrl_c_to_process(process:subprocess.Popen) -> dict:
    '''
    Sends a Ctrl+C termination signal to a process.

    Args:
        process: The process to send the Ctrl+C signal to

    Returns:
        dict: A dictionary containing:
            'success': True if the process was terminated, False otherwise
            'message': A message describing the result
            'was_offline': True if the process was already offline, False otherwise
    '''

    try:
        if process.poll() is None:  # check if process is still running via poll(), which returns None if a process is still running 
            if platform.system() == 'Windows':
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            
            try:
                process.wait(timeout=3) # Wait a bit for the process to terminate gracefully
            except subprocess.TimeoutExpired:
                print("\n\nProcess did not terminate within timeout, will be force-killed.\n\n")
                process.kill()  # Sends 'SIGKILL' on Unix-like to force-kill immediately / 'TerminateProcess' on Windows which still allows for graceful termination
                process.wait()
            
            if process.poll() is not None:
                print("\n\nProcess terminated successfully.\n\n")
                return {'success': True, 'message': f"Successfully terminated process: {process}", 'was_offline':False}
            else:
                raise Exception("\n\nProcess still running after force kill attempt.\n\n")
        else:
            return {'success': True, 'message': f"Confirmed process {process} is not running.", 'was_offline':True}
    
    except Exception as e:
        err_msg = f"Could not force-kill process, encountered error: {e}"
        return {'success': False, 'message': err_msg, 'was_offline':False}


def shutdown_local_llm_server_process(process:subprocess.Popen) -> dict:
    '''
    Provides a wrapper around `send-ctrl_c_to_process` to serve as a similar interface to `shutdown_waitress_server`.\n
    `send-ctrl_c_to_process` ensures the process is terminated and the result is returned, and handles any errors gracefully.\n
    This method will typically be used to shutdown command-line LLM servers without a dedicated shutdown API, such as llama.cpp.\n

    Args:
        process: The process to shutdown

    Returns:

        dict: generated by `send_ctrl_c_to_process`
    '''
    return send_ctrl_c_to_process(process)


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
        return [], 0
    finally:
        try:
            nvmlShutdown()  # Shutdown NVML (important!)
        except Exception:
            pass    # This can happen if nvmlInit() failed

    return gpu_info_list, math.ceil(total_free_memory_mib)


def ensure_minimum_free_vram(vram_amount_mib:int=5120, shutdown_server_at_url_to_make_room:list[str]=['http://localhost:9069']) -> dict:
    '''
    Handles requests to ensure a certain amount of free VRAM.

    Args:
        vram_amount_mib: The amount of free VRAM to ensure (in MiB)
        shutdown_server_at_url_to_make_room: A list of server URLs to shut down if necessary, in order of priority

    Returns:
        dict: A dictionary containing:
            'success': True if the minimum free VRAM was ensured, False otherwise
            'message': A message describing the result
            'was_offline': True if the server was already offline, False otherwise

    Raises:
        Exception: If the minimum free VRAM was not ensured, or if an error occurs during the process
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
        err_msg = f"Could not ensure minimum free GPU VRAM, encountered error: {e}"
        return {'success': False, 'message': err_msg, 'was_offline':False}

