from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, BitsAndBytesConfig, QuantoConfig, HqqConfig, T5EncoderModel, CLIPTextModel, AutoProcessor, GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from huggingface_hub import login, snapshot_download, scan_cache_dir
import torch

try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2DynamicJob
    from exllamav2 import ExLlamaV2Cache, ExLlamaV2Cache_8bit, ExLlamaV2Cache_Q4, ExLlamaV2Cache_Q6, ExLlamaV2Cache_Q8
except ImportError:
    print("exllamav2 is not installed. Skipping import.")

from diffusers import FluxPipeline, FluxTransformer2DModel

try:
    from optimum.quanto import freeze, qfloat8, quantize
except ImportError:
    print("optimum.quanto is not installed. Skipping import.")

try:
    from transformers import MllamaForConditionalGeneration
except ImportError:
    print("transformers version is below 4.45.0 required from Llama3.2-Vision. Skipping MllamaForConditionalGeneration import.")

try:
    import prompt_formatting as prompt_formatting_module
except ImportError:
    print("Prompt Formatter module `prompt_formatting.py` is not present. Skipping import. Must be present for exl2 bulk-summary generation.")

from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

import multiprocessing
import subprocess
import threading
import traceback
import argparse
import platform
import datetime
import logging
import pathlib
import base64
import shutil   # Shell Utilities is part of Python's standard library and is used for file operations
import queue
import time
import json
import uuid
import sys
import ast
import os
import io

from functools import wraps
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from waitress import serve

if not os.path.exists(os.path.join(os.getcwd(), 'exllamav2')):
    subprocess.run(['git', 'clone', '-b', 'v0.3.2', 'https://github.com/turboderp-org/exllamav2.git'], check=True)  # check=True raises an exception on non-zero exit code

app = Flask(__name__)
CORS(app)


@app.route('/serve_generated_image/<path:filename>')
def serve_generated_image(filename:str):
    print(f"\n\nserving generated image: {filename}\n\n")
    generated_images_folder = "generated_images"
    try:
        generated_images_folder = read_config(['generated_images_folder'])['generated_images_folder']
    except Exception as e:
        handle_error_no_return("Could not read generated_images_folder from hf_config.json, using default: generated_images in the current working directory. Encountered error: ", e)

    return send_from_directory(generated_images_folder, filename)

@app.route('/serve_uploaded_file/<path:filename>')
def serve_uploaded_file(filename:str):
    print(f"\n\nserving uploaded file: {filename}\n\n")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



#########################------------------GLOBALS!----------------------###############################
PIPE = None
VISION_MODEL = None
EXL2_MODEL = None
EXL2_TOKENIZER = None
EXL2_CACHE = None
AUTO_TOKENIZER = None

STOP_GENERATION = False
llm_semaphore = threading.Semaphore(1)
config_writer_semaphore = threading.Semaphore(1)
error_logging_semaphore = threading.Semaphore(1)
reader_semaphore = threading.Semaphore(3)

###---Complete List of HF-Transformers Environment Variables: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables

# os.environ['HUGGINGFACE_HUB_CACHE'] = transformer_models_folder
# os.environ['TRANSFORMERS_CACHE'] = transformer_models_folder
# os.environ['HF_HOME'] = transformer_models_folder
os.environ['HF_HUB_ENABLE_EMERGENCY_RETRY'] = 'true'
os.environ['HF_HUB_EMERGENCY_RETRY_WAIT_TIME'] = '10'
os.environ['HF_XET_HIGH_PERFORMANCE'] = 'true'
os.environ['HF_HUB_DISABLE_XET'] = 'true'

#########################------------------------------------------------###############################



########################---------------------config setup---------------------###############################
def _early_os_default_base_dir():
    try:
        sysname = platform.system()
    except Exception as e:
        print(f"Could not determine OS, defaulting to 'app' dir. Encountered error: {e}")
        sysname = ''
    if sysname == 'Windows':
        return 'C:/waitress_storage'
    elif sysname == 'Linux':
        return '/app/lars_storage'
    else:   # For Darwin (Mac) and otherwise
        return 'app'


def _early_resolve_base_and_config():
    # Code-local bootstrap pointer (same dir as app.py)
    bootstrap_path = os.path.join(os.getcwd(), 'waitress_storage_config.json')
    base = _early_os_default_base_dir()
    cfg_path = os.path.join(os.getcwd(), 'hf_config.json')
    if os.path.exists(bootstrap_path):
        try:
            with open(bootstrap_path, 'r') as f:
                boot = json.load(f) or {}
            base = boot.get('base_directory', base)
            cfg_path = boot.get('config_path', cfg_path)
        except Exception as e:
            print(f"Could not read waitress_storage_config.json, defaulting to base dir: {base}. Encountered error: {e}")

    try:
        os.makedirs(base, exist_ok=True)
    except Exception as e:
        print(f"Could not create base directory: {base}. Encountered error: {e}")
    
    return base, cfg_path, bootstrap_path

BASE_DIRECTORY, CONFIG_PATH, BOOTSTRAP_PATH = _early_resolve_base_and_config()

# Create real config if missing - no error handling as an exception should stop execution!
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as file:
        json.dump({}, file, indent=4)

# Update config with base_directory
try:
    with open(CONFIG_PATH, 'r+') as file:   # Unlike w, r+ allows updates without overwriting the whole file, but requires seek & truncation alongside the dump!
        config = json.load(file)
        '''
        TODO: `if config.get('base_directory') != BASE_DIRECTORY:`, then move contents of old dir to new dir! 
        Invoke: `_move_contents_of_old_dir_to_new_dir(old_dir, new_dir)`
        Verify if referencing system can handle moves first!
        '''
        config['base_directory'] = BASE_DIRECTORY
        file.seek(0)    # move file-pointer back to the start of the file before writing!
        json.dump(config, file, indent=4)
        file.truncate()    # truncate the file in case new config data is shorter than the original data! Eg: 'very_long_dir_name' -> 'short_dir_name'!
except Exception as e:
    print(f"Could not read config.json, encountered error: {e}")

# Ensure botstrap contains only base_directory (so users can move by editing this one knob!)
try:
    with open(BOOTSTRAP_PATH, 'w') as file:
        json.dump({'base_directory': str(pathlib.Path(BASE_DIRECTORY).resolve()), 'config_path': str(pathlib.Path(CONFIG_PATH).resolve())}, file, indent=4)
except Exception as e:
    print(f"Could not write bootstrap storage_config.json, encountered error: {e}")

#######################################################################################################



#########################------------Setup & Handle Logging-------------###############################
LOGS_DIR = os.path.join(os.getcwd(), 'waitress_server_logs')
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOGS_DIR, 'hf_server_log.log')

try:
    # 1 - Create a logger
    LOGGER = logging.getLogger('my_logger')
    LOGGER.setLevel(logging.ERROR)

    # 2 - Create a RotatingFileHandler
    # maxBytes: 1024 * 1024 * 5 Bytes = 5MB max file size per log, 2 backups = 3 files total
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1024*1024*5, backupCount=2)
    handler.setLevel(logging.ERROR)

    # 3 - Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)

    # 4 - Add the handler to the logger for final LOGGER - Usage: LOGGER.error(f"This is an error message with error {e}")
    LOGGER.addHandler(handler)
except Exception as e:
    print(f"\n\nCould not establish logger, encountered error: {e}")


def central_error_logging(message:str, exception:Exception=None):
    with error_logging_semaphore:
        error_message = f"\n\n{message} {str(exception) if exception else '; No exception info.'}\n\n"
        
        # traceback.format_exc() is most reliable when called directly from within an except block. If passing an exception object, it's best to handle it more explicitly!
        if exception:
            # To get the traceback of the passed 'exception' object
            traceback_details = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        else:
            # If no specific exception, format_exc() might give current stack if in an except block, or minimal info
            traceback_details = traceback.format_exc() if sys.exc_info()[0] else "No active exception."
        
        full_message = f"\n\n{error_message}\n\nTraceback: {traceback_details}\n\n"

        if LOGGER:
            LOGGER.error(full_message)
            print(error_message)
        else:
            print(error_message)
    
    return error_message


def handle_api_error(message:str, exception:Exception=None):
    error_message = central_error_logging(message, exception)
    return jsonify(success=False, error=error_message), 500 #internal server error


def handle_local_error(message:str, exception:Exception=None):
    _ = central_error_logging(message, exception)
    raise Exception(exception)


def handle_error_no_return(message:str, exception:Exception=None):
    _ = central_error_logging(message, exception)


def set_load_safe_defaults():
    try:
        write_config({'load_safe_defaults': True})
    except Exception as e:
        handle_error_no_return("Could not set load_safe_defaults to true in hf_config.json, encountered error: ", e)

def handle_model_loading_error(message:str, exception:Exception=None, target:str="local"):
    
    try:
        set_load_safe_defaults()
    except Exception as e:
        handle_error_no_return("Could not set load_safe_defaults to true in hf_config.json, encountered error: ", e)
    
    if target == "local":
        handle_local_error(message, exception)
    elif target == "api":
        return handle_api_error(message, exception)

############################----------------------------------------------###############################



############################------------configuration manager-------------###############################

def write_config(config_updates:dict, filename:str=None) -> dict:
    '''
    Method to write app configuration to hf_config.json.\n
    Acquires a semaphore to prevent concurrent writes to the file.
    
    Args:
        - config_updates: dict of key:values to be written to hf_config.json
        - filename: name of the file to write to, defaults to None which sets to CONFIG_PATH

    Returns:
        - Confirmation of success: {success: True}

    Raises:
        - Exception: If the file cannot be written to
    '''

    filename = filename or CONFIG_PATH

    with config_writer_semaphore:

        # Open hf_config file to read-in all current params:
        try:
            with open(filename, 'r') as file:
                hf_config = json.load(file)
        except Exception as e:
            hf_config = {}     #init emply hf_config dict
            handle_error_no_return("Could not read hf_config.json when attempting to write, encountered error: ", e)

        #restart logic in write_config() might be unnecessary, circle back later
        restart_required = False
        hard_reboot_required = False
        model_changed = False
        triggers_for_hf_restart = [
            'torch_device_map',
            'torch_dtype',
            'model_id',
            'awq',
            'attn_implementation',
            'pipeline_task',
            'quantize',
            'quant_level',
            'port',
            'use_flash_attention_2',
            'hqq_group_size',
            'flux_diffusers',
            'flux_low_vram_optimizations',
            'load_quantized_flux',
            'vision',
            'exl2',
            'exl2_bpw',
            'exl2_cache_type',
            'exl2_max_seq_len',
            'exl2_force_regenerate_measurement'
        ]

        triggers_for_hard_reboot = [
            'exl2',
            'exl2_bpw',
            'exl2_max_seq_len',
            'exl2_cache_type'
        ]
        
        for key in config_updates:
            if key in triggers_for_hf_restart and config_updates[key] != hf_config.get(key):
                restart_required = True
                if key == 'model_id': model_changed = True
                if key in triggers_for_hard_reboot: hard_reboot_required = True

        if config_updates.get('exl2', False) and model_changed:
            print("ExL2 status changed and model changed, setting hard_reboot_required to True")
            hard_reboot_required = True

        # Auto-detect Flux and Llama-3.2-Vision models
        if "flux" in config_updates.get('model_id', '').lower():
            print("Flux model auto-detected, setting flux_diffusers=True")
            config_updates['flux_diffusers'] = True
        else:
            config_updates['flux_diffusers'] = False

        if "llama-3.2" in config_updates.get('model_id', '').lower() and "vision" in config_updates.get('model_id', '').lower():
            print("Llama-3.2-Vision model auto-detected, setting vision=True")
            config_updates['vision'] = True
        else:
            config_updates['vision'] = False

        hf_config.update(config_updates)

        # Write updated hf_config.json:
        try:
            with open(filename, 'w') as file:
                json.dump(hf_config, file, indent=4)
        except Exception as e:
            handle_local_error("Could not update hf_config.json, encountered error: ", e)
        
        return {'success': True, 'restart_required':restart_required, 'hard_reboot_required':hard_reboot_required}
            

def read_config(keys:list, default_value=None, filename=None) -> dict:
    '''
    Method to read app configuration from hf_config.json. Central method to configure safe application defaults.
    Acquires a semaphore to prevent concurrent reads to the file.
    
    Args:
        - keys: list of keys to read from hf_config.json
        - default_value: default value to return if a key is not found in hf_config.json, defaults to None
        - filename: name of the file to read from, defaults to None which sets to CONFIG_PATH

    Returns:
        - dict of key:values read from hf_config.json

    Raises:
        - KeyError: If a key is not found in hf_config.json and no default value has been defined
    '''

    filename = filename or CONFIG_PATH

    with reader_semaphore:
    
        # Open hf_config file to read-in all current params:
        try:
            with open(filename, 'r') as file:
                hf_config = json.load(file)
        except Exception as e:
            handle_error_no_return("Could not read hf_config.json, encountered error: ", e)
            return {key: default_value for key in keys}     #because a read scenario wherein hf_config.json does not exist shouldn't occur!
        
        return_dict = {}
        update_config_dict = {}
        base_directory = hf_config.get('base_directory', BASE_DIRECTORY)   # specifying default if not found

        for key in keys:
            if key in hf_config:
                return_dict[key] = hf_config[key]
            else:
                default_value = {
                    'upload_folder':base_directory + '/uploaded_files_for_vision_inferencing',
                    'generated_images_folder':base_directory + '/generated_images',
                    'transformer_models_folder':base_directory + '/transformer_models',
                    'knowledge_graph_cache_dir': base_directory + '/knowledge_graph_cache_dir',
                    'access_gated':False,
                    'access_token':"",
                    'model_id':"Qwen/Qwen2.5-1.5B-Instruct",
                    'exl2':False,
                    'exl2_bpw':3.0,
                    'exl2_cache_type':"ExLlamaV2Cache",
                    'exl2_max_seq_len':2048,
                    'exl2_force_regenerate_measurement':False,
                    'exl2_no_flash_attn':False,
                    'reuse_graph_extraction_cache':True,    # dev flag: controlled by X-Reuse-Extraction-Cache header
                    'reuse_graph_summary_cache':True,       # dev flag: controlled by X-Reuse-Summary-Cache header
                    'gguf':False,
                    'awq':False,
                    'flux_diffusers':False,
                    'flux_low_vram_optimizations':True,
                    'load_quantized_flux':False,
                    'vision':False,
                    'gguf_model_id':None,
                    'gguf_filename':None,
                    'quantize':"quanto",
                    'quant_level':"int8",
                    'hqq_group_size':64,
                    'push_to_hub':False,
                    'torch_device_map':"auto", 
                    'torch_dtype':"auto", 
                    'trust_remote_code':True, 
                    'use_flash_attention_2':False, 
                    'pipeline_task':"text-generation", 
                    'max_new_tokens':700, 
                    'return_full_text':False, 
                    'temperature':0.1,
                    'do_sample':True, 
                    'top_k':40, 
                    'top_p':0.95, 
                    'min_p':0.05, 
                    'n_keep':0,
                    'port':9069,
                    'host':'0.0.0.0',
                    'load_safe_defaults':False,
                    'model_list': [
                                    'meta-llama/Llama-3.2-11B-Vision-Instruct',
                                    'meta-llama/Llama-3.2-1B-Instruct',
                                    'meta-llama/Llama-3.2-3B-Instruct',
                                    'black-forest-labs/FLUX.1-schnell',
                                    'black-forest-labs/FLUX.1-dev',
                                    'mistralai/Mistral-Nemo-Instruct-2407', 
                                    'meta-llama/Meta-Llama-3.1-8B-Instruct', 
                                    'meta-llama/Meta-Llama-3.1-70B-Instruct', 
                                    'meta-llama/Meta-Llama-3.1-405B-Instruct-FP8',
                                    'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
                                    'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
                                    'microsoft/Phi-3.5-mini-instruct',
                                    'microsoft/Phi-3.5-MoE-instruct',
                                    'microsoft/Phi-3-mini-4k-instruct',
                                    'microsoft/Phi-3-mini-128k-instruct',
                                    'microsoft/Phi-3-small-8k-instruct',
                                    'microsoft/Phi-3-small-128k-instruct',
                                    'microsoft/Phi-3-medium-4k-instruct',
                                    'microsoft/Phi-3-medium-128k-instruct',
                                    'CohereForAI/c4ai-command-r-plus',
                                    'CohereForAI/c4ai-command-r-v01',
                                    'google/gemma-2-2b-it',
                                    'google/gemma-2-9b-it',
                                    'google/gemma-2-27b-it',
                                    'Qwen/Qwen2.5-1.5B-Instruct',
                                    'Qwen/Qwen2-7B-Instruct',
                                    'Qwen/Qwen2-72B-Instruct',
                                    'alpindale/goliath-120b',
                                    'TheBloke/goliath-120b-AWQ'
                                   ]
                }.get(key, 'undefined')

                if default_value == 'undefined':
                    raise KeyError(f"Key \'{key}\' not found in hf_config.json and no default value has been defined either.\n")
                
                return_dict[key] = default_value
                update_config_dict[key] = default_value
        
        if update_config_dict:
            # Write defaults
            try:
                write_config(update_config_dict)
            except Exception as e:
                handle_error_no_return("Could not write defaults to hf_config.json. Encountered error: ", e)

        return return_dict


# Method for API route to read from hf_config.json
# Deviates from typical RESTful principals to use a POST call to fetch values but practical & justifyable because we:
# 1. Do not want to make the URL huge with a ever-growing list of query-params 2. Do not wish to expose values via query-params
@app.route('/hf_config_reader_api', methods=['POST'])
def hf_config_reader_api():
    # keys = request.args.getlist('keys') # Assuming keys are passed as query parameters
    
    try:
        keys = request.json.get('keys', []) # Could also do keys = request.json['keys'] but this way we can provide a default list should 'keys' be missing!
    except Exception as e:
        handle_api_error("Server-side error - could not read keys for hf_config_reader_api request. Encountered error:", e)

    try:
        values = read_config(keys)  # send list of keys, get dict of key:values
    except Exception as e:
        handle_api_error("Server-side error - could not read keys from hf_config.json. Encountered error: ", e)
    
    return jsonify(success=True, values=values)


# Method for API route to write to hf_config.json
@app.route('/hf_config_writer_api', methods=['POST'])
def hf_config_writer_api():

    try:
        config_updates = request.json['config_updates']
        print(f"\n\nconfig_updates for hf_config_writer_api:\n{config_updates}\n\n")
    except Exception as e:
        handle_api_error("Server-side error - could not read values for hf_config_writer_api request. Encountered error: ", e)
    
    try:
        write_return = write_config(config_updates)
    except Exception as e:
        handle_api_error("Server-side error - could not write keys to hf_config.json. Encountered error: ", e)
    
    return jsonify({"success": write_return['success'], "restart_required": write_return['restart_required'], "hard_reboot_required": write_return['hard_reboot_required']})

############################----------------------------------------------###############################



############################------------Security Config Manager-------------###############################
SECURITY_CONFIG_FILE = 'security_config.json'

def read_security_config():
    try:    # Create with defaults if absent
        if not os.path.exists(SECURITY_CONFIG_FILE):
            default_security_config = {
                'shutdown_allowed_ips': ['127.0.0.1', '::1'],   # IPv4 and IPv6 localhost
                '_comment': 'Edit this file manually to add IPs. No APIs exist to modify this config for security.'
            }

            with open(SECURITY_CONFIG_FILE, 'w') as file:
                json.dump(default_security_config, file, indent=4)

            print(f"\n🔐 Security config created: {SECURITY_CONFIG_FILE}")
            print(f"   📝 Edit manually to add IPs - no API access for security\n")

            return default_security_config
    except Exception as e:
        handle_error_no_return("Could not create security_config.json, returning localhost IPs. Encountered error: ", e)
        return {'shutdown_allowed_ips': ['127.0.0.1', '::1']}

    try:    # Read and return, updating if necessary
        config = {}
        with open(SECURITY_CONFIG_FILE, 'r') as file:
            config = json.load(file)

        if 'shutdown_allowed_ips' not in config:
            config['shutdown_allowed_ips'] = ['127.0.0.1', '::1']
            # Only writing if modified
            with open(SECURITY_CONFIG_FILE, 'w') as file:
                json.dump(config, file, indent=4)
            print(f"🔐 'shutdown_allowed_ips' added to {SECURITY_CONFIG_FILE}")

        return config

    except Exception as e:
        handle_error_no_return("Could not read security_config.json, returning localhost IPs. Encountered error: ", e)
        return {'shutdown_allowed_ips': ['127.0.0.1', '::1']}


def get_client_ip(request):
    '''
    Get the requesting client's IP address from the request headers.

    1. X-Forwarded-For:
        - For reverse proxies, they'll add a forwarded-for header. Without this you'll get the proxies address instead of the actual requesting client!
        - 'X-Forwarded-For' might be a comma-separated list of IPs (e.g., client_ip, proxy1_ip, proxy2_ip)
        - Typically we'll want the leftmost IP address in the X-Forwarded-For list, as this is usually the original client. 
        - However, this relies on trusting your immediate upstream proxy not to allow spoofing of this header or to correctly set/append to it!
    2. X-Real-IP:
        - Some proxies (like Nginx) might also set this header, typically with just the original client's IP.
    3. REMOTE_ADDR:
        - The WSGI server (like Gunicorn, uWSGI, or Flask's built-in dev server) populates the environ dictionary with various details about the incoming HTTP request. 
        - This is a safe way to access the dict.
    '''
    try:
        if 'X-Forwarded-For' in request.headers:
            return request.headers['X-Forwarded-For'].split(',')[0].strip() 
        elif 'X-Real-IP' in request.headers:
            return request.headers['X-Real-IP'] 
        else:
            return request.environ.get('REMOTE_ADDR', 'unknown')
    except Exception as e:
        handle_error_no_return("Could not get client IP, checking for REMOTE_ADDR or returning 'unknown'. Encountered error: ", e)
        return request.environ.get('REMOTE_ADDR', 'unknown')


def is_shutdown_allowed(client_ip):
    security_config = read_security_config()
    allowed_ips = security_config.get('shutdown_allowed_ips', ['127.0.0.1', '::1'])
    return client_ip in allowed_ips


'''
The /shutdown_hf_waitress API below is deliberately a POST request even though no data is being sent. 
This is for security purposes and as per standard best practices:

1. Semantic Correctness - Shutdown is a state-changing operation (not idempotent), which makes POST more appropriate than GET according to HTTP semantics

2. Security Best Practice - GET requests:
    - Are logged in web server access logs (with full URL)
    - Can be cached by browsers/proxies
    - Might be prefetched by browsers
    - Could be triggered accidentally by bots/crawlers

3. CSRF Protection - POST requests are harder to trigger accidentally via cross-site requests

4. Industry Standard - Most shutdown/restart APIs use POST (Docker, Kubernetes, etc.)

5. You Could Use GET, but this could be dangerous because:
    - A simple curl http://localhost:9069/shutdown_hf_waitress would shut down the server
    - Browser prefetching could accidentally trigger it
    - Web crawlers might hit it

6. The POST requirement adds a small layer of protection against accidental shutdowns, which is valuable for a destructive operation like server shutdown.
'''
@app.route('/shutdown_hf_waitress', methods=['POST'])
def shutdown_hf_waitress():
    try:
    
        try:
            client_ip = get_client_ip(request)
        except Exception as e:
            return handle_api_error("Server-side error - could not get client IP for shutdown_hf_waitress request. Encountered error: ", e)

        if not is_shutdown_allowed(client_ip):
            return jsonify(success=False, error="Shutdown not allowed for this IP.")
        
        shutdown_message = f"\n\n🔓 SHUTDOWN AUTHORIZED from IP {client_ip}\n🕐 Shutdown time: {datetime.datetime.now()}\n\n"
        print(shutdown_message)
        if LOGGER:
            try:
                LOGGER.log(logging.INFO, shutdown_message)  # Logger.log() requires the logging level (INFO, ERROR, etc.) and the message to log
            except Exception as e:
                handle_error_no_return("Could not log shutdown message to logger, skipping. Encountered error: ", e)

        with llm_semaphore:
            with config_writer_semaphore:
                with error_logging_semaphore:
                    print("🔒 All semaphores acquired, proceeding with shutdown...")

                    def cleanup():
                        try:
                            shutdown_vision_model()
                            shutdown_pipe()
                            shutdown_exl2()

                            print("✅ All models, pipes, and exl2 caches shut down...")
                            print("✅ Cleanup completed successfully")
                        except Exception as e:
                            handle_error_no_return("Could not shutdown model, pipe, or exl2, proceeding to force-kill. Encountered error: ", e)

                    cleanup_thread = threading.Thread(target=cleanup)
                    cleanup_thread.start()

                    print("⏰ Starting cleanup with 7-second timeout...")
                    cleanup_thread.join(timeout=7)    # Wait max 7 seconds

                    if cleanup_thread.is_alive():
                        print("⚠️  Cleanup timeout reached (7 seconds) - proceeding with force shutdown")
                    else:
                        print("✅ Graceful shutdown completed")

                    response = jsonify(success=True, message="Server shutting down gracefully...")

                    def delayed_shutdown():
                        time.sleep(1)   # Giving a chance for the HTTP response to be sent to the client!
                        print("🔚 Terminating server process...")
                        os._exit(0) # os._exit() will terminate the Server process.

                    threading.Thread(target=delayed_shutdown, daemon=True).start()  # Daemon ensures that the thread will terminate when the main process terminates, as a daemon is a child process of the main process.
                    return response
    
    except Exception as e:
        return handle_api_error("Shutdown error: ", e)

############################----------------------------------------------################################



#########################------------Setup Directories-------------###############################


###---Notes on the above workflow:---###
# 1. Everytime the app runs, the OS platform is detected and the appropriate OS-specific base directory is requested above
# 2. If this is the very first run:
#   a. read_config does not find the directory data in config.json
#   b. the else clause is triggered and defaults are written to config.json and subsequently returned
# 3. If this isn't the very first run, read_config simply returns the OS specific directory (windows_base_directory, unix_and_docker_base_directory, and mac_base_directory)
# 4. Basis this, BASE_DIRECTORY is written to config.json
# 5. This setup ensures that:
#   a. directories are set correctly at each run
#   b. The user can set their preferred directory by easily editing config.json!

# Having set the values for the directories above, proceed to actually create them on disk IF they don't alread exist!

try:
    read_return = read_config(['upload_folder', 'generated_images_folder', 'transformer_models_folder', 'knowledge_graph_cache_dir'])
except Exception as e:
    handle_local_error("Could not read paths for app directories (upload_folder, generated_images_folder) from config.json on boot, encountered error: ", e)

try:
    os.makedirs(read_return['upload_folder'], exist_ok=True)
    os.makedirs(read_return['generated_images_folder'], exist_ok=True)
    os.makedirs(read_return['transformer_models_folder'], exist_ok=True)
    os.makedirs(read_return['knowledge_graph_cache_dir'], exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create app directories, encountered error: ", e)

app.config['UPLOAD_FOLDER'] = read_return['upload_folder']

############################----------------------------------------------###############################



############################------------File & Folder Management-------------###############################

def load_json_file(file_path: str) -> dict | None:
    '''
    - Loads a JSON file from the given file path.
    - Returns:
        - dict: The JSON data from the file, or None if the file does not exist or cannot be loaded.
    - Returning None because:
        - isinstance({}, dict) will return True so better to return None!
        - It's more accurate to return None rather than an empty dict, as the former indicates that the file does not exist or cannot be loaded, 
        while the latter might be misconstrued as indicating that the file exists but is empty.
    '''
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            handle_local_error("Could not load JSON file, encountered error: ", e)
    else:
        return None


def update_and_save_json_file(data: dict, file_path: str) -> bool:
    '''
    - Updates a JSON file with the given data.
    - Returns:
        - bool: True if the file was updated and saved successfully, False otherwise.
    '''
    current_cache = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                current_cache = json.load(file)
        except Exception as e:
            handle_local_error("Could not save JSON file, encountered error: ", e)
    
    try:
        current_cache.update(data)
        with open(file_path, 'w') as file:
            json.dump(current_cache, file, indent=4)
    except Exception as e:
        handle_local_error("Could not save JSON file, encountered error: ", e)

    return True


def overwrite_json_file(data: dict, file_path: str) -> bool:
    '''
    - Overwrites a JSON file with the given data.
    - Returns:
        - bool: True if the file was overwritten and saved successfully, False otherwise.
    '''
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        handle_local_error("Could not save JSON file, encountered error: ", e)

    return True


def remove_file_from_filepath(filepath):
    print(f"\n\nRemoving file from filepath: {filepath}\n\n")
    try:
        os.remove(filepath)
        print(f"Successfully deleted file: {filepath}")
    except Exception as e:
        handle_local_error(f"Could not remove file from filepath: {filepath}, encountered error: ", e)


def safe_remove_file_from_filepath(filepath):
    try:
        remove_file_from_filepath(filepath)
    except Exception as e:
        handle_error_no_return(f"Could not remove file from filepath: {filepath}, encountered error: ", e)


def remove_folder_from_filepath(folderpath):
    print(f"\n\nRemoving folder from filepath: {folderpath}\n\n")
    try:
        shutil.rmtree(folderpath)
        print(f"Successfully deleted folder: {folderpath}")
    except Exception as e:
        handle_local_error(f"Could not remove folder from filepath: {folderpath}, encountered error: ", e)
    

def safe_remove_folder_from_filepath(folderpath):
    try:
        remove_folder_from_filepath(folderpath)
    except PermissionError:
        print(f"\nPermissionError: Ensure OneDrive or other file sync services aren't running! If not, manually delete the folder at the path that follows and try again. Error -- Could not remove folder from filepath: {folderpath}\n")
    except FileNotFoundError:   # Raised when the path doesn't exist at all
        print(f"\nFileNotFoundError: Could not remove folder from filepath: {folderpath}\n")
    except NotADirectoryError:   # Raised when the path exists but is not a directory (shutil.rmtree() only works on directories)
        print(f"\nNotADirectoryError: Could not remove folder from filepath: {folderpath}\n")
    except Exception as e:
        handle_error_no_return(f"Could not remove folder from filepath: {folderpath}, encountered error: ", e)

############################----------------------------------------------###############################



############################---------------Shutdown Methods----------------###############################

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


def safe_empty_cuda_cache(timeout=10):
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


def shutdown_vision_model():
    print("\n\nShutting down vision-model\n\n")
    global VISION_MODEL
    if VISION_MODEL:
        try:
            print("Attempting graceful offload of model")
            del VISION_MODEL
            print("Model graceful-offload successful")
        except Exception as e:
            handle_error_no_return("Could not gracefully offload model. Proceeding to directly force-offload. Encountered error: ", e)
        finally:
            VISION_MODEL = None
    print("\n\nModel offloading complete\n\n")
    return True


def shutdown_pipe():
    global PIPE
    print("\n\nShutting down pipeline\n\n")
    if PIPE:
        try:
            print("Attempting graceful offload of pipeline")
            del PIPE
            print("Pipeline graceful-offload successful")
        except Exception as e:
            handle_error_no_return("Could not gracefully offload pipeline. Proceeding to directly force-offload pipeline. Encountered error: ", e)
        finally:
            PIPE = None
    print("\n\nPipeline offloading complete\n\n")
    return True


def shutdown_exl2():
    print("\n\nShutting down ExLlamaV2 model\n\n")
    global EXL2_CACHE, EXL2_TOKENIZER, EXL2_MODEL, AUTO_TOKENIZER

    if EXL2_MODEL:
        try:
            print("Attempting to free ExLlamaV2 model")
            del EXL2_MODEL
            print("ExLlamaV2 model freed successfully")
        except Exception as e:
            handle_error_no_return("Could not free ExLlamaV2 model, encountered error: ", e)
        finally:
            EXL2_MODEL = None
    
    if EXL2_CACHE:
        try:
            print("Attempting to free ExLlamaV2 cache")
            del EXL2_CACHE
            print("ExLlamaV2 cache freed successfully")
        except Exception as e:
            handle_error_no_return("Could not free ExLlamaV2 cache, encountered error: ", e)
        finally:
            EXL2_CACHE = None
    
    if EXL2_TOKENIZER:
        try:
            print("Attempting to free ExLlamaV2 tokenizer")
            del EXL2_TOKENIZER
            print("ExLlamaV2 tokenizer freed successfully")
        except Exception as e:
            handle_error_no_return("Could not free ExLlamaV2 tokenizer, encountered error: ", e)
        finally:
            EXL2_TOKENIZER = None

    if AUTO_TOKENIZER:
        try:
            print("Attempting to free AutoTokenizer")
            del AUTO_TOKENIZER
            print("AutoTokenizer freed successfully")
        except Exception as e:
            handle_error_no_return("Could not free AutoTokenizer, encountered error: ", e)
        finally:
            AUTO_TOKENIZER = None
        
    print("\n\nExLlamaV2 cleanup complete\n\n")

############################-----------------------------------------------###############################




def safe_int(value, default):
    if value is None:
        handle_error_no_return("Null value, cannot convert to integer type. Proceeding with default value.")
        return default
    try:
        return int(value)
    except(ValueError, TypeError) as e:
        handle_error_no_return(f"Could not convert {value} to an integer, proceeding with default value {default}. Encountered error: ", e)
        return default


def safe_float(value, default):
    if value is None:
        handle_error_no_return("Null value, cannot convert to float type. Proceeding with default value.")
        return default
    try:
        return float(value)
    except(ValueError, TypeError) as e:
        handle_error_no_return(f"Could not convert {value} to a float, proceeding with default value {default}. Encountered error: ", e)
        return default


def os_sanitize_path(path):
    normalized_path = os.path.normpath(path)    # Normalize the path to handle any platform-specific separators
    expanded_path = os.path.expanduser(normalized_path) # Expand any tilde (~) to the user's home directory
    return expanded_path


def download_model_from_hf_hub(model_to_download):
    print(f"\n\nAttempting to download model {model_to_download} from HF-Hub...\n\n")
    try:
        latest_snapshot_path = snapshot_download(repo_id=model_to_download)
        print(f"\n\nDownload Successful. Latest snapshot path for {model_to_download} is:\n\n{latest_snapshot_path}\n\n")
        return os_sanitize_path(latest_snapshot_path)
    except Exception as e:
        handle_local_error("Could not download model from HF-Hub, encountered error: ", e)


def get_repo_info_for_model(model_id):
    print(f"\n\nScanning cache for model {model_id}...\n\n")
    try:
        cache_info = scan_cache_dir()
        print("\nCache Scan Complete.\n")
    except Exception as e:
        handle_local_error(f"Could not get cache_info for model {model_id}, encountered error: ", e)
    
    repo_info = next((repo for repo in cache_info.repos if repo.repo_id == model_id), None)
    if repo_info is None: return None
    
    return repo_info


def get_latest_revision_for_model(model_id):
    print(f"\n\nAttempting to get latest local revision for model {model_id}\n\n")

    try:
        repo_info = get_repo_info_for_model(model_id)
    except Exception as e:
        handle_error_no_return(f"Could not get repo_info for model {model_id}, encountered error: ", e)
    
    if repo_info is None:
        print(f"No cache info found for {model_id}, attempting to download model from HF-Hub")
        try:
            download_model_from_hf_hub(model_id)
            repo_info = get_repo_info_for_model(model_id)
        except Exception as e:
            handle_local_error(f"Could not download model {model_id} from HF-Hub, encountered error: ", e)
    
    try:
        revisions = list(repo_info.revisions)
        latest_revision = max(revisions, key=lambda rev: rev.last_modified)
        print(f"Determined latest revision details for {model_id}:")
        print(f"Commit Hash: {latest_revision.commit_hash}")
        print(f"Snapshot Path: {latest_revision.snapshot_path}")
        print(f"Last modified: {latest_revision.last_modified}")
        print(f"Number of Files: {len(latest_revision.files)}")
        return latest_revision
    except Exception as e:
        handle_local_error(f"Could not determine latest revision details for {model_id}, encountered error: ", e)


def hf_login_for_gated_models():
    access_token = ""
    try:
        read_return = read_config(['access_token'])
        access_token = str(read_return['access_token'])
    except Exception as e:
        handle_local_error("403 - No access token found, please submit an access token via the /hf_login endpoint")

    try:
        login(token=access_token)   # imported from huggingface_hub
    except Exception as e:
        handle_local_error("Unable to login to the HuggingFace-Hub, please ensure the correct access token has been provided. Encountered error: ", e)


def parse_arguments():

    try:
        parser = argparse.ArgumentParser(description="Server for HuggingFace Transformers models")
    except Exception as e:
        handle_local_error("Could not create parser to parse_arguments(), proceeding with defaults. Encountered error: ", e)

    # Even if a parser object could not be created, a read_request will write & return defaults 
    try:
        '''
        Reading values from the config will ensure defaults are set
        However we are only read_return-ing those values which we wish to remember the previous settings for, and using them as the `default` parameter for the parser.add_argument() below.
        For flags such as 'exl2', 'exl2_no_flash_attn', etc. we are not read_return-ing the values, as we do not wish to remember the previous settings for these flags and instead require the
        user to explicitly set them at launch, defaulting to False/None in parser otherwise.
        '''
        read_return = read_config(
            [
                'access_gated',
                'access_token',
                'model_id',
                'exl2',
                'exl2_bpw',
                'exl2_cache_type',
                'exl2_max_seq_len',
                'exl2_force_regenerate_measurement',
                'exl2_no_flash_attn',
                'gguf',
                'awq',
                'flux_diffusers',
                'flux_low_vram_optimizations',
                'load_quantized_flux',
                'vision',
                'gguf_model_id',
                'gguf_filename',
                'quantize',
                'quant_level',
                'hqq_group_size',
                'push_to_hub',
                'torch_device_map',
                'torch_dtype',
                'trust_remote_code',
                'use_flash_attention_2',
                'pipeline_task',
                'max_new_tokens',
                'return_full_text',
                'temperature',
                'do_sample',
                'top_k',
                'top_p',
                'min_p',
                'n_keep',
                'port',
                'host',
                'load_safe_defaults'
            ]
        )
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

    if parser:

        parser.add_argument("--reset_to_defaults", action="store_true", default=False, help="Use default settings")
        parser.add_argument("--access_gated", action="store_true", default=read_return['access_gated'], help="Specify True if you will be accessing gated models you've been approved to access")
        parser.add_argument("--access_token", type=str, default=read_return['access_token'], help="Access Token obtained from HF-Settings -> Access Tokens")
        parser.add_argument("--model_id", type=str, default=read_return['model_id'], help="model_id for for LLM in HF-Transformers format obtained from the model card. Remembers previously set value. Default: Phi3-mini-4k-instruct")
        parser.add_argument("--gguf", action="store_true", default=False, help="Add this flag if you'll be loading a GGUF LLM. Defaults to False.")
        parser.add_argument("--awq", action="store_true", default=False, help="Add this flag when loading AWQ-quantized models directly off the HF-Hub.")
        parser.add_argument("--flux_diffusers", action="store_true", default=False, help="Add this flag when loading FLUX-diffusers models directly off the HF-Hub.")
        parser.add_argument("--flux_low_vram_optimizations", action="store_true", default=read_return['flux_low_vram_optimizations'], help="Save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power")
        parser.add_argument("--load_quantized_flux", action="store_true", default=read_return['load_quantized_flux'], help="Add this flag when loading quantized FLUX models directly off the HF-Hub.")
        parser.add_argument("--vision", action="store_true", default=False, help="Add this flag when loading vision models directly off the HF-Hub.")
        parser.add_argument("--gguf_model_id", type=str, default=None, help="GGUF model_id of the target repo. Defaults to None")
        parser.add_argument("--gguf_filename", type=str, default=None, help="GGUF filename from the target repo. Defaults to None")
        parser.add_argument("--quantize", type=str, default=read_return['quantize'], help="Quantization method to be utilized. Simply type 'n' to not use quantization. Remembers previously set value. Default: bitsandbytes")
        parser.add_argument("--quant_level", type=str, default=read_return['quant_level'], help="Specify quantization level. Valid values -  BitsAndBytes: int8 & int4; Quanto: int8, int4 and int2; HQQ: int8, int4, int3, int2, int1. Remembers previously set value. Default: int8")
        parser.add_argument("--hqq_group_size", type=int, default=read_return['hqq_group_size'], help="Specify group_size for HQQ quantization. No restrictions as long as weight.numel() is divisible by the group_size. Remembers previously set value. Default: 64")
        parser.add_argument("--push_to_hub", action="store_true", default=read_return['push_to_hub'], help="Push quantized LLM to your HF-hub. Remembers previously set value. Default: False")
        parser.add_argument("--torch_device_map", type=str, default=read_return['torch_device_map'], help="Specify inference device, example: cuda. Remembers previously set value. Default: auto")
        parser.add_argument("--torch_dtype", type=str, default=read_return['torch_dtype'], help="Specify model tensor type, example: bfloat16. Remembers previously set value. Default: auto")
        parser.add_argument("--trust_remote_code", action="store_true", default=read_return['trust_remote_code'], help="Allows the model to execute custom code that's part of the model's HF-repository. Remembers previously set value. Default: False")
        parser.add_argument("--use_flash_attention_2", action="store_true", default=False, help="Set to True to attempt using Flash Attention 2. Defaults to False. Failed attempt to use FA2 will proceed to load the model without FA2.")
        parser.add_argument("--pipeline_task", type=str, default=read_return['pipeline_task'], help="Defaults to text-generation. For more details, open a Python shell, `import transformers`, and Run `help(transfomers.pipeline)`.")
        parser.add_argument("--max_new_tokens", type=int, default=read_return['max_new_tokens'], help="Set a hard limit on the maximum number of tokens an LLM can generate when responding. Remembers previously set value. Default: 500")
        parser.add_argument("--return_full_text", action="store_true", default=read_return['return_full_text'], help="When set to True, the LLM response contains the entire messages list with the latest response appended at the end.")
        parser.add_argument("--temperature", type=float, default=read_return['temperature'], help="Set LLM temperature on a scale of 0.0 to 2.0. Remembers previously set value. Default: 0.1")
        parser.add_argument("--do_sample", action="store_true", default=read_return['do_sample'], help="Perform sampling when selecting response tokens. Remembers previously set value. Default: True. Must be set to True when temperature is above 0.0. For greedy decoding, leave this as False and set temp to 0.0")
        parser.add_argument("--top_k", type=int, default=read_return['top_k'], help="Limit the next token selection to the K most probable tokens. Remembers previously set value. Default: 40")
        parser.add_argument("--top_p", type=float, default=read_return['top_p'], help="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Remembers previously set value. Default: 0.95")
        parser.add_argument("--min_p", type=float, default=read_return['min_p'], help="The minimum probability for a token to be considered, relative to the probability of the most likely token. Remembers previously set value. Default: 0.05")
        parser.add_argument("--n_keep", type=int, default=read_return['n_keep'], help="Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded. Remembers previously set value. Default: 0. Use -1 to retain all tokens from the prompt.")
        parser.add_argument("--port", type=int, default=read_return['port'], help="Specify the port to be used by the server. Remembers previously set value. Default: 9069")
        parser.add_argument("--host", type=str, default=read_return['host'], help="Specify the host to be used by the server. Remembers previously set value. Default: 0.0.0.0")

        # ExLlamaV2:
        parser.add_argument("--exl2", action="store_true", default=False, help="Add this flag when loading models via ExLlamaV2. Defaults to False.")
        parser.add_argument("--exl2_bpw", type=float, default=read_return['exl2_bpw'], help="Specify the bpw to be used when quantizing ExLlamaV2 models. Remembers previously set value and falls-back to 3.0 as the default.")
        parser.add_argument("--exl2_force_regenerate_measurement", action="store_true", default=read_return['exl2_force_regenerate_measurement'], help="Add this flag required to re-generate the measurement file for ExLlamaV2 models. Defaults to False.")
        parser.add_argument("--exl2_cache_type", type=str, default=read_return['exl2_cache_type'], help="Specify the cache type to be used when loading ExLlamaV2 models. Remembers previously set value and falls-back to full ExLlamaV2Cache as the default.")
        parser.add_argument("--exl2_max_seq_len", type=int, default=read_return['exl2_max_seq_len'], help="Specify the max sequence length (context size) to be used when loading ExLlamaV2 models. Remembers previously set value and falls-back to 2048 as the default.")
        parser.add_argument("--exl2_no_flash_attn", action="store_true", default=False, help="Use this flag to disable Flash Attention 2 for ExLlamaV2 models. Defaults to False.")
        
        args = parser.parse_args()
        print(f"\n\nparser.parse_args():\n\n{args}\n\n")

        if args.reset_to_defaults or read_return['load_safe_defaults']:
            print("\n\nLoading Server with Safe Defaults\n\n")
            try:
                # Empty hf_config.json
                config_writer_semaphore.acquire()
                with open(CONFIG_PATH, 'w') as file:
                    json.dump({}, file, indent=4)
                config_writer_semaphore.release()
                
                # Set defaults by triggering read on an empty file
                read_config([
                    'access_gated',
                    'access_token',
                    'model_id',
                    'exl2',
                    'exl2_bpw',
                    'exl2_cache_type',
                    'exl2_max_seq_len',
                    'exl2_force_regenerate_measurement',
                    'exl2_no_flash_attn',
                    'gguf',
                    'awq',
                    'flux_diffusers',
                    'flux_low_vram_optimizations',
                    'load_quantized_flux',
                    'vision',
                    'gguf_model_id',
                    'gguf_filename',
                    'quantize',
                    'quant_level',
                    'hqq_group_size',
                    'push_to_hub',
                    'torch_device_map',
                    'torch_dtype',
                    'trust_remote_code',
                    'use_flash_attention_2',
                    'pipeline_task',
                    'max_new_tokens',
                    'return_full_text',
                    'temperature',
                    'do_sample',
                    'top_k',
                    'top_p',
                    'min_p',
                    'n_keep',
                    'port',
                    'host'
                ])

            except Exception as e:
                handle_local_error("Could not reset hf_config.json, encountered error: ", e)
        else:
            try:
                # Auto-detect Flux and Llama-3.2-Vision models
                if "flux" in args.model_id.lower():
                    print("Flux model auto-detected, setting flux_diffusers=True")
                    args.flux_diffusers = True
                    args.vision = False
                    args.gguf = False
                    args.awq = False
                    args.exl2 = False
                else:
                    args.flux_diffusers = False
                
                if "llama-3.2" in args.model_id.lower() and "vision" in args.model_id.lower():
                    print("Llama-3.2-Vision model auto-detected, setting vision=True")
                    args.vision = True
                    args.flux_diffusers = False
                    args.gguf = False
                    args.awq = False
                    args.exl2 = False
                else:
                    args.vision = False
                
                write_config({
                    'access_gated':args.access_gated,
                    'access_token':args.access_token,
                    'model_id':args.model_id,
                    'exl2':args.exl2,
                    'exl2_bpw':args.exl2_bpw,
                    'exl2_force_regenerate_measurement':args.exl2_force_regenerate_measurement,
                    'exl2_cache_type':args.exl2_cache_type,
                    'exl2_max_seq_len':args.exl2_max_seq_len,
                    'exl2_no_flash_attn':args.exl2_no_flash_attn,
                    'gguf':args.gguf,
                    'awq':args.awq,
                    'gguf_model_id':args.gguf_model_id,
                    'gguf_filename':args.gguf_filename,
                    'quantize':args.quantize,
                    'quant_level':args.quant_level,
                    'hqq_group_size':args.hqq_group_size,
                    'push_to_hub':args.push_to_hub, 
                    'torch_device_map':args.torch_device_map, 
                    'torch_dtype':args.torch_dtype, 
                    'trust_remote_code':args.trust_remote_code, 
                    'use_flash_attention_2':args.use_flash_attention_2, 
                    'flux_diffusers':args.flux_diffusers,
                    'flux_low_vram_optimizations':args.flux_low_vram_optimizations,
                    'load_quantized_flux':args.load_quantized_flux,
                    'vision':args.vision,
                    'pipeline_task':args.pipeline_task, 
                    'max_new_tokens':args.max_new_tokens, 
                    'return_full_text':args.return_full_text, 
                    'temperature':args.temperature,
                    'do_sample':args.do_sample, 
                    'top_k':args.top_k, 
                    'top_p':args.top_p, 
                    'min_p':args.min_p, 
                    'n_keep':args.n_keep,
                    'port':args.port,
                    'host':args.host
                })
            except Exception as e:
                handle_local_error("Could not write launch arguments to hf_config.json, encountered error: ", e)

            if args.access_gated:
                try:
                    hf_login_for_gated_models()
                except Exception as e:
                    handle_error_no_return("Login to HF-Hub unsuccessful, encountered error: ", e)

        return args

    # Return None if parser was not created
    return None


def str_to_torch_dtype(dtype_str):

    print(f"\n\nstr_to_torch_dtype({dtype_str})\n\n")

    dtype_map = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.int8": torch.int8,
        "torch.int16": torch.int16,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
        "torch.bfloat16": torch.bfloat16,
        "auto":"auto"
    }
    return dtype_map.get(dtype_str, None)


def get_model_params():

    print("\n\ninitializing model parameters\n\n")

    global PIPE

    try:
        read_return = read_config([
            'gguf',
            'awq',
            'gguf_model_id',
            'gguf_filename',
            'quantize',
            'quant_level',
            'hqq_group_size',
            'torch_device_map',
            'torch_dtype',
            'trust_remote_code',
            'use_flash_attention_2',
            'pipeline_task',
            'vision'
        ])
        gguf = str(read_return['gguf']).lower() == 'true'
        awq = str(read_return['awq']).lower() == 'true'
        gguf_model_id = str(read_return['gguf_model_id'])
        gguf_filename = str(read_return['gguf_filename'])
        quantize = str(read_return['quantize'])
        quant_level = str(read_return['quant_level'])
        hqq_group_size = int(read_return['hqq_group_size'])
        torch_device_map = str(read_return['torch_device_map'])
        torch_dtype = str(read_return['torch_dtype'])
        trust_remote_code = str(read_return['trust_remote_code']).lower() == 'true'
        use_flash_attention_2 = str(read_return['use_flash_attention_2']).lower() == 'true'
        pipeline_task = str(read_return['pipeline_task'])
        vision = str(read_return['vision']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to get model_params, encountered error: ", e)

    if gguf:
        print(gguf)
        print("\n\nLoading GGUF\n\n")
        try:
            model = AutoModelForCausalLM.from_pretrained(gguf_model_id, gguf_file=gguf_filename)
        except Exception as e:
            handle_local_error("Could not create AutoModelForCausalLM, encountered error: ", e)
        try:
            tokenizer = AutoTokenizer.from_pretrained(gguf_model_id, gguf_file=gguf_filename)
        except Exception as e:
            handle_local_error("Could not set AutoTokenizer, encountered error: ", e)
        try:
            PIPE = pipeline(
                pipeline_task,
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as e:
            handle_local_error("Could not create model PIPELINE, encountered error: ", e)

        return True

    if awq:
        print("Proceed to load AWQ-quantized model from the HF-Hub, setting torch_dtype=torch.float16 and quantize=n and proceeding.")
        torch_dtype_obj = torch.float16
        quantize = "n"
    else:
        try:
            torch_dtype_obj = str_to_torch_dtype(torch_dtype)
        except Exception as e:
            handle_error_no_return("Error determining torch data-type, setting to auto and proceeding: ", e)
            torch_dtype_obj = "auto"
        if torch_dtype_obj is None:
            handle_error_no_return("Could not obtain torch dtype object, check if the value passed is correct. Setting to auto and proceeding.")
            torch_dtype_obj = "auto"

    if vision:
        print("Vision model detected, setting torch_dtype=torch.bfloat16")
        torch_dtype_obj = torch.bfloat16

    model_params = {
        "device_map": torch_device_map,
        "torch_dtype": torch_dtype_obj,
        "trust_remote_code": trust_remote_code,
    }

    if use_flash_attention_2 and not vision:
        model_params["attn_implementation"] = "flash_attention_2"

    quantize = quantize.lower().strip()
    if quantize != "n":
        try:
            if quantize == "bitsandbytes":
                print("Quantizing with BitsAndBytes")
                quant_level = quant_level.lower().strip()

                if quant_level == "int8":
                    print("Proceeding with BitsAndBytes-Int8 Quant")
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int4":
                    print("Proceeding with BitsAndBytes-Int4 Quant")
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    model_params["quantization_config"] = quantization_config
                else:
                    print(f"Invalid quant_level setting, BitsAndBytes supports only int8 and int4 quants but you set {quant_level}; proceeding with BitsAndBytes-Int4 Quant")
                    print("Proceeding with BitsAndBytes-Int4 Quant")
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    model_params["quantization_config"] = quantization_config
            elif quantize == "quanto":
                print("Quanto-Quantizing")
                quant_level = quant_level.lower().strip()

                if quant_level == "int8":
                    print("Proceeding with Quanto-Int8 Weights")
                    quantization_config  = QuantoConfig(weights="int8")
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "float8":
                    print("Proceeding with Quanto-Float8 Weights")
                    quantization_config  = QuantoConfig(weights="float8")
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int4":
                    print("Proceeding with Quanto-Int4 Weights")
                    quantization_config  = QuantoConfig(weights="int4")
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int2":
                    print("Proceeding with Quanto-Int2 Weights")
                    quantization_config  = QuantoConfig(weights="int2")
                    model_params["quantization_config"] = quantization_config
                else:
                    print(f"Invalid quant_level setting, Quanto supports only int8, int4 and int2 quants but you set {quant_level}; proceeding with Quanto-Int4 Quant")
                    quantization_config  = QuantoConfig(weights="int4")
                    model_params["quantization_config"] = quantization_config
            elif quantize == "hqq":
                print("HQQ-Quantizing - Force-setting torch_dtype to torch.bfloat16")
                model_params["torch_dtype"] = torch.bfloat16
                quant_level = quant_level.lower().strip()

                if quant_level == "int8":
                    print("Proceeding with HQQ-Int8 Weights")
                    quantization_config  = HqqConfig(nbits=8, group_size=hqq_group_size)
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int4":
                    print("Proceeding with HQQ-Int4 Weights")
                    quantization_config  = HqqConfig(nbits=4, group_size=hqq_group_size)
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int3":
                    print("Proceeding with HQQ-Int3 Weights")
                    quantization_config  = HqqConfig(nbits=3, group_size=hqq_group_size)
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int2":
                    print("Proceeding with HQQ-Int2 Weights")
                    quantization_config  = HqqConfig(nbits=2, group_size=hqq_group_size)
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int1":
                    print("Proceeding with HQQ-Int1 Weights")
                    quantization_config  = HqqConfig(nbits=1, group_size=hqq_group_size)
                    model_params["quantization_config"] = quantization_config
                else:
                    print(f"Invalid quant_level setting, HQQ supports int8, int4, int3, int2 & int1 quants but you set {quant_level}; proceeding with HQQ-Int4 Quant")
                    quantization_config  = HqqConfig(nbits=4, group_size=hqq_group_size)
                    model_params["quantization_config"] = quantization_config
        except Exception as e:
            handle_local_error("Could not create quantization_config when attempting to get model_params, encountered error: ", e)

    return model_params


def load_flux_pipeline(pipeline):

    print("\n\nLoading Flux Pipeline\n\n")
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python' # Sets Protocol Buffers to use the pure Python implementation instead of the default C++ implementation. This is significantly slower but must be done for FLUX to work. That's why this environment variable is deleted whenever other models are loaded.

    try:
        read_return = read_config(['model_id', 'flux_low_vram_optimizations', 'load_quantized_flux'])
        model_id = str(read_return['model_id'])
        flux_low_vram_optimizations = str(read_return['flux_low_vram_optimizations']).lower() == 'true'
        load_quantized_flux = str(read_return['load_quantized_flux']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to load_flux_pipeline(), encountered error: ", e)

    if load_quantized_flux:
        print("Loading quantized Flux Pipeline")
        bfl_repo = model_id
        dtype = torch.bfloat16

        quantized_checkpoint = ""
        if "schnell" in model_id.lower():
            quantized_checkpoint = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8-e4m3fn.safetensors"
        elif "dev" in model_id.lower():
            quantized_checkpoint = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors"
        
        print(f"\n\nLoading FLUX FP8 Quantized Checkpoint from: {quantized_checkpoint}\n\n")

        try:
            print(f"\n\nSetting and quantizing Transformer for Quantized Flux Pipeline from {bfl_repo}\n\n")
            transformer = FluxTransformer2DModel.from_single_file(quantized_checkpoint, torch_dtype=dtype)
            quantize(transformer, weights=qfloat8)
            freeze(transformer)

            print(f"\n\nSetting and quantizing Text Encoder 2 for Quantized Flux Pipeline from {bfl_repo}\n\n")
            text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
            quantize(text_encoder_2, weights=qfloat8)
            freeze(text_encoder_2)

            print(f"\n\nQuatization complete! Loading Quantized Flux Pipeline...\n\n")
            pipeline = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype)
            pipeline.transformer = transformer
            pipeline.text_encoder_2 = text_encoder_2

            print(f"\n\nEnabling model CPU offload for Quantized Flux Pipeline\n\n")
            pipeline.enable_model_cpu_offload()
        except Exception as e:
            handle_model_loading_error("Could not load quantized Flux Pipeline, encountered error: ", e)
            return False
    else:    
        try:
            pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            if flux_low_vram_optimizations:
                pipeline.enable_sequential_cpu_offload()
                pipeline.vae.enable_slicing()
                pipeline.vae.enable_tiling()
                pipeline.to(torch.float16)  # Casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once
        except Exception as e:
            handle_model_loading_error("Could not load Flux Pipeline, encountered error: ", e)
            return False
    
    print(f"\n{model_id} loaded successfully!\n")
    return pipeline


def load_vision_pipeline(pipeline, model_params):

    global VISION_MODEL

    try:
        read_return = read_config(['model_id', 'torch_device_map'])
        model_id = str(read_return['model_id'])
        torch_device_map = str(read_return['torch_device_map'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to load vision-pipeline, encountered error: ", e)

    model_params.pop('trust_remote_code', None)

    try:
        print(f"\nInitializing vision model: {model_id} with device_map: {torch_device_map}\n")
        VISION_MODEL = MllamaForConditionalGeneration.from_pretrained(model_id, **model_params)
       
        try:
            print(f"Your vision-model's memory footprint is: {VISION_MODEL.get_memory_footprint()}")
        except Exception as e:
            handle_error_no_return("Could not determine the model's memory footprint, encountered error: ", e)

        print(f"\nInitializing processor for vision model: {model_id}\n")
        pipeline = AutoProcessor.from_pretrained(model_id)  # Using 'pipeline' instead of 'processor' to maintain consistency with the server code. AutoProcessor is used to process images and text inputs for the vision model.
        
        print(f"\nVision Model & Processor Loaded Successfully!\n")
        return pipeline
    except Exception as e:
        handle_model_loading_error("Could not load Vision Pipeline, encountered error: ", e)
        return False


def generate_exllama_measurement_file_for_model(model_id: str, model_snapshot_path: os.PathLike) -> os.PathLike:
    print(f"\n\nAttempting to generate measurement file for model {model_id}...\n\n")

    try:
        read_return = read_config(['transformer_models_folder', 'exl2_force_regenerate_measurement'])
        transformer_models_folder = str(read_return['transformer_models_folder'])
        exl2_force_regenerate_measurement = str(read_return['exl2_force_regenerate_measurement']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to generate-exllama_measurement_file_for_model(), encountered error: ", e)

    try:
        temp_dir = os.path.join(os.getcwd(), "exllamav2", "temp-converter-files")
        os.makedirs(temp_dir, exist_ok=True)

        measurement_file_path = os.path.join(transformer_models_folder, model_id, "exllama-measurements-file", "measurement.json")
        os.makedirs(os.path.dirname(measurement_file_path), exist_ok=True)
    except Exception as e:
        handle_local_error("Could not create measurement file directory when attempting to generate-exllama_measurement_file_for_model(), encountered error: ", e)
    
    if os.path.exists(measurement_file_path) and not exl2_force_regenerate_measurement:
        print(f"\nMeasurement file for {model_id} already exists. Skipping measurement file generation.\n")
        return measurement_file_path
    
    convert_script_path = os.path.normpath(os.path.join(os.getcwd(), "exllamav2", "convert.py"))
    command = [
        'python' if platform.system() == 'Windows' else 'python3',
        convert_script_path,
        '-i', model_snapshot_path,
        '-o', temp_dir,
        '-nr',
        '-om', measurement_file_path
    ]

    try:
        print(f"\nRunning ExLlamaV2 measurement file generator for {model_id}...\n")
        subprocess.run(command, check=True) # check=True ensures that the command will raise an exception if it fails
        print(f"\nExLlamaV2 measurement file generator for {model_id} completed successfully!\n")
    except Exception as e:
        safe_remove_folder_from_filepath(temp_dir)  # Since measurement-generation errored out, restarting afresh by clearing the temp dir is safer
        handle_local_error("Could not run ExLlamaV2 measurement file generator, encountered error: ", e)

    return measurement_file_path


def exllama_bpw_quantize_model(model_id: str, measurement_file_path: os.PathLike, model_snapshot_path: os.PathLike, exl2_bpw: float) -> os.PathLike:
    print(f"\n\nAttempting to quantize model {model_id} to {exl2_bpw}bpw...\n\n")
    
    try:
        read_return = read_config(['transformer_models_folder'])
        transformer_models_folder = str(read_return['transformer_models_folder'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to exllama-bpw_quantize_model(), encountered error: ", e)

    try:
        temp_dir = os.path.join(os.getcwd(), "exllamav2", "temp-converter-files")
        os.makedirs(temp_dir, exist_ok=True)

        quantized_model_path = os.path.join(transformer_models_folder, model_id, "exl2-qaunts", f"{exl2_bpw}bpw")
        os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)   # Create parent directory structure - final `{exl2_bpw}bpw` directory will be created by ExLlamaV2 converter
    except Exception as e:
        handle_local_error("Could not create directory to store quantized model when attempting to exllama-bpw_quantize_model(), encountered error: ", e)

    if os.path.exists(quantized_model_path):
        print(f"\nQuantized model for {model_id} already exists. Skipping quantization.\n")
        return quantized_model_path
    
    convert_script_path = os.path.normpath(os.path.join(os.getcwd(), "exllamav2", "convert.py"))
    command = [
        'python' if platform.system() == 'Windows' else 'python3',
        convert_script_path,
        '-i', model_snapshot_path,
        '-o', temp_dir,
        '-nr',
        '-m', measurement_file_path,
        '-cf', quantized_model_path,
        '-b', str(exl2_bpw)
    ]

    try:
        print(f"\nRunning ExLlamaV2 bpw quantizer for {model_id}...\n")
        subprocess.run(command, check=True) # check=True ensures that the command will raise an exception if it fails
        print(f"\nExLlamaV2 Conversion of {model_id} to {exl2_bpw}bpw completed successfully!\n")
        safe_remove_folder_from_filepath(temp_dir)  # conversion completed, deleting temp dir to free space
    except Exception as e:
        safe_remove_folder_from_filepath(temp_dir)  # Since conversion errored out, restarting afresh by clearing the temp dir is safer
        handle_local_error("Could not run ExLlamaV2 bpw quantizer, encountered error: ", e)

    return quantized_model_path


def get_exl2_cache_type(exl2_cache_type: str):  #  -> ExLlamaV2Cache; commenting out as it will cause the server to error out if ExLlamaV2 is not installed!
    print(f"\nDetermining ExLlamaV2 cache type for {exl2_cache_type}...\n")
    try:
        if exl2_cache_type == "c8":
            return ExLlamaV2Cache_8bit
        elif exl2_cache_type == "cq4":
            return ExLlamaV2Cache_Q4
        elif exl2_cache_type == "cq6":
            return ExLlamaV2Cache_Q6
        elif exl2_cache_type == "cq8":
                return ExLlamaV2Cache_Q8
        else:
            return ExLlamaV2Cache
    except Exception as e:
        handle_local_error("Could not get ExLlamaV2 cache type, encountered error: ", e)


def define_exllama_generator_components(quantized_model_path: os.PathLike, exl2_no_flash_attn: bool):    # -> ExLlamaV2DynamicGenerator: commenting out as it will cause the server to error out if ExLlamaV2 is not installed!
    print(f"\n\nAttempting to define ExLlamaV2 Generator Components for Model: {quantized_model_path}...\n\n")

    try:
        if exl2_no_flash_attn:
            config = ExLlamaV2Config()
            config.model_dir = quantized_model_path
            config.no_flash_attn = True # From ExLlamaV2 model_init.py, line 111
        else:
            config = ExLlamaV2Config(quantized_model_path)
        print("\nConfig defined successfully\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV2 config, encountered error: ", e)
    
    try:
        global EXL2_MODEL
        EXL2_MODEL = ExLlamaV2(config)
        print("\nModel defined successfully\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV2 model, encountered error: ", e)
    
    try:
        exl2_cache_type = str(read_config(['exl2_cache_type'])['exl2_cache_type'])
        exl2_max_seq_len = int(read_config(['exl2_max_seq_len'])['exl2_max_seq_len'])
    except Exception as e:
        handle_local_error("Could not read cache type from hf_config.json, encountered error: ", e)
    
    try:
        cache_type = get_exl2_cache_type(exl2_cache_type)
        print(f"\nCache type determined successfully: {cache_type}\n")
    except Exception as e:
        handle_local_error("Could not get ExLlamaV2 cache type, encountered error: ", e)

    try:
        global EXL2_CACHE
        EXL2_CACHE = cache_type(EXL2_MODEL, max_seq_len = exl2_max_seq_len, lazy = True)
        print(f"\nCache defined successfully with max_seq_len: {exl2_max_seq_len}\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV2 cache, encountered error: ", e)

    try:
        print(f"\nLoading model...\n")
        EXL2_MODEL.load_autosplit(EXL2_CACHE, progress=True)
        print("\nModel loaded with autosplit successfully\n")
    except Exception as e:
        handle_local_error("Could not load ExLlamaV2 model with autosplit, encountered error: ", e)

    try:
        global EXL2_TOKENIZER
        EXL2_TOKENIZER = ExLlamaV2Tokenizer(config)
        print("\nTokenizer defined successfully\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV2 tokenizer, encountered error: ", e)

    return True


def load_exllama_pipeline():
    print("\n\nLoading ExLlamaV2 Pipeline\n\n")
    
    try:
        read_return = read_config(['model_id', 'exl2_bpw', 'exl2_no_flash_attn'])
        model_id = str(read_return['model_id'])
        exl2_bpw = float(read_return['exl2_bpw'])
        exl2_no_flash_attn = str(read_return['exl2_no_flash_attn']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to load the ExLlamaV2 pipeline, encountered error: ", e)

    latest_snapshot_path = None
    try:
        latest_snapshot_path = download_model_from_hf_hub(model_id)
    except Exception as e:
        handle_error_no_return(f"Could not download {model_id} from HF-Hub. Attempting to scan for pre-existing local snapshots. Encountered error: ", e)
        try:
            latest_revision = get_latest_revision_for_model(model_id)
            latest_snapshot_path = os_sanitize_path(latest_revision.snapshot_path)
        except Exception as e:
            handle_local_error(f"Error attempting to work with local snapshot for {model_id}. Encountered error: ", e)
    
    if latest_snapshot_path is None:
        handle_local_error(f"Could not find a local snapshot for {model_id}. Please check your connection and access token if you're using a private model.")
    
    try:
        measurement_file_path = generate_exllama_measurement_file_for_model(model_id, latest_snapshot_path)
    except Exception as e:
        handle_local_error(f"Error generating ExLlamaV2 measurement file for {model_id}. Encountered error: ", e)

    try:
        quantized_model_path = exllama_bpw_quantize_model(model_id, measurement_file_path, latest_snapshot_path, exl2_bpw)
    except Exception as e:
        handle_local_error(f"Error ExLlamaV2 quantizing {model_id} to {exl2_bpw} bits per word. Encountered error: ", e)

    try:
        define_exllama_generator_components(quantized_model_path, exl2_no_flash_attn)
    except Exception as e:
        handle_local_error(f"Error loading ExLlamaV2 quantized model from {quantized_model_path}. Encountered error: ", e)

    try:
        global AUTO_TOKENIZER
        AUTO_TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)    # Using Transformers' AutoTokenizer as ExLlamaV2's ExLlamaV2Tokenizer does not contain an equivalent apply_chat_template() method!
        print("\nTransformers-AutoTokenizer configured successfully for automated prompt-formatting\n")
    except Exception as e:
        handle_local_error(f"Error loading AutoTokenizer for {model_id}. Encountered error: ", e)

    try:
        print(f"Model's context-length (max_seq_len) is: {EXL2_MODEL.config.max_seq_len}")
    except Exception as e:
        handle_error_no_return("Could not determine the model's context-length (max_seq_len), encountered error: ", e)
    
    try:
        print(f"Model's context-length (max_input_len per forward-pass) is: {EXL2_MODEL.config.max_input_len}")
    except Exception as e:
        handle_error_no_return("Could not determine the model's context-length (max_input_len), encountered error: ", e)

    print("\n\nExLlamaV2 Pipeline Loaded Successfully!\n\n")
    return True



class CustomStream(io.StringIO):
    def __init__(self, callback=None):  # this callback stream to handle written data is not thread-safe!
        super().__init__()
        self.callback = callback

    def write(self, data):
        # If we have a callback, call it
        if self.callback:
            self.callback(data)

        return super().write(data)  # writing directly to the in-memory string buffer is not thread-safe!

class ThreadSafeStream(io.StringIO):
    def __init__(self, queue):  # Python's queue's are thread-safe as they are internally synchronized using locks (thread synchronized primitives) to ensure operations like get and put are atomic & thread-safe
        super().__init__()
        self.queue = queue

    def write(self, data):
        self.queue.put(data)    # bypassing the in-memory string buffer to avoid race-conditions and writing to the thread-safe queue directly instead!
        # nothing returned as return-value is not directly used by the caller

@app.route('/restart_server_stream')
def restart_server_stream():

    llm_semaphore.acquire()
    print("\n\nrestart-server-stream acquired llm_semaphore, proceeding...\n\n")
    config_writer_semaphore.acquire()
    print("\n\nrestart-server-stream acquired config_writer_semaphore, proceeding...\n\n")
    error_logging_semaphore.acquire()
    print("\n\nrestart-server-stream acquired error_logging_semaphore, proceeding...\n\n")

    print("\n\nrestarting server with stream\n\n")

    shutdown_vision_model()
    shutdown_pipe()
    shutdown_exl2()
    safe_empty_cuda_cache()

    try:
        read_return = read_config(['model_id', 'pipeline_task', 'flux_diffusers', 'vision', 'exl2'])
        model_id = str(read_return['model_id'])
        pipeline_task = str(read_return['pipeline_task'])
        flux_diffusers = str(read_return['flux_diffusers']).lower() == 'true'
        vision = str(read_return['vision']).lower() == 'true'
        exl2 = str(read_return['exl2']).lower() == 'true'
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read values from hf_config.json when attempting restart-server-stream, encountered error: ", e)

    model_params = {}

    if not flux_diffusers:
        model_params = get_model_params()   # We need to do so before redirecting output-streams!
        print(f"Setting model-parameters: {model_params}")
    
    stop_thread = threading.Event()

    output_queue = queue.Queue()
    
    custom_stdout = ThreadSafeStream(output_queue)
    custom_stderr = ThreadSafeStream(output_queue)

    original_stdout = sys.stdout    # seperate stream objects!
    original_stderr = sys.stderr

    def model_initialization_task():

        global PIPE

        try:
            sys.stdout = custom_stdout
            sys.stderr = custom_stderr
            logging.basicConfig(stream=custom_stdout, level=logging.INFO)   # logging level is set to INFO to ensure all logs are captured by the stream
            
            if flux_diffusers:
                print("\nFlux Diffusers Selected - Loading...\n")
                PIPE = load_flux_pipeline(PIPE)
            else:
                if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' in os.environ:
                    del os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']    # Better to delete as the default behavior is to try the C++ implementation first and fall back to Python if needed, which is more robust than simply setting it to 'cpp'.
                
                if exl2:
                    print("\n\nExLlamaV2 Selected - Loading...\n\n")
                    load_exllama_pipeline()
                elif vision:
                    print("\nVision Model Selected - Loading...\n")
                    PIPE = load_vision_pipeline(PIPE, model_params)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_id, **model_params)
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    print("\nInitializing inference pipeline...")
                    PIPE = pipeline(
                        pipeline_task,
                        model=model,
                        tokenizer=tokenizer,
                    )

                    try:
                        print(f"Your model's memory footprint is: {model.get_memory_footprint()}")
                    except Exception as e:
                        handle_error_no_return("Could not determine the model's memory footprint, encountered error: ", e)
            
            print(f"\n{model_id} loaded successfully!\n")
        except Exception as e:
            return handle_model_loading_error("Model loading failed, encountered error: ", e, "api")
        finally:
            output_queue.put(None)
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logging.basicConfig(stream=sys.stderr, level=logging.WARNING)  # reset logging to the default WARNING level so only warnings and more severe messages (ERROR, CRITICAL, etc.) are logged, as is more appropriate for production for a production environment
            llm_semaphore.release()
            config_writer_semaphore.release()
            error_logging_semaphore.release()
            stop_thread.set()

    def output_reader():
        while True:
            line = output_queue.get()
            if line is None:
                print("\nNone read, breaking and stopping thread\n")
                break
            yield f"data: {json.dumps(line)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"
        print("\nrestart-server-stream done\n")

    thread = threading.Thread(target=model_initialization_task)
    thread.start()

    print(f"\nModel Initialization Begins - Loading {model_id}\n")
    return Response(output_reader(), content_type='text/event-stream')


def initialize_model():

    global PIPE

    try:
        read_return = read_config(['model_id', 'push_to_hub', 'quant_level', 'pipeline_task', 'flux_diffusers', 'vision', 'exl2'])
        model_id = str(read_return['model_id'])
        push_to_hub = str(read_return['push_to_hub']).lower() == 'true'
        quant_level = str(read_return['quant_level'])
        pipeline_task = str(read_return['pipeline_task'])
        flux_diffusers = str(read_return['flux_diffusers']).lower() == 'true'
        vision = str(read_return['vision']).lower() == 'true'
        exl2 = str(read_return['exl2']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to initialize_model(), encountered error: ", e)
    
    print(f"\n\nInitializing HF-Waitress LLM Server for {model_id}\n\n")

    if flux_diffusers:
        print("\n\nFlux Diffusers Selected - Loading...\n\n")
        PIPE = load_flux_pipeline(PIPE)

    else:
        # Remove explicit protobuf implementation setting to use system default
        if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' in os.environ:
            del os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']    # Better to delete as the default behavior is to try the C++ implementation first and fall back to Python if needed, which is more robust than simply setting it to 'cpp'.

        if exl2:
            print("\n\nExLlamaV2 Selected - Loading...\n\n")
            load_exllama_pipeline()
        
        else:
            model_params = get_model_params()
            print(f"Setting model-parameters: {model_params}")
            
            if vision:
                print("\n\nVision Model Selected - Loading...\n\n")
                PIPE = load_vision_pipeline(PIPE, model_params)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_params)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                print("\nInitializing inference pipeline...")
                PIPE = pipeline(
                    pipeline_task,
                    model=model,
                    tokenizer=tokenizer,
                )
        
            try:
                print(f"Your model's memory footprint is: {model.get_memory_footprint()}")
            except Exception as e:
                handle_error_no_return("Could not determine the model's memory footprint, encountered error: ", e)
    
    print(f"\n{model_id} loaded successfully!\n")

    if push_to_hub:
        try:
            model.push_to_hub(f"{model_id}-{quant_level}")
        except Exception as e:
            handle_error_no_return("Could not push the model to your hub, encountered error: ", e)

    return True


def generate_flux_image(request):
    print("\n\nFlux Diffusers Selected - Generating Image\n\n")

    try:
        data = request.json
        messages = data.get('messages', [])
    except Exception as e:
        handle_local_error("Could not read POST-request messages for /completions, encountered error: ", e)
        return False

    generated_images_folder = "generated_images"
    try:
        generated_images_folder = read_config(['generated_images_folder'])['generated_images_folder']
    except Exception as e:
        handle_error_no_return("Could not read generated_images_folder from hf_config.json, using default: generated_images in the current working directory. Encountered error: ", e)

    try:
        flux_generation_args = {
            "guidance_scale": float(request.headers.get('X-Guidance-Scale', 0)),
            "height": int(request.headers.get('X-Height', 768)),
            "width": int(request.headers.get('X-Width', 1360)),
            "num_inference_steps": int(request.headers.get('X-Num-Inference-Steps', 5)),
            "max_sequence_length": int(request.headers.get('X-Max-Sequence-Length', 256)),
            "num_images_per_prompt": int(request.headers.get('X-Num-Images-Per-Prompt', 1))
        }
        image = PIPE(
            messages[0]["prompt"],
            **flux_generation_args
        ).images[0]

        stream_session_id = str(uuid.uuid4())
        image_name = "output_" + stream_session_id + ".png"
        image_path = generated_images_folder + "/" + image_name

        try:
            os.makedirs(generated_images_folder, exist_ok=True)
            image.save(image_path)
        except Exception as e:
            handle_error_no_return("Could not store image in generated_images folder, saving in current working directory instead. Encountered error: ", e)
            try:    
                image.save(image_name)
            except Exception as e:
                handle_error_no_return("Could not store image in current working directory. Proceeding to simply return the image as a base64 encoded string. Encountered error: ", e)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str, image_name
    except Exception as e:
        handle_local_error("Could not generate image with FLUX Diffusers. Encountered error: ", e)
        return False


def convert_pdf_to_images_list(filepath, dpi=300):
    print(f"\n\nConverting PDF to images list\n\n")
    try:
        return convert_from_path(filepath, dpi)
    except Exception as e:
        handle_local_error("Could not convert PDF to images list, encountered error: ", e)
        return False


def convert_to_pdf_with_unoconv(input_file_path, output_file_path):
    print("\n\nConverting non-PDF document to PDF format\n\n")
    if platform.system() == 'Windows':
        subprocess.run(['python', 'unoconv.py', '-f', 'pdf', '-o', output_file_path, input_file_path], check=True)
    else:
        subprocess.run(['unoconv', '-f', 'pdf', '-o', output_file_path, input_file_path], check=True)


def convert_non_pdf_to_pdf_with_unoconv(filename, filepath):
    print("Converting to PDF file")

    try:
        conv_filename = os.path.splitext(filename)[0] + '.pdf'
        conv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], conv_filename)

        convert_to_pdf_with_unoconv(filepath, conv_filepath)

        return conv_filename, conv_filepath
    except subprocess.CalledProcessError as e:
        return handle_api_error("Could not convert file to PDF, encountered error: ", e)
    except Exception as e:
        return handle_api_error("Unexpected error when converting file to PDF, encountered error: ", e)


def get_pil_image_objects_for_file(filename, filepath, dpi=300):
    print(f"\n\nGetting PIL image objects for file: {filename}\n\n")

    if not filename.lower().endswith('.pdf'):
        _, filepath = convert_non_pdf_to_pdf_with_unoconv(filename, filepath)

    try:
        pil_image_object_list = convert_pdf_to_images_list(filepath, dpi)
        return pil_image_object_list
    except Exception as e:
        return handle_api_error("Could not get PIL image objects for file, encountered error: ", e)


def get_input_params_for_vision_model(request):
    print("\n\nGetting input params for vision model\n\n")

    vision_file_present = False

    input_file = None
    try:
        if 'file' in request.files:
            print("\n\nVision file present in request\n\n")
            input_file = request.files['file']
            vision_file_present = True
        messages = json.loads(request.form.get('messages', '[]'))
    except Exception as e:
        handle_local_error("Could not read POST-request messages when attempting to get input_params for vision-model, encountered error: ", e)
        return False

    try:
        read_return = read_config(['max_new_tokens'])
        max_new_tokens = int(read_return['max_new_tokens'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to get input_params for vision-model, encountered error: ", e)

    try:
        dpi = int(request.headers.get('X-DPI', 300))
        try:
            generation_config = {
                "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
                "use_cache": True
            }
        except Exception as e:
            handle_error_no_return("Could not set generation-arguments when attempting to get input_params for vision-model, proceeding without them. Encountered error: ", e)
    except Exception as e:
        handle_error_no_return("Could not read DPI from request headers, proceeding with default: 300. Encountered error: ", e)

    filename = ""
    pil_image_object_list = []
    if input_file:
        filename = secure_filename(input_file.filename) # Ensure the filename is secure
        if "PDF" in filename:
            filename = filename.replace("PDF", "pdf")

        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            print("Loading new file - filename: ", filename)
            print("Loading new file - filepath: ", filepath)

            # Save the uploaded file to the specified path
            input_file.save(filepath)
        except Exception as e:
            return handle_api_error("Failed to save document to app folder, encountered error: ", e)
        
        try:
            pil_image_object_list = get_pil_image_objects_for_file(filename, filepath, dpi)
        except Exception as e:
            return handle_api_error("Could not get PIL image objects for file, encountered error: ", e)

    try:
        print(f"\n\nApplying Chat Template for messages: {messages}\n\n")
        input_text = PIPE.apply_chat_template(messages, add_generation_prompt=True)
    except Exception as e:
        handle_local_error("Could not apply chat template, encountered error: ", e)
        return False

    return input_text, pil_image_object_list, generation_config, filename, vision_file_present


def get_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)    # Try to load Arial from the system
    except IOError:
        pass

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)    # Try to load a common sans-serif font available on most systems
    except IOError:
        pass
    
    print("\n\nCould not load a truetype font. Using default font\n\n")
    return ImageFont.load_default()     # If all else fails, use the default font


def get_blank_pil_image_object():
    print("\n\nNo vision file present, generating sys-prompt image for text-only input\n\n")
    img = Image.new('RGB', (800, 600), color='white')
    d = ImageDraw.Draw(img)
    font = get_font(16)
    d.text((10,10), "Answer the user's question as accurately as possible.", fill=(0,0,0), font=font)   # (x, y) co-ordinate: start 10 pixels from the left and top of the image

    img_path = os.path.join(os.getcwd(), "text_only_input_image.png")   # Save the image to the current directory
    img.save(img_path)
    print(f"\n\nImage saved to: {img_path}\n\n")

    return img


def inference_with_vision_model(request):
    print("\n\nvision-completions route triggered") # No need to acquire LLM semaphore here, as the invoking method already has it!
    
    try:
        input_text, pil_image_object_list, generation_config, filename, vision_file_present = get_input_params_for_vision_model(request)
    except Exception as e:
        handle_local_error("Could not get input params in vision-completions, encountered error: ", e)

    if not vision_file_present:
        pil_image_object_list.append(get_blank_pil_image_object())

    inference_output = ""
    for page_number, image in enumerate(pil_image_object_list, start=1): # start=1 to match the page numbers in the PDF

        if vision_file_present: 
            print(f"\n\nProcessing Page: {page_number} from file: {filename}\n\n")
        
        try:
            print("\n\nLoading Input to Model\n\n")
            inputs = PIPE(image, input_text, return_tensors="pt").to(VISION_MODEL.device)
        except Exception as e:
            handle_local_error("Could not load input to model, encountered error: ", e)
            return False

        try:
            print("\n\nGenerating Output\n\n")
            output = VISION_MODEL.generate(**inputs, **generation_config)    # `output` is a tensor and needs to be decoded!

            # Get length of the input sequence as follows:
            # 1. The `inputs` object returned by processor is typically a dictionary-like object containing various tensors needed for the model's forward pass (output generation).
            # 2. The `input_ids` tensor is usually the most important one among them, representing the tokenized input to the model.
            # 3. `inputs.input_ids.shape` is a tensor with shape: (batch_size, sequence_length) 
            # 4. The `batch_size`, i.e. `inputs.input_ids.shape[0]` in our case is 1, indicating we're processing one image/prompt at a time
            # 5. The `sequence_length`, i.e. `inputs.input_ids.shape[1]`, gives us the the length of the input sequence, i.e. the number of tokens in the input.
            # 6. The model's `generate` method returns a tensor that includes both, the input tokens and the newly generated tokens.
            # 7. By slicing from `input_length` onwards, we're effectively saying "give me all the tokens after the input sequence", which are the newly generated tokens.
            input_length = inputs.input_ids.shape[1] 
            
            # Slice the tensor and decode only the output!
            decoded_output = PIPE.decode(output[0][input_length:], skip_special_tokens=True)    # Setting skip_special_tokens=True to remove: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.

            print(f"\n\ndecoded_output: {decoded_output}\n\n")
            inference_output += decoded_output
        except Exception as e:
            handle_local_error("Could not generate output, encountered error: ", e)
            return False
    
    return inference_output


@app.route('/completions', methods=['POST'])
def completions():

    print("\n\ncompletions route triggered - attempting to acquire LLM semaphore\n\n")

    with llm_semaphore:

        print("\n\nLLM semaphore acquired by /completions\n\n")

        try:
            read_return = read_config(['max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep', 'flux_diffusers', 'vision'])
            max_new_tokens = int(read_return['max_new_tokens'])
            return_full_text = str(read_return['return_full_text']).lower() == 'true'
            temperature = float(read_return['temperature'])
            do_sample = str(read_return['do_sample']).lower() == 'true'
            top_k = int(read_return['top_k'])
            top_p = float(read_return['top_p'])
            min_p = float(read_return['min_p'])
            n_keep = int(read_return['n_keep'])
            flux_diffusers = str(read_return['flux_diffusers']).lower() == 'true'
            vision = str(read_return['vision']).lower() == 'true'
        except Exception as e:
            handle_local_error("Could not read values from hf_config.json when attempting /completions, encountered error: ", e)

        if flux_diffusers:
            try:
                image_str, image_name = generate_flux_image(request)
                return jsonify({"success": True, "response": image_str, "image_name": image_name})
            except Exception as e:
                return handle_api_error("Could not generate image with FLUX Diffusers. Encountered error: ", e)

        if vision:
            try:
                response = inference_with_vision_model(request)
                return jsonify({"success": True, "response": response})
            except Exception as e:
                return handle_api_error("Could not generate image with Vision Model. Encountered error: ", e)

        try:
            data = request.json
            messages = data.get('messages', [])
        except Exception as e:
            return handle_api_error("Could not read POST-request messages for /completions, encountered error: ", e)

        try:
            generation_config = GenerationConfig(
                max_new_tokens=int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
                temperature=float(request.headers.get('X-Temperature', str(temperature))),
                do_sample=request.headers.get('X-Do-Sample', str(do_sample)).lower() == 'true',
                top_k=int(request.headers.get('X-Top-K', str(top_k))),
                top_p=float(request.headers.get('X-Top-P', str(top_p))),
                min_p=float(request.headers.get('X-Min-P', str(min_p))),
                use_cache=True
            )
            # use_cache=True by default, setting explictily to True for clarity. Intra-call optimization: tells the generator, "During this single generation call, please be efficient.
            # As you process the prompt and generate new tokens, create and use a KV cache internally so you don't have to re-calculate everything for every single new token." 
        except Exception as e:
            handle_error_no_return("Could not set generation-arguments for /completions, proceeding without them. Encountered error: ", e)
            generation_config = GenerationConfig(max_new_tokens=max_new_tokens, use_cache=True)

        try:
            print(f"\n\nApplying Chat Template for messages: {messages}\n\n")
            inputs = PIPE.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        except Exception as e:
            handle_local_error("Could not apply chat template, encountered error: ", e)
            return False

        try:
            print("\n\nLoading Input to Model\n\n")
            inputs.to(PIPE.model.device)
        except Exception as e:
            handle_local_error("Could not load input to model, encountered error: ", e)
            return False

        inference_output = ""
        try:
            print("\n\nGenerating Output\n\n")
            output = PIPE.model.generate(**inputs, generation_config=generation_config)
            input_length = inputs.input_ids.shape[1]   # Check inference_with_vision_model(request) for detailed explanation!
            
            # Slice the tensor and decode only the output!
            decoded_output = PIPE.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)    # Setting skip_special_tokens=True to remove: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.

            print(f"\n\ndecoded_output: {decoded_output}\n\n")
            inference_output += decoded_output
        except Exception as e:
            handle_local_error("Could not generate output, encountered error: ", e)
            return False

        print("\n\nCompletions done - releasing LLM semaphore\n\n")

        return jsonify({"success": True, "response": inference_output})



class StopOnEvent(StoppingCriteria):    # custom StoppingCriteria to stop generation when an event is set - inherits from StoppingCriteria, which is an abstract class defined in Hugging Face's transformers library. StoppingCriteria is used to define custom stopping conditions for the generation process.
    def __init__(self, event: threading.Event): # A threading.Event object is taken as an instantiating argument by our custom class's constructor.
        self.event = event  # The inheritance of StoppingCriteria means our class is a type of StoppingCriteria, initialized with an event object.

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.event.is_set()
    # Note on input_ids: torch.LongTensor and scores: torch.FloatTensor:
    # These type hints are specified by the StoppingCriteria class in the transformers library.
    # input_ids: This is the input IDs tensor, which is a tensor of token IDs representing the current input sequence. Tokens are typically represented as integers, hence LongTensor (64-bit integer).
    # scores: This is a tensor of scores (logits) output by the model for each token in the input sequence. These are floating-point values, hence FloatTensor.
    # The __call__ method is called during the generation process, after each generated token, to check if the stopping condition is met. We must define this method in our concrete class because StoppingCriteria is an abstract class.
    # It takes the input IDs, scores, and any additional keyword arguments (**kwargs) as arguments. However, it ignores the input IDs and scores, and only uses the keyword argument event to determine if the stopping condition is met.
    # The method should return True if the stopping condition is met (e.g., if an event is set), and False otherwise.



@app.route('/vision_stream', methods=['POST'])
def vision_stream():
    print("\n\nvision_stream route triggered - attempting to acquire LLM semaphore\n\n")

    llm_semaphore.acquire()

    print("\n\nLLM semaphore acquired by /vision_stream\n\n")

    try:
        input_text, pil_image_object_list, generation_config, filename, vision_file_present = get_input_params_for_vision_model(request)
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not get input params in vision_stream, encountered error: ", e)
    
    if not vision_file_present:
        pil_image_object_list.append(get_blank_pil_image_object())

    stop_event = threading.Event()
    generation_config["stopping_criteria"] = StoppingCriteriaList([StopOnEvent(stop_event)])  # StoppingCriteriaList is a container that holds a list of StoppingCriteria objects. In our case, we have only one such object, which is our custom StoppingCriteria class, initialized with the stop_event object.

    data_queue = queue.Queue()

    def llm_task():

        try:
            for page_number, image in enumerate(pil_image_object_list, start=1): # start=1 to match the page numbers in the PDF
                
                if vision_file_present: 
                    status_string = f"Processing Page: {page_number} from file: {filename}\n\n"
                    data_queue.put(status_string)
                    print(status_string)
                
                print("\n\nLoading Input to Model\n\n")
                inputs = PIPE(image, input_text, return_tensors="pt").to(VISION_MODEL.device)

                print("\n\nGenerating Output\n\n")
                output = VISION_MODEL.generate(**inputs, **generation_config)    # `output` is a tensor and needs to be decoded!

                # Get length of the input sequence - look for detailed comment in inference_with_vision_model() !
                input_length = inputs.input_ids.shape[1] 
                
                # Slice the tensor and decode only the output!
                decoded_output = PIPE.decode(output[0][input_length:], skip_special_tokens=True)    # Setting skip_special_tokens=True to remove: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.

                print(f"\n\ndecoded_output: {decoded_output}\n\n")
                data_queue.put(decoded_output)
                data_queue.put("\n\n\n")

        finally:
            data_queue.put(None)
            print("\n\nLLM stream done, releasing semaphore\n\n")
            llm_semaphore.release()
    
    def generate():

        global STOP_GENERATION
        STOP_GENERATION = False

        thread = threading.Thread(target=llm_task)
        thread.start()

        while True:
            if STOP_GENERATION:
                print("\n\nStopping generation with stop_event\n\n")
                stop_event.set()
                thread.join()
                break
            output = data_queue.get()
            if output is None:
                print("\n\nNone read, breaking and stopping thread\n\n")
                thread.join()
                break
            yield f"data: {json.dumps(output)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"

        STOP_GENERATION = False
            
    print("\n\nInferencing Begins!\n\n")
    return Response(generate(), content_type='text/event-stream')



class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_special_tokens=True, skip_prompt=True, **kwargs):
        super().__init__(tokenizer, skip_special_tokens=skip_special_tokens, skip_prompt=skip_prompt, **kwargs)
        self.callback = None
        self.buffer = io.StringIO()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.callback:
            self.callback(text)
        return self.buffer.write(text)
    
    def flush(self):
        self.buffer.flush()

@app.route('/completions_stream', methods=['POST'])
def completions_stream():

    print("\n\ncompletions_stream route triggered - attempting to acquire LLM semaphore\n\n")

    llm_semaphore.acquire()

    print("\n\nLLM semaphore acquired by /completions_stream\n\n")

    try:
        data = request.json
        if isinstance(data, str):   # should be a list
            data = json.loads(data)
        messages = data.get('messages', [])
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read POST-request messages for /completions_stream, encountered error: ", e)

    try:
        read_return = read_config(['max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep'])
        max_new_tokens = int(read_return['max_new_tokens'])
        return_full_text = str(read_return['return_full_text']).lower() == 'true'
        temperature = float(read_return['temperature'])
        do_sample = str(read_return['do_sample']).lower() == 'true'
        top_k = int(read_return['top_k'])
        top_p = float(read_return['top_p'])
        min_p = float(read_return['min_p'])
        n_keep = int(read_return['n_keep'])
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read values from hf_config.json when attempting /completions_stream, encountered error: ", e)

    try:    # Create a GenerationConfig object
        generation_config = {
            "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
            "return_full_text": request.headers.get('X-Return-Full-Text', str(return_full_text)).lower() == 'true',
            "temperature": float(request.headers.get('X-Temperature', str(temperature))),
            "do_sample": request.headers.get('X-Do-Sample', str(do_sample)).lower() == 'true',
            "top_k": int(request.headers.get('X-Top-K', str(top_k))),
            "top_p": float(request.headers.get('X-Top-P', str(top_p))),
            "min_p": float(request.headers.get('X-Min-P', str(min_p))),
            "use_cache": True
        }
    except Exception as e:
        handle_error_no_return("Could not set generation-arguments for /completions_stream, proceeding without them. Encountered error: ", e)
        generation_config = {"max_new_tokens": max_new_tokens,"use_cache": True}

    try:
        print(f"\n\nApplying Chat Template for messages: {messages}\n\n")
        inputs = PIPE.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")    
        # `PIPE.tokenizer.apply_chat_template` outputs a dict to `inputs` with keys: input_ids, attention_mask, labels, token_type_ids
        # type(inputs): <class 'transformers.tokenization_utils_base.BatchEncoding'>
        # type(inputs['input_ids']): <class 'torch.Tensor'>
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not apply chat template, encountered error: ", e)

    try:
        # Slice the tensor and decode only the input!
        decoded_inputs = PIPE.tokenizer.decode(inputs['input_ids'][0].tolist(), skip_special_tokens=False)    # Setting skip_special_tokens=False to keep: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.
        print(f"\n\ndecoded_inputs: {decoded_inputs}\n\n")
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not decode inputs, encountered error: ", e)

    stop_event = threading.Event()
    data_queue = queue.Queue()

    def callback(data):
        data_queue.put(data)

    custom_streamer = CustomTextStreamer(PIPE.tokenizer, skip_special_tokens=True, skip_prompt=True)    # special tokens need not be streamed though!
    custom_streamer.callback = callback

    def llm_task():

        global PIPE

        try:
            generation_config["streamer"] = custom_streamer
            generation_config["stopping_criteria"] = StoppingCriteriaList([StopOnEvent(stop_event)])  # StoppingCriteriaList is a container that holds a list of StoppingCriteria objects. In our case, we have only one such object, which is our custom StoppingCriteria class, initialized with the stop_event object.
            output = PIPE(decoded_inputs, **generation_config)
        finally:
            data_queue.put(None)
            print("\n\nLLM stream done, releasing semaphore\n\n")
            llm_semaphore.release()

    def generate():        
        
        global STOP_GENERATION
        STOP_GENERATION = False

        thread = threading.Thread(target=llm_task)
        thread.start()

        while True:
            if STOP_GENERATION:
                print("\n\nStopping generation with stop_event\n\n")
                stop_event.set()
                thread.join()
                break
            line = data_queue.get()
            if line is None:
                print("\n\nNone read, breaking and stopping thread\n\n")
                thread.join()
                break
            yield f"data: {json.dumps(line)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"

        STOP_GENERATION = False
            
    print("\n\nInferencing Begins!\n\n")
    return Response(generate(), content_type='text/event-stream')


def exl2_prompt_fits_within_max_context_length(prompt: str) -> bool:
    """
    Checks if a given prompt fits within the max context length of the model.

    A model's max context length sets a hard-limit on how large a prompt can be.
    No input (prompt + generation) can exceed this value. If it does, the models' attention layers cannot process it correctly.
    Accordingly, ExLlamaV2 will discard or ignore extra tokens, processing only upto N tokens.
    This auto-truncation can lead to unexpected responses for long prompts, which is problematic.
    Thus the need to check if the prompt fits within the max context length.

    Args:
        prompt: The prompt to check the context length of.

    Returns:
        True if the prompt fits within the max context length, False otherwise.
    """

    print(f"\n\nChecking if exl2-prompt fits within max context length...\n\n")
    try:
        prompt_tokens = EXL2_TOKENIZER.encode(prompt, encode_special_tokens=True)
        # print(f"\n\nPrompt tokens shape: {prompt_tokens.shape}\n\n") # Outputs the tensor's shape (dimensions): torch.Size([<batch_size>, <num_tokens_in_prompt>]), example: torch.Size([1, 602])
        # shape[-1] "gives the shape of the last dimension", prompt_tokens.shape is multi-dimentional of shape [<batch_size>, <num_tokens_in_prompt>]
        # so -1, the last dimention = num_tokens_in_prompt
        
        if prompt_tokens.shape[-1] <= EXL2_MODEL.config.max_seq_len:   # max_seq_len is the max context length of the model
            print(f"\n\nPrompt fits within context-window\n\n")
            return True
        else:
            print(f"\n\nPrompt does not fit within max context length: {prompt_tokens.shape[-1]} > {EXL2_MODEL.config.max_seq_len}\n\n")
            return False
    except Exception as e:
        handle_error_no_return(f"Could not check if exl2-prompt fits within max context length, enabling auto-truncation as a fallback. Encountered error: ", e)
        return True


def set_global_exl2_dynamic_generator(batch_mode: bool = False):
    """
    Defines and returns an ExLlamaV2DynamicGenerator object.
    The generator object's cache builds up over time which is why it's best to create a new generator object for new chat requests as they may eb from different users.

    Args:
        batch_mode: Whether to use batch mode. Default is False.

    Returns:
        The ExLlamaV2DynamicGenerator object.
    """

    try:
        if batch_mode:
            print("\nDefining ExLlamaV2DynamicGenerator in batch mode\n")
            max_batch_size = read_config(['max_batch_size'])
            max_tokens_per_sequence = read_config(['max_tokens_per_sequence'])
            exl2_dynamic_generator = ExLlamaV2DynamicGenerator(
                model = EXL2_MODEL,
                cache = EXL2_CACHE,
                tokenizer = EXL2_TOKENIZER,
                max_batch_size = max_batch_size,
                max_q_size = max_tokens_per_sequence
            )
        else:
            print("\nDefining ExLlamaV2DynamicGenerator in single-sequence mode\n")
            exl2_dynamic_generator = ExLlamaV2DynamicGenerator(model = EXL2_MODEL, cache = EXL2_CACHE, tokenizer = EXL2_TOKENIZER)
        
        print("\nGenerator defined successfully\n")
        return exl2_dynamic_generator
    except Exception as e:
        handle_local_error("Could not define ExLlamaV2 generator, encountered error: ", e)


def get_exl2_gen_settings(request):

    try:
        config_data = read_config(['max_new_tokens', 'temperature', 'top_k', 'top_p', 'knowledge_graph_cache_dir'])
        requested_max_new_tokens = config_data['max_new_tokens']    # safe value for return
    except Exception as e:  # Not using `handle_local_error` as the necessary params may be in the request headers so why error out here?
        handle_error_no_return("Could not read values from hf_config.json when attempting exl2-grapher, relying on request headers instead. Encountered error: ", e)
        config_data = {}

    try:
        print("\nExLlamaV2Sampler.Settings In-Progress...\n")
        gen_settings = ExLlamaV2Sampler.Settings(
            temperature = float(request.headers.get('X-Temperature', str(config_data.get('temperature', '')))),
            top_k = int(request.headers.get('X-Top-K', str(config_data.get('top_k', '')))),
            top_p = float(request.headers.get('X-Top-P', str(config_data.get('top_p', ''))))
        )
        requested_max_new_tokens = int(request.headers.get('X-Max-New-Tokens', str(config_data.get('max_new_tokens', ''))))
        print("\nExLlamaV2Sampler.Settings Defined Successfully\n")
    except Exception as e:
        handle_error_no_return("Could not set generation-arguments for exl2-grapher, proceeding without them. Encountered error: ", e)
        gen_settings = None

    return gen_settings, requested_max_new_tokens, config_data.get('knowledge_graph_cache_dir', '/')


def exl2_test_encoding_logic(tokenized_messages: str):
    """
    Tests the encoding logic of the ExLlamaV2Tokenizer object.

    Args:
        tokenized_messages: The tokenized messages to test the encoding logic of.

    Returns:
        None.

    Example Output:
    
    Tokenized messages: <|im_start|>system<|im_sep|>You are a helpful assistant.<|im_end|><|im_start|>user<|im_sep|>Hi!<|im_end|><|im_start|>assistant<|im_sep|>

    Case 1 -- encode_special_tokens=False: Template tags not properly encoded:
    ["systemYou are a helpful assistant.userHi!assistant"]

    Case 2 -- encode_special_tokens=True: Template tags properly encoded:
    ["<|im_start|>system<|im_sep|>You are a helpful assistant.<|im_end|><|im_start|>user<|im_sep|>Hi!<|im_end|><|im_start|>assistant<|im_sep|>"]

    Case 3 -- add_bos=True: Unwanted extra Beginning-of-String (BOS) token <|endoftext|> added at the beginning! NOTE - encode_special_tokens defaults to False but Template tags properly encoded!
    ["<|endoftext|><|im_start|>system<|im_sep|>You are a helpful assistant.<|im_end|><|im_start|>user<|im_sep|>Hi!<|im_end|><|im_start|>assistant<|im_sep|>"]

    Case 4 -- add_eos=True: Unwanted extra End-of-String (EOS) token <|im_end|> added at the end! NOTE - encode_special_tokens defaults to False but Template tags properly encoded!
    ["<|im_start|>system<|im_sep|>You are a helpful assistant.<|im_end|><|im_start|>user<|im_sep|>Hi!<|im_end|><|im_start|>assistant<|im_sep|><|im_end|>"]

    Case 5 -- add_bos=True, encode_special_tokens=True, add_eos=True: Unwanted extras at both ends!:
    ["<|endoftext|><|im_start|>system<|im_sep|>You are a helpful assistant.<|im_end|><|im_start|>user<|im_sep|>Hi!<|im_end|><|im_start|>assistant<|im_sep|><|im_end|>"]
    """
    try:
        exl2_donot_encode_special_tokens = EXL2_TOKENIZER.encode(tokenized_messages, encode_special_tokens=False)
        exl2_encode_special_tokens = EXL2_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True)
        exl2_add_bos = EXL2_TOKENIZER.encode(tokenized_messages, add_bos=True)
        exl2_add_eos = EXL2_TOKENIZER.encode(tokenized_messages, add_eos=True)
        exl2_add_all = EXL2_TOKENIZER.encode(tokenized_messages, add_bos=True, encode_special_tokens=True, add_eos=True)

        decoded_exl2_donot_encode_special_tokens = EXL2_TOKENIZER.decode(exl2_donot_encode_special_tokens, decode_special_tokens=False)
        decoded_exl2_encode_special_tokens = EXL2_TOKENIZER.decode(exl2_encode_special_tokens, decode_special_tokens=True)
        decoded_exl2_add_bos = EXL2_TOKENIZER.decode(exl2_add_bos, decode_special_tokens=True)
        decoded_exl2_add_eos = EXL2_TOKENIZER.decode(exl2_add_eos, decode_special_tokens=True)
        decoded_exl2_add_all = EXL2_TOKENIZER.decode(exl2_add_all, decode_special_tokens=True)

        print(f"EXL2-TOKENIZER.encode(encode_special_tokens=False): {decoded_exl2_donot_encode_special_tokens}\n")
        print(f"EXL2-TOKENIZER.encode(encode_special_tokens=True): {decoded_exl2_encode_special_tokens}\n")
        print(f"EXL2-TOKENIZER.encode(add_bos=True): {decoded_exl2_add_bos}\n")
        print(f"EXL2-TOKENIZER.encode(add_eos=True): {decoded_exl2_add_eos}\n")
        print(f"EXL2-TOKENIZER.encode(add_bos=True, encode_special_tokens=True, add_eos=True): {decoded_exl2_add_all}\n")
    except Exception as e:
        handle_error_no_return("Could not test-encode messages for exl2-stream, encountered error: ", e)


@app.route('/exl2_stream', methods=['POST'])
def exl2_stream():

    print("\n\nexl2-stream route triggered - attempting to acquire LLM semaphore\n\n")

    llm_semaphore.acquire()

    print("\nLLM semaphore acquired by exl2-stream\n")

    try:
        data = request.json
        if isinstance(data, str):   # must convert to a list
            data = json.loads(data)
        messages = data.get('messages', [])
        gen_settings, max_new_tokens, _  = get_exl2_gen_settings(request)
        # print(f"\nRead request - message received: {messages}\n")
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read POST-request messages for exl2-stream, encountered error: ", e)
    
    try:
        exl2_dynamic_generator = set_global_exl2_dynamic_generator()
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not set global ExLlamaV2DynamicGenerator, encountered error: ", e)
    
    try:
        tokenized_messages = AUTO_TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # print(f"\nTokenized messages: {tokenized_messages}\n")
        # exl2_test_encoding_logic(tokenized_messages)
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not tokenize messages for exl2-stream, encountered error: ", e)
    
    if not exl2_prompt_fits_within_max_context_length(tokenized_messages):
        return handle_api_error("Prompt does not fit within max context length, enabling auto-truncation as a fallback. Encountered error: ", e)

    try:
        # print(f"AUTO_TOKENIZER.eos_token_id: {AUTO_TOKENIZER.eos_token_id}")
        # print(f"EXL2_TOKENIZER.eos_token_id: {EXL2_TOKENIZER.eos_token_id}")
        print("\nCreating ExLlamaV2-DynamicJob Object...\n")
        job = ExLlamaV2DynamicJob(
            input_ids= EXL2_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True),
            max_new_tokens = int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
            stop_conditions = [EXL2_TOKENIZER.eos_token_id, AUTO_TOKENIZER.eos_token_id],
            gen_settings = gen_settings
        )
        exl2_dynamic_generator.enqueue(job)
        print("\nExLlamaV2-DynamicJob Defined & Enqueued Successfully\n")
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not create ExLlamaV2-DynamicJob object for exl2-stream, encountered error: ", e)

    stop_thread = threading.Event()
    output_queue = queue.Queue()

    def llm_task():

        try:            
            while exl2_dynamic_generator.num_remaining_jobs():
                # output_queue.put(exl2_dynamic_generator.iterate()[0])  # Will print dict with keys: dict_keys(['job', 'stage', 'eos', 'serial', 'text', 'token_ids']) ### USE: yield f"data: {(line)}\n\n"
                current_token = exl2_dynamic_generator.iterate()   # directly trying to access the 'text' key here will result in a KeyError as iteration may not have completed yet!
                if len(current_token) == 1 and 'text' in current_token[0]:
                    output_queue.put(current_token[0]['text'])  # thus best to capture the current iteration's output and then access the 'text' key!
                elif len(current_token) > 1:
                    for job in current_token:
                        if 'stage' in job and job['stage'] == 'streaming':
                            if 'text' in job: 
                                output_queue.put(job['text'])   
                # output_queue.put(current_token[2]['text']) if current_token[0]['stage'] == 'started' else output_queue.put(current_token[0]['text'])
        except Exception as e:
            return handle_error_no_return("Response generation failed, encountered error: ", e)
        finally:
            output_queue.put(None)
            print("\n\nLLM stream done, releasing semaphore\n\n")
            llm_semaphore.release()
            stop_thread.set()

    def generate():

        global STOP_GENERATION
        STOP_GENERATION = False

        thread = threading.Thread(target=llm_task)
        thread.start()

        while True:
            if STOP_GENERATION:
                print("\n\nStopping generation with stop_event\n\n")
                output_queue.put(None)
                STOP_GENERATION = False
                thread.join()
            
            line = output_queue.get()
            if line is None:
                print("\nNone read, breaking and stopping thread\n")
                thread.join()
                break
            yield f"data: {json.dumps(line)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"

        print("\nexl2-stream done\n")

    print("\n\nInferencing Begins!\n\n")
    return Response(generate(), content_type='text/event-stream')



### Exl2 Graph Helper Functions ###

def create_and_execute_exl2_job(payload:str, max_new_tokens:int, gen_settings):
    try:    # Step 1: Create Exl2 Generator & Job
        exl2_dynamic_generator = set_global_exl2_dynamic_generator()
        job = ExLlamaV2DynamicJob(
            input_ids= EXL2_TOKENIZER.encode(payload, encode_special_tokens=True),
            max_new_tokens = max_new_tokens,
            stop_conditions = [EXL2_TOKENIZER.eos_token_id, AUTO_TOKENIZER.eos_token_id],
            gen_settings = gen_settings
        )
        exl2_dynamic_generator.enqueue(job)
    except AssertionError as e:
        handle_local_error(f"Could not create ExLlamaV2-DynamicJob object for payload {payload} - likely due to insufficient cache. Ensure the `max_seq_len` is set to a value that allows for the entire payload to be processed. Encountered error: ", e)
    except Exception as e:
        handle_local_error(f"Could not create ExLlamaV2-DynamicJob object for payload {payload}, encountered error: ", e)

    try:    # Step 2: Iterate Over Jobs & Generate Response(s)
        print("\nProcessing Exl2 Job...\n")
        full_response = ""
        while exl2_dynamic_generator.num_remaining_jobs():
            current_token = exl2_dynamic_generator.iterate()
            
            if len(current_token) == 1 and 'text' in current_token[0]:
                full_response += current_token[0]['text']
            
            elif len(current_token) > 1:
                for job in current_token:
                    if 'stage' in job and job['stage'] == 'streaming':
                        if 'text' in job: 
                            full_response += job['text']

        exl2_dynamic_generator = None    # release reference to the generator to help with garbage collection
        return full_response
    except Exception as e:
        handle_local_error(f"Could not generate response for payload {payload}, encountered error: ", e)


def get_request_payload_for_graph_entity_extraction(chunk_text: str):
    try:
        chunk_payload = "Extract nodes and relationships from the following text:\n" + chunk_text + "\n<knowledge_graph>"
        full_payload = AUTO_TOKENIZER.apply_chat_template([{"role": "user", "content": chunk_payload}], add_generation_prompt=True, tokenize=False)
        return full_payload
    except Exception as e:
        handle_local_error(f"Could not get request payload for graph entity extraction, encountered error: ", e)


def remove_blank_nodes_and_relationships(entities_and_relationships: dict):
    '''
    Filters out nodes and relationships with blank/empty required fields.
    
    Args:
        entities_and_relationships: Pre-validated dict with 'nodes' and 'relationships' lists
        
    Returns:
        Dict with filtered nodes and relationships
        
    Note:
        Assumes input has already been validated by core-dict_validation_logic_for_cache_reuse()
    '''
    try:    # at this point, the entities_and_relationships has already been validated so we can directly iterate over the nodes and relationships

        def is_valid_node(node: dict) -> bool:
            """Check if node has non-empty required fields"""
            return (str(node.get('name') or '').strip() and
                    str(node.get('type') or '').strip())
        
            '''
            NOTE - Works because:
            1. A blank string evaluates to False in a boolean context
            2. strip() is crucial for handling whitespace edge-cases like '    ' !
            3. `name = str(node.get('name', ''))` will lead to an edge-case error, wherein an input like`{'name': None}` will result in a string 'None', causing the eval to incorrectly pass!
            4. The above means a follow-on check like `name = '' if name is None else name` will never execute!
            5. `str(node.get('name') or '').strip()` handles all edge-cases, as a valid value will be returned or a blank '' for empty/None values, with .strip() handling whitespaces.
            '''

        def is_valid_relationship(relationship: dict) -> bool:
            """Check if relationship has non-empty required fields"""
            return (str(relationship.get('source') or '').strip() and
                    str(relationship.get('target') or '').strip() and
                    str(relationship.get('relationship') or '').strip())

        return {
            'nodes': [node for node in entities_and_relationships['nodes'] if is_valid_node(node)],
            'relationships': [relationship for relationship in entities_and_relationships['relationships'] if is_valid_relationship(relationship)]
        }

    except Exception as e:
        handle_error_no_return(f"Could not remove blank nodes and relationships returning unchanged response. Encountered error: ", e)
        return entities_and_relationships


def core_dict_validation_logic_for_cache_reuse(extracted_dict: dict):
    try:
        if not isinstance(extracted_dict, dict):
            raise ValueError(f"Invalid dictionary validation request - expected a dict, got {type(extracted_dict).__name__}")

        # Check for required keys
        if 'nodes' not in extracted_dict or 'relationships' not in extracted_dict:
            raise ValueError("Missing required keys in extraction response: 'nodes' and/or 'relationships'")
        
        # Check if keys are of the correct type
        if not isinstance(extracted_dict['nodes'], list) or not isinstance(extracted_dict['relationships'], list):
            raise ValueError(f"Expected 'nodes' and 'relationships' to be lists, got {type(extracted_dict['nodes']).__name__} and {type(extracted_dict['relationships']).__name__}")
        
        # Check if nodes and relationships are lists of dicts - Consider using a JSON schema validation library (like jsonschema) if the expected structure becomes more complex.
        if not all(isinstance(node, dict) for node in extracted_dict['nodes']):
            raise ValueError("Expected all elements in 'nodes' list to be dicts.")
        if not all(isinstance(relationship, dict) for relationship in extracted_dict['relationships']):
            raise ValueError("Expected all elements in 'relationships' list to be dicts.")

        return extracted_dict

    except Exception as e:
        handle_error_no_return(f"Could not validate entity extraction response, encountered error: ", e)
        return None


def validate_entity_extraction_response(extraction_response: str):
    '''
    This function attempts to validate the response from the entity extraction task.
    It first attempts to evaluate the response thoroughly for both type and structure.
    If this fails, it attempts to trim the response and then evaluate again.
    If this also fails, it returns the unchanged response and a flag indicating that the response is invalid.
    This is a very robust validation logic, and is designed to handle a wide range of edge cases.
    '''
    def try_validation(response_str):
        try:
            response_dict = ast.literal_eval(response_str)

            # Check if it's a dict - ast.literal_eval can return other Python objects such as lists, strings, etc., so we need to check for that!
            if not isinstance(response_dict, dict):
                raise ValueError(f"Expected a dict, got {type(response_dict).__name__}")
            
            validated_response = core_dict_validation_logic_for_cache_reuse(response_dict)
            if validated_response is not None:
                return {'validated_response': validated_response, 'is_valid': True}
            return None
        except Exception as e:
            handle_error_no_return(f"Could not validate entity extraction response, encountered error: ", e)
            return None
    
    # Try direct response
    result = try_validation(extraction_response)
    if result is not None:
        return result
    
    # Try with trimming
    trimmed_response = prompt_formatting_module.trim_response(extraction_response, '{"nodes":', '"}]}', include_start_substring=True, include_end_substring=True)
    result = try_validation(trimmed_response)
    if result is not None:
        print("\n\nTrimmed response validated successfully\n\n")
        return result
    
    print(f"\nResponse validation failed even after trimming.\n")
    return {'validated_response': extraction_response, 'is_valid': False}


def process_nodes_and_relationships(
        nodes_and_relationships: dict,
        chunk_text: str,
        source_doc_name: str,
        page_number_list: list,
        requested_max_new_tokens: int = 1000,
        gen_settings = None
    ):
    '''
    This function takes a dictionary containing nodes and relationships, and a chunk of text.
    It generates a comprehensive summary for the chunk covering all nodes and relationships, and saves the summary to every node and relationship.
    It returns an updated dictionary comprising all nodes and relationships with their comprehensive summaries.
    '''
    try:
        comprehensive_summary_request_prompt = prompt_formatting_module.get_user_query_for_comprehensive_summary(nodes_and_relationships, chunk_text)
        formatted_prompt = AUTO_TOKENIZER.apply_chat_template([{"role": "user", "content": comprehensive_summary_request_prompt}], add_generation_prompt=True, tokenize=False)
        
        if exl2_prompt_fits_within_max_context_length(formatted_prompt):
            full_response = create_and_execute_exl2_job(payload=formatted_prompt, max_new_tokens=requested_max_new_tokens, gen_settings=gen_settings)
        else:   # No errors are raised if the prompt is larger than the max context length because it'll be auto-truncated which we don't want so best to handle manually!
            minimal_summary_request_prompt = prompt_formatting_module.get_minimal_query_for_summary(chunk_text)
            formatted_prompt = AUTO_TOKENIZER.apply_chat_template([{"role": "user", "content": minimal_summary_request_prompt}], add_generation_prompt=True, tokenize=False)
            
            if exl2_prompt_fits_within_max_context_length(formatted_prompt):
                full_response = create_and_execute_exl2_job(payload=formatted_prompt, max_new_tokens=requested_max_new_tokens, gen_settings=gen_settings)
            else:
                handle_error_no_return(f"Could not generate comprehensive summary for chunk of document {source_doc_name} - too long even with minimal data! Encountered error: ", e)
                return [""]
        
        print("Summary generated, post-processing...\n")
        full_response = prompt_formatting_module.trim_response(full_response, '"summary":', '}').replace("'", "") + "\n{Source Document Name: " + source_doc_name + "}\n{Page Number(s): " + str(page_number_list) + "}\n\n"
        print("\nSummary generation completed for present document chunk, proceeding...\n")
        return [str(full_response)]
    except Exception as e:
        handle_error_no_return(f"Could not prepare comprehensive-summary query for document {source_doc_name}, encountered error: ", e)
        return [""]

    
### End of Helper Functions ###


@app.route('/exl2_graph_extractor', methods=['POST'])
def exl2_graph_extractor():
    '''
    Appends the `entities_and_relationships` key to each chunk_entities dict, returning the following structure:

    chunk_entities = {
        '<graph_chunk_number>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>'
        }
    }
    '''

    print("\n\nexl2-graph-extractor route triggered - attempting to acquire LLM semaphore\n\n")
    llm_semaphore.acquire()
    print("\nLLM semaphore acquired by /exl2-graph-extractor\n")

    try:
        chunk_entities = request.json.get('chunk_entities')
        rag_response_mode = request.json.get('rag_response_mode', False)
        gen_settings, requested_max_new_tokens, knowledge_graph_cache_dir = get_exl2_gen_settings(request)
        reuse_graph_extraction_cache = str(request.headers.get('X-Reuse-Extraction-Cache', str(read_config(['reuse_graph_extraction_cache'])['reuse_graph_extraction_cache']).lower())).lower() == 'true'
        # print(f"\nchunk_entities received:\n\n{chunk_entities}\n")
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read POST-request messages for /exl2-graph-extractor, encountered error: ", e)

    stop_thread = threading.Event()
    output_queue = queue.Queue()
    cache_queue = queue.Queue()

    def extraction_task():
        # BATCH_SIZE can be tuned based on memory and performance testing.
        BATCH_SIZE = 1000

        def load_and_fire_off_cache():
            '''
            Checks if any entries in the received chunk_entities dict are present in the local cache.
            If so, the cached data is streamed back right-away, leaving only the tasks to be processed by the LLM.
            This ensures maximum GPU utilization as each iteration purely executes the LLM task, without checking for / saving to cache which can be very I/O expensive for large files! 
            '''

            cache_data_map = {}     # declaring here in-case multiple source-doc_names are present in the chunk_entities dict!
            new_chunks_for_llm = [] # we add to this instead of deleting from the chunck_entities dict as "Filter & Return" is a superior design pattern to "Iterate & Mutate"
            # In the latter, a shared structure (chunk_entities) outside the scope of the function is mutated, which is not thread-safe!

            for chunk_number, chunk_data in chunk_entities.items():
                try:
                    source_doc_name = os.path.splitext(os.path.basename(chunk_data['source_doc_name']))[0]
                    
                    print(f"\nChecking for existing cache of previously extracted nodes and relationships for chunk {chunk_number} of document {source_doc_name}...\n")
                    
                    if source_doc_name not in cache_data_map:
                        extraction_cache_file_path = os.path.join(knowledge_graph_cache_dir, f"{source_doc_name}_extraction_cache.json")
                        cached_data = None
                        try:
                            cached_data = load_json_file(extraction_cache_file_path)
                        except Exception as e:
                            handle_error_no_return(f"Could not load extraction-cache file at path {extraction_cache_file_path} for document {source_doc_name}, proceeding to extract afresh. Encountered error: ", e)
                        
                        cache_data_map[source_doc_name] = {
                            'data': cached_data if isinstance(cached_data, dict) else {},
                            'file_path': extraction_cache_file_path
                        }   # Because the next step expects `source_doc_name` to be a key in the `cache_data_map` dict!
                    
                    
                    cached_chunk_data = cache_data_map.get(source_doc_name, {}).get('data', {}).get(chunk_number, {})
                    
                    if cached_chunk_data and cached_chunk_data.get('entities_and_relationships'):
                        validation_result = core_dict_validation_logic_for_cache_reuse(cached_chunk_data['entities_and_relationships'])
                        
                        if validation_result is not None:
                            # Path 1: Valid cache found. Use it and DO NOT add to the list.
                            chunk_entities[chunk_number]['entities_and_relationships'] = remove_blank_nodes_and_relationships(cached_chunk_data['entities_and_relationships'])    # NOTE: Only the 'entities_and_relationships' key is updated in the chunk_entities dict received in the POST request!
                            output_queue.put(chunk_entities[chunk_number])
                            print(f"\nFound existing cache of previously extracted nodes and relationships for chunk {chunk_number} of document {source_doc_name}, returning cached data...\n")
                        else:
                            # Path 2: Cache found but is invalid. Add to list for reprocessing.
                            new_chunks_for_llm.append(chunk_number)
                            print(f"\nCached data for chunk {chunk_number} of document {source_doc_name} failed validation, proceeding to extract afresh...\n")
                    else:
                        # Path 3: No cache entry exists for this chunk. Add to list.
                        new_chunks_for_llm.append(chunk_number)
                
                except Exception as e:
                    # Path 4: A generic error occurred. Default to reprocessing for safety.
                    handle_error_no_return(f"Error processing cache for chunk {chunk_number} of document {source_doc_name}, will be processed afresh for safety. Encountered error: ", e)
                    new_chunks_for_llm.append(chunk_number)

            return new_chunks_for_llm
                
        try:
            # --- PHASE 1: DETERMINE FULL WORKLOAD ---
            entries_to_process = list(chunk_entities.keys())    # Default to all chunks entries
            
            if reuse_graph_extraction_cache and not rag_response_mode:   # When responding to a query, we don't want to use the cache which is meant only for the file-processing step!
                try:
                    entries_to_process = load_and_fire_off_cache()  # If any entries were previously cached, those are returned and only new entries are processed by the LLM
                except Exception as e:
                    handle_error_no_return(f"Error processing cache, proceeding to process all chunks afresh. Encountered error: ", e)
                    entries_to_process = list(chunk_entities.keys())

            # --- PHASE 2 & 3: BATCH AND EXECUTE ---
            total_entry_count = len(entries_to_process)
            processed_entries = 0
            print(f"\nBeginning LLM processing for {total_entry_count} chunks in batches of {BATCH_SIZE}...\n")

            for i in range(0, len(entries_to_process), BATCH_SIZE):
                batch_keys = entries_to_process[i:i + BATCH_SIZE]   # Get the keys for the current batch

                batch_chunk_entities = {key: chunk_entities[key] for key in batch_keys} # CRITICAL STEP: Create a small, temporary dictionary for this batch ONLY - Details below:
                '''
                Technical Note: Optimizing the GPU Processing Pipeline
                
                1. Executive Summary:
                
                The initial implementation of the data processing method suffered from severe performance bottlenecks, causing GPU utilization to drop to near-zero between processing items in large jobs (>1000 items). 
                This led to poor performance, low cost-efficiency, and unnecessary hardware stress. 
                
                Through a two-phase optimization process, we addressed these issues, resulting in a sustained GPU utilization of ~75% when using Gemma2-2B on large workloads. Larger models will likely see a higher utilization.
                
                The two primary bottlenecks identified and solved were:
                1. Disk I/O Contention: Heavy file I/O for caching was performed sequentially within the main processing loop.
                2. CPU Cache Misses: Iterating over a very large central dictionary (chunk_entities) caused constant stalls in the CPU pipeline as it waited for data to be fetched from slow system RAM.

                The processing cycle was dramatically improved:

                Previously: Stall (cache-miss) -> Prepare -> GPU Work -> Stall (File I/O) -> Repeat
                Phase 1: Stall (cache-miss) -> Prepare -> GPU Work -> Repeat
                Phase 2: Prepare -> GPU Work -> Repeat

                
                2. Phase 1: Decoupling I/O with a Producer-Consumer Architecture

                The most apparent bottleneck was that disk caching operations were blocking the main thread.

                + Problem: The original loop would process a chunk, save the result to an on-disk cache, and then move to the next chunk. This heavy file I/O stalled the entire pipeline.
                
                + Solution: We implemented a "producer-consumer-consumer" pattern. The main processing thread (producer) now places results onto two separate queues. 
                One queue feeds the client response stream (consumer 1), and a new, dedicated thread handles all disk-caching operations (consumer 2).
                
                + Result: This dramatically improved performance, but a significant bottleneck remained. GPU utilization still dropped to ~20% between items, indicating that I/O was not the only issue.

                
                3. Phase 2: Solving CPU Cache Misses with Batching
                
                While dictionary lookups in Python are algorithmically O(1), this theoretical speed is only realized if the data is readily available to the CPU.
                
                + The Deeper Problem: Poor Data Locality. The central chunk_entities dictionary was often too large to fit into the CPU's fast cache. When iterating, moving from one item to the next 
                required the CPU to fetch data from a new, distant location in main RAM. This is a cache miss, and it stalls the CPU, preventing it from preparing the next job for the GPU. 
                The GPU would finish its work and sit idle, waiting.
                
                + The Solution: Cache Warming via Batching. Instead of iterating over the entire dictionary at once, we now process it in small, manageable batches (e.g., 1000 items):
                    ```
                    batch_chunk_entities = {key: chunk_entities[key] for key in batch_keys}
                        for chunk_number, chunk_data in batch_chunk_entities.items():
                    ```

                    The initial dictionary comprehension pulls all the necessary data for the upcoming batch into the CPU's fast cache at once. The subsequent "hot loop" then operates on this localized data, 
                    resulting in consistent cache hits. The CPU can prepare the next GPU task instantly, eliminating the pipeline stall.

                
                4. Final Benefits:

                This two-pronged approach ensures the GPU is fed a continuous, uninterrupted stream of data, leading to significant real-world benefits:
                    + Maximum Performance: Job completion time is drastically reduced.
                    + Improved Cost-Efficiency: We extract the maximum value from GPU resources by keeping them active.
                    + Increased Hardware Lifespan: Steady-state operation avoids the voltage fluctuations and thermal cycles of a bursty, inconsistent workload.

                Highly likely all this would have been missed if we were using one of AMD's X3D CPUs from the get-go!
                '''

                print(f"\nProcessing batch {i//BATCH_SIZE + 1} of {len(entries_to_process)//BATCH_SIZE + 1}...\n")

                for chunk_number, chunk_data in batch_chunk_entities.items():

                    try:
                        source_doc_name = os.path.splitext(os.path.basename(chunk_data['source_doc_name']))[0]

                        print(f"\nAttempting to extract entities and relationships from chunk {chunk_number} of document {source_doc_name}...processing item {processed_entries + 1} of total {total_entry_count} items...\n")
                        full_payload = get_request_payload_for_graph_entity_extraction(chunk_data['chunk_text'])
                        
                        response_validation_result = None
                        if exl2_prompt_fits_within_max_context_length(full_payload):
                            extraction_response_first_attempt = create_and_execute_exl2_job(payload=full_payload, max_new_tokens=requested_max_new_tokens, gen_settings=gen_settings)
                            response_validation_result = validate_entity_extraction_response(extraction_response_first_attempt)
                        
                        if response_validation_result is not None and response_validation_result['is_valid']:
                            full_response = remove_blank_nodes_and_relationships(response_validation_result['validated_response'])  # remove-blank_nodes_and_relationships() will either return a cleaned-up dict, or the unchanged dict on error or if no changes were made!
                        else:
                            print(f"Extraction response failed validation - Chunk is likely too large. Attempting to split and retry entity extraction for chunk {chunk_number} of document {source_doc_name}...")

                            def retry_entity_extraction(chunk_text, max_depth=5, current_depth=0):
                                '''
                                Splits the chunk into two halves and retries entity extraction for each half.
                                Recursively repeats until valid responses are obtained.
                                The reason for this design is that the model may have been overwhelmed by the chunk size, and relying on the overlap is not robust enough for two reasons:
                                    1. It may not be a big enough overlap for offsetting to help.
                                    2. By trimming both the leading and trailing overlap, imagine the scenario for consecutive chunks that fail to process:
                                        a) Say a chunk fails so you remove the overlap, which includes the last 300 chars as those are expected to be the first 300 chars of the next chunk anyways.
                                        b) Now if the next chunk also fails, and you remove the leading and trailing overlap, you have outright lost those 300 chars and never even processed them!
                                The best way to mitigate this is via clever recursion!
                                '''
                                
                                # Prevent infinite recursion
                                if current_depth >= max_depth:
                                    print(f"Max depth reached - could not extract all entities and relationships for chunk {chunk_number} of document {source_doc_name}. Returning empty response...")
                                    return {'nodes': [], 'relationships': []}
                                
                                # Prevent splitting text that's too small to be meaningful
                                if len(chunk_text.strip()) < 100:
                                    print(f"Chunk text too small - could not extract all entities and relationships for chunk {chunk_number} of document {source_doc_name}. Returning empty response...")
                                    return {'nodes': [], 'relationships': []}
                                
                                try:
                                    payload = get_request_payload_for_graph_entity_extraction(chunk_text)
                                    response = create_and_execute_exl2_job(payload=payload, max_new_tokens=requested_max_new_tokens, gen_settings=gen_settings)
                                    response_validation_result = validate_entity_extraction_response(response)
                                except Exception as e:
                                    handle_error_no_return(f"Could not extract entities and relationships for sub-chunk of chunk {chunk_number} from document {source_doc_name}. Current recursive depth: {current_depth} of {max_depth}. Encountered error: ", e)
                                    return {'nodes': [], 'relationships': []}
                                
                                if response_validation_result['is_valid']:
                                    print(f"Recursive Entity Extraction Successful: Sub-chunk of chunk {chunk_number} from document {source_doc_name} successfully validated!")
                                    return remove_blank_nodes_and_relationships(response_validation_result['validated_response'])
                                else:
                                    print(f"Sub-chunk still too large - continuing recursive extraction - attempt {current_depth + 1} of {max_depth} for chunk {chunk_number} of document {source_doc_name}...")
                                    # Split and recurse
                                    mid_point = len(chunk_text)//2
                                    recursive_response_1 = retry_entity_extraction(chunk_text[:mid_point], max_depth, current_depth + 1)
                                    recursive_response_2 = retry_entity_extraction(chunk_text[mid_point:], max_depth, current_depth + 1)

                                    # Properly merge the lists within the dictionaries
                                    merged_response = {
                                        'nodes': recursive_response_1.get('nodes', []) + recursive_response_2.get('nodes', []),
                                        'relationships': recursive_response_1.get('relationships', []) + recursive_response_2.get('relationships', [])
                                    }
                                    return merged_response

                            # Split and recurse
                            large_chunk_text = chunk_data['chunk_text']
                            mid_point = len(large_chunk_text)//2
                            response_1 = retry_entity_extraction(large_chunk_text[:mid_point])   # Floor division will round down to the nearest integer, thus handling odd-length chunks
                            response_2 = retry_entity_extraction(large_chunk_text[mid_point:])
                            
                            full_response = {
                                'nodes': response_1.get('nodes', []) + response_2.get('nodes', []),
                                'relationships': response_1.get('relationships', []) + response_2.get('relationships', [])
                            }

                        chunk_entities[chunk_number]['entities_and_relationships'] = full_response
                        output_queue.put(chunk_entities[chunk_number])
                        if not rag_response_mode:
                            cache_queue.put({chunk_number: chunk_entities[chunk_number]})

                        processed_entries += 1
                
                    except Exception as e:
                        handle_error_no_return(f"Could not extract entities and relationships from chunk {chunk_number} of document {source_doc_name}, skipping. Encountered error: ", e)
                        chunk_entities[chunk_number]['entities_and_relationships'] = {'nodes': [], 'relationships': []}     # Set empty or default value in case of complete failure
                        output_queue.put(chunk_entities[chunk_number])
                        if not rag_response_mode:
                            cache_queue.put({chunk_number: chunk_entities[chunk_number]})
                        continue

        except Exception as e:
            handle_error_no_return(f"Could not extract entities and relationships, encountered error: ", e)
        finally:
            # Signal the client-facing stream to stop. This always runs.
            output_queue.put(None)

            # ONLY signal the caching thread to stop IF it was started.
            if not rag_response_mode:
                cache_queue.put(None)
            
            print("\n\nLLM stream done, releasing semaphore\n\n")
            llm_semaphore.release()
            stop_thread.set()


    def save_to_local_cache(line):
        try:
            for chunk_number, chunk_data in line.items():
                source_doc_name = os.path.splitext(os.path.basename(chunk_data['source_doc_name']))[0]
                extraction_cache_file_path = os.path.join(knowledge_graph_cache_dir, f"{source_doc_name}_extraction_cache.json")
                update_and_save_json_file({chunk_number: chunk_data}, extraction_cache_file_path)
                print(f"\nSaved identified nodes and relationships from chunk {chunk_number} of document {source_doc_name} to cache file at path {extraction_cache_file_path}\n")
        except Exception as e:
            handle_error_no_return(f"Could not cache identified nodes and relationships from document {source_doc_name} to cache file at path {extraction_cache_file_path}, skipping. Encountered error: ", e)

    
    def caching_thread():
        while True:
            line =  cache_queue.get()
            if line is None:
                print("\nNone read, breaking and stopping cache-thread\n")
                break
            save_to_local_cache(line)

        print("\n\nCaching complete\n\n")


    def generate():
        while True:
            line = output_queue.get()
            if line is None:
                print("\nNone read, breaking and stopping task-thread\n")
                break
            yield f"data: {json.dumps(line)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"

        print("\n/exl2-graph-extractor done\n")

    task_thread = threading.Thread(target=extraction_task)
    task_thread.start()

    if not rag_response_mode:
        cache_thread = threading.Thread(target=caching_thread)
        cache_thread.start()

    print("\n\nInferencing Begins!\n\n")
    return Response(generate(), content_type='text/event-stream')



@app.route('/exl2_graph_summarizer', methods=['POST'])
def exl2_graph_summarizer():
    '''
    Generates and appends the `summary` key to each entry in the chunk_entities dict, returning the following structure:

    chunk_entities = {
        '<graph_chunk_number>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>',
            '<summary>': '<summary_text>'
        }
    }
    '''

    print("\n\nexl2-graph-summarizer route triggered - attempting to acquire LLM semaphore\n\n")
    llm_semaphore.acquire()
    print("\nLLM semaphore acquired by /exl2-graph-summarizer\n")

    try:
        chunk_entities = request.json.get('chunk_entities')
        gen_settings, requested_max_new_tokens, knowledge_graph_cache_dir = get_exl2_gen_settings(request)
        reuse_graph_summary_cache = str(request.headers.get('X-Reuse-Summary-Cache', str(read_config(['reuse_graph_summary_cache'])['reuse_graph_summary_cache']).lower())).lower() == 'true'
        # print(f"\nchunk_entities received:\n\n{chunk_entities}\n")
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read POST-request messages for /exl2-graph-summarizer, encountered error: ", e)

    stop_thread = threading.Event()
    output_queue = queue.Queue()
    cache_queue = queue.Queue()

    def summary_generation_task():
        # BATCH_SIZE can be tuned based on memory and performance testing.
        BATCH_SIZE = 1000

        def load_and_fire_off_cache():
            '''
            Checks if any entries in the received chunk_entities dict are present in the local cache.
            If so, the cached data is streamed back right-away, leaving only the tasks to be processed by the LLM.
            This ensures maximum GPU utilization as each iteration purely executes the LLM task, without checking for / saving to cache which can be very I/O expensive for large files! 
            '''
            
            cache_data_map = {}     # declaring here in-case multiple source-doc_names are present in the chunk_entities dict!
            new_chunks_for_llm = [] # we add to this instead of deleting from the chunck_entities dict as "Filter & Return" is a superior design pattern to "Iterate & Mutate"
            # In the latter, a shared structure (chunk_entities) outside the scope of the function is mutated, which is not thread-safe!

            for chunk_number, chunk_data in chunk_entities.items():
                try:
                    source_doc_name = os.path.splitext(os.path.basename(chunk_data['source_doc_name']))[0]

                    print(f"\nChecking for existing summary cache for chunk {chunk_number} of document {source_doc_name}...\n")

                    if source_doc_name not in cache_data_map:
                        summary_cache_file_path = os.path.join(knowledge_graph_cache_dir, f"{source_doc_name}_summary_cache.json")
                        cached_data = None
                        try:
                            cached_data = load_json_file(summary_cache_file_path)
                        except Exception as e:
                            handle_error_no_return(f"Could not load summary-cache file at path {summary_cache_file_path} for document {source_doc_name}, proceeding to generate afresh. Encountered error: ", e)
                        
                        cache_data_map[source_doc_name] = {
                            'data': cached_data if isinstance(cached_data, dict) else {},
                            'file_path': summary_cache_file_path
                        }   # Because the next step expects `source_doc_name` to be a key in the `cache_data_map` dict!
                        
                    cached_chunk_data = cache_data_map.get(source_doc_name, {}).get('data', {}).get(chunk_number, {})

                    if cached_chunk_data and 'summary' in cached_chunk_data:
                        
                        if cached_chunk_data['summary'] is not None and isinstance(cached_chunk_data['summary'], list) and len(cached_chunk_data['summary']) > 0:
                            # Path 1: Valid cache found. Use it and DO NOT add to the list.
                            chunk_entities[chunk_number]['summary'] = cached_chunk_data['summary']
                            output_queue.put(chunk_entities[chunk_number])
                            print(f"\nFound existing summary cache for chunk {chunk_number}, returning cached data...\n")
                        else:
                            # Path 2: Cache found but is invalid. Add to list for reprocessing.
                            new_chunks_for_llm.append(chunk_number)
                            print(f"\nCached data for chunk {chunk_number} from document {source_doc_name} failed validation, proceeding to generate summary afresh...\n")
                    else:
                        # Path 3: No cache entry exists for this chunk. Add to list.
                        new_chunks_for_llm.append(chunk_number)
                
                except Exception as e:
                    # Path 4: A generic error occurred. Default to reprocessing for safety.
                    handle_error_no_return(f"Error processing cache for chunk {chunk_number} of document {source_doc_name}, will be processed afresh for safety. Encountered error: ", e)
                    new_chunks_for_llm.append(chunk_number)
                
            return new_chunks_for_llm

        try:
            # --- PHASE 1: DETERMINE FULL WORKLOAD ---
            entries_to_process = list(chunk_entities.keys())    # Default to all chunks entries

            if reuse_graph_summary_cache:
                try:
                    entries_to_process = load_and_fire_off_cache()  # If any entries were previously cached, those are returned and only new entries are processed by the LLM
                except Exception as e:
                    handle_error_no_return(f"Error processing cache, proceeding to process all chunks afresh. Encountered error: ", e)
                    entries_to_process = list(chunk_entities.keys())
            
            # --- PHASE 2 & 3: BATCH AND EXECUTE ---
            total_entry_count = len(entries_to_process)
            processed_entries = 0
            print(f"\nBeginning LLM processing for {total_entry_count} chunks in batches of {BATCH_SIZE}...\n")

            for i in range(0, len(entries_to_process), BATCH_SIZE):
                batch_keys = entries_to_process[i:i + BATCH_SIZE]   # Get the keys for the current batch

                batch_chunk_entities = {key: chunk_entities[key] for key in batch_keys} # CRITICAL STEP: Create a small, temporary dictionary for this batch ONLY - Details in exl2_graph_extractor() above!

                print(f"\nProcessing batch {i//BATCH_SIZE + 1} of {len(entries_to_process)//BATCH_SIZE + 1}...\n")

                for chunk_number, chunk_data in batch_chunk_entities.items():

                    try:
                        source_doc_name = os.path.splitext(os.path.basename(chunk_data['source_doc_name']))[0]

                        print(f"\nAttempting to generate a summary report for chunk {chunk_number} of document {source_doc_name}...processing item {processed_entries + 1} of total {total_entry_count} items...\n")

                        if chunk_data['entities_and_relationships']['nodes'] == [] and chunk_data['entities_and_relationships']['relationships'] == []:
                            print(f"\nNo nodes or relationships to summarize for chunk {chunk_number} of document {source_doc_name}, skipping summary generation...\n")
                            chunk_entities[chunk_number]['summary'] = []
                            output_queue.put(chunk_entities[chunk_number])
                            continue

                        chunk_summary = process_nodes_and_relationships(
                            nodes_and_relationships=chunk_data['entities_and_relationships'],
                            chunk_text=chunk_data['chunk_text'],
                            source_doc_name=source_doc_name,
                            page_number_list=chunk_data['page_number'],
                            requested_max_new_tokens=requested_max_new_tokens,
                            gen_settings=gen_settings
                        )
                        
                        chunk_entities[chunk_number]['summary'] = chunk_summary
                        output_queue.put(chunk_entities[chunk_number])
                        cache_queue.put({chunk_number: chunk_entities[chunk_number]})
                        
                        processed_entries += 1
                    
                    except Exception as e:
                        handle_error_no_return(f"Could not generate summary for nodes and relationships for chunk {chunk_number} from document {source_doc_name}, skipping. Encountered error: ", e)
                        chunk_entities[chunk_number]['summary'] = []
                        output_queue.put(chunk_entities[chunk_number])
                        cache_queue.put({chunk_number: chunk_entities[chunk_number]})
                        continue

        except Exception as e:
            handle_error_no_return("Summary generation failed, encountered error: ", e)
        finally:
            output_queue.put(None)
            cache_queue.put(None)
            print("\n\nLLM stream done, releasing semaphore\n\n")   # TODO: investigate hanging here - likely caused by (previously) uncaught exception in `create-and_execute_exl2_job` that led to unexpected behavior. Added error-handling, ready for re-test.
            llm_semaphore.release()
            stop_thread.set()


    def save_to_local_cache(line):
        try:
            for chunk_number, chunk_data in line.items():
                source_doc_name = os.path.splitext(os.path.basename(chunk_data['source_doc_name']))[0]
                summary_cache_file_path = os.path.join(knowledge_graph_cache_dir, f"{source_doc_name}_summary_cache.json")
                update_and_save_json_file({chunk_number: chunk_data}, summary_cache_file_path)
                print(f"\nSaved generated summary for chunk {chunk_number} of document {source_doc_name} to cache file at path {summary_cache_file_path}\n")
        except Exception as e:
            handle_error_no_return(f"Could not cache summary for document {source_doc_name} to cache file at path {summary_cache_file_path}, skipping. Encountered error: ", e)


    def caching_thread():
        while True:
            line =  cache_queue.get()
            if line is None:
                print("\nNone read, breaking and stopping cache-thread\n")
                break
            save_to_local_cache(line)
        
        print("\n\nCaching complete\n\n")


    def generate():
        while True:
            line = output_queue.get()
            if line is None:
                print("\nNone read, breaking and stopping task-thread\n")
                break
            yield f"data: {json.dumps(line)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"

        print("\n/exl2-graph-summarizer done\n")

    task_thread = threading.Thread(target=summary_generation_task)
    task_thread.start()

    cache_thread = threading.Thread(target=caching_thread)
    cache_thread.start()

    print("\n\nInferencing Begins!\n\n")
    return Response(generate(), content_type='text/event-stream')


######################################################----End Exl2-Graph Logic----########################################################


@app.route('/health')
def health():

    def throw_health_check_error(attrib, e):
        print(f"\nHF-Waitress Server online but could not determine LLM's '{attrib}' attribute. Continuing...\n")
        return True

    print("\n\nHF-Waitress LLM health-check in-progress...\n\n")
    
    with reader_semaphore:
    
        try:
            if PIPE is None and (EXL2_MODEL is None or EXL2_CACHE is None or EXL2_TOKENIZER is None or AUTO_TOKENIZER is None):
                return jsonify(status="error", message="Model not loaded"), 503 # Service Unavailable
            
            model_info = {}

            # print(f"\n\nmodel details: {PIPE.model}\n\n")
            # print(f"\n\nmodel.config details: {PIPE.model.config}\n\n")
            # print(f"\n\ntokenizer details: {PIPE.tokenizer}\n\n")
            
            try:
                model_info["model_id"] = str(PIPE.model.config._name_or_path)
            except Exception as e:
                throw_health_check_error("model_id", e)

            try:
                model_info["transformers_version"] = str(PIPE.model.config.transformers_version)
            except Exception as e:
                throw_health_check_error("transformers_version", e)

            try:
                model_info["architecture"] = str(PIPE.model.config.architectures)
            except Exception as e:
                throw_health_check_error("model architecture", e)

            try:
                model_info["model_type"] = str(PIPE.model.config.model_type)
            except Exception as e:
                throw_health_check_error("model_type", e)

            try:
                model_info["torch_dtype"] = str(PIPE.model.config.torch_dtype)
            except Exception as e:
                throw_health_check_error("torch_dtype", e)

            try:
                model_info["device"] = str(PIPE.device)
            except Exception as e:
                throw_health_check_error("inference device", e)

            try:
                if hasattr(PIPE.model.config, "quantization_config"):
                    model_info["is_quantized"] = True
                    model_info["quant_method"] = str(PIPE.model.config.quantization_config.quant_method)
                    model_info["quantization_config"] = str(PIPE.model.config.quantization_config)
                else:
                    model_info["is_quantized"] = False
            except Exception as e:
                throw_health_check_error("quantization status", e)

            try:
                model_info["memory_footprint"] = str(PIPE.model.get_memory_footprint())
            except Exception as e:
                throw_health_check_error("memory_footprint", e)

            try:
                model_info["model_vocab_size"] = str(PIPE.model.config.vocab_size)
            except Exception as e:
                throw_health_check_error("model_vocab_size", e)
                try:
                    model_info["tokenizer_vocab_length"] = len(PIPE.tokenizer)
                except Exception as e:
                    throw_health_check_error("tokenizer_vocab_length", e)
            
            try:
                model_info["tokenizer_vocab_size"] = str(PIPE.tokenizer.vocab_size)
            except Exception as e:
                throw_health_check_error("tokenizer_vocab_size", e)
            
            try:
                model_info["number_of_hidden_layers"] = str(PIPE.model.config.num_hidden_layers)
            except Exception as e:
                throw_health_check_error("number_of_hidden_layers", e)
            
            try:
                model_info["number_of_attention_heads"] = str(PIPE.model.config.num_attention_heads)
            except Exception as e:
                throw_health_check_error("number_of_attention_heads", e)

            try:
                model_info["hidden_dimensions"] = str(PIPE.model.config.head_dim)
            except Exception as e:
                throw_health_check_error("hidden_dimensions", e)

            try:
                model_info["number_of_key_value_heads"] = str(PIPE.model.config.num_key_value_heads)
            except Exception as e:
                throw_health_check_error("number_of_key_value_heads", e)
            
            try:
                model_info["hidden_activation"] = str(PIPE.model.config.hidden_act)
            except Exception as e:
                throw_health_check_error("hidden_activation", e)
            
            try:
                model_info["hidden_size"] = str(PIPE.model.config.hidden_size)
            except Exception as e:
                throw_health_check_error("hidden_size", e)

            try:
                model_info["intermediate_size"] = str(PIPE.model.config.intermediate_size)
            except Exception as e:
                throw_health_check_error("intermediate_size", e)

            try:
                model_info["max_position_embeddings"] = str(PIPE.model.config.max_position_embeddings)
            except Exception as e:
                throw_health_check_error("max_position_embeddings", e)

            try:
                model_info["tokenizer"] = str(PIPE.tokenizer.name_or_path)
            except Exception as e:
                throw_health_check_error("tokenizer", e)

            try:
                model_info["max_seq_length"] = str(PIPE.tokenizer.model_max_length)
            except Exception as e:
                throw_health_check_error("max_seq_length", e)

            print(f"HF-Waitress LLM-server health-check completed successfully, returning.\n")
            return jsonify(status="ok", model_info=model_info), 200

        except Exception as e:
            handle_api_error("Error checking hf-server health, encountered error: ", e)


@app.route('/restart_server')
def restart_server():
    
    with llm_semaphore:
        print("\n\nrestart-server acquired llm_semaphore, proceeding...\n\n")
        with config_writer_semaphore:
            print("\n\nrestart-server acquired config_writer_semaphore, proceeding...\n\n")
            with error_logging_semaphore:
                print("\n\nrestart-server acquired error_logging_semaphore, proceeding...\n\n")

                try:
                    shutdown_vision_model()
                    shutdown_pipe()
                    shutdown_exl2()
                    safe_empty_cuda_cache()

                    initialize_model()
                except Exception as e:
                    handle_api_error("Could not restart server, encountered error: ", e)
                
                return jsonify(success=True)


@app.route('/stop_generation')
def stop_generation():
    global STOP_GENERATION
    STOP_GENERATION = True
    return jsonify(success=True)


def get_host_and_port():
    try:
        read_return = read_config(['host', 'port'])
        host = str(read_return['host'])
        port = int(read_return['port'])
        return host, port
    except Exception as e:
        handle_error_no_return("Could not get host and port from hf_config.json, encountered error: ", e)


if __name__ == '__main__':
    _ = parse_arguments()
    initialize_model()
    host, port = get_host_and_port()
    serve(app, host=host, port=port)