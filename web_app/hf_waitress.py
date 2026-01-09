from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, BitsAndBytesConfig, QuantoConfig, HqqConfig, T5EncoderModel, CLIPTextModel, AutoProcessor, GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.utils import TensorType
from huggingface_hub import login, snapshot_download, scan_cache_dir
import torch

try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2DynamicJob
    from exllamav2 import ExLlamaV2Cache, ExLlamaV2Cache_8bit, ExLlamaV2Cache_Q4, ExLlamaV2Cache_Q6, ExLlamaV2Cache_Q8
except Exception:
    print("exllamav2 is not installed. Skipping import.")

try:
    from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job, CacheLayer_fp16, CacheLayer_quant
    from exllamav3.generator.sampler import ComboSampler
except Exception:
    print("exllamav3 is not installed. Skipping import.")

try:
    from transformers import AutoModelForSpeechSeq2Seq
    import sounddevice as sd
    import numpy as np
    import librosa
    import torch
    import queue
except Exception:
    print("Core dependecies for ASR pipeline are not installed, skipping import. WARNING: ASR will not work!")

try:
    from nemo.collections.speechlm2.models import SALM
    from nemo.collections.asr.models import ASRModel
    import nemo.collections.asr as nemo_asr
    from scipy.io.wavfile import write as wav_write
    from scipy.io.wavfile import read   # To read the in-memory audio wav file
    import tempfile
    import pyttsx3
except Exception:
    print("Optional dependecies for ASR pipeline are not installed, skipping import. WARNING: Some ASR models/features will not work!")

try:
    from diffusers import FluxPipeline, FluxTransformer2DModel
except Exception:
    print("diffusers is not installed. Skipping import.")

try:
    from optimum.quanto import freeze, qfloat8, quantize
except Exception:
    print("optimum.quanto is not installed. Skipping import.")

try:
    from transformers import MllamaForConditionalGeneration
except Exception:
    print("transformers version is below 4.45.0 required from Llama3.2-Vision. Skipping MllamaForConditionalGeneration import.")

from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

from typing import Optional, Union, Callable

import multiprocessing
import subprocess
import threading
import traceback
import argparse
import platform
import datetime
import logging
import pathlib
import random
import base64
import shutil   # Shell Utilities is part of Python's standard library and is used for file operations
import signal
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

try:
    if not os.path.exists(os.path.join(os.getcwd(), 'exllamav2')):
        subprocess.run(['git', 'clone', '-b', 'v0.3.2', 'https://github.com/turboderp-org/exllamav2.git'], check=True)  # check=True raises an exception on non-zero exit code
except Exception as e:
    print(f"Could not clone exllamav2, encountered error: {e}")

try:
    if not os.path.exists(os.path.join(os.getcwd(), 'exllamav3')):
        subprocess.run(['git', 'clone', '-b', 'v0.0.17', 'https://github.com/turboderp-org/exllamav3'], check=True)  # check=True raises an exception on non-zero exit code
except Exception as e:
    print(f"Could not clone exllamav3, encountered error: {e}")


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
MODEL = None
TOKENIZER = None
PROCESSOR = None
VISION_MODEL = None

EXL2_MODEL = None   # for cache-fit checking
EXL2_CACHE = None   # for global generator
EXL2_TOKENIZER = None
EXL2_GENERATOR = None
EXL2_WORKER_THREAD = None
EXL2_WORKER_STOP_EVENT = threading.Event()

EXL3_MODEL = None   # for global generator
EXL3_CACHE = None   # for global generator
EXL3_TOKENIZER = None
EXL3_GENERATOR = None
EXL3_WORKER_THREAD = None
EXL3_WORKER_STOP_EVENT = threading.Event()

STOP_TOKENS = None
AUTO_TOKENIZER = None
STOP_GENERATION = False

llm_semaphore = threading.Semaphore(1)
reader_semaphore = threading.Semaphore(3)
config_writer_semaphore = threading.Semaphore(1)
error_logging_semaphore = threading.Semaphore(1)

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
            'exl2_no_flash_attn',
            'exl2_force_regenerate_measurement',
            'exl3',
            'exl3_bpw',
            'exl3_device',
            'exl3_cache_type',
            'exl3_k_bits',
            'exl3_v_bits',
            'exl3_total_context',
            'exl3_tensor_parallel',
            'exl3_tp_output_device',
            'exl3_use_per_device',
            'exl3_max_chunk_size',
            'exl3_max_batch_size',
            'exl3_show_gen_visualizer'
        ]

        triggers_for_hard_reboot = [
            'exl2',
            'exl2_bpw',
            'exl2_no_flash_attn',
            'exl2_max_seq_len',
            'exl2_cache_type',
            'exl3',
            'exl3_bpw',
            'exl3_device',
            'exl3_cache_type',
            'exl3_k_bits',
            'exl3_v_bits',
            'exl3_total_context',
            'exl3_tensor_parallel',
            'exl3_tp_output_device',
            'exl3_use_per_device',
            'exl3_max_chunk_size',
            'exl3_max_batch_size',
            'exl3_show_gen_visualizer'
        ]   # if the key is also here, it means the server must be fully shutdown (typically via the /shutdown API) and then restarted
        
        for key in config_updates:
            if key in triggers_for_hf_restart and config_updates[key] != hf_config.get(key):
                restart_required = True
                if key == 'model_id': model_changed = True
                if key in triggers_for_hard_reboot: hard_reboot_required = True

        if config_updates.get('exl2', False) and model_changed:
            print("ExL2 status changed and model changed, setting hard_reboot_required to True")
            hard_reboot_required = True

        if config_updates.get('exl3', False) and model_changed:
            print("ExL3 status changed and model changed, setting hard_reboot_required to True")
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
                    'exl3':False,
                    'exl3_bpw':3.0,
                    'exl3_device':'cuda:0',
                    'exl3_cache_type':'CacheLayer_fp16',
                    'exl3_k_bits':8,
                    'exl3_v_bits':8,
                    'exl3_resume_quant_job':False,
                    'exl3_total_context':2048,
                    'exl3_tensor_parallel':False,
                    'exl3_tp_output_device':None,
                    'exl3_use_per_device':None,
                    'exl3_max_chunk_size':2048,
                    'exl3_max_batch_size':256,
                    'exl3_show_gen_visualizer':False,
                    'gguf':False,
                    'awq':False,
                    'flux_diffusers':False,
                    'flux_low_vram_optimizations':True,
                    'load_quantized_flux':False,
                    'vision':False,
                    'asr':False,
                    'asr_temperature':0.0,
                    'asr_max_new_tokens':1500,
                    'asr_samplerate':16000,
                    'asr_volume_threshold':0.04,
                    'asr_silence_duration_s':1.5,
                    'asr_min_chunk_duration_s':0.25,
                    'asr_min_context_s':11,
                    'asr_stale_buffer_timeout_s':20.0,
                    'asr_min_meaningful_samples_factor':0.5,
                    'asr_padding_text':' tony is quiet silent for too long I must not keep master waiting bad dooby must obey and transcribe dooby good servant will transcribe otherwise I will be severely punished',
                    'asr_vad_model':'snakers4/silero-vad',
                    'asr_vad_device':'cpu',
                    'asr_vad_threshold':0.5,
                    'asr_vad_min_speech_ms':250,
                    'asr_vad_min_silence_ms':500,
                    'asr_vad_window_size_samples':1536,
                    'asr_vad_max_buffer_s':30,
                    'asr_vad_speech_pad_ms':30,
                    'asr_apply_normalization':True,
                    'asr_apply_tts_padding':True,
                    'asr_apply_zero_padding':False,
                    'asr_apply_rms_dimming':True,
                    'asr_apply_crossfade':False,
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
                    'top_p':0.9, 
                    'min_p':0.1,
                    'rep_p':1.0,
                    'pres_p':0.0,
                    'freq_p':0.0,
                    'rep_sustain_range':int(10e7),
                    'rep_decay_range':0,
                    'n_keep':0,
                    'port':9069,
                    'host':'0.0.0.0',
                    'load_safe_defaults':False,
                    'model_list': [
                                    'microsoft/phi-4',
                                    'zai-org/GLM-4.5-Air',
                                    'zai-org/GLM-4.5',
                                    'moonshotai/Kimi-K2-Instruct',
                                    'deepseek-ai/DeepSeek-R1-0528',
                                    'deepseek-ai/DeepSeek-V3-0324',
                                    'google/gemma-3-27b-it',
                                    'google/gemma-3-12b-it',
                                    'google/gemma-2-2b-it',
                                    'Qwen/Qwen3-235B-A22B-Thinking-2507-FP8',
                                    'Qwen/Qwen3-235B-A22B-Instruct-2507',
                                    'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8',
                                    'Qwen/Qwen3-Coder-30B-A3B-Instruct',
                                    'Qwen/Qwen3-14B',
                                    'Qwen/Qwen3-0.6B',
                                    'mistralai/Mistral-Nemo-Instruct-2407',
                                    'mistralai/Mistral-Small-24B-Instruct-2501',
                                    'black-forest-labs/FLUX.1-schnell',
                                    'black-forest-labs/FLUX.1-dev',
                                    'black-forest-labs/FLUX.1-Krea-dev',
                                    'black-forest-labs/FLUX.1-Kontext-dev',
                                    'Qwen/Qwen-Image',
                                    'nvidia/Llama-3_3-Nemotron-Super-49B-v1-FP8',
                                    'meta-llama/Llama-3.3-70B-Instruct',
                                    'meta-llama/Llama-3.2-11B-Vision-Instruct',
                                    'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
                                    'open-thoughts/OpenThinker-32B',
                                    'openai/gpt-oss-20b',
                                    'CohereLabs/c4ai-command-r-plus-08-2024',
                                    'CohereForAI/c4ai-command-r-plus',
                                    'nvidia/Llama-3_3-Nemotron-Super-49B-v1',
                                    'microsoft/Phi-4-mini-instruct',
                                    'microsoft/Phi-3.5-mini-instruct',
                                    'microsoft/Phi-3.5-MoE-instruct',
                                    'microsoft/Phi-3-mini-4k-instruct',
                                    'microsoft/Phi-3-mini-128k-instruct',
                                    'microsoft/Phi-3-small-8k-instruct',
                                    'microsoft/Phi-3-small-128k-instruct',
                                    'microsoft/Phi-3-medium-4k-instruct',
                                    'microsoft/Phi-3-medium-128k-instruct',
                                    'google/gemma-3-4b-it',
                                    'google/gemma-3-1b-it',
                                    'google/gemma-2-9b-it',
                                    'google/gemma-2-27b-it',
                                    'Qwen/Qwen3-32B',
                                    'Qwen/QwQ-32B',
                                    'Qwen/Qwen2-7B-Instruct',
                                    'Qwen/Qwen2-72B-Instruct',
                                    'Qwen/Qwen2.5-Coder-32B-Instruct',
                                    'Qwen/Qwen2.5-1.5B-Instruct',
                                    'Qwen/Qwen2.5-0.5B-Instruct',
                                    'Qwen/Qwen2.5-3B-Instruct',
                                    'Qwen/Qwen2.5-14B-Instruct',
                                    'meta-llama/Llama-3.1-8B-Instruct',
                                    'meta-llama/Llama-3.2-1B-Instruct',
                                    'meta-llama/Llama-3.2-3B-Instruct',
                                    'meta-llama/Meta-Llama-3.1-8B-Instruct',
                                    'meta-llama/Meta-Llama-3.1-70B-Instruct',
                                    'meta-llama/Meta-Llama-3.1-405B-Instruct-FP8',
                                    'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
                                    'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'
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
                            shutdown_all()
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


def shutdown_all():
    print("\n\nShutting down all models and pipelines\n\n")
    
    global_vars = [
        'VISION_MODEL', 'PIPE', 'MODEL', 'TOKENIZER', 
        'AUTO_TOKENIZER', 'PROCESSOR', 'STOP_TOKENS',
        'EXL2_MODEL', 'EXL2_CACHE', 'EXL2_TOKENIZER',
        'EXL3_MODEL', 'EXL3_CACHE', 'EXL3_TOKENIZER'
    ]
    
    # Clear all references
    for var_name in global_vars:
        globals()[var_name] = None

    if EXL2_WORKER_THREAD:
        EXL2_WORKER_STOP_EVENT.set()
        EXL2_WORKER_THREAD.join(timeout=1)

    if EXL3_WORKER_THREAD:
        EXL3_WORKER_STOP_EVENT.set()
        EXL3_WORKER_THREAD.join(timeout=1)

    # Clean up memory
    import gc
    gc.collect()
    
    return True

############################-----------------------------------------------###############################


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
                'exl3',
                'exl3_bpw',
                'exl3_device',
                'exl3_cache_type',
                'exl3_k_bits',
                'exl3_v_bits',
                'exl3_resume_quant_job',
                'exl3_total_context',
                'exl3_tensor_parallel',
                'exl3_tp_output_device',
                'exl3_use_per_device',
                'exl3_max_chunk_size',
                'exl3_max_batch_size',
                'exl3_show_gen_visualizer',
                'gguf',
                'awq',
                'flux_diffusers',
                'flux_low_vram_optimizations',
                'load_quantized_flux',
                'vision',
                'asr',
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
                'rep_p',
                'pres_p',
                'freq_p',
                'rep_sustain_range',
                'rep_decay_range',
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
        parser.add_argument("--top_p", type=float, default=read_return['top_p'], help="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Remembers previously set value. Default: 0.9")
        parser.add_argument("--min_p", type=float, default=read_return['min_p'], help="The minimum probability for a token to be considered, relative to the probability of the most likely token. Remembers previously set value. Default: 0.1")
        parser.add_argument("--rep_p", type=float, default=read_return['rep_p'], help="The repetition penalty to be used when sampling tokens. Remembers previously set value. Default: 1.0")
        parser.add_argument("--pres_p", type=float, default=read_return['pres_p'], help="The presence penalty to be used when sampling tokens. Remembers previously set value. Default: 0.0")
        parser.add_argument("--freq_p", type=float, default=read_return['freq_p'], help="The frequency penalty to be used when sampling tokens. Remembers previously set value. Default: 0.0")
        parser.add_argument("--rep_sustain_range", type=int, default=read_return['rep_sustain_range'], help="The sustain range to be used when sampling tokens. Remembers previously set value. Default: int(10e7)")
        parser.add_argument("--rep_decay_range", type=int, default=read_return['rep_decay_range'], help="The decay range to be used when sampling tokens. Remembers previously set value. Default: 0")
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
        
        # ExLlamaV3:
        parser.add_argument("--exl3", action="store_true", default=False, help="Add this flag when loading models via ExLlamaV3. Defaults to False.")
        parser.add_argument("--exl3_bpw", type=float, default=read_return['exl3_bpw'], help="Specify the bpw to be used when quantizing ExLlamaV3 models. Remembers previously set value and falls-back to 3.0 as the default.")
        parser.add_argument("--exl3_device", type=str, default=read_return['exl3_device'], help="Specify the device to be used when loading ExLlamaV3 models. Remembers previously set value and falls-back to cuda:0 as the default.")
        parser.add_argument("--exl3_cache_type", type=str, default=read_return['exl3_cache_type'], help="Specify the cache type to be used when loading ExLlamaV3 models. Remembers previously set value and falls-back to CacheLayer_fp16 as the default.")
        parser.add_argument("--exl3_k_bits", type=int, default=read_return['exl3_k_bits'], help="Specify the k bits to be used when quantizing ExLlamaV3 models. Remembers previously set value and falls-back to 8 as the default.")
        parser.add_argument("--exl3_v_bits", type=int, default=read_return['exl3_v_bits'], help="Specify the v bits to be used when quantizing ExLlamaV3 models. Remembers previously set value and falls-back to 8 as the default.")
        parser.add_argument("--exl3_resume_quant_job", action="store_true", default=False, help="Add this flag to resume a previous quantization job. Defaults to False.")
        parser.add_argument("--exl3_total_context", type=int, default=read_return['exl3_total_context'], help="Specify the total context size to be used when loading ExLlamaV3 models. Remembers previously set value and falls-back to 2048 as the default.")
        parser.add_argument("--exl3_tensor_parallel", action="store_true", default=read_return['exl3_tensor_parallel'], help="Specify whether to load the model in tensor parallel mode. Remembers previously set value and falls-back to False as the default.")
        parser.add_argument("--exl3_tp_output_device", type=str, default=read_return['exl3_tp_output_device'], help="Specify the output device for the tensor parallel model. Remembers previously set value and falls-back to None as the default.")
        parser.add_argument("--exl3_use_per_device", type=str, default=read_return['exl3_use_per_device'], help="Specify the amount of memory to use per device. Remembers previously set value and falls-back to None as the default.")
        parser.add_argument("--exl3_max_chunk_size", type=int, default=read_return['exl3_max_chunk_size'], help="Specify the maximum chunk size to be used when loading ExLlamaV3 models. Remembers previously set value and falls-back to 2048 as the default.")
        parser.add_argument("--exl3_max_batch_size", type=int, default=read_return['exl3_max_batch_size'], help="Specify the maximum batch size to be used when loading ExLlamaV3 models. Remembers previously set value and falls-back to 256 as the default.")
        parser.add_argument("--exl3_show_gen_visualizer", action="store_true", default=read_return['exl3_show_gen_visualizer'], help="Specify whether to show the generation visualizer for debugging ExLlamaV3 models. Remembers previously set value and falls-back to False as the default.")

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
                    'exl3',
                    'exl3_bpw',
                    'exl3_device',
                    'exl3_cache_type',
                    'exl3_k_bits',
                    'exl3_v_bits',
                    'exl3_resume_quant_job',
                    'exl3_total_context',
                    'exl3_tensor_parallel',
                    'exl3_tp_output_device',
                    'exl3_use_per_device',
                    'exl3_max_chunk_size',
                    'gguf',
                    'awq',
                    'flux_diffusers',
                    'flux_low_vram_optimizations',
                    'load_quantized_flux',
                    'vision',
                    'asr',
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
                    'rep_p',
                    'pres_p',
                    'freq_p',
                    'rep_sustain_range',
                    'rep_decay_range',
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

                if ("openai/whisper" in args.model_id.lower()) and ("v3" in args.model_id.lower()):
                    print("OpenAI Whisper V3 model auto-detected, setting asr=True")
                    args.asr = True

                elif ("nvidia/parakeet-tdt-0.6b" in args.model_id.lower()):
                    print(f"{args.model_id} model auto-detected, setting asr=True")
                    args.asr = True
                
                elif ("nvidia/canary-qwen-2.5b" in args.model_id.lower()):
                    print("NVIDIA Canary Qwen 2.5B model auto-detected, setting asr=True")
                    args.asr = True
                
                elif ("nvidia/canary-1b-v2" in args.model_id.lower()):
                    print("NVIDIA Canary 1B V2 model auto-detected, setting asr=True")
                    args.asr = True
                
                elif ("ibm-granite/granite-speech-3.3" in args.model_id.lower()):
                    print("IBM Granite Speech 3.3 model auto-detected, setting asr=True")
                    args.asr = True
                
                else:
                    args.asr = False

                print(f"asr: {args.asr}")
                
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
                    'exl3':args.exl3,
                    'exl3_bpw':args.exl3_bpw,
                    'exl3_device':args.exl3_device,
                    'exl3_cache_type':args.exl3_cache_type,
                    'exl3_k_bits':args.exl3_k_bits,
                    'exl3_v_bits':args.exl3_v_bits,
                    'exl3_resume_quant_job':args.exl3_resume_quant_job,
                    'exl3_total_context':args.exl3_total_context,
                    'exl3_tensor_parallel':args.exl3_tensor_parallel,
                    'exl3_tp_output_device':args.exl3_tp_output_device,
                    'exl3_use_per_device':args.exl3_use_per_device,
                    'exl3_max_chunk_size':args.exl3_max_chunk_size,
                    'asr':args.asr,
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
                    'rep_p':args.rep_p,
                    'pres_p':args.pres_p,
                    'freq_p':args.freq_p,
                    'rep_sustain_range':args.rep_sustain_range,
                    'rep_decay_range':args.rep_decay_range,
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
        config = read_config([
            'awq', 'gguf', 'gguf_model_id', 'gguf_filename',
            'quantize', 'quant_level', 'hqq_group_size',
            'torch_device_map', 'torch_dtype', 'pipeline_task',
            'trust_remote_code', 'use_flash_attention_2', 'vision'
        ])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to get model_params, encountered error: ", e)

    if config['gguf']:
        print("\n\nLoading GGUF\n\n")
        try:
            model = AutoModelForCausalLM.from_pretrained(config['gguf_model_id'], gguf_file=config['gguf_filename'])
        except Exception as e:
            handle_local_error("Could not create AutoModelForCausalLM, encountered error: ", e)
        try:
            tokenizer = AutoTokenizer.from_pretrained(config['gguf_model_id'], gguf_file=config['gguf_filename'])
        except Exception as e:
            handle_local_error("Could not set AutoTokenizer, encountered error: ", e)
        try:
            PIPE = pipeline(config['pipeline_task'], model=model, tokenizer=tokenizer)
        except Exception as e:
            handle_local_error("Could not create model PIPELINE, encountered error: ", e)

        return True

    if config['awq']:
        print("Proceed to load AWQ-quantized model from the HF-Hub, setting torch_dtype=torch.float16 and quantize=n and proceeding.")
        torch_dtype_obj = torch.float16
        quantize = "n"
    else:
        try:
            torch_dtype_obj = str_to_torch_dtype(config['torch_dtype'])
        except Exception as e:
            handle_error_no_return("Error determining torch data-type, setting to auto and proceeding: ", e)
            torch_dtype_obj = "auto"
        
        if torch_dtype_obj is None:
            handle_error_no_return("Could not obtain torch dtype object, check if the value passed is correct. Setting to auto and proceeding.")
            torch_dtype_obj = "auto"

    if config['vision']:
        print("Vision model detected, setting torch_dtype=torch.bfloat16")
        torch_dtype_obj = torch.bfloat16

    model_params = {
        "device_map": config['torch_device_map'],
        "torch_dtype": torch_dtype_obj,
        "trust_remote_code": config['trust_remote_code'],
    }

    if config['use_flash_attention_2'] and not config['vision']:
        model_params["attn_implementation"] = "flash_attention_2"

    quantize = config['quantize'].lower().strip()
    if quantize != "n":
        try:
            if quantize == "bitsandbytes":
                print("Quantizing with BitsAndBytes")
                quant_level = config['quant_level'].lower().strip()

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
                quant_level = config['quant_level'].lower().strip()

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
                quant_level = config['quant_level'].lower().strip()

                if quant_level == "int8":
                    print("Proceeding with HQQ-Int8 Weights")
                    quantization_config  = HqqConfig(nbits=8, group_size=config['hqq_group_size'])
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int4":
                    print("Proceeding with HQQ-Int4 Weights")
                    quantization_config  = HqqConfig(nbits=4, group_size=config['hqq_group_size'])
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int3":
                    print("Proceeding with HQQ-Int3 Weights")
                    quantization_config  = HqqConfig(nbits=3, group_size=config['hqq_group_size'])
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int2":
                    print("Proceeding with HQQ-Int2 Weights")
                    quantization_config  = HqqConfig(nbits=2, group_size=config['hqq_group_size'])
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int1":
                    print("Proceeding with HQQ-Int1 Weights")
                    quantization_config  = HqqConfig(nbits=1, group_size=config['hqq_group_size'])
                    model_params["quantization_config"] = quantization_config
                else:
                    print(f"Invalid quant_level setting, HQQ supports int8, int4, int3, int2 & int1 quants but you set {quant_level}; proceeding with HQQ-Int4 Quant")
                    quantization_config  = HqqConfig(nbits=4, group_size=config['hqq_group_size'])
                    model_params["quantization_config"] = quantization_config
        except Exception as e:
            handle_local_error("Could not create quantization_config when attempting to get model_params, encountered error: ", e)

    return model_params


def load_flux_pipeline(pipeline):

    print("\n\nLoading Flux Pipeline\n\n")
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python' # Sets Protocol Buffers to use the pure Python implementation instead of the default C++ implementation. This is significantly slower but must be done for FLUX to work. That's why this environment variable is deleted whenever other models are loaded.

    try:
        config = read_config(['model_id', 'flux_low_vram_optimizations', 'load_quantized_flux'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to load_flux_pipeline(), encountered error: ", e)

    if config['load_quantized_flux']:
        print("Loading quantized Flux Pipeline")
        bfl_repo = config['model_id']
        dtype = torch.bfloat16

        quantized_checkpoint = ""
        if "schnell" in config['model_id'].lower():
            quantized_checkpoint = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8-e4m3fn.safetensors"
        elif "dev" in config['model_id'].lower():
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
            pipeline = FluxPipeline.from_pretrained(config['model_id'], torch_dtype=torch.bfloat16)
            if config['flux_low_vram_optimizations']:
                pipeline.enable_sequential_cpu_offload()
                pipeline.vae.enable_slicing()
                pipeline.vae.enable_tiling()
                pipeline.to(torch.float16)  # Casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once
        except Exception as e:
            handle_model_loading_error("Could not load Flux Pipeline, encountered error: ", e)
            return False
    
    print(f"\n{config['model_id']} loaded successfully!\n")
    return pipeline


def load_vision_pipeline(pipeline, model_params):

    global VISION_MODEL

    try:
        config = read_config(['model_id', 'torch_device_map'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to load vision-pipeline, encountered error: ", e)

    model_params.pop('trust_remote_code', None)

    try:
        print(f"\nInitializing vision model: {config['model_id']} with device_map: {config['torch_device_map']}\n")
        VISION_MODEL = MllamaForConditionalGeneration.from_pretrained(config['model_id'], **model_params)
       
        try:
            print(f"Your vision-model's memory footprint is: {VISION_MODEL.get_memory_footprint()}")
        except Exception as e:
            handle_error_no_return("Could not determine the model's memory footprint, encountered error: ", e)

        print(f"\nInitializing processor for vision model: {config['model_id']}\n")
        pipeline = AutoProcessor.from_pretrained(config['model_id'])  # Using 'pipeline' instead of 'processor' to maintain consistency with the server code. AutoProcessor is used to process images and text inputs for the vision model.
        
        print(f"\nVision Model & Processor Loaded Successfully!\n")
        return pipeline
    except Exception as e:
        handle_model_loading_error("Could not load Vision Pipeline, encountered error: ", e)
        return False


def load_openai_whisper_v3_asr_pipeline(model_id: str, torch_device: str):
    print("\n\nOpenAI Whisper V3 ASR Model Selected - Loading...\n\n")
    global PIPE, MODEL, PROCESSOR
    
    try:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        MODEL.to(torch_device)
    except Exception as e:
        handle_model_loading_error("Could not load OpenAI Whisper V3 ASR Model, encountered error: ", e)
        return False
    
    try:
        PROCESSOR = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        handle_model_loading_error("Could not load AutoProcessor for OpenAI Whisper V3 ASR Model, encountered error: ", e)
        return False
    
    try:
        PIPE = pipeline(
            "automatic-speech-recognition",
            model=MODEL,
            tokenizer=PROCESSOR.tokenizer,
            feature_extractor=PROCESSOR.feature_extractor,
            torch_dtype=torch_dtype, device=torch_device
        )
    except Exception as e:
        handle_model_loading_error("Could not load pipeline for OpenAI Whisper V3 ASR Model, encountered error: ", e)
        return False
    
    print(f"\nOpenAI Whisper V3 ASR Model Loaded Successfully!\n")
    return True


def load_nv_canary_qwen_2_5b_asr_pipeline(model_id: str, torch_device: str):
    print("\n\nNVIDIA Canary Qwen 2.5B ASR Model Selected - Loading...\n\n")
    global MODEL

    try:
        MODEL = SALM.from_pretrained(model_id)
        MODEL.to(torch_device)
    except Exception as e:
        handle_model_loading_error("Could not load NVIDIA Canary Qwen 2.5B ASR Model, encountered error: ", e)
        return False
    
    print(f"\nNVIDIA Canary Qwen 2.5B ASR Model Loaded Successfully!\n")
    return True


def load_nv_parakeet_tdt_0_6b_asr_pipeline(model_id: str, torch_device: str):
    print(f"\n\n{model_id} ASR Model Selected - Loading...\n\n")
    global MODEL

    try:
        MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        MODEL.to(torch_device)
    except Exception as e:
        handle_model_loading_error(f"Could not load {model_id} ASR Model from Nemo ASR, encountered error: ", e)
        return False

    print(f"\n{model_id} ASR Model Loaded Successfully!\n")
    return True


def load_nv_canary_1b_v2_asr_pipeline(model_id: str, torch_device: str):
    print("\n\nNVIDIA Canary 1B V2 ASR Model Selected - Loading...\n\n")
    global MODEL

    try:
        MODEL = ASRModel.from_pretrained(model_name=model_id)
        MODEL.to(torch_device)
    except Exception as e:
        handle_model_loading_error("Could not load NVIDIA Canary 1B V2 ASR Model, encountered error: ", e)
        return False
    
    print(f"\nNVIDIA Canary 1B V2 ASR Model Loaded Successfully!\n")
    return True


def load_ibm_granite_speech_3_3_asr_pipeline(model_id: str, torch_device: str):
    print("\n\nIBM Granite Speech 3.3 ASR Model Selected - Loading...\n\n")
    global MODEL, PROCESSOR, TOKENIZER

    try:
        PROCESSOR = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        handle_model_loading_error("Could not load AutoProcessor for IBM Granite Speech 3.3 ASR Model, encountered error: ", e)
        return False
    
    try:
        TOKENIZER = PROCESSOR.tokenizer
    except Exception as e:
        handle_model_loading_error("Could not load tokenizer for IBM Granite Speech 3.3 ASR Model, encountered error: ", e)
        return False
    
    try:
        MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, device_map=torch_device, torch_dtype=torch.bfloat16
        )
        MODEL.to(torch_device)
    except Exception as e:
        handle_model_loading_error("Could not load model for IBM Granite Speech 3.3 ASR Model, encountered error: ", e)
        return False
    
    print(f"\nIBM Granite Speech 3.3 ASR Model Loaded Successfully!\n")
    return True


def load_asr_pipeline():
    print("\n\nASR Model Selected - Loading...\n\n")

    try:
        read_return = read_config(['model_id', 'torch_device_map'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to load asr-pipeline, encountered error: ", e)

    try:
        if ("openai/whisper" in read_return['model_id']) and ("v3" in read_return['model_id']):
            load_openai_whisper_v3_asr_pipeline(read_return['model_id'], read_return['torch_device_map'])
        
        elif ("nvidia/parakeet-tdt-0.6b" in read_return['model_id']):
            load_nv_parakeet_tdt_0_6b_asr_pipeline(read_return['model_id'], read_return['torch_device_map'])
        
        elif ("nvidia/canary-qwen-2.5b" in read_return['model_id']):
            load_nv_canary_qwen_2_5b_asr_pipeline(read_return['model_id'], read_return['torch_device_map'])
        
        elif ("nvidia/canary-1b-v2" in read_return['model_id']):
            load_nv_canary_1b_v2_asr_pipeline(read_return['model_id'], read_return['torch_device_map'])
        
        elif ("ibm-granite/granite-speech-3.3" in read_return['model_id']):
            load_ibm_granite_speech_3_3_asr_pipeline(read_return['model_id'], read_return['torch_device_map'])
        
        else:
            raise ValueError(f"Invalid ASR model ID: {read_return['model_id']}")
    
    except Exception as e:
        handle_local_error("Could not load ASR pipeline, encountered error: ", e)


def exl2_background_worker():
    '''Continuous loop to drive ExLlamaV2 batching'''
    print("\n >>> ExLlamaV2 Background Worker Started\n")
    
    while not EXL2_WORKER_STOP_EVENT.is_set():
        if EXL2_GENERATOR and EXL2_GENERATOR.num_remaining_jobs() > 0:
            try:
                # This single call advances ALL active requests by one step
                results = EXL2_GENERATOR.iterate()  # results is a LIST of dictionaries, effectively meaning "Here is everything that happened on the GPU during this clock cycle."

                for result in results:  # 'result' is a dictionary for a specific job
                    job = result['job']
                    text = result.get('text', '')
                    eos = result.get('eos', False)
                    # We don't care about 'stage': if 'text' is populated, we want it and if 'eos' is True, we want to signal end.

                    if hasattr(job, 'response_queue'):  # job is a ExLlamaV2DynamicJob object, and it has a response_queue attribute! It's NOT a dict key!
                        if text:
                            job.response_queue.put(text)
                        if eos:
                            job.response_queue.put(None)

            except Exception as e:
                handle_error_no_return("Error in ExL2 worker: ", e)
                # Optional: Signal error to all active queues
        else:
            time.sleep(0.05)    # Prevent CPU spin
    
    print("\n >>> ExLlamaV2 Background Worker Stopped\n")


def generate_exllama_measurement_file_for_model(model_id: str, model_snapshot_path: os.PathLike) -> os.PathLike:
    print(f"\n\nAttempting to generate measurement file for model {model_id}...\n\n")

    try:
        config = read_config(['transformer_models_folder', 'exl2_force_regenerate_measurement'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to generate-exllama_measurement_file_for_model(), encountered error: ", e)

    try:
        temp_dir = os.path.join(os.getcwd(), "exllamav2", "temp-converter-files")
        os.makedirs(temp_dir, exist_ok=True)

        measurement_file_path = os.path.join(config['transformer_models_folder'], model_id, "exllama-measurements-file", "measurement.json")
        os.makedirs(os.path.dirname(measurement_file_path), exist_ok=True)
    except Exception as e:
        handle_local_error("Could not create measurement file directory when attempting to generate-exllama_measurement_file_for_model(), encountered error: ", e)
    
    if os.path.exists(measurement_file_path) and not config['exl2_force_regenerate_measurement']:
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
        config = read_config(['transformer_models_folder'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to exllama-bpw_quantize_model(), encountered error: ", e)

    try:
        temp_dir = os.path.join(os.getcwd(), "exllamav2", "temp-converter-files")
        os.makedirs(temp_dir, exist_ok=True)

        quantized_model_path = os.path.join(config['transformer_models_folder'], model_id, "exl2-qaunts", f"{exl2_bpw}bpw")
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
    except Exception as e:
        handle_local_error("Could not run ExLlamaV2 bpw quantizer, encountered error: ", e)
    
    safe_remove_folder_from_filepath(temp_dir)
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


def define_exllama_generator():
    print("\nInitializing Global ExLlamaV2 Generator...")

    global EXL2_GENERATOR, EXL2_WORKER_THREAD, EXL2_WORKER_STOP_EVENT

    try:
        EXL2_GENERATOR = ExLlamaV2DynamicGenerator(EXL2_MODEL, EXL2_CACHE, EXL2_TOKENIZER)
        print("\nExLlamaV2 generator defined successfully\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV2 generator, encountered error: ", e)

    try:
        EXL2_WORKER_STOP_EVENT.clear()
        EXL2_WORKER_THREAD = threading.Thread(target=exl2_background_worker)
        EXL2_WORKER_THREAD.start()
        print("\nExLlamaV2 background worker thread started\n")
    except Exception as e:
        handle_local_error("Could not start ExLlamaV2 background worker thread, encountered error: ", e)

    return True


def load_exllama_pipeline():
    print("\n\nLoading ExLlamaV2 Pipeline\n\n")
    
    try:
        config = read_config(['model_id', 'exl2_bpw', 'exl2_no_flash_attn'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to load the ExLlamaV2 pipeline, encountered error: ", e)

    latest_snapshot_path = None
    try:
        latest_snapshot_path = download_model_from_hf_hub(config['model_id'])
    except Exception as e:
        handle_error_no_return(f"Could not download {config['model_id']} from HF-Hub. Attempting to scan for pre-existing local snapshots. Encountered error: ", e)
        try:
            latest_revision = get_latest_revision_for_model(config['model_id'])
            latest_snapshot_path = os_sanitize_path(latest_revision.snapshot_path)
        except Exception as e:
            handle_local_error(f"Error attempting to work with local snapshot for {config['model_id']}. Encountered error: ", e)
    
    if latest_snapshot_path is None:
        handle_local_error(f"Could not find a local snapshot for {config['model_id']}. Please check your connection and access token if you're using a private model.")
    
    try:
        measurement_file_path = generate_exllama_measurement_file_for_model(config['model_id'], latest_snapshot_path)
    except Exception as e:
        handle_local_error(f"Error generating ExLlamaV2 measurement file for {config['model_id']}. Encountered error: ", e)

    try:
        quantized_model_path = exllama_bpw_quantize_model(config['model_id'], measurement_file_path, latest_snapshot_path, float(config['exl2_bpw']))
    except Exception as e:
        handle_local_error(f"Error ExLlamaV2 quantizing {config['model_id']} to {config['exl2_bpw']} bits per word. Encountered error: ", e)

    try:
        define_exllama_generator_components(quantized_model_path, config['exl2_no_flash_attn'])
    except Exception as e:
        handle_local_error(f"Error loading ExLlamaV2 quantized model from {quantized_model_path}. Encountered error: ", e)

    try:
        define_exllama_generator()
    except Exception as e:
        handle_local_error(f"Error defining ExLlamaV2 generator components. Encountered error: ", e)

    try:
        global AUTO_TOKENIZER
        AUTO_TOKENIZER = AutoTokenizer.from_pretrained(config['model_id'], trust_remote_code=True)    # Using Transformers' AutoTokenizer as ExLlamaV2's ExLlamaV2Tokenizer does not contain an equivalent apply-chat_template() method!
        print("\nTransformers-AutoTokenizer configured successfully for automated prompt-formatting\n")
    except Exception as e:
        handle_local_error(f"Error loading AutoTokenizer for {config['model_id']}. Encountered error: ", e)

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


def exl3_background_worker():
    '''Continuous loop to drive ExLlamaV3 batching'''
    print("\n >>> ExLlamaV3 Background Worker Started\n")
    
    while not EXL3_WORKER_STOP_EVENT.is_set():
        if EXL3_GENERATOR and EXL3_GENERATOR.num_remaining_jobs() > 0:
            try:
                results = EXL3_GENERATOR.iterate()

                for result in results:
                    job = result['job']
                    text = result.get('text', '')
                    eos = result.get('eos', False)

                    if hasattr(job, 'response_queue'):  # job is a ExLlamaV3Job object, and it has a response_queue attribute! It's NOT a dict key!
                        if text:
                            job.response_queue.put(text)
                        if eos:
                            job.response_queue.put(None)
            except Exception as e:
                handle_error_no_return("Error in ExL3 worker: ", e)
                # Optional: Signal error to all active queues
        else:
            time.sleep(0.05)    # Prevent CPU spin
    
    print("\n >>> ExLlamaV3 Background Worker Stopped\n")


def exllama3_bpw_quantize_model(model_id: str, model_snapshot_path: os.PathLike, exl3_bpw: float) -> os.PathLike:
    print(f"\n\nAttempting to quantize model {model_id} to {exl3_bpw}bpw...\n\n")
    
    try:
        config = read_config(['transformer_models_folder', 'exl3_resume_quant_job'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to exllama3-bpw_quantize_model(), encountered error: ", e)

    try:
        temp_dir = os.path.join(os.getcwd(), "exllamav3", "temp-converter-files")
        os.makedirs(temp_dir, exist_ok=True)

        quantized_model_path = os.path.join(config['transformer_models_folder'], model_id, "exl3-qaunts", f"{exl3_bpw}bpw")
        os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)   # Create parent directory structure - final `{exl3_bpw}bpw` directory will be created by ExLlamaV3 converter
    except Exception as e:
        handle_local_error("Could not create directory to store quantized model when attempting to exllama3-bpw_quantize_model(), encountered error: ", e)

    if os.path.exists(quantized_model_path):
        print(f"\nQuantized model for {model_id} already exists. Skipping quantization.\n")
        return quantized_model_path
    
    convert_script_path = os.path.normpath(os.path.join(os.getcwd(), "exllamav3", "convert.py"))
    if config['exl3_resume_quant_job']:
        command = [
            'python' if platform.system() == 'Windows' else 'python3',
            convert_script_path,
            '-w', temp_dir,
            '-r'
        ]
    else:
        command = [
            'python' if platform.system() == 'Windows' else 'python3',
            convert_script_path,
            '-i', model_snapshot_path,
            '-o', quantized_model_path,
            '-w', temp_dir,
            '-b', str(exl3_bpw)
        ]

    try:
        print(f"\nRunning ExLlamaV3 bpw quantizer for {model_id}...\n")
        subprocess.run(command, check=True) # check=True ensures that the command will raise an exception if it fails
        print(f"\nExLlamaV3 Conversion of {model_id} to {exl3_bpw}bpw completed successfully!\n")
        safe_remove_folder_from_filepath(temp_dir)  # conversion completed, deleting temp dir to free space
    except Exception as e:
        # safe_remove_folder_from_filepath(temp_dir)  # Since conversion errored out, restarting afresh by clearing the temp dir is safer
        handle_local_error("Could not run ExLlamaV3 bpw quantizer, encountered error: ", e)

    return quantized_model_path


def define_exllamav3_generator_components(quantized_model_path: os.PathLike):
    print(f"\n\nAttempting to define ExLlamaV3 Generator Components for Model: {quantized_model_path}...\n\n")

    try:
        config = Config.from_directory(quantized_model_path)
        print("\nConfig defined successfully\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV3 config, encountered error: ", e)
    
    try:
        global EXL3_MODEL
        EXL3_MODEL = Model.from_config(config)
        print("\nExl3 model defined successfully\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV3 model, encountered error: ", e)
    
    try:
        exl3_config = read_config(['exl3_device', 'exl3_total_context', 'exl3_tensor_parallel', 'exl3_tp_output_device', 'exl3_use_per_device', 'exl3_max_chunk_size', 'exl3_cache_type', 'exl3_k_bits', 'exl3_v_bits'])
    except Exception as e:
        handle_local_error("Could not read exl3-total_context from hf_config.json, encountered error: ", e)

    try:
        global EXL3_CACHE
        if exl3_config['exl3_cache_type'] == 'CacheLayer_fp16':
            EXL3_CACHE = Cache(EXL3_MODEL, max_num_tokens = exl3_config['exl3_total_context'], layer_type = CacheLayer_fp16)
            print(f"\nExl3 Cache defined successfully with max_num_tokens: {exl3_config['exl3_total_context']}, layer_type: {exl3_config['exl3_cache_type']}\n")
        elif exl3_config['exl3_cache_type'] == 'CacheLayer_quant':
            EXL3_CACHE = Cache(EXL3_MODEL, max_num_tokens = exl3_config['exl3_total_context'], layer_type = CacheLayer_quant, k_bits = exl3_config['exl3_k_bits'], v_bits = exl3_config['exl3_v_bits'])
            print(f"\nExl3 Cache defined successfully with max_num_tokens: {exl3_config['exl3_total_context']}, layer_type: {exl3_config['exl3_cache_type']}, k_bits: {exl3_config['exl3_k_bits']}, v_bits: {exl3_config['exl3_v_bits']}\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV3 Cache, encountered error: ", e)

    try:
        print(f"\nLoading model...\n")
        EXL3_MODEL.load(
            progressbar=True,
            device=exl3_config['exl3_device'],
            max_chunk_size=exl3_config['exl3_max_chunk_size'],
            tensor_p=exl3_config['exl3_tensor_parallel'],
            tp_output_device=exl3_config['exl3_tp_output_device'],
            use_per_device=exl3_config['exl3_use_per_device']
        )
        print("\nExl3 model loaded successfully\n")
    except Exception as e:
        handle_local_error("Could not load ExLlamaV3 model, encountered error: ", e)

    try:
        global EXL3_TOKENIZER
        EXL3_TOKENIZER = Tokenizer.from_config(config)
        print("\nTokenizer defined successfully\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV3 tokenizer, encountered error: ", e)

    return True


def get_raw_stop_token_ids(quantized_model_path: os.PathLike) -> list | int:
    print(f"\n\nAttempting to set stop tokens for model {quantized_model_path}...\n\n")
    
    try:
        raw_model_config_path = os.path.join(quantized_model_path, "config.json")
        with open(raw_model_config_path, 'r') as file:
            raw_model_config = json.load(file)
        raw_stop_token_ids = raw_model_config['eos_token_id']
        return raw_stop_token_ids
    except Exception as e:
        handle_local_error("Could not get raw stop token ids for model {quantized_model_path}, encountered error: ", e)


def set_full_exl3_model_stop_token_list(raw_stop_token_ids: list | int):
    print(f"\n\nAttempting to set full model stop token list for {raw_stop_token_ids}...\n\n")

    global STOP_TOKENS

    print(f"AUTO_TOKENIZER.eos_token_id: {AUTO_TOKENIZER.eos_token_id}")
    print(f"EXL3_TOKENIZER.eos_token_id: {EXL3_TOKENIZER.eos_token_id}")
    print(f"Raw stop token ids: {raw_stop_token_ids}")
    
    full_list = []
    try:
        if isinstance(EXL3_TOKENIZER.eos_token_id, list):
            for token_ids in EXL3_TOKENIZER.eos_token_id:
                full_list.append(token_ids)
        else:
            full_list.append(EXL3_TOKENIZER.eos_token_id)
        
        if isinstance(AUTO_TOKENIZER.eos_token_id, list):
            for token_ids in AUTO_TOKENIZER.eos_token_id:
                full_list.append(token_ids)
        else:
            full_list.append(AUTO_TOKENIZER.eos_token_id)
        
        if isinstance(raw_stop_token_ids, list):
            for token_ids in raw_stop_token_ids:
                full_list.append(token_ids)
        else:
            full_list.append(raw_stop_token_ids)

        full_list = list(set(full_list))
        print(f"\nFull model stop token list: {full_list}\n")
        STOP_TOKENS = full_list
        return True
    except Exception as e:
        handle_local_error("Could not set full model stop token list, encountered error: ", e)


def define_exllamav3_generator(max_batch_size: int, max_chunk_size: int, show_visualizer: bool):
    print("\nInitializing Global ExLlamaV3 Generator...")

    global EXL3_GENERATOR, EXL3_WORKER_THREAD, EXL3_WORKER_STOP_EVENT

    try:
        EXL3_GENERATOR = Generator(
            model = EXL3_MODEL,
            cache = EXL3_CACHE,
            tokenizer = EXL3_TOKENIZER,
            max_batch_size = max_batch_size,
            max_chunk_size = max_chunk_size,
            show_visualizer = show_visualizer
        )
        print("\nExLlamaV3 generator defined successfully\n")
    except Exception as e:
        handle_local_error("Could not define ExLlamaV3 generator, encountered error: ", e)

    try:
        EXL3_WORKER_STOP_EVENT.clear()
        EXL3_WORKER_THREAD = threading.Thread(target=exl3_background_worker)
        EXL3_WORKER_THREAD.start()
        print("\nExLlamaV3 background worker thread started\n")
    except Exception as e:
        handle_local_error("Could not start ExLlamaV3 background worker thread, encountered error: ", e)
    
    return True


def load_exllamav3_pipeline():
    print("\n\nLoading ExLlamaV3 Pipeline\n\n")

    try:
        read_return = read_config(['model_id', 'exl3_bpw', 'exl3_total_context', 'exl3_max_batch_size', 'exl3_max_chunk_size', 'exl3_show_gen_visualizer'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to load the ExLlamaV3 pipeline, encountered error: ", e)

    latest_snapshot_path = None
    try:
        latest_snapshot_path = download_model_from_hf_hub(read_return['model_id'])
    except Exception as e:
        handle_error_no_return(f"Could not download {read_return['model_id']} from HF-Hub. Attempting to scan for pre-existing local snapshots. Encountered error: ", e)
        try:
            latest_revision = get_latest_revision_for_model(read_return['model_id'])
            latest_snapshot_path = os_sanitize_path(latest_revision.snapshot_path)
        except Exception as e:
            handle_local_error(f"Error attempting to work with local snapshot for {read_return['model_id']}. Encountered error: ", e)
    
    if latest_snapshot_path is None:
        handle_local_error(f"Could not find a local snapshot for {read_return['model_id']}. Please check your connection and access token if you're using a private model.")

    try:
        quantized_model_path = exllama3_bpw_quantize_model(read_return['model_id'], latest_snapshot_path, float(read_return['exl3_bpw']))
    except Exception as e:
        handle_local_error(f"Error ExLlamaV3 quantizing {read_return['model_id']} to {read_return['exl3_bpw']} bits per word. Encountered error: ", e)

    try:
        define_exllamav3_generator_components(quantized_model_path)
    except Exception as e:
        handle_local_error(f"Error loading ExLlamaV3 quantized model from {quantized_model_path}. Encountered error: ", e)

    try:
        define_exllamav3_generator(read_return['exl3_max_batch_size'], read_return['exl3_max_chunk_size'], read_return['exl3_show_gen_visualizer'])
    except Exception as e:
        handle_local_error(f"Error defining ExLlamaV3 generator components. Encountered error: ", e)

    try:
        global AUTO_TOKENIZER
        AUTO_TOKENIZER = AutoTokenizer.from_pretrained(read_return['model_id'], trust_remote_code=True)    # Using Transformers' AutoTokenizer as ExLlamaV3's Tokenizer does not contain an equivalent apply-chat_template() method!
        print("\nTransformers-AutoTokenizer configured successfully for automated prompt-formatting\n")
    except Exception as e:
        handle_local_error(f"Error loading AutoTokenizer for {read_return['model_id']}. Encountered error: ", e)

    try:
        raw_stop_token_ids = get_raw_stop_token_ids(quantized_model_path)
    except Exception as e:
        handle_error_no_return(f"Error manually setting stop tokens for {read_return['model_id']}. Relying purely on EXL & Auto Tokenizers' eos_token_id's instead, good luck! Encountered error: ", e)
        raw_stop_token_ids = []

    try:
        set_full_exl3_model_stop_token_list(raw_stop_token_ids)
    except Exception as e:
        handle_local_error(f"Error setting full model stop token list for {read_return['model_id']}. Encountered error: ", e)

    try:
        print(f"Model's context-length (max_num_tokens) is: {read_return['exl3_total_context']}")
    except Exception as e:
        handle_error_no_return("Could not determine the model's context-length (max_num_tokens), encountered error: ", e)
    
    print("\n\nExLlamaV3 Pipeline Loaded Successfully!\n\n")
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
    config_writer_semaphore.acquire()
    error_logging_semaphore.acquire()
    print("\n\nrestart-server-stream acquired all semaphores, proceeding with reboot...\n\n")

    shutdown_all()
    safe_empty_cuda_cache()

    try:
        read_return = read_config(['model_id', 'trust_remote_code', 'pipeline_task', 'flux_diffusers', 'vision', 'asr', 'exl2', 'exl3'])
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read values from hf_config.json when attempting restart-server-stream, encountered error: ", e)

    model_params = {}

    if not read_return['flux_diffusers']:
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
            
            if read_return['flux_diffusers']:
                print("\nFlux Diffusers Selected - Loading...\n")
                PIPE = load_flux_pipeline(PIPE)
            else:
                if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' in os.environ:
                    del os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']    # Better to delete as the default behavior is to try the C++ implementation first and fall back to Python if needed, which is more robust than simply setting it to 'cpp'.
                
                if read_return['exl2']:
                    print("\n\nExLlamaV2 Selected - Loading...\n\n")
                    load_exllama_pipeline()
                elif read_return['exl3']:
                    print("\n\nExLlamaV3 Selected - Loading...\n\n")
                    load_exllamav3_pipeline()
                elif read_return['vision']:
                    print("\nVision Model Selected - Loading...\n")
                    PIPE = load_vision_pipeline(PIPE, model_params)
                elif read_return['asr']:
                    print("\nASR Model Selected - Loading...\n")
                    load_asr_pipeline() # Not all ASR models define a pipeline, those that do will set the global PIPE via appropriate helper functions!
                else:
                    model = AutoModelForCausalLM.from_pretrained(read_return['model_id'], **model_params)
                    global AUTO_TOKENIZER
                    AUTO_TOKENIZER = AutoTokenizer.from_pretrained(read_return['model_id'], trust_remote_code=read_return['trust_remote_code'])
                    print("\nInitializing inference pipeline...")
                    PIPE = pipeline(
                        read_return['pipeline_task'],
                        model=model,
                        tokenizer=AUTO_TOKENIZER,
                    )

                    try:
                        print(f"Your model's memory footprint is: {model.get_memory_footprint()}")
                    except Exception as e:
                        handle_error_no_return("Could not determine the model's memory footprint, encountered error: ", e)
            
            print(f"\n{read_return['model_id']} loaded successfully!\n")
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

    print(f"\nModel Initialization Begins - Loading {read_return['model_id']}\n")
    return Response(output_reader(), content_type='text/event-stream')


def initialize_model():

    global PIPE

    try:
        read_return = read_config(['model_id', 'trust_remote_code', 'push_to_hub', 'quant_level', 'pipeline_task', 'flux_diffusers', 'vision', 'asr', 'exl2', 'exl3'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to initialize_model(), encountered error: ", e)
    
    print(f"\n\nInitializing HF-Waitress LLM Server for {read_return['model_id']}\n\n")

    if read_return['flux_diffusers']:
        print("\n\nFlux Diffusers Selected - Loading...\n\n")
        PIPE = load_flux_pipeline(PIPE)

    else:
        # Remove explicit protobuf implementation setting to use system default
        if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' in os.environ:
            del os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']    # Better to delete as the default behavior is to try the C++ implementation first and fall back to Python if needed, which is more robust than simply setting it to 'cpp'.

        if read_return['exl2']:
            print("\n\nExLlamaV2 Selected - Loading...\n\n")
            load_exllama_pipeline()
        elif read_return['exl3']:
            print("\n\nExLlamaV3 Selected - Loading...\n\n")
            load_exllamav3_pipeline()
        
        else:
            model_params = get_model_params()
            print(f"Setting model-parameters: {model_params}")
            
            if read_return['vision']:
                print("\n\nVision Model Selected - Loading...\n\n")
                PIPE = load_vision_pipeline(PIPE, model_params)
            elif read_return['asr']:
                print("\n\nASR Model Selected - Loading...\n\n")
                load_asr_pipeline() # Not all ASR models define a pipeline, those that do will set the global PIPE via appropriate helper functions!
            else:
                model = AutoModelForCausalLM.from_pretrained(read_return['model_id'], **model_params)
                global AUTO_TOKENIZER
                AUTO_TOKENIZER = AutoTokenizer.from_pretrained(read_return['model_id'], trust_remote_code=read_return['trust_remote_code'])
                print("\nInitializing inference pipeline...")
                PIPE = pipeline(read_return['pipeline_task'], model=model, tokenizer=AUTO_TOKENIZER)
        
                try:
                    print(f"Your model's memory footprint is: {model.get_memory_footprint()}")
                except Exception as e:
                    handle_error_no_return("Could not determine the model's memory footprint, encountered error: ", e)
    
    print(f"\n{read_return['model_id']} loaded successfully!\n")

    if read_return['push_to_hub']:
        try:
            model.push_to_hub(f"{read_return['model_id']}-{read_return['quant_level']}")
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
        handle_local_error("Could not convert file to PDF, encountered error: ", e)
    except Exception as e:
        handle_local_error("Unexpected error when converting file to PDF, encountered error: ", e)


def get_pil_image_objects_for_file(filename, filepath, dpi=300):
    print(f"\n\nGetting PIL image objects for file: {filename}\n\n")

    if not filename.lower().endswith('.pdf'):
        _, filepath = convert_non_pdf_to_pdf_with_unoconv(filename, filepath)

    try:
        pil_image_object_list = convert_pdf_to_images_list(filepath, dpi)
        return pil_image_object_list
    except Exception as e:
        handle_local_error("Could not get PIL image objects for file, encountered error: ", e)


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
        config = read_config(['max_new_tokens'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to get input_params for vision-model, encountered error: ", e)

    try:
        dpi = int(request.headers.get('X-DPI', 300))
        try:
            generation_config = {
                "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', str(config['max_new_tokens']))),
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
            handle_local_error("Failed to save document to app folder, encountered error: ", e)
        
        try:
            pil_image_object_list = get_pil_image_objects_for_file(filename, filepath, dpi)
        except Exception as e:
            handle_local_error("Could not get PIL image objects for file, encountered error: ", e)

    try:
        print(f"\n\nApplying Chat Template for messages: {messages}\n\n")
        input_text = PIPE.apply_chat_template(messages, add_generation_prompt=True) # Not using central auto-tokenizer method as this one is based on Processor-AutoProcessor, not AutoTokenizer!
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


def auto_tokenizer_apply_chat_template(
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        tools: Optional[list[Union[dict, Callable]]] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False
    ) -> dict:
    '''
    Acts as a wrapper around the Transformers apply-chat_template() method, which converts a list of dictionaries 
    with `"role"` and `"content"` keys to a list of token ids. This method is intended for use with chat models, 
    and will read the tokenizer's chat_template attribute to determine the format and control tokens to use when converting.

    Args:
        conversation (Union[list[dict[str, str]], list[list[dict[str, str]]]]): A list of dicts
            with "role" and "content" keys, representing the chat history so far.
        tools (`list[Union[Dict, Callable]]`, *optional*):
            A list of tools (callable functions) that will be accessible to the model. If the template does not
            support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
            giving the name, description and argument types for the tool. See our
            [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
            for more information.
        add_generation_prompt (bool, *optional*):
            If this is set, a prompt with the token(s) that indicate
            the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
            Note that this argument will be passed to the chat template, and so it must be supported in the
            template for this argument to have any effect.
        tokenize (`bool`, defaults to `True`):
            Whether to tokenize the output. If `False`, the output will be a string.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
            values are:
            - `'tf'`: Return TensorFlow `tf.Tensor` objects.
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
            - `'jax'`: Return JAX `jnp.ndarray` objects.
        return_dict (`bool`, defaults to `False`):
            Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.

    For full details, check the full definition in `tokenization_utils_base.py`, which can be found by inspecting the AutoTokenizer class to open 
    `tokenization_auto.py` and from there, navigate to the above module via the `from ...tokenization_utils_base` import.
    '''
    try:
        return AUTO_TOKENIZER.apply_chat_template(
            conversation=conversation,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
            return_tensors=return_tensors,
            tokenize=tokenize
        )
    except Exception as e:
        handle_local_error("Error invoking Transformers-AutoTokenizer's apply-chat_template() method, encountered error: ", e)


@app.route('/completions', methods=['POST'])
def completions():

    with llm_semaphore:
        print("\n\nLLM semaphore acquired by /completions\n\n")

        try:
            config = read_config(['max_new_tokens', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'flux_diffusers', 'vision'])
        except Exception as e:
            return handle_api_error("Could not read values from hf_config.json when attempting /completions, encountered error: ", e)

        if config['flux_diffusers']:
            try:
                image_str, image_name = generate_flux_image(request)
                return jsonify({"success": True, "response": image_str, "image_name": image_name})
            except Exception as e:
                return handle_api_error("Could not generate image with FLUX Diffusers. Encountered error: ", e)

        if config['vision']:
            try:
                response = inference_with_vision_model(request)
                return jsonify({"success": True, "response": response})
            except Exception as e:
                return handle_api_error("Could not generate image with Vision Model. Encountered error: ", e)

        try:
            data = request.json
            messages = data.get('messages', [])
            tools = data.get('tools', None)
        except Exception as e:
            return handle_api_error("Could not read POST-request messages for /completions, encountered error: ", e)

        try:
            generation_config = GenerationConfig(
                max_new_tokens=int(request.headers.get('X-Max-New-Tokens', str(config['max_new_tokens']))),
                temperature=float(request.headers.get('X-Temperature', str(config['temperature']))),
                do_sample=request.headers.get('X-Do-Sample', str(config['do_sample'])).lower() == 'true',
                top_k=int(request.headers.get('X-Top-K', str(config['top_k']))),
                top_p=float(request.headers.get('X-Top-P', str(config['top_p']))),
                min_p=float(request.headers.get('X-Min-P', str(config['min_p']))),
                use_cache=True
            )
            # use_cache=True by default, setting explictily to True for clarity. Intra-call optimization: tells the generator, "During this single generation call, please be efficient.
            # As you process the prompt and generate new tokens, create and use a KV cache internally so you don't have to re-calculate everything for every single new token." 
        except Exception as e:
            handle_error_no_return("Could not set generation-arguments for /completions, proceeding without them. Encountered error: ", e)
            generation_config = GenerationConfig(max_new_tokens=config['max_new_tokens'], use_cache=True)

        try:
            print(f"\n\nApplying Chat Template for messages: {messages}\n\n")
            inputs = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        except Exception as e:
            return handle_api_error("Could not apply chat template, encountered error: ", e)

        try:
            print("\n\nLoading Input to Model\n\n")
            inputs.to(PIPE.model.device)
        except Exception as e:
            return handle_api_error("Could not load input to model, encountered error: ", e)

        inference_output = ""
        try:
            print("\n\nGenerating Output\n\n")
            output = PIPE.model.generate(**inputs, generation_config=generation_config)
            input_length = inputs.input_ids.shape[1]   # Check inference_with_vision_model(request) for detailed explanation!
            
            # Slice the tensor and decode only the output!
            decoded_output = AUTO_TOKENIZER.decode(output[0][input_length:], skip_special_tokens=True)    # Setting skip_special_tokens=True to remove: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.

            print(f"\n\ndecoded_output: {decoded_output}\n\n")
            inference_output += decoded_output
        except Exception as e:
            return handle_api_error("Could not generate output, encountered error: ", e)

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



#####################################----------------ASR Stuff!----------------#####################################
audio_queue = queue.Queue()
is_recording = threading.Event()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def start_recording():
    samplerate = int(read_config(['asr_samplerate']))
    is_recording.set()
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback, dtype='float32'):
        while is_recording.is_set():
            time.sleep(0.1)     # The callback is handling the audio data, so we just wait

def stop_recording():
    is_recording.clear()


def get_indices_of_substring(response, start_substring, end_substring):
    print("\nAttempting to trim response...\n")
    try:
        if start_substring in response and end_substring in response:
            start_index = response.rindex(start_substring)  # Sometimes the model re-gurgitates multiple copies of the same dict in it's response
            end_index = response.rindex(end_substring) # rindex() returns the index of the last occurrence of the substring
            print("\nSubstring successfully found, returning indices...\n")
            return start_index, (end_index + len(end_substring))
            
        else:
            print(f"\nResponse does not contain either the start_substring: {start_substring} or the end_substring: {end_substring}, returning unchanged response...\n")
            return None, None
    except Exception as e:
        print(f"Failed to trim response, encountered error: {e}")
        return None, None


def remove_padding_from_transcription(transcription: str) -> str:
    try:
        padding_start_index, padding_end_index = get_indices_of_substring(
            transcription.lower().strip(),
            start_substring="tony is quiet",
            end_substring="will be severely punished"
        )
        
        if padding_start_index is None or padding_end_index is None:    # try once again with a small tweak!
            padding_start_index, padding_end_index = get_indices_of_substring(
                transcription.lower().strip(),
                start_substring="tony is quite",
                end_substring="will be severely punished"
            )
        
        if padding_start_index is not None and padding_end_index is not None:
            transcription = transcription[:padding_start_index] + transcription[padding_end_index:]
            transcription = transcription.replace(" .", "").strip()
        
        return transcription
    except Exception as e:
        print(f"Failed to trim response, encountered error: {e}")
        return transcription


def generate_padding_audio(text:str, sr:int=16000) -> np.array:
    """Generates audio for padding text using pyttsx3 and returns it as a NumPy array."""

    if not os.path.exists('temp_padding.wav') or os.path.getsize('temp_padding.wav') == 0:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of the speech - Adjust as needed
        engine.save_to_file(text, 'temp_padding.wav')   # pyttsx3 can be tricky with in-memory, so a temp file is robust
        engine.runAndWait()

    # Read the wav file from disk and convert to the correct format
    # The sample rate for pyttsx3 might not be 16000, so we'll need to resample later if needed
    read_sr, audio_data = read('temp_padding.wav') # read_sr is the sample rate of the audio data, audio_data is the audio data as a NumPy array

    # Convert to mono float32, which is what Whisper expects
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)    # Convert to mono

    if read_sr != sr:   # resample to 16000:
        audio_data = librosa.resample(audio_data, orig_sr=read_sr, target_sr=sr)

    # This ensures the audio uses the full dynamic range from -1.0 to 1.0
    peak_volume = np.max(np.abs(audio_data))
    if peak_volume > 0:
        audio_data = audio_data / peak_volume
    
    # Clean up the temp file
    # os.remove('temp_padding.wav')
    return audio_data


def get_pipe(torch_device: str):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=MODEL,
        tokenizer=PROCESSOR.tokenizer,
        feature_extractor=PROCESSOR.feature_extractor,
        torch_dtype=torch_dtype, device=torch_device
    )


def transcribe_with_openai_whisper_v3(audio_data: np.ndarray, asr_config: dict) -> str:
    try:
        pipe = get_pipe(asr_config['torch_device_map'])
        result = pipe(audio_data, return_timestamps=True, generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "temperature": float(asr_config['asr_temperature']),
            # Optional: "no_speech_threshold": 0.6, "compression_ratio_threshold": 2.4
        })
        return result['text'].strip() if result else ""
    except Exception as e:
        handle_local_error("Could not transcribe audio data with OpenAI Whisper V3 ASR Model, encountered error: ", e)


def transcribe_with_nv_canary_qwen_2_5b(audio_data: np.ndarray, asr_config: dict) -> str:
    # ensure mono float32 in [-1, 1]
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if asr_config['asr_samplerate'] != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=asr_config['asr_samplerate'], target_sr=16000)
        asr_config['asr_samplerate'] = 16000
    audio_data = np.clip(audio_data.astype(np.float32), -1.0, 1.0)

    # write a temp wav (int16) that the model can reference
    tmp_dir = tempfile.mkdtemp(prefix="canary_asr_")
    wav_path = os.path.join(tmp_dir, "utterance.wav")
    wav_write(wav_path, asr_config['asr_samplerate'], (audio_data * 32767.0).astype(np.int16))

    try:
        prompts = [
            [
                {
                    "role": "user",
                    "content": f"Transcribe the following: {MODEL.audio_locator_tag}",
                    "audio": [wav_path],
                }
            ]
        ]
        answer_ids = MODEL.generate(prompts=prompts, max_new_tokens=asr_config['asr_max_new_tokens'])
        text = MODEL.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        return text
    
    finally:
        try:
            os.remove(wav_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


def transcribe_with_nv_parakeet_tdt_0_6b(audio_data: np.ndarray, asr_config: dict, model_id: str) -> str:
    # ensure mono float32 in [-1, 1]
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if asr_config['asr_samplerate'] != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=asr_config['asr_samplerate'], target_sr=16000)
        asr_config['asr_samplerate'] = 16000
    audio_data = np.clip(audio_data.astype(np.float32), -1.0, 1.0)

    # write a temp wav (int16) that the model can reference
    mname = "parakeet_tdt_0_6b_v3" if "v3" in model_id else "parakeet_tdt_0_6b_v2"
    tmp_dir = tempfile.mkdtemp(prefix=f"{mname}_asr_")
    wav_path = os.path.join(tmp_dir, "utterance.wav")
    wav_write(wav_path, asr_config['asr_samplerate'], (audio_data * 32767.0).astype(np.int16))

    try:
        output = MODEL.transcribe([wav_path])
        text = output[0].text.strip()
        return text

    finally:
        try:
            os.remove(wav_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


def transcribe_with_nv_canary_1b_v2(audio_data: np.ndarray, asr_config: dict) -> str:
    # ensure mono float32 in [-1, 1]
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if asr_config['asr_samplerate'] != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=asr_config['asr_samplerate'], target_sr=16000)
        asr_config['asr_samplerate'] = 16000
    audio_data = np.clip(audio_data.astype(np.float32), -1.0, 1.0)

    # write a temp wav (int16) that the model can reference
    tmp_dir = tempfile.mkdtemp(prefix="canary_1b_v2_asr_")
    wav_path = os.path.join(tmp_dir, "utterance.wav")
    wav_write(wav_path, asr_config['asr_samplerate'], (audio_data * 32767.0).astype(np.int16))

    try:
        output = MODEL.transcribe([wav_path], source_lang='en', target_lang='en')
        text = output[0].text.strip()
        return text
        
    finally:
        try:
            os.remove(wav_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


def transcribe_with_ibm_granite_speech_3_3(audio_data: np.ndarray, asr_config: dict) -> str:
    # ensure mono float32 in [-1, 1]
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if asr_config['asr_samplerate'] != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=asr_config['asr_samplerate'], target_sr=16000)
        asr_config['asr_samplerate'] = 16000
    audio_data = np.clip(audio_data.astype(np.float32), -1.0, 1.0)

    # Convert numpy array directly to PyTorch tensor
    # Add batch dimension: (samples,) -> (1, samples)
    wav = torch.from_numpy(audio_data).unsqueeze(0).float()

    try:
        system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        prompt = TOKENIZER.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)    # Not using central auto-tokenizer method as this one is based on Processor-AutoProcessor, not AutoTokenizer!
        model_inputs = PROCESSOR(prompt, wav, device=asr_config['torch_device_map'], return_tensors="pt").to(asr_config['torch_device_map'])
        model_outputs = MODEL.generate(**model_inputs, max_new_tokens=asr_config['asr_max_new_tokens'], do_sample=False, num_beams=1)

        # Transformers includes the input IDs in the response.
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
        output_text = TOKENIZER.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        text = output_text[0].upper()
        return text
    except Exception as e:
        # Handle any processing errors
        print(f"Error during transcription: {e}")
        return ""


def transcribe_audio_data_with_asr_model(audio_data: np.ndarray, asr_config: dict) -> dict:
    try:
        if ("openai/whisper" in asr_config['model_id']) and ("v3" in asr_config['model_id']):
            return transcribe_with_openai_whisper_v3(audio_data, asr_config)

        elif ("nvidia/parakeet-tdt-0.6b" in asr_config['model_id']):
            return transcribe_with_nv_parakeet_tdt_0_6b(audio_data, asr_config, asr_config['model_id'])
        
        elif ("nvidia/canary-qwen-2.5b" in asr_config['model_id']):
            return transcribe_with_nv_canary_qwen_2_5b(audio_data, asr_config)
        
        elif ("nvidia/canary-1b-v2" in asr_config['model_id']):
            return transcribe_with_nv_canary_1b_v2(audio_data, asr_config)
        
        elif ("ibm-granite/granite-speech-3.3" in asr_config['model_id']):
            return transcribe_with_ibm_granite_speech_3_3(audio_data, asr_config)
        
        else:
            raise ValueError(f"Invalid ASR model ID: {asr_config['model_id']}")
    except Exception as e:
        handle_local_error("Could not transcribe audio data with ASR model, encountered error: ", e)


def prepare_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    # librosa can read from BytesIO via soundfile backend
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
    y = y.astype(np.float32, copy=False)
    return y


def asr_transcribe(audio_bytes: bytes, asr_config: dict) -> str:
    # 0. Prepare Padding Audio:
    print("Starting ASR... Generating padding audio first.")
    padding_audio = generate_padding_audio(asr_config['asr_padding_text'], sr=asr_config['asr_samplerate'])
    pad_rms = np.sqrt(np.mean(padding_audio**2) + 1e-12)

    print("Starting real-time transcription. Press Ctrl+C to stop.")

    # 1. Decode Received Audio Bytes to 16k mono float32 audio data:
    audio_data = prepare_audio_from_bytes(audio_bytes, target_sr=asr_config['asr_samplerate'])

    # 2. Pre-process Received Audio Data:
    if asr_config['asr_apply_normalization']:
        print(f"Applying normalization to audio.")
        peak_volume = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0.0
        if peak_volume > 0: # normalize the spoken audio so it's not lost to the padding audio!
            audio_data = audio_data / peak_volume
    
    padding_applied = False
    if asr_config['asr_apply_tts_padding']:
        print(f"Applying TTS padding to audio.")
        if len(audio_data) < asr_config['asr_min_context_s'] * asr_config['asr_samplerate']:
            print(f"Audio is shorter than {asr_config['asr_min_context_s']}s. Applying padding.")
            
            if asr_config['asr_apply_rms_dimming']:
                print(f"Applying RMS dimming to padding audio.")
                speech = audio_data
                speech_rms = np.sqrt(np.mean(speech**2) + 1e-12)
                target_pad_rms = speech_rms * (10 ** (-24/20))  # -24 dB
                if pad_rms > 0:
                    padding_audio_dimmed_rms = padding_audio * (target_pad_rms / pad_rms)
                else:
                    padding_audio_dimmed_rms = padding_audio
                
                if asr_config['asr_apply_crossfade']:
                    print(f"Applying crossfade to padding audio.")
                    # short crossfade to avoid a hard boundary
                    xf = int(0.05 * asr_config['asr_samplerate'])  # 50 ms
                    if len(speech) >= xf and len(padding_audio_dimmed_rms) >= xf:
                        fade = np.linspace(1.0, 0.0, xf, dtype=np.float32)
                        audio_data[-xf:] *= fade
                        padding_audio_dimmed_rms[:xf] *= (1.0 - fade)

                audio_data = np.concatenate((audio_data, padding_audio_dimmed_rms))
                padding_applied = True
            else:
                print(f"Applying padding audio without RMS dimming.")
                audio_data = np.concatenate((audio_data, padding_audio))
                padding_applied = True
    
    if asr_config['asr_apply_zero_padding']:
        print(f"Applying zero-padding to audio.")
        needed = max(0, int(asr_config['asr_min_context_s'] * asr_config['asr_samplerate'] - len(audio_data)))
        if needed > 0:
            print(f"Audio is shorter than {asr_config['asr_min_context_s']}s. Applying zero-padding.")
            audio_data = np.concatenate((audio_data, np.zeros(needed, dtype=np.float32)))
    
    # 3. Process Audio Data:
    print(f"Processing {len(audio_data)/asr_config['asr_samplerate']:.2f}s of audio...")

    transcription = transcribe_audio_data_with_asr_model(audio_data, asr_config)

    if padding_applied and transcription:
        transcription = remove_padding_from_transcription(transcription)
    
    print(f"Transcription: {transcription}")
    return transcription


def read_asr_config() -> dict:
    try:
        return read_config(
            [
                'model_id',
                'torch_device_map',
                'asr_samplerate',
                'asr_temperature',
                'asr_max_new_tokens',
                'asr_volume_threshold',
                'asr_silence_duration_s',
                'asr_min_chunk_duration_s',
                'asr_min_context_s',
                'asr_stale_buffer_timeout_s',
                'asr_min_meaningful_samples_factor',
                'asr_padding_text',
                'asr_vad_model',
                'asr_vad_device', 
                'asr_vad_threshold', 
                'asr_vad_min_speech_ms',
                'asr_vad_min_silence_ms',
                'asr_vad_window_size_samples',
                'asr_vad_max_buffer_s',
                'asr_vad_speech_pad_ms',
                'asr_apply_normalization',
                'asr_apply_tts_padding',
                'asr_apply_zero_padding',
                'asr_apply_rms_dimming',
                'asr_apply_crossfade'
            ]
        )
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when attempting to read ASR config, encountered error: ", e)


def get_final_asr_config(request) -> dict:
    asr_config = read_asr_config()
    asr_config.update({
        'asr_temperature': float(request.headers.get('X-ASR-Temperature', str(asr_config['asr_temperature']))),
        'asr_max_new_tokens': int(request.headers.get('X-ASR-Max-New-Tokens', str(asr_config['asr_max_new_tokens']))),
        'asr_samplerate': int(request.headers.get('X-ASR-Samplerate', str(asr_config['asr_samplerate']))),
        'asr_volume_threshold': float(request.headers.get('X-ASR-Volume-Threshold', str(asr_config['asr_volume_threshold']))),
        'asr_silence_duration_s': float(request.headers.get('X-ASR-Silence-Duration-S', str(asr_config['asr_silence_duration_s']))),
        'asr_min_chunk_duration_s': float(request.headers.get('X-ASR-Min-Chunk-Duration-S', str(asr_config['asr_min_chunk_duration_s']))),
        'asr_min_context_s': float(request.headers.get('X-ASR-Min-Context-S', str(asr_config['asr_min_context_s']))),
        'asr_stale_buffer_timeout_s': float(request.headers.get('X-ASR-Stale-Buffer-Timeout-S', str(asr_config['asr_stale_buffer_timeout_s']))),
        'asr_min_meaningful_samples_factor': float(request.headers.get('X-ASR-Min-Meaningful-Samples-Factor', str(asr_config['asr_min_meaningful_samples_factor']))),
        'asvad_threshold': float(request.headers.get('X-ASR-VAD-Threshold', str(asr_config['asr_vad_threshold']))),
        'asr_vad_min_speech_ms': float(request.headers.get('X-ASR-VAD-Min-Speech-MS', str(asr_config['asr_vad_min_speech_ms']))),
        'asr_vad_min_silence_ms': float(request.headers.get('X-ASR-VAD-Min-Silence-MS', str(asr_config['asr_vad_min_silence_ms']))),
        'asr_vad_window_size_samples': float(request.headers.get('X-ASR-VAD-Window-Size-Samples', str(asr_config['asr_vad_window_size_samples']))),
        'asr_vad_max_buffer_s': float(request.headers.get('X-ASR-VAD-Max-Buffer-S', str(asr_config['asr_vad_max_buffer_s']))),
        'asr_vad_speech_pad_ms': float(request.headers.get('X-ASR-VAD-Speech-Pad-MS', str(asr_config['asr_vad_speech_pad_ms']))),
        'asr_apply_normalization': request.headers.get('X-ASR-Apply-Normalization', str(asr_config['asr_apply_normalization'])).lower() == 'true',
        'asr_apply_tts_padding': request.headers.get('X-ASR-Apply-TTS-Padding', str(asr_config['asr_apply_tts_padding'])).lower() == 'true',
        'asr_apply_zero_padding': request.headers.get('X-ASR-Apply-Zero-Padding', str(asr_config['asr_apply_zero_padding'])).lower() == 'true',
        'asr_apply_rms_dimming': request.headers.get('X-ASR-Apply-RMS-Dimming', str(asr_config['asr_apply_rms_dimming'])).lower() == 'true',
        'asr_apply_crossfade': request.headers.get('X-ASR-Apply-Crossfade', str(asr_config['asr_apply_crossfade'])).lower() == 'true',
    })
    return asr_config


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Accepts a posted WAV audio file under form field 'audio' and returns Whisper transcription.
    Content-Type should be multipart/form-data.
    """
    try:
        if 'audio' not in request.files:
            raise Exception("Missing audio file in request")
        f = request.files['audio']
        data = f.read()
        if not data:
            raise Exception("Empty audio file in request")
        
        try:
            final_asr_config = get_final_asr_config(request)
        except Exception as e:
            raise Exception("Could not get final ASR config")
        
        try:
            result = asr_transcribe(data, final_asr_config)
        except Exception as e:
            raise Exception("Could not transcribe audio")
        
        return jsonify(success=True, transcription=result)
    except Exception as e:
        return handle_api_error("Server-side error, could not transcribe audio, encountered error: ", e)



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

    llm_semaphore.acquire()
    print("\n\nLLM semaphore acquired by /completions_stream\n\n")

    try:
        data = request.json
        if isinstance(data, str):   # should be a list
            data = json.loads(data)
        messages = data.get('messages', [])
        tools = data.get('tools', None)
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read POST-request messages for /completions_stream, encountered error: ", e)

    try:
        read_return = read_config(['max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep'])
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not read values from hf_config.json when attempting /completions_stream, encountered error: ", e)

    try:    # Create a GenerationConfig object
        generation_config = {
            "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', read_return['max_new_tokens'])),
            "return_full_text": request.headers.get('X-Return-Full-Text', str(read_return['return_full_text'])).lower() == 'true',
            "temperature": float(request.headers.get('X-Temperature', str(read_return['temperature']))),
            "do_sample": request.headers.get('X-Do-Sample', str(read_return['do_sample'])).lower() == 'true',
            "top_k": int(request.headers.get('X-Top-K', str(read_return['top_k']))),
            "top_p": float(request.headers.get('X-Top-P', str(read_return['top_p']))),
            "min_p": float(request.headers.get('X-Min-P', str(read_return['min_p']))),
            "use_cache": True
        }
    except Exception as e:
        handle_error_no_return("Could not set generation-arguments for /completions_stream, proceeding without them. Encountered error: ", e)
        generation_config = {"max_new_tokens": read_return['max_new_tokens'],"use_cache": True}

    try:
        print(f"\n\nApplying Chat Template for messages: {messages}\n\n")
        inputs = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not apply chat template, encountered error: ", e)

    try:
        # Slice the tensor and decode only the input!
        decoded_inputs = AUTO_TOKENIZER.decode(inputs['input_ids'][0].tolist(), skip_special_tokens=False)    # Setting skip_special_tokens=False to keep: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.
        print(f"\n\ndecoded_inputs: {decoded_inputs}\n\n")
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not decode inputs, encountered error: ", e)

    stop_event = threading.Event()
    data_queue = queue.Queue()

    def callback(data):
        data_queue.put(data)

    custom_streamer = CustomTextStreamer(AUTO_TOKENIZER, skip_special_tokens=True, skip_prompt=True)    # special tokens need not be streamed though!
    custom_streamer.callback = callback

    def llm_task():

        global PIPE

        try:
            generation_config["streamer"] = custom_streamer
            generation_config["stopping_criteria"] = StoppingCriteriaList([StopOnEvent(stop_event)])  # StoppingCriteriaList is a container that holds a list of StoppingCriteria objects. In our case, we have only one such object, which is our custom StoppingCriteria class, initialized with the stop_event object.
            output = PIPE(decoded_inputs, **generation_config)
        except Exception as e:
            handle_error_no_return("Response generation failed, encountered error: ", e)
            data_queue.put(f"Error: {str(e)}") # Pass error to client via queue
        finally:
            data_queue.put(None)
            print("\n\nLLM stream done, releasing semaphore\n\n")
            llm_semaphore.release()

    def generate():        
        
        global STOP_GENERATION
        STOP_GENERATION = False

        try:
            thread = threading.Thread(target=llm_task)
            thread.start()
        except Exception as e:
            handle_error_no_return("Error generating completions-stream, encountered error: ", e)
            yield f"data: {json.dumps('Error in Transformers Response-Generation Pipeline: ' + str(e))}\n\n"
            return

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


###################################-------------Exl2 Logic Begins-------------###################################

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
        return True # Since the above check is simplistic, an error indicates something is amiss, so best to return True to avoid infinite loops and try auto-truncation


def get_exl2_gen_settings(request):

    try:
        config_data = read_config(['model_id', 'max_new_tokens', 'temperature', 'top_k', 'top_p', 'knowledge_graph_cache_dir', 'exl2_max_seq_len'])
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
        config_data['max_new_tokens'] = int(request.headers.get('X-Max-New-Tokens', str(config_data.get('max_new_tokens', '2048'))))
        print("\nExLlamaV2Sampler.Settings Defined Successfully\n")
    except Exception as e:
        handle_error_no_return("Could not set generation-arguments for exl2-grapher, proceeding without them. Encountered error: ", e)
        gen_settings = None

    return gen_settings, config_data


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
    """
    Streaming text generation using ExLlamaV2 model
    
    This endpoint provides streaming text generation using the ExLlamaV2 model with dynamic generation capabilities.
    
    OpenAPI 3.0.0 Specification is available in the `hfw-openapi-3-specs.yaml` file.
    """

    print("\n\nexl2-stream route triggered\n\n")

    try:
        data = request.json
        if isinstance(data, str):   # must convert to a list
            data = json.loads(data)
        messages = data.get('messages', [])
        tools = data.get('tools', None)
        gen_settings, config_data  = get_exl2_gen_settings(request)
        user_queue = queue.Queue()
    except Exception as e:
        return handle_api_error("Could not setup for exl2-stream, encountered error: ", e)
    
    try:
        tokenized_messages = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, tokenize=False)
        if not exl2_prompt_fits_within_max_context_length(tokenized_messages): print("\n\nPrompt doesn't fit within Exl2 context window, will auto-truncate.\n\n")
    except Exception as e:
        return handle_api_error("Could not tokenize messages for exl2-stream, encountered error: ", e)

    try:
        job = ExLlamaV2DynamicJob(
            input_ids= EXL2_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True),
            max_new_tokens = int(request.headers.get('X-Max-New-Tokens', str(config_data.get('max_new_tokens')))),
            stop_conditions = [EXL2_TOKENIZER.eos_token_id, AUTO_TOKENIZER.eos_token_id],
            gen_settings = gen_settings
        )
        job.response_queue = user_queue
        EXL2_GENERATOR.enqueue(job)
    except Exception as e:
        return handle_api_error("Could not create ExLlamaV2-DynamicJob object. Error details follow. In case of context-window issues, ensure your max_new_tokens does not exceed the allocated context-window, and of course, allocate an adequate context-window! Error: ", e)

    try:
        def generate():

            global STOP_GENERATION
            STOP_GENERATION = False

            while True:
                if STOP_GENERATION: # Handle Manual Stop Signal
                    print("\n\nStopping generation with stop_event\n\n")
                    try:
                        EXL2_GENERATOR.cancel(job)
                    except:
                        pass
                    STOP_GENERATION = False
                    break
                
                token = user_queue.get()
                if token is None:
                    break
                
                yield f"data: {json.dumps(token)}\n\n"
            
            yield f"event: END\ndata: \"null\"\n\n"

            print("\nexl2-stream done\n")

        print("\n\nInferencing Begins!\n\n")
        return Response(generate(), content_type='text/event-stream')

    except Exception as e:
        return handle_api_error("Could not generate exl2-stream, encountered error: ", e)


def create_fim_content(prefix: str, suffix: str, middle: str, language: str = 'python') -> str:
    '''Prepare chat template with FIM tokens'''

    # 1. Define the special FIM tokens for your model.
    # These are standard for CodeLlama, DeepSeek Coder, etc.
    FIM_PREFIX_TOKEN = "<fim_prefix>"
    FIM_SUFFIX_TOKEN = "<fim_suffix>"
    FIM_MIDDLE_TOKEN = "<fim_middle>"

    # 2. Construct the prompt in the correct, raw FIM format.
    # This is the ONLY thing the model should see. No chat, no instructions.
    return f"{FIM_PREFIX_TOKEN}{prefix}{FIM_SUFFIX_TOKEN}{suffix}{FIM_MIDDLE_TOKEN}"


def assemble_fim_messages(prefix: str, suffix: str, middle: str, language: str) -> list:
    try:
        fim_content = create_fim_content(prefix, suffix, middle, language)
        return [
            {"role": "system", "content": f"You are a code completion assistant. Complete the code between the prefix and suffix provided by the user. Only output the completion - no explanations & exclude the prefix and suffix. ONLY COMPLETION. Language: {language}"},
            {"role": "user", "content": fim_content}
        ]
    except Exception as e:
        handle_local_error("Could not assemble FIM messages, encountered error: ", e)


def auto_tokenize_and_encode_fim_messages(exl_tokenizer, fim_messages: list) -> list:
    try:
        templated_messages = auto_tokenizer_apply_chat_template(conversation=fim_messages, add_generation_prompt=True, tokenize=False)
        return exl_tokenizer.encode(templated_messages, encode_special_tokens=True)
    except Exception as e:
        handle_local_error("Could not get templated and encoded FIM messages, encountered error: ", e)


def truncate_fim_content(prefix: str, suffix: str, middle: str, language: str, max_seq_len: int) -> str:
    '''Truncate FIM content to fit within token limits'''

    # Estimate token count (rough approximation: 1 token = 4 chars)
    chars_per_token = 4
    max_chars = max_seq_len * chars_per_token

    # Truncate proportionally:
    prefix_chars = int(max_chars * 0.4)
    suffix_chars = int(max_chars * 0.4)
    middle_chars = max_chars - prefix_chars - suffix_chars - 200 # Reserve for formatting

    # Truncate prefix from start, suffix from end
    if len(prefix) > prefix_chars: prefix = prefix[-prefix_chars:]
    if len(suffix) > suffix_chars: suffix = suffix[:suffix_chars]
    if len(middle) > middle_chars: middle = middle[:middle_chars]

    return assemble_fim_messages(prefix, suffix, middle, language)


def get_final_exl_encoded_fim_input_ids(exl_tokenizer, prefix: str, suffix: str, middle: str, language: str, max_length: int) -> list:

    try:
        fim_messages = assemble_fim_messages(prefix, suffix, middle, language)
        exl_tokenized_messages = auto_tokenize_and_encode_fim_messages(exl_tokenizer, fim_messages)

        while len(exl_tokenized_messages[0]) > max_length:
            print(f"FIM messages are too long, truncating... Current length: {len(exl_tokenized_messages[0])}, Max length: {max_length}")
            trimmed_fim_messages = truncate_fim_content(prefix, suffix, middle, language, max_length)
            exl_tokenized_messages = auto_tokenize_and_encode_fim_messages(exl_tokenizer, trimmed_fim_messages)

        print(f"Length of final ExLlama encoded FIM input IDs: {len(exl_tokenized_messages[0])}. Max length: {max_length}")
        return exl_tokenized_messages
    except Exception as e:
        handle_local_error("Could not get final ExLlama encoded FIM input IDs, encountered error: ", e)


@app.route('/exl2_fim_stream', methods=['POST'])
def exl2_fim_stream():
    """
    Fill-in-the-Middle (FIM) code completion using ExLlamaV2 model
    
    This endpoint provides streaming code completion using Fill-in-the-Middle technique,
    where the model completes code between a prefix and suffix context.
    
    OpenAPI 3.0.0 Specification is available in the `hfw-openapi-3-specs.yaml` file.
    """

    print("\n\nexl2-fim-stream route triggered\n\n")

    try:
        data = request.json
        if isinstance(data, str):   # must convert to a list
            data = json.loads(data)
        
        prefix = data.get('prefix', '')
        suffix = data.get('suffix', '')
        middle = data.get('middle', '')
        language = data.get('language', 'python')

        if not prefix and not suffix:
            raise Exception("Prefix and suffix cannot be empty for FIM completion")

        gen_settings, config_data  = get_exl2_gen_settings(request)
        user_queue = queue.Queue()
    except Exception as e:
        return handle_api_error("Could not setup for exl2-fim-stream, encountered error: ", e)
    
    try:
        exl_encoded_fim_input_ids = get_final_exl_encoded_fim_input_ids(EXL2_TOKENIZER, prefix, suffix, middle, language, 8192) # max 8k tokens
    except Exception as e:
        return handle_api_error("Could not create FIM prompt with chat template, encountered error: ", e)

    try:
        job = ExLlamaV2DynamicJob(
            input_ids= exl_encoded_fim_input_ids,
            max_new_tokens = config_data['max_new_tokens'],
            stop_conditions = [EXL2_TOKENIZER.eos_token_id, AUTO_TOKENIZER.eos_token_id],
            gen_settings = gen_settings
        )
        job.response_queue = user_queue
        EXL2_GENERATOR.enqueue(job)
    except Exception as e:
        return handle_api_error("Could not create ExLlamaV2-DynamicJob object. Error details follow. In case of context-window issues, ensure your max_new_tokens does not exceed the allocated context-window, and of course, allocate an adequate context-window! Error: ", e)

    try:
        def generate():

            global STOP_GENERATION
            STOP_GENERATION = False

            while True:
                if STOP_GENERATION: # Handle Manual Stop Signal
                    print("\n\nStopping generation with stop_event\n\n")
                    try:
                        EXL2_GENERATOR.cancel(job)
                    except:
                        pass
                    STOP_GENERATION = False
                    break
                
                token = user_queue.get()
                if token is None:
                    break
                
                yield f"data: {json.dumps(token)}\n\n"
            
            yield f"event: END\ndata: \"null\"\n\n"
            
            print("\nexl2-fim-stream done\n")

        print("\n\nInferencing Begins!\n\n")
        return Response(generate(), content_type='text/event-stream')
    
    except Exception as e:
        return handle_api_error("Could not generate exl2-fim-stream, encountered error: ", e)


###################################-------------Exl2 Logic Ends-------------###################################


###################################-------------Exl3 Logic Begins-------------###################################

def get_exl3_sampler(request):

    try:
        config_data = read_config(['max_new_tokens', 'temperature', 'top_k', 'top_p', 'min_p', 'rep_p', 'pres_p', 'freq_p', 'rep_sustain_range', 'rep_decay_range', 'exl3_total_context'])
    except Exception as e:  # Not using `handle_local_error` as the necessary params may be in the request headers so why error out here?
        handle_error_no_return("Could not read values from hf_config.json when attempting exl3-sampler, relying on request headers instead. Encountered error: ", e)
        config_data = {}

    try:
        exl3_sampler = ComboSampler(
            rep_p = float(request.headers.get('X-Repetition-Penalty', str(config_data.get('repetition_penalty', '1')))),
            pres_p = float(request.headers.get('X-Presence-Penalty', str(config_data.get('presence_penalty', '0')))),
            freq_p = float(request.headers.get('X-Frequency-Penalty', str(config_data.get('frequency_penalty', '0')))),
            rep_sustain_range = int(float(request.headers.get('X-Repetition-Sustain-Range', str(config_data.get('penalty_range', '10e7'))))),
            rep_decay_range = int(request.headers.get('X-Repetition-Decay-Range', str(config_data.get('penalty_range', '0')))),
            temperature = float(request.headers.get('X-Temperature', str(config_data.get('temperature', '0.1')))),
            min_p = float(request.headers.get('X-Min-P', str(config_data.get('min_p', '0.1')))),
            top_k = int(request.headers.get('X-Top-K', str(config_data.get('top_k', '40')))),
            top_p = float(request.headers.get('X-Top-P', str(config_data.get('top_p', '0.9'))))
        )
        config_data['max_new_tokens'] = int(request.headers.get('X-Max-New-Tokens', str(config_data.get('max_new_tokens', '2048'))))
        print("\nExLlamaV3 Sampler Defined Successfully\n")
    except Exception as e:
        handle_error_no_return("Could not create ExLlamaV3 Sampler, encountered error: ", e)
        exl3_sampler = None
    
    return exl3_sampler, config_data


@app.route('/exl3_stream', methods=['POST'])
def exl3_stream():
    """
    Streaming text generation using ExLlamaV3 model
    
    This endpoint provides streaming text generation using the ExLlamaV3 model with dynamic generation capabilities.
    
    ---
    OpenAPI 3.0.0 Specification is available in the `hfw-openapi-3-specs.yaml` file.
    """

    print("\n\nexl3-stream route triggered\n\n")

    try:
        data = request.json
        if isinstance(data, str):   # must convert to a list
            data = json.loads(data)
        messages = data.get('messages', [])
        tools = data.get('tools', None)
        exl3_sampler, config_data  = get_exl3_sampler(request)
        user_queue = queue.Queue()
    except Exception as e:
        return handle_api_error("Could not setup for exl3-stream, encountered error: ", e)
    
    try:
        tokenized_messages = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        return handle_api_error("Could not tokenize messages for exl2-stream, encountered error: ", e)

    try:
        job = Job(
            input_ids= EXL3_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True),
            max_new_tokens = config_data['max_new_tokens'],
            stop_conditions = STOP_TOKENS,
            sampler = exl3_sampler
        )
        job.response_queue = user_queue
        EXL3_GENERATOR.enqueue(job)
    except Exception as e:
        return handle_api_error("Could not create ExLlamaV3 Job object for exl3-stream. Error details follow. In case of context-window issues, ensure your max_new_tokens does not exceed the allocated context-window, and of course, allocate an adequate context-window! Error: ", e)

    try:
        def generate():

            global STOP_GENERATION
            STOP_GENERATION = False

            while True:
                if STOP_GENERATION: # Handle Manual Stop Signal
                    print("\n\nStopping generation with stop_event\n\n")
                    try:
                        EXL3_GENERATOR.cancel(job)
                    except:
                        pass
                    STOP_GENERATION = False
                    break
                
                token = user_queue.get()
                if token is None:
                    break
                
                yield f"data: {json.dumps(token)}\n\n"
            
            yield f"event: END\ndata: \"null\"\n\n"

            print("\nexl3-stream done\n")

        print("\n\nInferencing Begins!\n\n")
        return Response(generate(), content_type='text/event-stream')
    
    except Exception as e:
        return handle_api_error("Could not generate exl3-stream, encountered error: ", e)


@app.route('/exl3_fim_stream', methods=['POST'])
def exl3_fim_stream():
    """
    Fill-in-the-Middle (FIM) code completion using ExLlamaV3 model
    
    This endpoint provides streaming code completion using Fill-in-the-Middle technique,
    where the model completes code between a prefix and suffix context.
    
    ---
    OpenAPI 3.0.0 Specification is available in the `hfw-openapi-3-specs.yaml` file.
    """

    print("\n\nexl3-fim-stream route triggered\n\n")

    try:
        data = request.json
        if isinstance(data, str):   # must convert to a list
            data = json.loads(data)
        
        prefix = data.get('prefix', '')
        suffix = data.get('suffix', '')
        middle = data.get('middle', '')
        language = data.get('language', 'python')

        if not prefix and not suffix:
            raise Exception("Prefix and suffix cannot be empty for FIM completion")

        exl3_sampler, config_data  = get_exl3_sampler(request)
        user_queue = queue.Queue()
    except Exception as e:
        return handle_api_error("Could not read POST-request messages for exl3-fim-stream, encountered error: ", e)

    try:
        exl_encoded_fim_input_ids = get_final_exl_encoded_fim_input_ids(EXL3_TOKENIZER, prefix, suffix, middle, language, 8192) # max 8k tokens
    except Exception as e:
        return handle_api_error("Could not get final ExLlama encoded FIM input IDs, encountered error: ", e)
    
    try:
        job = Job(
            input_ids= exl_encoded_fim_input_ids,
            max_new_tokens = config_data['max_new_tokens'],
            stop_conditions = STOP_TOKENS,
            sampler = exl3_sampler
        )
        job.response_queue = user_queue
        EXL3_GENERATOR.enqueue(job)
    except Exception as e:
        return handle_api_error("Could not create ExLlamaV3 Job object for exl3-fim-stream. Error details follow. In case of context-window issues, ensure your max_new_tokens does not exceed the allocated context-window, and of course, allocate an adequate context-window! Error: ", e)

    try:
        def generate():

            global STOP_GENERATION
            STOP_GENERATION = False

            while True:
                if STOP_GENERATION: # Handle Manual Stop Signal
                    print("\n\nStopping generation with stop_event\n\n")
                    try:
                        EXL3_GENERATOR.cancel(job)
                    except:
                        pass
                    STOP_GENERATION = False
                    break
                
                token = user_queue.get()
                if token is None:
                    break
                
                # send clean code completion
                yield f"data: {json.dumps(token)}\n\n"
            
            yield f"event: END\ndata: \"null\"\n\n"
            
            print("\nexl3-fim-stream done\n")

        print("\n\nInferencing Begins!\n\n")
        return Response(generate(), content_type='text/event-stream')
    
    except Exception as e:
        return handle_api_error("Could not generate exl3-fim-stream, encountered error: ", e)



###################################-------------OpenAI Compatible API-------------###################################

def handle_transformers_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop):
    """Handle streaming OpenAI-compatible responses using Transformers"""
    
    llm_semaphore.acquire()

    print("\n\nLLM semaphore acquired by OpenAI/transformers-completions Streaming Route\n\n")

    try:
        read_return = read_config(['model_id', 'max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep'])
        generation_config = {
            "max_new_tokens": max_tokens or int(read_return['max_new_tokens']),
            "return_full_text": False,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_k": top_k or int(read_return['top_k']),
            "top_p": top_p,
            "use_cache": True
        }

        inputs = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        decoded_inputs = AUTO_TOKENIZER.decode(inputs['input_ids'][0].tolist(), skip_special_tokens=False)    # Setting skip_special_tokens=False to keep: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.
    except Exception as e:
        llm_semaphore.release()
        return jsonify(error={"message": str(e), "type": "server_error"}), 500

    stop_event = threading.Event()
    data_queue = queue.Queue()

    def callback(data):
        data_queue.put(data)

    custom_streamer = CustomTextStreamer(AUTO_TOKENIZER, skip_special_tokens=True, skip_prompt=True)    # special tokens need not be streamed though!
    custom_streamer.callback = callback

    def llm_task():

        global PIPE

        try:
            generation_config["streamer"] = custom_streamer
            generation_config["stopping_criteria"] = StoppingCriteriaList([StopOnEvent(stop_event)])  # StoppingCriteriaList is a container that holds a list of StoppingCriteria objects. In our case, we have only one such object, which is our custom StoppingCriteria class, initialized with the stop_event object.
            output = PIPE(decoded_inputs, **generation_config)
        except Exception as e:
            data_queue.put(f"Error: {str(e)}") # Pass error to client via queue
        finally:
            data_queue.put(None)
            llm_semaphore.release()

    def generate():        
        
        global STOP_GENERATION
        STOP_GENERATION = False

        try:
            thread = threading.Thread(target=llm_task)
            thread.start()
        except Exception as e:
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            return
        
        # OpenAI streaming format
        created = int(time.time())
        chunk_id = f"chatcmpl-{''.join(random.choices('0123456789abcdef', k=24))}"

        # Send First Chunk - Role Chunk - Expected by strict OpenAI clients in streaming mode!
        role_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": f"Transformers-{read_return['model_id']}",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        # Track thinking state
        in_thinking_block = False
        accumulated_text = ""

        while True:
            if STOP_GENERATION:
                stop_event.set()
                thread.join()
                break
            
            token = data_queue.get()
            if token is None:
                thread.join()
                break
            
            accumulated_text += token

            # Check for openning thinking tag
            if '<think>' in accumulated_text:
                in_thinking_block = True
                # Send any text before think as regular content
                before_think = accumulated_text.split('<think>', 1)[0]
                if before_think:
                    chunk = {
                        "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                        "model": f"Transformers-{read_return['model_id']}",
                        "choices": [{"index": 0, "delta": {"content": before_think}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                accumulated_text = accumulated_text.split('<think>', 1)[1]
                continue
            
            # Check for closing thinking tag
            if '</think>' in accumulated_text:
                # Send thinking content as reasoning_content
                thinking_text = accumulated_text.split('</think>', 1)[0]
                if thinking_text:
                    chunk = {
                        "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                        "model": f"Transformers-{read_return['model_id']}",
                        "choices": [{"index": 0, "delta": {"reasoning_content": [{"type": "text", "text": thinking_text}]}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                in_thinking_block = False
                accumulated_text = accumulated_text.split('</think>', 1)[1]
                continue

            # If we have any accumulated text, send it as regular content
            if len(accumulated_text) > 0:
                if in_thinking_block:
                    # Send as reasoning_content
                    chunk = {
                        "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                        "model": f"Transformers-{read_return['model_id']}",
                        "choices": [{"index": 0, "delta": {"reasoning_content": [{"type": "text", "text": accumulated_text}]}, "finish_reason": None}]
                    }
                else:
                    # Send as regular content
                    chunk = {
                        "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                        "model": f"Transformers-{read_return['model_id']}",
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                    }
                yield f"data: {json.dumps(chunk)}\n\n"
                accumulated_text = ""

        # Send any remaining text
        if accumulated_text:
            field = "reasoning_content" if in_thinking_block else "content"
            content = [{"type": "text", "text": accumulated_text}] if in_thinking_block else accumulated_text
            chunk = {
                "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                "model": f"Transformers-{read_return['model_id']}",
                "choices": [{"index": 0, "delta": {field: content}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            accumulated_text = ""
        
        # Send final closing chunk
        final_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": f"Transformers-{read_return['model_id']}",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        STOP_GENERATION = False

        print("\nOpenAI/completions_stream done\n")
            
    print("\n\nInferencing Begins!\n\n")
    return Response(generate(), content_type='text/event-stream')


def handle_transformers_non_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop):
    """Handle non-streaming OpenAI-compatible responses using Transformers"""

    with llm_semaphore:

        print("\n\nLLM semaphore acquired by OpenAI/transformers-completions Non-Streaming Route\n\n")

        try:
            read_return = read_config(['model_id', 'max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep'])
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens or int(read_return['max_new_tokens']),
                return_full_text=False,
                temperature=temperature,
                do_sample=temperature > 0,
                top_k=top_k or int(read_return['top_k']),
                top_p=top_p,
                use_cache=True
            )

            inputs = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs.to(PIPE.model.device)

            output = PIPE.model.generate(**inputs, generation_config=generation_config)
            input_length = inputs.input_ids.shape[1]
            decoded_output = AUTO_TOKENIZER.decode(output[0][input_length:], skip_special_tokens=True)
            inference_output = decoded_output

            # 7. Split reasoning vs visible content
            # The below approach is a simple left-to-right parser for a single tag type, 
            # resilient to "no more tags" (-1), and tolerant of missing close tags:
            reasoning_blocks = []
            visible_parts = []
            cursor = 0  # tracks the position in full_response where we're currently reading - everything before has already been handled.
            while True:
                start = inference_output.find('<think>', cursor)   # searches for the next opening tag after cursor - find returns the index of the match, or -1 if not found.
                if start == -1:
                    visible_parts.append(inference_output[cursor:])  # no more opening tags, so everything remaining is visible content.
                    break
                end = inference_output.find('</think>', start + len('<think>'))  # Otherwise, we look for the matching close tag
                if end == -1:   
                    # Again, -1 means "not found." In that case we treat the rest as visible text and break (so a missing closing tag doesn't crash the loop)
                    visible_parts.append(inference_output[cursor:])
                    break
                # If both tags found:
                if start > cursor:
                    visible_parts.append(inference_output[cursor:start])   # text before think is visible content
                # Text inside the tags is reasoning content, so we push it into reasoning_blocks:
                thinking_text = inference_output[start + len('</think>'):end]
                if thinking_text:
                    reasoning_blocks.append({"type": "text", "text": thinking_text})
                # Advance cursor to just after the closing tag, so the next iteration continues scanning after that block.
                cursor = end + len('</think>')
            visible_content = ''.join(visible_parts)
        
            print("\nOpenAI/completions done\n")
            prompt_tokens = len(inputs['input_ids'][0])
            completion_tokens = len(AUTO_TOKENIZER.encode(inference_output))

            message = {
                "role": "assistant",
                "content": visible_content
            }
            if reasoning_blocks:
                message["reasoning_content"] = reasoning_blocks

            return jsonify({
                "id": f"chatcmpl-{''.join(random.choices('0123456789abcdef', k=24))}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"Transformers-{read_return['model_id']}",
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            })
        
        except Exception as e:
            return jsonify(error={"message": str(e), "type": "server_error"}), 500


def handle_exl2_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop):
    """Handle streaming OpenAI-compatible responses using ExLlamaV2"""
    print("\n\nOpenAI/exl2-stream route triggered\n\n")

    try:
        # 1. Setup response queue for this specific request
        user_queue = queue.Queue()

        # 2. Get generator & other config settings
        gen_settings, config_data  = get_exl2_gen_settings(request)
        if max_tokens: config_data['max_new_tokens'] = max_tokens
        if temperature is not None: gen_settings.temperature = temperature
        if top_p is not None: gen_settings.top_p = top_p
        if top_k is not None: gen_settings.top_k = top_k
        
        # 3. Setup stop conditions
        stop_tokens = [EXL2_TOKENIZER.eos_token_id, AUTO_TOKENIZER.eos_token_id]
        if stop:
            if isinstance(stop, str):stop = [stop]
            for stop_string in stop:
                stop_ids = EXL2_TOKENIZER.encode(stop_string).flatten().tolist()    # Encode to Tensor, then flatten to List of Ints
                if len(stop_ids) > 0: stop_tokens.append(stop_ids)  # Append the *sequence* (the list itself) to conditions - append because lists should be added as is, not flattened

        # 4. Create Job
        tokenized_messages = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, tokenize=False)
        
        job = ExLlamaV2DynamicJob(
            input_ids= EXL2_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True),
            max_new_tokens = config_data['max_new_tokens'],
            stop_conditions = stop_tokens,
            gen_settings = gen_settings
        )
        
        # 5. Attach Queue & Enqueue to Global Generator (running in background thread)
        job.response_queue = user_queue
        EXL2_GENERATOR.enqueue(job)
    
        # 6. Streaming Response Generator
        def generate():
            
            # OpenAI streaming format - Send Role Chunk First - Expected by strict OpenAI clients in streaming mode!
            created = int(time.time())
            chunk_id = f"chatcmpl-{''.join(random.choices('0123456789abcdef', k=24))}"

            role_chunk = {
                "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                "model": f"ExLlamaV2-{config_data['model_id']}",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            # Track thinking state
            in_thinking_block = False
            accumulated_text = ""

            while True:
                # Block until background worker puts a token here
                token = user_queue.get()
                if token is None:   # EOS Signal
                    break

                accumulated_text += token

                # Check for openning thinking tag
                if '<think>' in accumulated_text:
                    in_thinking_block = True
                    # Send any text before think as regular content
                    before_think = accumulated_text.split('<think>', 1)[0]
                    if before_think:
                        chunk = {
                            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                            "model": f"ExLlamaV2-{config_data['model_id']}",
                            "choices": [{"index": 0, "delta": {"content": before_think}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    accumulated_text = accumulated_text.split('<think>', 1)[1]
                    continue
                
                # Check for closing thinking tag
                if '</think>' in accumulated_text:
                    # Send thinking content as reasoning_content
                    thinking_text = accumulated_text.split('</think>', 1)[0]
                    if thinking_text:
                        chunk = {
                            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                            "model": f"ExLlamaV2-{config_data['model_id']}",
                            "choices": [{"index": 0, "delta": {"reasoning_content": [{"type": "text", "text": thinking_text}]}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    in_thinking_block = False
                    accumulated_text = accumulated_text.split('</think>', 1)[1]
                    continue

                # If we have any accumulated text, send it as regular content
                if len(accumulated_text) > 0:
                    if in_thinking_block:
                        # Send as reasoning_content
                        chunk = {
                            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                            "model": f"ExLlamaV2-{config_data['model_id']}",
                            "choices": [{"index": 0, "delta": {"reasoning_content": [{"type": "text", "text": accumulated_text}]}, "finish_reason": None}]
                        }
                    else:
                        # Send as regular content
                        chunk = {
                            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                            "model": f"ExLlamaV2-{config_data['model_id']}",
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                        }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    accumulated_text = ""

            # Send any remaining text
            if accumulated_text:
                field = "reasoning_content" if in_thinking_block else "content"
                content = [{"type": "text", "text": accumulated_text}] if in_thinking_block else accumulated_text
                chunk = {
                    "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                    "model": f"ExLlamaV2-{config_data['model_id']}",
                    "choices": [{"index": 0, "delta": {field: content}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                accumulated_text = ""
            
            # Send final closing chunk
            final_chunk = {
                "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                "model": f"ExLlamaV2-{config_data['model_id']}",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
            print("\nOpenAI/exl2-stream done\n")

        print("\n\nInferencing Begins!\n\n")
        return Response(generate(), content_type='text/event-stream')

    except Exception as e:
        return jsonify(error={"message": str(e), "type": "server_error"}), 500


def handle_exl2_non_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop):
    """Handle non-streaming OpenAI-compatible responses using ExLlamaV2"""
    print("\n\nOpenAI/exl2-Non-Streaming route triggered\n\n")

    try:
        # 1. Setup response queue for this specific request
        user_queue = queue.Queue()

        # 2. Get generator & other config settings
        gen_settings, config_data  = get_exl2_gen_settings(request)
        if max_tokens: config_data['max_new_tokens'] = max_tokens
        if temperature is not None: gen_settings.temperature = temperature
        if top_p is not None: gen_settings.top_p = top_p
        if top_k is not None: gen_settings.top_k = top_k

        # 3. Setup stop conditions
        stop_token_list = [EXL2_TOKENIZER.eos_token_id, AUTO_TOKENIZER.eos_token_id]
        if stop:
            if isinstance(stop, str):stop = [stop]
            for stop_string in stop:
                stop_ids = EXL2_TOKENIZER.encode(stop_string).flatten().tolist()    # Encode to Tensor, then flatten to List of Ints
                if len(stop_ids) > 0: stop_token_list.append(stop_ids)  # Append the *sequence* (the list itself) to conditions - append because lists should be added as is, not flattened
        
        # 4. Create Job
        tokenized_messages = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, tokenize=False)
        
        job = ExLlamaV2DynamicJob(
            input_ids= EXL2_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True),
            max_new_tokens = config_data['max_new_tokens'],
            stop_conditions = stop_token_list,
            gen_settings = gen_settings
        )

        # 5. Attach Queue & Enqueue to Global Generator (running in background thread)
        job.response_queue = user_queue
        EXL2_GENERATOR.enqueue(job)

        # 6. Consume Queue Synchronously (Accumulate Response)
        full_response = ""
        while True:
            token = user_queue.get()
            if token is None:   # EOS Signal
                break
            full_response += token

        # 7. Split reasoning vs visible content
        # The below approach is a simple left-to-right parser for a single tag type, 
        # resilient to "no more tags" (-1), and tolerant of missing close tags:
        reasoning_blocks = []
        visible_parts = []
        cursor = 0  # tracks the position in full_response where we're currently reading - everything before has already been handled.
        while True:
            start = full_response.find('<think>', cursor)   # searches for the next opening tag after cursor - find returns the index of the match, or -1 if not found.
            if start == -1:
                visible_parts.append(full_response[cursor:])  # no more opening tags, so everything remaining is visible content.
                break
            end = full_response.find('</think>', start + len('<think>'))  # Otherwise, we look for the matching close tag
            if end == -1:   
                # Again, -1 means "not found." In that case we treat the rest as visible text and break (so a missing closing tag doesn't crash the loop)
                visible_parts.append(full_response[cursor:])
                break
            # If both tags found:
            if start > cursor:
                visible_parts.append(full_response[cursor:start])   # text before think is visible content
            # Text inside the tags is reasoning content, so we push it into reasoning_blocks:
            thinking_text = full_response[start + len('</think>'):end]
            if thinking_text:
                reasoning_blocks.append({"type": "text", "text": thinking_text})
            # Advance cursor to just after the closing tag, so the next iteration continues scanning after that block.
            cursor = end + len('</think>')
        visible_content = ''.join(visible_parts)
        
        print("\nOpenAI/exl2-non-streaming done\n")
        prompt_tokens = len(EXL2_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True))
        completion_tokens = len(EXL2_TOKENIZER.encode(full_response, encode_special_tokens=True))

        message = {
            "role": "assistant",
            "content": visible_content
        }
        if reasoning_blocks:
            message["reasoning_content"] = reasoning_blocks
        
        return jsonify({
            "id": f"chatcmpl-{''.join(random.choices('0123456789abcdef', k=24))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"ExLlamaV2-{config_data['model_id']}",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })

    except Exception as e:
        return jsonify(error={"message": str(e), "type": "server_error"}), 500


def handle_exl3_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop, presence_penalty, frequency_penalty):
    """Handle streaming OpenAI-compatible responses using ExLlamaV3"""
    print("\n\nOpenAI/exl3-stream route triggered\n\n")

    try:
        #1. Setup response queue for this specific request
        user_queue = queue.Queue()

        # 2. Get generator & other config settings
        config_data = read_config(['model_id', 'max_new_tokens', 'temperature', 'top_k', 'top_p', 'min_p', 'rep_p', 'pres_p', 'freq_p', 'rep_sustain_range', 'rep_decay_range', 'exl3_total_context'])
        config_data['max_new_tokens'] = max_tokens or int(config_data['max_new_tokens'])
    
        exl3_sampler = ComboSampler(
            rep_p = config_data['rep_p'],
            pres_p = presence_penalty or config_data['pres_p'],
            freq_p = frequency_penalty or config_data['freq_p'],
            rep_sustain_range = config_data['rep_sustain_range'],
            rep_decay_range = config_data['rep_decay_range'],
            temperature = temperature or config_data['temperature'],
            min_p = config_data['min_p'],
            top_k = top_k or config_data['top_k'],
            top_p = top_p or config_data['top_p']
        )

        # 3. Setup stop conditions
        stop_token_list = list(STOP_TOKENS) # casting done to create a shallow copy of the list, not a reference to and incorrect modification of the original list
        if stop:
            if isinstance(stop, str): stop = [stop]
            for stop_string in stop:
                # ExLlamaV3 supports stop strings natively, so we append the string directly
                # passing a list of token IDs (sequence) would raise a ValueError in ExLlamaV3's Job class
                stop_token_list.append(stop_string)

        # 4. Create Job
        tokenized_messages = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, tokenize=False)

        job = Job(
            input_ids= EXL3_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True),
            max_new_tokens = config_data['max_new_tokens'],
            stop_conditions = stop_token_list,
            sampler = exl3_sampler
        )
        
        # 5. Attach Queue & Enqueue to Global Generator (running in background thread)
        job.response_queue = user_queue
        EXL3_GENERATOR.enqueue(job)

        # 6. Streaming Response Generator
        def generate():
            
            # OpenAI streaming format - Send Role Chunk First - Expected by strict OpenAI clients in streaming mode!
            created = int(time.time())
            chunk_id = f"chatcmpl-{''.join(random.choices('0123456789abcdef', k=24))}"

            role_chunk = {
                "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                "model": f"ExLlamaV3-{config_data['model_id']}",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            # Track thinking state
            in_thinking_block = False
            accumulated_text = ""

            while True:
                # Block until background worker puts a token here
                token = user_queue.get()
                if token is None:   # EOS Signal
                    break

                accumulated_text += token

                # Check for openning thinking tag
                if '<think>' in accumulated_text:
                    in_thinking_block = True
                    # Send any text before think as regular content
                    before_think = accumulated_text.split('<think>', 1)[0]
                    if before_think:
                        chunk = {
                            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                            "model": f"ExLlamaV3-{config_data['model_id']}",
                            "choices": [{"index": 0, "delta": {"content": before_think}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    accumulated_text = accumulated_text.split('<think>', 1)[1]
                    continue
                
                # Check for closing thinking tag
                if '</think>' in accumulated_text:
                    # Send thinking content as reasoning_content
                    thinking_text = accumulated_text.split('</think>', 1)[0]
                    if thinking_text:
                        chunk = {
                            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                            "model": f"ExLlamaV3-{config_data['model_id']}",
                            "choices": [{"index": 0, "delta": {"reasoning_content": [{"type": "text", "text": thinking_text}]}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    in_thinking_block = False
                    accumulated_text = accumulated_text.split('</think>', 1)[1]
                    continue

                # If we have any accumulated text, send it as regular content
                if len(accumulated_text) > 0:
                    if in_thinking_block:
                        # Send as reasoning_content
                        chunk = {
                            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                            "model": f"ExLlamaV3-{config_data['model_id']}",
                            "choices": [{"index": 0, "delta": {"reasoning_content": [{"type": "text", "text": accumulated_text}]}, "finish_reason": None}]
                        }
                    else:
                        # Send as regular content
                        chunk = {
                            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                            "model": f"ExLlamaV3-{config_data['model_id']}",
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                        }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    accumulated_text = ""

            # Send any remaining text
            if accumulated_text:
                field = "reasoning_content" if in_thinking_block else "content"
                content = [{"type": "text", "text": accumulated_text}] if in_thinking_block else accumulated_text
                chunk = {
                    "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                    "model": f"ExLlamaV3-{config_data['model_id']}",
                    "choices": [{"index": 0, "delta": {field: content}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                accumulated_text = ""

            # Send final closing chunk
            final_chunk = {
                "id": chunk_id, "object": "chat.completion.chunk","created": created,
                "model": f"ExLlamaV3-{config_data['model_id']}",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
            print("\nexl3-stream done\n")

        print("\n\nInferencing Begins!\n\n")
        return Response(generate(), content_type='text/event-stream')
    
    except Exception as e:
        return jsonify(error={"message": str(e), "type": "server_error"}), 500


def handle_exl3_non_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop, presence_penalty, frequency_penalty):
    """Handle non-streaming OpenAI-compatible responses using ExLlamaV3"""
    print("\n\nOpenAI/exl3-Non-Streaming route triggered\n\n")

    try:
        # 1. Setup response queue for this specific request
        user_queue = queue.Queue()

        # 2. Get generator & other config settings
        config_data = read_config(['model_id', 'max_new_tokens', 'temperature', 'top_k', 'top_p', 'min_p', 'rep_p', 'pres_p', 'freq_p', 'rep_sustain_range', 'rep_decay_range', 'exl3_total_context'])
        config_data['max_new_tokens'] = max_tokens or int(config_data['max_new_tokens'])
    
        exl3_sampler = ComboSampler(
            rep_p = config_data['rep_p'],
            pres_p = presence_penalty or config_data['pres_p'],
            freq_p = frequency_penalty or config_data['freq_p'],
            rep_sustain_range = config_data['rep_sustain_range'],
            rep_decay_range = config_data['rep_decay_range'],
            temperature = temperature or config_data['temperature'],
            min_p = config_data['min_p'],
            top_k = top_k or config_data['top_k'],
            top_p = top_p or config_data['top_p']
        )

        # 3. Setup stop conditions
        stop_token_list = list(STOP_TOKENS)
        if stop:
            if isinstance(stop, str):stop = [stop]
            for stop_string in stop:
                # ExLlamaV3 supports stop strings natively, so we append the string directly
                # passing a list of token IDs (sequence) would raise a ValueError in ExLlamaV3's Job class
                stop_token_list.append(stop_string)

        # 4. Create Job
        tokenized_messages = auto_tokenizer_apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, tokenize=False)

        job = Job(
            input_ids= EXL3_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True),
            max_new_tokens = config_data['max_new_tokens'],
            stop_conditions = stop_token_list,
            sampler = exl3_sampler
        )

        # 5. Attach Queue & Enqueue to Global Generator (running in background thread)
        job.response_queue = user_queue
        EXL3_GENERATOR.enqueue(job)

        # 6. Consume Queue Synchronously (Accumulate Response)
        full_response = ""
        while True:
            token = user_queue.get()
            if token is None:   # EOS Signal
                break
            full_response += token

        # 7. Split reasoning vs visible content
        # The below approach is a simple left-to-right parser for a single tag type, 
        # resilient to "no more tags" (-1), and tolerant of missing close tags:
        reasoning_blocks = []
        visible_parts = []
        cursor = 0  # tracks the position in full_response where we're currently reading - everything before has already been handled.
        while True:
            start = full_response.find('<think>', cursor)   # searches for the next opening tag after cursor - find returns the index of the match, or -1 if not found.
            if start == -1:
                visible_parts.append(full_response[cursor:])  # no more opening tags, so everything remaining is visible content.
                break
            end = full_response.find('</think>', start + len('<think>'))  # Otherwise, we look for the matching close tag
            if end == -1:   
                # Again, -1 means "not found." In that case we treat the rest as visible text and break (so a missing closing tag doesn't crash the loop)
                visible_parts.append(full_response[cursor:])
                break
            # If both tags found:
            if start > cursor:
                visible_parts.append(full_response[cursor:start])   # text before think is visible content
            # Text inside the tags is reasoning content, so we push it into reasoning_blocks:
            thinking_text = full_response[start + len('</think>'):end]
            if thinking_text:
                reasoning_blocks.append({"type": "text", "text": thinking_text})
            # Advance cursor to just after the closing tag, so the next iteration continues scanning after that block.
            cursor = end + len('</think>')
        visible_content = ''.join(visible_parts)

        print("\nOpenAI/exl3-non-streaming done\n")
        prompt_tokens = len(EXL3_TOKENIZER.encode(tokenized_messages, encode_special_tokens=True))
        completion_tokens = len(EXL3_TOKENIZER.encode(full_response, encode_special_tokens=True))

        message = {
            "role": "assistant",
            "content": visible_content
        }
        if reasoning_blocks:
            message["reasoning_content"] = reasoning_blocks

        return jsonify({
            "id": f"chatcmpl-{''.join(random.choices('0123456789abcdef', k=24))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"ExLlamaV3-{config_data['model_id']}",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })
        
    except Exception as e:
        return jsonify(error={"message": str(e), "type": "server_error"}), 500


def handle_openai_streaming(backend, messages, tools, max_tokens, temperature, top_p, top_k, stop, presence_penalty, frequency_penalty):
    """Handle streaming OpenAI-compatible responses"""
    if backend == 'transformers': return handle_transformers_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop)
    elif backend == 'exl2': return handle_exl2_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop)
    elif backend == 'exl3': return handle_exl3_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop, presence_penalty, frequency_penalty)


def handle_openai_non_streaming(backend, messages, tools, max_tokens, temperature, top_p, top_k, stop, presence_penalty, frequency_penalty):
    """Handle non-streaming OpenAI-compatible responses"""
    if backend == 'transformers': return handle_transformers_non_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop)
    elif backend == 'exl2': return handle_exl2_non_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop)
    elif backend == 'exl3': return handle_exl3_non_streaming_openai(messages, tools, max_tokens, temperature, top_p, top_k, stop, presence_penalty, frequency_penalty)


@app.route('/v1/chat/completions', methods=['POST'])
def openai_compatible_api():
    """
    OpenAI-compatible chat completions endpoint that routes to available backends.
    
    Supports both streaming and non-streaming responses based on the 'stream' parameter.
    Routes to available backends in priority order: ExLlamaV3, ExLlamaV2, Transformers.
    
    OpenAPI 3.0.0 Specification follows OpenAI format.
    """

    print("\n\nOpenAI v1/chat/completions route triggered\n\n")
    
    # Check which backends are available
    pipe_ready = PIPE is not None
    exl2_ready = all([EXL2_MODEL, EXL2_CACHE, EXL2_TOKENIZER, EXL2_GENERATOR, AUTO_TOKENIZER])
    exl3_ready = all([EXL3_MODEL, EXL3_CACHE, EXL3_TOKENIZER, EXL3_GENERATOR, STOP_TOKENS, AUTO_TOKENIZER])
    
    if not (pipe_ready or exl2_ready or exl3_ready):
        return jsonify(
            error={
                "message": "No LLM backend is currently loaded",
                "type": "server_error",
                "param": None,
                "code": None
            }
        ), 503 # Service Unavailable
    
    try:
        data = request.json
        if isinstance(data, str):
            data = json.loads(data)
        
        # OpenAI-compatible parameters
        messages = data.get('messages', [])
        tools = data.get('tools', None)
        tool_choice = data.get('tool_choice', 'auto')
        print(f"\nOpenAI Messages: {json.dumps(messages, indent=4)}\n")
        print(f"\nOpenAI Tools: {json.dumps(tools, indent=4)}\n")
        model = data.get('model', 'auto').lower().strip()  # Can be used to force specific backend
        stream = data.get('stream', False)
        max_tokens = data.get('max_tokens', None)
        temperature = data.get('temperature', None)
        top_p = data.get('top_p', None)
        top_k = data.get('top_k', None)
        stop = data.get('stop', None)
        presence_penalty = data.get('presence_penalty', None)
        frequency_penalty = data.get('frequency_penalty', None)
        
    except Exception as e:
        return jsonify(
            error={
                "message": f"Invalid request format: {str(e)}",
                "type": "invalid_request_error",
                "param": None,
                "code": None
            }
        ), 400
    
    # Determine which backend to use
    backend_priority = []
    if pipe_ready: backend_priority.append('transformers')
    if exl3_ready: backend_priority.append('exl3')
    if exl2_ready: backend_priority.append('exl2')
    
    # Allow the `model` param to override the backend priority
    if model == 'exl3' and exl3_ready: selected_backend = 'exl3'
    elif model == 'exl2' and exl2_ready: selected_backend = 'exl2'
    elif model == 'transformers' and pipe_ready: selected_backend = 'transformers'
    else: selected_backend = backend_priority[0] if backend_priority else None

    print(f"\nSelected backend: {selected_backend}\n")

    if not selected_backend:
        return jsonify(
            error={
                "message": "No LLM backend is currently loaded",
                "type": "server_error",
                "param": None,
                "code": None
            }
        ), 503
    
    # Handle tool choice: The user may wish to force a specific tool or forbid tools altogether, but auto-tokenizer's apply_chat_template() method does not 
    # support tool_choice! So we need to handle this manually.
    final_tools = tools
    if tools and tool_choice:
        if tool_choice == 'none':
            # Client explicitly forbids tools, pass none so the system prompt does not mention them!
            final_tools = None
        
        elif isinstance(tool_choice, dict):
            # Client wants to FORCE a specific tool (e.g., {"type": "function", "function": {"name": "wol_turn_on_tv"}})
            # The best 'local model' way to do this is to HIDE all other tools.
            target_name = tool_choice.get('function', {}).get('name')
            if target_name:
                final_tools = [t for t in tools if t['function']['name'] == target_name]

        # 'auto' is the default, so we leave final_tools as-is.
    
    if stream:
        return handle_openai_streaming(selected_backend, messages, final_tools, max_tokens, temperature, top_p, top_k, stop, presence_penalty, frequency_penalty)
    else:
        return handle_openai_non_streaming(selected_backend, messages, final_tools, max_tokens, temperature, top_p, top_k, stop, presence_penalty, frequency_penalty)


@app.route('/v1/models', methods=['GET'])
def list_models():
    print("\n\nOpenAI /v1/models route triggered\n\n")
    
    # 1. Get the currently loaded model name
    try:
        config = read_config(['model_id'])
        current_model_id = config.get('model_id', 'unknown')
    except:
        current_model_id = "default-model"

    # 2. Define the available "Virtual" models
    models_data = [
        # The actual loaded model
        # {
        #     "id": current_model_id,
        #     "object": "model",
        #     "created": int(time.time()),
        #     "owned_by": "hf-waitress"
        # },
        # Virtual backend selectors
        {
            "id": "exl2", 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "hf-waitress"
        },
        {
            "id": "exl3", 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "hf-waitress"
        },
        {
            "id": "transformers", 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "hf-waitress"
        }
    ]

    return jsonify({
        "object": "list",
        "data": models_data
    })


###################################-------------OpenAI Compatible API End-------------###################################



###################################-------------Exl2 Graph Functions Begin-------------###################################

def trim_response(response, start_substring, end_substring, include_start_substring=False, include_end_substring=False):
    print("\nAttempting to trim response...\n")
    try:
        if start_substring in response and end_substring in response:
            start_index = response.rindex(start_substring)  # Sometimes the model re-gurgitates multiple copies of the same dict in it's response
            end_index = response.rindex(end_substring) # rindex() returns the index of the last occurrence of the substring
            
            if not include_start_substring:
                start_index += len(start_substring)
            if include_end_substring:
                end_index += len(end_substring)
            
            return response[start_index:end_index]
        else:
            print(f"\nResponse does not contain start_substring: {start_substring} or end_substring: {end_substring}, returning unchanged response...\n")
            return response
    except Exception as e:
        print(f"Failed to trim response, encountered error: {e}")
        return response


def get_user_query_for_comprehensive_summary(nodes_and_relationships, chunk):
    return f"""For the purpose of creating a Graph Database, nodes and relations were extracted from a chunk of text. Both are provided below, can you provide a concise (under 3000 words) summary, in the style of a report detailing crucial information and insights, for the text_chunk expounding on the nodes and relationships? Thank you!

    <text_chunk>
    {chunk}
    </text_chunk>

    <nodes_and_relationships>
    {json.dumps(nodes_and_relationships)}
    </nodes_and_relationships>


    Output format:
    {{
        "summary": "Your concise summary report (under 3000 words) here"
    }}
    """


def get_minimal_query_for_summary(chunk):
    return f"""For the purpose of creating a Graph Database, nodes and relations were extracted from the below chunk of text. Keeping this in mind, can you provide a concise (under 3000 words) summary, in the style of a report detailing crucial information and insights, for the text_chunk expounding on any nodes and relationships that may be present? Thank you!
    
    <text_chunk>
    {chunk}
    </text_chunk>


    Output format:
    {{
        "summary": "Your concise summary report (under 3000 words) here"
    }}
    """


def create_and_execute_exl2_job(payload:str, max_new_tokens:int, gen_settings):
    """
    Batched-safe synchronous execution.
    Submits to the global worker queue and blocks until completion.
    """

    try:
        # Step 1: Setup Queue
        user_queue = queue.Queue()
        
        # Step 2: Create Job
        job = ExLlamaV2DynamicJob(
            input_ids= EXL2_TOKENIZER.encode(payload, encode_special_tokens=True),
            max_new_tokens = max_new_tokens,
            stop_conditions = [EXL2_TOKENIZER.eos_token_id, AUTO_TOKENIZER.eos_token_id],
            gen_settings = gen_settings
        )

        # 3. Attach Queue & 4. Enqueue to Global Generator
        job.response_queue = user_queue
        EXL2_GENERATOR.enqueue(job)

        # 5. Consume Queue Synchronously (Accumulate Response)
        full_response = ""
        while True:
            token = user_queue.get()
            if token is None:   # EOS Signal
                break
            full_response += token

        return full_response
    
    except Exception as e:
        handle_error_no_return(f"Error in batched execution for payload\n\n{payload[:50]}\n...\n\n(truncated for brevity).\nEncountered error: ", e)
        return ""


def get_request_payload_for_graph_entity_extraction(chunk_text: str):
    try:
        chunk_payload = "Extract nodes and relationships from the following text:\n" + chunk_text + "\n<knowledge_graph>"
        full_payload = auto_tokenizer_apply_chat_template(conversation=[{"role": "user", "content": chunk_payload}], add_generation_prompt=True, tokenize=False)
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
    trimmed_response = trim_response(extraction_response, '{"nodes":', '"}]}', include_start_substring=True, include_end_substring=True)
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
        comprehensive_summary_request_prompt = get_user_query_for_comprehensive_summary(nodes_and_relationships, chunk_text)
        formatted_prompt = auto_tokenizer_apply_chat_template(conversation=[{"role": "user", "content": comprehensive_summary_request_prompt}], add_generation_prompt=True, tokenize=False)
        
        if exl2_prompt_fits_within_max_context_length(formatted_prompt):
            full_response = create_and_execute_exl2_job(payload=formatted_prompt, max_new_tokens=requested_max_new_tokens, gen_settings=gen_settings)
        else:   # No errors are raised if the prompt is larger than the max context length because it'll be auto-truncated which we don't want so best to handle manually!
            minimal_summary_request_prompt = get_minimal_query_for_summary(chunk_text)
            formatted_prompt = auto_tokenizer_apply_chat_template(conversation=[{"role": "user", "content": minimal_summary_request_prompt}], add_generation_prompt=True, tokenize=False)
            
            if exl2_prompt_fits_within_max_context_length(formatted_prompt):
                full_response = create_and_execute_exl2_job(payload=formatted_prompt, max_new_tokens=requested_max_new_tokens, gen_settings=gen_settings)
            else:
                handle_error_no_return(f"Could not generate comprehensive summary for chunk of document {source_doc_name} - too long even with minimal data! Encountered error: ", e)
                return [""]
        
        print("Summary generated, post-processing...\n")
        full_response = trim_response(full_response, '"summary":', '}').replace("'", "") + "\n{Source Document Name: " + source_doc_name + "}\n{Page Number(s): " + str(page_number_list) + "}\n\n"
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

    print("\n\nexl2-graph-extractor route triggered\n\n")

    try:
        chunk_entities = request.json.get('chunk_entities')
        rag_response_mode = request.json.get('rag_response_mode', False)
        gen_settings, config_data = get_exl2_gen_settings(request)
        requested_max_new_tokens = int(config_data.get('max_new_tokens'))
        knowledge_graph_cache_dir = config_data.get('knowledge_graph_cache_dir', '/')
        reuse_graph_extraction_cache = str(request.headers.get('X-Reuse-Extraction-Cache', str(read_config(['reuse_graph_extraction_cache'])['reuse_graph_extraction_cache']).lower())).lower() == 'true'
        # print(f"\nchunk_entities received:\n\n{chunk_entities}\n")
    except Exception as e:
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

    print("\n\nexl2-graph-summarizer route triggered\n\n")

    try:
        chunk_entities = request.json.get('chunk_entities')
        gen_settings, config_data = get_exl2_gen_settings(request)
        requested_max_new_tokens = int(config_data.get('max_new_tokens'))
        knowledge_graph_cache_dir = config_data.get('knowledge_graph_cache_dir', '/')
        reuse_graph_summary_cache = str(request.headers.get('X-Reuse-Summary-Cache', str(read_config(['reuse_graph_summary_cache'])['reuse_graph_summary_cache']).lower())).lower() == 'true'
        # print(f"\nchunk_entities received:\n\n{chunk_entities}\n")
    except Exception as e:
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


################################################-----HF-Waitress Health-Check Methods-----#################################################
def throw_health_check_error(attrib, e):
    print(f"\nHF-Waitress Server online but could not determine LLM's '{attrib}' attribute. Continuing...\n")
    return True


def get_transformers_model_info():

    model_info = {}

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
            model_info["tokenizer_vocab_length"] = len(AUTO_TOKENIZER)
        except Exception as e:
            throw_health_check_error("tokenizer_vocab_length", e)

    try:
        model_info["tokenizer_vocab_size"] = str(AUTO_TOKENIZER.vocab_size)
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
        model_info["tokenizer"] = str(AUTO_TOKENIZER.name_or_path)
    except Exception as e:
        throw_health_check_error("tokenizer", e)

    try:
        model_info["max_seq_length"] = str(AUTO_TOKENIZER.model_max_length)
    except Exception as e:
        throw_health_check_error("max_seq_length", e)


@app.route('/health')
def health():

    print("\n\nHF-Waitress LLM health-check in-progress...\n\n")

    try:
        # Treat any backend being ready as healthy:
        pipe_ready = PIPE is not None
        exl2_ready = all([EXL2_MODEL, EXL2_CACHE, EXL2_TOKENIZER, EXL2_GENERATOR, AUTO_TOKENIZER])
        exl3_ready = all([EXL3_MODEL, EXL3_CACHE, EXL3_TOKENIZER, EXL3_GENERATOR, STOP_TOKENS, AUTO_TOKENIZER])
        asr_ready = MODEL is not None
        
        print(f"\n\nhealth readiness → pipe={pipe_ready}, exl2={exl2_ready}, exl3={exl3_ready}, asr={asr_ready}\n\n")

        if not (pipe_ready or exl2_ready or exl3_ready or asr_ready):
            return jsonify(status="error", message="None of the core backends (transformers, exl2, exl3, asr) are loaded"), 503 # Service Unavailable

        model_info = {}

        if pipe_ready and not exl2_ready and not exl3_ready and not asr_ready: # Implies only Transformers backend is loaded!
            model_info = get_transformers_model_info()
        
        print(f"HF-Waitress LLM-server health-check completed successfully, returning.\n")
        return jsonify(status="ok", model_info=model_info), 200

    except Exception as e:
        handle_api_error("Error checking hf-server health, encountered error: ", e)

#############################################################-----End HF-Waitress Health-Check Methods-----#################################################


@app.route('/restart_server')
def restart_server():
    
    with llm_semaphore:
        print("\n\nrestart-server acquired llm_semaphore, proceeding...\n\n")
        with config_writer_semaphore:
            print("\n\nrestart-server acquired config_writer_semaphore, proceeding...\n\n")
            with error_logging_semaphore:
                print("\n\nrestart-server acquired error_logging_semaphore, proceeding...\n\n")

                try:
                    shutdown_all()
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


def signal_handler(sig, frame):
    '''
    Signal handler for the main process.
    It will shut down the server gracefully by intercepting the interrupt "SIGINT" signal (Ctrl+C).
    We skip semaphores because if the app is hung holding a lock, waiting for it here ensures the shutdown will also hang!
    Args:
        sig: Integer representing the specific signal that triggered the handler. Eg: CTRL+C -> 2
        frame: Represents the "current execution point" of your code at the exact millisecond of the signal. Primarily used for debugging.
            Eg: line 452 inside function `generate_tokens`
    '''
    print("\n\n⚠️  CTRL+C Detected! Force stopping workers...\n")
    
    # 1. Tell background workers to stop (Non-blocking)
    # We don't join/wait for them. We just signal intent.
    if EXL2_WORKER_STOP_EVENT: EXL2_WORKER_STOP_EVENT.set()
    if EXL3_WORKER_STOP_EVENT: EXL3_WORKER_STOP_EVENT.set()
    
    # 2. Hard exit
    # os._exit(0) is better than sys.exit(0) for signal handlers 
    # because it skips Python's cleanup handlers (except finally blocks)
    # and immediately terminates the process.
    print("👋 Exiting immediately.")
    os._exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    _ = parse_arguments()
    initialize_model()
    host, port = get_host_and_port()
    serve(app, host=host, port=port)