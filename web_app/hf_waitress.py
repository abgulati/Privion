from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, BitsAndBytesConfig, QuantoConfig, HqqConfig, T5EncoderModel, CLIPTextModel, AutoProcessor
from transformers import StoppingCriteria, StoppingCriteriaList
from huggingface_hub import login
import torch

from diffusers import FluxPipeline, FluxTransformer2DModel

try:
    from optimum.quanto import freeze, qfloat8, quantize
except ImportError:
    print("optimum.quanto is not installed. Skipping import.")

try:
    from transformers import MllamaForConditionalGeneration
except ImportError:
    print("transformers version is below 4.45.0 required from Llama3.2-Vision. Skipping MllamaForConditionalGeneration import.")

from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

import subprocess
import threading
import traceback
import argparse
import platform
import logging
import base64
import queue
import time
import json
import uuid
import sys
import os
import io

from functools import wraps
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from waitress import serve

app = Flask(__name__)
CORS(app)

PIPE = None
MODEL = None

STOP_GENERATION = False
llm_semaphore = threading.Semaphore(1)
config_writer_semaphore = threading.Semaphore(1)
error_logging_semaphore = threading.Semaphore(1)
reader_semaphore = threading.Semaphore(3)


#########################------------Setup & Handle Logging-------------###############################
try:
    # 1 - Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.ERROR)

    # 2 - Create a RotatingFileHandler
    # maxBytes: max file size of log file after which a new file is created; set to 1024 * 1024 * 5 for 5MB: 1024x1024 is 1MB, then a multiplyer for the number of MB
    # backupCount: number of backup files to keep specifying how many old log files to keep
    handler = RotatingFileHandler('hf_server_log.log', maxBytes=1024*1024*5, backupCount=2)
    handler.setLevel(logging.ERROR)

    # 3 - Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)

    # 4 - Add the handler to the logger
    logger.addHandler(handler)
    # Logger ready! Usage: logger.error(f"This is an error message with error {e}")
except Exception as e:
    print(f"\n\nCould not establish logger, encountered error: {e}")



def central_error_logging(message, exception=None):
    with error_logging_semaphore:
        error_message = f"\n\n{message} {str(exception) if exception else '; No exception info.'}\n\n"
        traceback_details = traceback.format_exc()
        full_message = f"\n\n{error_message}\n\nTraceback: {traceback_details}\n\n"

        if logger:
            logger.error(full_message)
            print(error_message)
        else:
            print(error_message)
    
    return error_message


def handle_api_error(message, exception=None):
    error_message = central_error_logging(message, exception)
    return jsonify(success=False, error=error_message), 500 #internal server error


def handle_local_error(message, exception=None):
    _ = central_error_logging(message, exception)
    raise Exception(exception)


def handle_error_no_return(message, exception=None):
    _ = central_error_logging(message, exception)


def set_load_safe_defaults():
    try:
        write_config({'load_safe_defaults': True})
    except Exception as e:
        handle_error_no_return("Could not set load_safe_defaults to true in hf_config.json, encountered error: ", e)

def handle_model_loading_error(message, exception=None, target="local"):
    
    try:
        set_load_safe_defaults()
    except Exception as e:
        handle_error_no_return("Could not set load_safe_defaults to true in hf_config.json, encountered error: ", e)
    
    if target == "local":
        return handle_local_error(message, exception)
    elif target == "api":
        return handle_api_error(message, exception)



############################----------------------------------------------###############################



############################------------configuration manager-------------###############################

if not os.path.exists('hf_config.json'):
    with config_writer_semaphore:
        try:
            with open('hf_config.json', 'w') as file:
                json.dump({}, file)
        except Exception as e:
            handle_error_no_return("Could not init config.json. Multiple app restarts may be required to get the app to init correctly. Printing error and proceeding: ", e)


# Method to write to hf_config.json | input- dict of key:values to be written to hf_config.json
def write_config(config_updates, filename='hf_config.json'):

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
        triggers_for_hf_restart = ['torch_device_map', 'torch_dtype', 'model_id', 'awq', 'attn_implementation', 'pipeline_task', 'quantize', 'quant_level', 'port', 'use_flash_attention_2', 'hqq_group_size', 'flux_diffusers', 'flux_low_vram_optimizations', 'load_quantized_flux', 'vision']
        for key in config_updates:
            if key in triggers_for_hf_restart and config_updates[key] != hf_config.get(key):
                restart_required = True

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
        
        return {'success': True, 'restart_required':restart_required}
            

# Method to read from hf_config.json | input- list of keys to be read from hf_config.json; output- dict of key:value pairs; MANAGE DEFAULTS HERE!
def read_config(keys, default_value=None, filename='hf_config.json'):

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
        base_directory = hf_config.get('base_directory', '/app/waitress_storage')   # specifying default if not found

        for key in keys:
            if key in hf_config:
                return_dict[key] = hf_config[key]
            else:
                default_value = {
                    'windows_base_directory':'C:/waitress_storage',
                    'unix_and_docker_base_directory':'/app/waitress_storage',
                    'mac_base_directory':'waitress_storage',
                    'upload_folder':base_directory + '/uploaded_files_for_vision_inferencing',
                    'generated_images_folder':base_directory + '/generated_images',
                    'access_gated':False,
                    'access_token':"",
                    'model_id':"microsoft/Phi-3-mini-4k-instruct",
                    'gguf':False,
                    'awq':False,
                    'flux_diffusers':False,
                    'flux_low_vram_optimizations':False,
                    'load_quantized_flux':False,
                    'vision':False,
                    'gguf_model_id':None,
                    'gguf_filename':None,
                    'quantize':"n",
                    'quant_level':"int4",
                    'hqq_group_size':64,
                    'push_to_hub':False,
                    'torch_device_map':"auto", 
                    'torch_dtype':"auto", 
                    'trust_remote_code':True, 
                    'use_flash_attention_2':False, 
                    'pipeline_task':"text-generation", 
                    'max_new_tokens':500, 
                    'return_full_text':False, 
                    'temperature':0.0,
                    'do_sample':False, 
                    'top_k':40, 
                    'top_p':0.95, 
                    'min_p':0.05, 
                    'n_keep':0,
                    'port':9069,
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
    
    return jsonify({"success": write_return['success'], "restart_required": write_return['restart_required']})


############################----------------------------------------------###############################


#########################------------Setup Directories-------------###############################
BASE_DIRECTORY = ""

if platform.system() == 'Windows':
    try:
        read_return = read_config(['windows_base_directory'])   #passing list of values to read
        BASE_DIRECTORY = str(read_return['windows_base_directory']) #received dict of key:values
    except Exception as e:
        handle_local_error("Could not read windows_base_directory on boot, encountered error: ", e)

elif platform.system() == 'Linux':
    try:
        read_return = read_config(['unix_and_docker_base_directory'])
        BASE_DIRECTORY = str(read_return['unix_and_docker_base_directory'])
    except Exception as e:
        handle_local_error("Could not read unix_and_docker_base_directory on boot, encountered error: ", e)

else:   #Likely 'Darwin' and hence MacOS
    try:
        read_return = read_config(['mac_base_directory'])
        BASE_DIRECTORY = str(read_return['mac_base_directory'])
    except Exception as e:
        handle_local_error("Could not read mac_base_directory on boot, encountered error: ", e)

try:
    write_config({'base_directory':BASE_DIRECTORY})
except Exception as e:
    handle_local_error("Could not write OS BASE_DIRECTORY on boot, encountered error: ", e)


###---Notes on the above workflow:---###
# 1. Everytime the app runs, the OS platform is detected
# 2. Following which the apporpriate base directory is requested as above
# 3. If this is the very first run:
#   a. read_config does not find the directory data in config.json
#   b. the else clause is triggered and defaults set for both, write_config and return
# 4. If this isn't the very first run, read_config simply returns the OS specific directory
# 5. On return, BASE_DIRECTORY is set and write_config has os specific directories are subsequently set (windows_base_directory, unix_and_docker_base_directory, and mac_base_directory)
# 6. write_config is then invoked for BASE_DIRECTORY
# 7. This setup ensures that:
#   a. directories are set correctly at each run
#   b. The user can set their preferred directory by easily editing config.json!


# Having set the values for the directories above, proceed to actually create them on disk IF they don't alread exist!

try:
    os.makedirs(BASE_DIRECTORY, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create Base App Directory, encountered error: ", e)

try:
    read_return = read_config(['upload_folder', 'generated_images_folder'])
    upload_folder = read_return['upload_folder']
    generated_images_folder = read_return['generated_images_folder']
except Exception as e:
    handle_local_error("Could not read paths for app directories (upload_folder, generated_images_folder) from config.json on boot, encountered error: ", e)

try:
    os.makedirs(upload_folder, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create Uploaded Files Directory (upload_folder), encountered error: ", e)

try:
    os.makedirs(generated_images_folder, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create Generated Images Directory (generated_images_folder), encountered error: ", e)


app.config['UPLOAD_FOLDER'] = upload_folder

############################----------------------------------------------###############################


@app.route('/serve_generated_image/<path:filename>')
def serve_generated_image(filename):
    print(f"\n\nserving generated image: {filename}\n\n")
    generated_images_folder = "generated_images"
    try:
        generated_images_folder = read_config(['generated_images_folder'])['generated_images_folder']
    except Exception as e:
        handle_error_no_return("Could not read generated_images_folder from hf_config.json, using default: generated_images in the current working directory. Encountered error: ", e)

    return send_from_directory(generated_images_folder, filename)


@app.route('/serve_uploaded_file/<path:filename>')
def serve_uploaded_file(filename):
    print(f"\n\nserving uploaded file: {filename}\n\n")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


############################---------------Shutdown Methods----------------###############################

def empty_cuda_cache():
    print("\n\nEmptying CUDA cache\n\n")
    # check if torch.cuda is available
    if torch.cuda.is_available():
        try:
            print("Attempting to empty cuda cache")
            torch.cuda.empty_cache()
            print("CUDA cache successfully emptied")
        except Exception as e:
            handle_error_no_return("Could not empty cuda cache, encountered error: ", e)
    else:
        print("\n\nCUDA is not available, skipping cache-emptying\n\n")
        return


def shutdown_model():
    print("\n\nShutting down model\n\n")
    global MODEL
    if MODEL:
        try:
            print("Attempting graceful offload of model")
            if hasattr(MODEL, 'cpu'):
                MODEL.cpu() # Moving to CPU ensures that all GPU operations are completed and the model is fully synchronized before deletion.
            del MODEL
            print("Model graceful-offload successful")
        except Exception as e:
            handle_error_no_return("Could not gracefully offload model. Proceeding to directly force-offload. Encountered error: ", e)
        finally:
            MODEL = None
            empty_cuda_cache()
    print("\n\nModel offloading complete\n\n")


def shutdown_pipe():
    global PIPE
    print("\n\nShutting down pipeline\n\n")
    if PIPE:
        try:
            print("Attempting graceful offload of pipeline")
            if hasattr(PIPE, 'model') and hasattr(PIPE.model, 'cpu'):
                PIPE.model.cpu() # Moving to CPU ensures that all GPU operations are completed and the model is fully synchronized before deletion.
            del PIPE
            print("Pipeline graceful-offload successful")
        except Exception as e:
            handle_error_no_return("Could not gracefully offload pipeline. Proceeding to directly force-offload pipeline. Encountered error: ", e)
        finally:
            PIPE = None
            empty_cuda_cache()
    print("\n\nPipeline offloading complete\n\n")
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


def hf_login_for_gated_models():
    access_token = ""
    try:
        read_return = read_config(['access_token'])
        access_token = str(read_return['access_token'])
    except Exception as e:
        handle_api_error("403 - No access token found, please submit an access token via the /hf_login endpoint")

    try:
        login(token=access_token)   # imported from huggingface_hub
    except Exception as e:
        handle_api_error("Unable to login to the HuggingFace-Hub, please ensure the correct access token has been provided. Encountered error: ", e)


def parse_arguments():

    try:
        parser = argparse.ArgumentParser(description="Server for HuggingFace Transformers models")
    except Exception as e:
        handle_local_error("Could not create parser to parse_arguments(), proceeding with defaults. Encountered error: ", e)

    # Even if a parser object could not be created, a read_request will write & return defaults 
    try:
        read_return = read_config([
            'access_gated',
            'access_token',
            'model_id',
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
            'load_safe_defaults'
        ])
        access_gated = str(read_return['access_gated']).lower() == 'true'
        access_token = str(read_return['access_token'])
        model_id = str(read_return['model_id'])
        quantize = str(read_return['quantize'])
        quant_level = str(read_return['quant_level'])
        hqq_group_size = int(read_return['hqq_group_size'])
        push_to_hub = str(read_return['push_to_hub']).lower() == 'true'
        torch_device_map = str(read_return['torch_device_map'])
        torch_dtype = str(read_return['torch_dtype'])
        trust_remote_code = str(read_return['trust_remote_code']).lower() == 'true'
        pipeline_task = str(read_return['pipeline_task'])
        max_new_tokens = int(read_return['max_new_tokens'])
        return_full_text = str(read_return['return_full_text']).lower() == 'true'
        temperature = float(read_return['temperature'])
        do_sample = str(read_return['do_sample']).lower() == 'true'
        top_k = int(read_return['top_k'])
        top_p = float(read_return['top_p'])
        min_p = float(read_return['min_p'])
        n_keep = int(read_return['n_keep'])
        port = int(read_return['port'])
        load_safe_defaults = str(read_return['load_safe_defaults']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

    if parser:

        parser.add_argument("--reset_to_defaults", action="store_true", default=False, help="Use default settings")
        parser.add_argument("--access_gated", action="store_true", default=access_gated, help="Specify True if you will be accessing gated models you've been approved to access")
        parser.add_argument("--access_token", type=str, default=access_token, help="Access Token obtained from HF-Settings -> Access Tokens")
        parser.add_argument("--model_id", type=str, default=model_id, help="model_id for for LLM in HF-Transformers format obtained from the model card. Remembers previously set value and falls-back to Phi3-mini-4k-instruct as the default.")
        parser.add_argument("--gguf", action="store_true", default=False, help="Add this flag if you'll be loading a GGUF LLM. Defaults to False.")
        parser.add_argument("--awq", action="store_true", default=False, help="Add this flag when loading AWQ-quantized models directly off the HF-Hub.")
        parser.add_argument("--flux_diffusers", action="store_true", default=False, help="Add this flag when loading FLUX-diffusers models directly off the HF-Hub.")
        parser.add_argument("--flux_low_vram_optimizations", action="store_true", default=False, help="Save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power")
        parser.add_argument("--load_quantized_flux", action="store_true", default=False, help="Add this flag when loading quantized FLUX models directly off the HF-Hub.")
        parser.add_argument("--vision", action="store_true", default=False, help="Add this flag when loading vision models directly off the HF-Hub.")
        parser.add_argument("--gguf_model_id", type=str, default=None, help="GGUF model_id of the target repo. Defaults to None")
        parser.add_argument("--gguf_filename", type=str, default=None, help="GGUF filename from the target repo. Defaults to None")
        parser.add_argument("--quantize", type=str, default=quantize, help="Quantization method to be utilized. Simply type 'n' to not use quantization. Remembers previously set value and falls-back to bitsandbytes as the default.")
        parser.add_argument("--quant_level", type=str, default=quant_level, help="Specify quantization level. Valid values -  BitsAndBytes: int8 & int4; Quanto: int8, int4 and int2; HQQ: int8, int4, int3, int2, int1. Remembers previously set value and falls-back to int8 as the default.")
        parser.add_argument("--hqq_group_size", type=int, default=hqq_group_size, help="Specify group_size for HQQ quantization. No restrictions as long as weight.numel() is divisible by the group_size. Remembers previously set value and falls-back to 64 as a default.")
        parser.add_argument("--push_to_hub", action="store_true", default=push_to_hub, help="Push quantized LLM to your HF-hub. Remembers previously set value and falls-back to False as the default.")
        parser.add_argument("--torch_device_map", type=str, default=torch_device_map, help="Specify inference device, example: cuda. Remembers previously set value and falls-back to auto as the default.")
        parser.add_argument("--torch_dtype", type=str, default=torch_dtype, help="Specify model tensor type, example: bfloat16. Remembers previously set value and falls-back to auto as the default.")
        parser.add_argument("--trust_remote_code", action="store_true", default=trust_remote_code, help="Allows the model to execute custom code that's part of the model's HF-repository. Remembers previously set value and falls-back to False by default as a security measure to prevent potentially malicious code from running automatically.")
        parser.add_argument("--use_flash_attention_2", action="store_true", default=False, help="Set to True to attempt using Flash Attention 2. Defaults to False. Failed attempt to use FA2 will proceed to load the model without FA2.")
        parser.add_argument("--pipeline_task", type=str, default=pipeline_task, help="Defaults to text-generation. For more details, open a Python shell, `import transformers`, and Run `help(transfomers.pipeline)`.")
        parser.add_argument("--max_new_tokens", type=int, default=max_new_tokens, help="Set a hard limit on the maximum number of tokens an LLM can generate when responding. Remembers previously set value and falls-back to 500 as a default.")
        parser.add_argument("--return_full_text", action="store_true", default=return_full_text, help="When set to True, the LLM response contains the entire messages list with the latest response appended at the end.")
        parser.add_argument("--temperature", type=float, default=temperature, help="Set LLM temperature on a scale of 0.0 to 2.0. Remembers previously set value and falls-back to 0.0 as a default.")
        parser.add_argument("--do_sample", action="store_true", default=do_sample, help="Perform sampling when selecting response tokens. Remembers previously set value and falls-back to Flase as a default. Must be set to True when temperature is above 0.0. For greedy decoding, leave this as False and set temp to 0.0")
        parser.add_argument("--top_k", type=int, default=top_k, help="Limit the next token selection to the K most probable tokens. Remembers previously set value and falls-back to 40 as a default.")
        parser.add_argument("--top_p", type=float, default=top_p, help="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Remembers previously set value and falls-back to 0.95 as a default.")
        parser.add_argument("--min_p", type=float, default=min_p, help="The minimum probability for a token to be considered, relative to the probability of the most likely token. Remembers previously set value and falls-back to 0.05 as a default.")
        parser.add_argument("--n_keep", type=int, default=n_keep, help="Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded. Remembers previously set value and falls-back to 0 as a default, meaning no tokens are kept. Use -1 to retain all tokens from the prompt.")
        parser.add_argument("--port", type=int, default=port, help="Specify the port to be used by the server. Remembers previously set value and falls-back to 9069 as a default.")

        args = parser.parse_args()
        print(f"\n\nparser.parse_args():\n\n{args}\n\n")

        if args.reset_to_defaults or load_safe_defaults:
            print("\n\nLoading Server with Safe Defaults\n\n")
            try:
                # Empty hf_config.json
                config_writer_semaphore.acquire()
                with open('hf_config.json', 'w') as file:
                    json.dump({}, file, indent=4)
                config_writer_semaphore.release()
                
                # Set defaults
                read_config([
                    'access_gated',
                    'access_token',
                    'model_id',
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
                    'port'
                ])

            except Exception as e:
                handle_local_error("Could not reset hf_config.json, encountered error: ", e)
        else:
            try:
                # Auto-detect Flux and Llama-3.2-Vision models
                if "flux" in args.model_id.lower():
                    print("Flux model auto-detected, setting flux_diffusers=True")
                    args.flux_diffusers = True
                else:
                    args.flux_diffusers = False
                
                if "llama-3.2" in args.model_id.lower() and "vision" in args.model_id.lower():
                    print("Llama-3.2-Vision model auto-detected, setting vision=True")
                    args.vision = True
                else:
                    args.vision = False
                
                write_config({
                    'access_gated':args.access_gated,
                    'access_token':args.access_token,
                    'model_id':args.model_id,
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
                    'port':args.port
                })
            except Exception as e:
                handle_local_error("Could not write launch arguments to hf_config.json, encountered error: ", e)

            if args.access_gated:
                try:
                    hf_login_for_gated_models()
                except Exception as e:
                    handle_local_error("Login to HF-Hub unsuccessful, encountered error: ", e)

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
        handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

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

        if quantize == "bitsandbytes":
            print("Quantizing with BitsAndBytes")
            quant_level = quant_level.lower().strip()

            try:
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
            except Exception as e:
                handle_local_error("Could not set BitsAndBytes config to initialize_model(), encountered error: ", e)
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

    return model_params


def load_flux_pipeline(pipeline):

    print("\n\nLoading Flux Pipeline\n\n")

    try:
        read_return = read_config(['model_id', 'flux_low_vram_optimizations', 'load_quantized_flux'])
        model_id = str(read_return['model_id'])
        flux_low_vram_optimizations = str(read_return['flux_low_vram_optimizations']).lower() == 'true'
        load_quantized_flux = str(read_return['load_quantized_flux']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to initialize_model(), encountered error: ", e)

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

    global MODEL

    try:
        read_return = read_config(['model_id', 'torch_device_map'])
        model_id = str(read_return['model_id'])
        torch_device_map = str(read_return['torch_device_map'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to initialize_model(), encountered error: ", e)

    model_params.pop('trust_remote_code', None)

    try:
        print(f"\nInitializing vision model: {model_id} with device_map: {torch_device_map}\n")
        MODEL = MllamaForConditionalGeneration.from_pretrained(model_id, **model_params)
       
        try:
            print(f"Your vision-model's memory footprint is: {MODEL.get_memory_footprint()}")
        except Exception as e:
            handle_error_no_return("Could not determine the model's memory footprint, encountered error: ", e)

        print(f"\nInitializing processor for vision model: {model_id}\n")
        pipeline = AutoProcessor.from_pretrained(model_id)  # Using 'pipeline' instead of 'processor' to maintain consistency with the server code. AutoProcessor is used to process images and text inputs for the vision model.
        
        print(f"\nVision Model & Processor Loaded Successfully!\n")
        return pipeline
    except Exception as e:
        handle_model_loading_error("Could not load Vision Pipeline, encountered error: ", e)
        return False



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
    print("\n\n/restart_server acquired llm_semaphore, proceeding...\n\n")
    config_writer_semaphore.acquire()
    print("\n\n/restart_server acquired config_writer_semaphore, proceeding...\n\n")
    error_logging_semaphore.acquire()
    print("\n\n/restart_server acquired error_logging_semaphore, proceeding...\n\n")

    print("\n\nrestarting server with stream\n\n")

    shutdown_pipe()
    if MODEL is not None:
        shutdown_model()

    try:
        read_return = read_config(['model_id', 'pipeline_task', 'flux_diffusers', 'vision'])
        model_id = str(read_return['model_id'])
        pipeline_task = str(read_return['pipeline_task'])
        flux_diffusers = str(read_return['flux_diffusers']).lower() == 'true'
        vision = str(read_return['vision']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to initialize_model(), encountered error: ", e)

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
        print("\nrestart_server_stream done\n")

    thread = threading.Thread(target=model_initialization_task)
    thread.start()

    print(f"\nModel Initialization Begins - Loading {model_id}\n")
    return Response(output_reader(), content_type='text/event-stream')


def initialize_model():

    global PIPE

    try:
        read_return = read_config(['model_id', 'push_to_hub', 'quant_level', 'pipeline_task', 'flux_diffusers', 'vision'])
        model_id = str(read_return['model_id'])
        push_to_hub = str(read_return['push_to_hub']).lower() == 'true'
        quant_level = str(read_return['quant_level'])
        pipeline_task = str(read_return['pipeline_task'])
        flux_diffusers = str(read_return['flux_diffusers']).lower() == 'true'
        vision = str(read_return['vision']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to initialize_model(), encountered error: ", e)
    
    print(f"\n\nInitializing HF-Waitress LLM Server for {model_id}\n\n")

    if flux_diffusers:
        print("\n\nFlux Diffusers Selected - Loading...\n\n")
        PIPE = load_flux_pipeline(PIPE)
    
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
        handle_local_error("Could not read POST-request messages when attempting to get_input_params_for_vision_model, encountered error: ", e)
        return False

    try:
        read_return = read_config(['max_new_tokens'])
        max_new_tokens = int(read_return['max_new_tokens'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to get_input_params_for_vision_model(), encountered error: ", e)

    try:
        dpi = int(request.headers.get('X-DPI', 300))
        try:
            generation_args = {
                "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
            }
        except Exception as e:
            handle_error_no_return("Could not set generation-arguments when attempting to get_input_params_for_vision_model, proceeding without them. Encountered error: ", e)
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

    return input_text, pil_image_object_list, generation_args, filename, vision_file_present


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
        input_text, pil_image_object_list, generation_args, filename, vision_file_present = get_input_params_for_vision_model(request)
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
            inputs = PIPE(image, input_text, return_tensors="pt").to(MODEL.device)
        except Exception as e:
            handle_local_error("Could not load input to model, encountered error: ", e)
            return False

        try:
            print("\n\nGenerating Output\n\n")
            output = MODEL.generate(**inputs, **generation_args)    # `output` is a tensor and needs to be decoded!

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

        try:
            empty_cuda_cache()
        except Exception as e:
            handle_error_no_return("Could not empty cuda cache, encountered error: ", e)
    
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
            handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

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
            generation_args = {
                "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
                "return_full_text": request.headers.get('X-Return-Full-Text', str(return_full_text)).lower() == 'true',
                "temperature": float(request.headers.get('X-Temperature', str(temperature))),
                "do_sample": request.headers.get('X-Do-Sample', str(do_sample)).lower() == 'true',
                "top_k": int(request.headers.get('X-Top-K', str(top_k))),
                "top_p": float(request.headers.get('X-Top-P', str(top_p))),
                "min_p": float(request.headers.get('X-Min-P', str(min_p)))
            }
        except Exception as e:
            handle_error_no_return("Could not set generation-arguments for /completions, proceeding without them. Encountered error: ", e)

        try:
            if generation_args:
                output = PIPE(messages, **generation_args)
            else:
                output = PIPE(messages)
        except Exception as e:
            handle_api_error("Could not generate output, encountered error: ", e)

        print("\n\nCompletions done - releasing LLM semaphore\n\n")

        try:
            empty_cuda_cache()
        except Exception as e:
            handle_error_no_return("Could not empty cuda cache, encountered error: ", e)

        return jsonify({"success": True, "response": output})



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
        input_text, pil_image_object_list, generation_args, filename, vision_file_present = get_input_params_for_vision_model(request)
    except Exception as e:
        llm_semaphore.release()
        return handle_api_error("Could not get input params in vision_stream, encountered error: ", e)
    
    if not vision_file_present:
        pil_image_object_list.append(get_blank_pil_image_object())

    stop_event = threading.Event()
    generation_args["stopping_criteria"] = StoppingCriteriaList([StopOnEvent(stop_event)])  # StoppingCriteriaList is a container that holds a list of StoppingCriteria objects. In our case, we have only one such object, which is our custom StoppingCriteria class, initialized with the stop_event object.

    data_queue = queue.Queue()

    def llm_task():

        try:
            for page_number, image in enumerate(pil_image_object_list, start=1): # start=1 to match the page numbers in the PDF
                
                if vision_file_present: 
                    status_string = f"Processing Page: {page_number} from file: {filename}\n\n"
                    data_queue.put(status_string)
                    print(status_string)
                
                print("\n\nLoading Input to Model\n\n")
                inputs = PIPE(image, input_text, return_tensors="pt").to(MODEL.device)

                print("\n\nGenerating Output\n\n")
                output = MODEL.generate(**inputs, **generation_args)    # `output` is a tensor and needs to be decoded!

                # Get length of the input sequence - look for detailed comment in inference_with_vision_model() !
                input_length = inputs.input_ids.shape[1] 
                
                # Slice the tensor and decode only the output!
                decoded_output = PIPE.decode(output[0][input_length:], skip_special_tokens=True)    # Setting skip_special_tokens=True to remove: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.

                print(f"\n\ndecoded_output: {decoded_output}\n\n")
                data_queue.put(decoded_output)
                data_queue.put("\n\n\n")

                empty_cuda_cache()
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

        try:
            empty_cuda_cache()
        except Exception as e:
            handle_error_no_return("Could not empty cuda cache, encountered error: ", e)
            
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
        messages = data.get('messages', [])
    except Exception as e:
        handle_api_error("Could not read POST-request messages for /completions_stream, encountered error: ", e)

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
        handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

    try:
        generation_args = {
            "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
            "return_full_text": request.headers.get('X-Return-Full-Text', str(return_full_text)).lower() == 'true',
            "temperature": float(request.headers.get('X-Temperature', str(temperature))),
            "do_sample": request.headers.get('X-Do-Sample', str(do_sample)).lower() == 'true',
            "top_k": int(request.headers.get('X-Top-K', str(top_k))),
            "top_p": float(request.headers.get('X-Top-P', str(top_p))),
            "min_p": float(request.headers.get('X-Min-P', str(min_p)))
        }
    except Exception as e:
        handle_error_no_return("Could not set generation-arguments for /completions_stream, proceeding without them. Encountered error: ", e)

    stop_event = threading.Event()

    data_queue = queue.Queue()

    def callback(data):
        data_queue.put(data)

    custom_streamer = CustomTextStreamer(PIPE.tokenizer, skip_special_tokens=True, skip_prompt=True)
    custom_streamer.callback = callback

    def llm_task():

        global PIPE

        try:                
            if generation_args:
                generation_args["streamer"] = custom_streamer
                generation_args["stopping_criteria"] = StoppingCriteriaList([StopOnEvent(stop_event)])  # StoppingCriteriaList is a container that holds a list of StoppingCriteria objects. In our case, we have only one such object, which is our custom StoppingCriteria class, initialized with the stop_event object.
                output = PIPE(messages, **generation_args)
            else:
                output = PIPE(messages, streamer=custom_streamer, stopping_criteria=StoppingCriteriaList([StopOnEvent(stop_event)]))
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

        try:
            empty_cuda_cache()
        except Exception as e:
            handle_error_no_return("Could not empty cuda cache, encountered error: ", e)
            
    print("\n\nInferencing Begins!\n\n")
    return Response(generate(), content_type='text/event-stream')


@app.route('/health')
def health():

    def throw_health_check_error(attrib, e):
        handle_error_no_return(f"Could not determine {attrib} while checking HF-Waitress server health. This is not a critical error and the HF-Waitress LLM server is online! More details of the issue encountered follows: ", e)
        return True

    print("\n\nHF-Waitress LLM health-check in-progress...\n\n")
    
    with reader_semaphore:
    
        try:
            if PIPE is None:
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
        print("\n\n/restart_server acquired llm_semaphore, proceeding...\n\n")
        with config_writer_semaphore:
            print("\n\n/restart_server acquired config_writer_semaphore, proceeding...\n\n")
            with error_logging_semaphore:
                print("\n\n/restart_server acquired error_logging_semaphore, proceeding...\n\n")

                try:
                    shutdown_pipe()
                    if MODEL is not None:
                        shutdown_model()
                    initialize_model()
                except Exception as e:
                    handle_api_error("Could not restart server, encountered error: ", e)
                
                return jsonify(success=True)


@app.route('/stop_generation')
def stop_generation():
    global STOP_GENERATION
    STOP_GENERATION = True
    return jsonify(success=True)

if __name__ == '__main__':
    args = parse_arguments()
    initialize_model()
    port = getattr(args, 'port', 9069)
    serve(app, host='0.0.0.0', port=port)