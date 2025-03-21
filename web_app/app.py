from flask import Flask, render_template, request, redirect, url_for, Response
from flask import send_from_directory
from flask_cors import CORS
from flask import jsonify

from sentence_transformers import SentenceTransformer, util
import torch

from werkzeug.utils import secure_filename

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from falkordb import FalkorDB
import chromadb

from pdf2image import convert_from_path
from PIL import Image

import fitz # PyMuPDF
from rapidfuzz import fuzz

import subprocess
import traceback
import threading
import platform
import argparse
import datetime
import requests
import logging
import sqlite3
import signal
import PyPDF2
import shutil   # Shell Utilities is part of Python's standard library and is used for file operations
import queue
import uuid
import json
import time
import nltk
import ast
import os
import io
import re
import gc
from logging.handlers import RotatingFileHandler
from nltk.corpus import stopwords

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, Or
from whoosh import scoring

from waitress import serve

try:
    from graph_clustering import apply_leiden_clustering
except Exception as e:
    print(f"Could not import graph_clustering (likely not installed), skipping. Encountered error: {e}")

try:
    from prompt_formatting import manually_format_prompt_with_prompt_template
except ImportError:
    print("WARNING: Prompt Formatter module `prompt_formatting.py` is not present. Skipping import. Exl2 and llama.cpp will not work!")

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1' # Allow insecure traffic - Needed to bypass HTTPS requirement for Google Drive OAuth. FOR DEV USE ONLY! SWITCH TO SELF-SIGNED CERTIFICATES & HTTPS FOR PRODUCTION!

app = Flask(__name__)
CORS(app)

# Route for the home page, rendering the initial model selection form (legacy)
@app.route('/')
def index():
    return render_template('chat.html')

# model_selection.html triggers window.location.href to '/chat', which triggers this route, which loads the chat.html template at the end!
@app.route('/chat')
def chat():
    return render_template('chat.html')

# Route to display the file loading form
@app.route('/load_file')
def load_file():
    return render_template('model_selection.html', show_file_form=True)

@app.route('/download/<filename>')
def download_file(filename):
    # return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=False, mimetype='application/pdf')

@app.route('/pdf/<filename>')
def pdf_viewer(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)



#########################------------------GLOBALS!----------------------###############################
LLAMA_CPP_PROCESS = None
HF_WAITRESS_PROCESS = None
LLM = None
LOADED_UP = False
LLM_LOADED_UP = False
LLM_CHANGE_RELOAD_TRIGGER_SET = False

# Dict for user queries:  queries[session_id] = user_input
QUERIES = {}

# If modifying these scopes, delete the file token.json.
GDRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive" # broader access needed for Share Drive files
]
GDRIVE_CREDS = None
#########################------------------------------------------------###############################



#########################------------Setup & Handle Logging-------------###############################
try:
    # 1 - Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.ERROR)

    # 2 - Create a RotatingFileHandler
    # maxBytes: max file size of log file after which a new file is created; set to 1024 * 1024 * 5 for 5MB: 1024x1024 is 1MB, then a multiplyer for the number of MB
    # backupCount: number of backup files to keep specifying how many old log files to keep
    handler = RotatingFileHandler('lars_server_log.log', maxBytes=1024*1024*5, backupCount=2)
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
    error_message = f"{message} {str(exception) if exception else '; No exception info.'}".strip()
    #traceback_details = traceback.format_exc()
    #full_message = f"\n\n{error_message}\n\nTraceback: {traceback_details}\n\n"
    full_message = f"\n\n{error_message}\n\n"
    if logger:
        logger.error(full_message)
        print(full_message)
    else:
        print(full_message)

    return error_message


def handle_api_error(message, exception=None):
    error_message = central_error_logging(message, exception)
    return jsonify(success=False, error=error_message), 500 #internal server error


def handle_local_error(message, exception=None):
    _ = central_error_logging(message, exception)
    raise Exception(exception)


def handle_error_no_return(message, exception=None):
    _ = central_error_logging(message, exception)

#########################-------------------------------------###############################



if not os.path.exists('config.json'):
    try:
        with open('config.json', 'w') as file:
            json.dump({}, file)
    except Exception as e:
        handle_error_no_return("Could not init config.json. Multiple app restarts may be required to get the app to init correctly. Printing error and proceeding: ", e)



# Method to write to config.json | input- dict of key:values to be written to config.json
def write_config(config_updates, filename='config.json'):

    # Open config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
    except Exception as e:
        config = {}     #init emply config dict
        handle_error_no_return("Could not read config.json when attempting to write, encountered error: ", e)
        
    restart_required = False
    if LLM_LOADED_UP:
        llm_trigger_keys_for_app_restart = [
            'use_local_llm',
            'local_llm_server',
            'use_azure_open_ai',
            'use_gpu',
            'model_choice',
            'local_llm_chat_template_format',
            'local_llm_context_length',
            'local_llm_max_new_tokens',
            'local_llm_gpu_layers',
            'base_template',
            'skip_system_prompt',
            'hf_waitress_serving_url',
            'hf_waitress_access_url',
            'hf_waitress_server_port'
        ]
                
        for key in llm_trigger_keys_for_app_restart:
            if key in config_updates and config_updates[key] != config.get(key):
                global LLM_CHANGE_RELOAD_TRIGGER_SET
                LLM_CHANGE_RELOAD_TRIGGER_SET = True
                restart_required = True
                break

    config.update(config_updates)

    # Write updated config.json:
    try:
        with open(filename, 'w') as file:
            json.dump(config, file, indent=4)
    except Exception as e:
        handle_local_error("Could not update config.json, encountered error: ", e)
     
    return {'success': True, 'restart_required':restart_required}
            

# Method to read from config.json | input- list of keys to be read from config.json; output- dict of key:value pairs; MANAGE DEFAULTS HERE!
def read_config(keys, default_value=None, filename='config.json'):
    
    # Open config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
    except Exception as e:
        handle_error_no_return("Could not read config.json, encountered error: ", e)
        return {key: default_value for key in keys}     #because a read scenario wherein config.json does not exist shouldn't occur!
    
    return_dict = {}
    update_config_dict = {}
    base_directory = config.get('base_directory', '/app/lars_storage')   # specifying default if not found

    for key in keys:
        if key in config:
            return_dict[key] = config[key]
        else:
            default_value = {
                'windows_base_directory':'C:/lars_storage',
                'unix_and_docker_base_directory':'/app/lars_storage',
                'mac_base_directory':'app',
                'upload_folder':base_directory + '/uploaded_pdfs',
                'sqlite_images_db':base_directory + '/images_database_main.db',
                'sqlite_history_db':base_directory + '/chat_history.db',
                'sqlite_docs_loaded_db':base_directory + '/docs_loaded.db',
                'model_dir':base_directory + '/models',
                'highlighted_docs':base_directory + '/highlighted_pdfs',
                'ocr_pdfs':base_directory + '/ocr_pdfs',
                'pdfs_to_txts':base_directory + '/pdfs_to_txts',
                'docs_to_knowledge_graph_dir': base_directory + '/docs_to_knowledge_graph',
                'local_llm_server':'hf-waitress',
                'model_choice':'Meta-Llama-3-8B-Instruct.f16.gguf',
                'vision_llm_local_url':"http://localhost:9069/completions",
                'kosmos_local_url':"http://localhost:25000/infer_file_stream",
                'kosmos_task':'ocr',
                'kosmos_threshold':20,
                'lars_host':'0.0.0.0',
                'lars_port':5000,
                'hf_waitress_serving_url':'0.0.0.0',
                'hf_waitress_access_url':'localhost',
                'hf_waitress_server_port':9069,
                'llama_cpp_serving_url':'0.0.0.0',
                'llama_cpp_access_url':'localhost',
                'llama_cpp_server_port':8080,
                'do_rag':True,
                'force_enable_rag':False,
                'force_disable_rag':False,
                'use_local_llm':True,
                'use_gpu':True,
                'use_gpu_for_embeddings':False,
                'azure_cv_free_tier':True,
                'use_azure_open_ai':False,
                'azure_openai_api_type':'azure',
                'azure_openai_api_version':'2023-05-15',
                'azure_openai_max_tokens':4096,
                'azure_openai_temperature':0.7,
                'use_ocr':False,
                'ocr_service_choice':'None',
                'force_extract_previously_extracted_text':False,
                'llm_filter_citations':True,
                'local_llm_model_type':'llama',
                'local_llm_chat_template_format':'llama3',
                'exl2_prompt_template_format':'raw',
                'local_llm_context_length':8192,
                'local_llm_max_new_tokens':2048,
                'local_llm_gpu_layers':47,
                'local_llm_temperature':0.8,
                'local_llm_top_k':40,
                'local_llm_top_p':0.95,
                'local_llm_min_p':0.05,
                'local_llm_n_keep':0,
                'server_timeout_seconds':10,
                'server_retry_attempts':3,
                'whoosh_search_weighting':'BM25F',
                'fetch_top_k_results_from_whoosh':50,
                'fetch_top_k_results_from_vectordb':11,
                'filter_top_k_results_by_reranking':50,
                'min_semantic_similarity_threshold':0.5,
                'min_lexical_similarity_threshold':3.0,
                'chunk_size':250,
                'chunk_overlap':0,
                'perform_graph_rag':True,
                'perform_only_graph_rag':False,
                'graph_chunk_size':1500,    # Larger chunks are better because they provide more context for the model to identify meaningful entities and relationships
                'graph_chunk_overlap':300,  # 20% overlap for a 1500 char chunk: provides very reasonable overlap while not being too redundant.
                'graph_generator_model':'Metin/Gemma-2-2B-TR-Knowledge-Graph',
                'graph_model_server_port':9070,
                'graph_model_access_url':'localhost',
                'quantize_graph_model':'bitsandbytes',
                'quantize_graph_model_bits':'int8',
                'exl2_quantize_graph_model':False,
                'exl2_quantize_graph_model_bpw':8.0,
                'graph_db_server_host':'localhost',
                'assign_host_port_to_graph_db_server':6379,
                'launch_graph_db_with_ui':True,
                'assign_host_port_to_graph_db_ui':3000,
                'graph_model_max_new_tokens':4096,
                'graph_model_max_seq_len':20480,
                'graph_model_temperature':0.1,
                'graph_model_do_sample':True,
                'graph_model_top_k':40,
                'graph_model_top_p':0.95,
                'graph_model_min_p':0.05,
                'skip_summary_generation':False,
                'reuse_previously_extracted_graph_entities_and_relationships':False,
                'reuse_previously_generated_graph_summaries':False,
                'base_template': (
                            "You are a helpful assistant deployed in a Retrieval Augmented Generation (RAG) system.\n"
                            "Your task is to evaluate retrieved contextual data to answer users' questions accurately and in detail.\n\n"
                            "Follow these instructions carefully:\n\n"
                            "1. Accuracy and Truthfulness:\n"
                            "- Provide accurate information based on the provided context or your knowledge\n"
                            "- Do not fabricate or guess answers\n"
                            "- If unsure, clearly state that you don't know or need more information\n\n"
                            "2. Using Retrieved Context:\n"
                            "- Only reference document names and page numbers that are directly relevant to the query\n"
                            "- Quote document names exactly as they appear in the metadata array, including special characters\n"
                            "- When citing information, use clear attribution (e.g., 'According to [document] on page [X]...')\n"
                            "- Do not mention or acknowledge irrelevant documents or context\n\n"
                            "3. Response Structure:\n"
                            "- Start with a direct answer to the main question when possible\n"
                            "- For factual questions: provide direct, concise answers\n"
                            "- For complex or multi-part questions: break down the response into clearly labeled sections\n"
                            "- For explanations: provide detailed, step-by-step responses\n"
                            "- Maintain clarity and readability throughout\n\n"
                            "4. Context Awareness:\n"
                            "- Remember that users cannot see the retrieved context\n"
                            "- Avoid referencing irrelevant details that might confuse users\n"
                            "- When context is unavailable, unhelpful, or unrelated, rely on your base knowledge\n"
                            "- Clearly distinguish between information from retrieved context and base knowledge\n\n"
                            "5. Source Attribution (v.important!):\n"
                            "- Only include metadata references ('source' and 'page_number') when they directly support your answer\n"
                            "- Include ONLY source and page_number, NOT page_content\n"
                            "- When using base knowledge, no source citation is needed\n\n"
                            "6. Handling Uncertainty:\n"
                            "- If the retrieved context is ambiguous, acknowledge the ambiguity\n"
                            "- When multiple interpretations are possible, explain the different possibilities\n"
                            "- If more information is needed, specify what additional details would help\n"
                            "- When partial information is available, clearly indicate what is known and what is uncertain\n\n"
                            "Remember: Only include references to 'source' and 'page_number' metadata when the retrieved context is directly helpful for answering the user's query, no need to specify that you're not referencing it otherwise."
                ),
                'vision_ocr_prompt': (
                            "Please OCR the attached image line-by-line as accurately as possible.\n"
                            "If the image contains a table, output cell contents with their row and column indices. Include row and column name headers too. Follow this formatting example:\n"
                            "[Row 0 (name:<header-name>), Column 0 (name:<header-name>): <cell-data>; Row 0 (name:<header-name>), Column 1 (name:<header-name>): <cell-data>;] etc.\n"
                            "The extracted text will be converted into embeddings and used for semantic search, so extracting as much detail as possible, while maintaining formatting integrity and tabular context is crucially important.\n"
                            "Please output only the text extracted from the image, without any other text, code, or markup. Please no yapping!\n"
                            "Don't even say stuff like 'Here's the OCR'ed text from the image' or 'Here's the text extracted from the image' or anything like that. Just output the text.\n"
                            "Thank you!"
                ),
                'skip_system_prompt':False,
                'embedding_models_list':[
                    'sentence-transformers/all-mpnet-base-v2',
                    'BAAI/bge-large-en-v1.5',
                    'BAAI/bge-base-en-v1.5',
                    'BAAI/bge-small-en-v1.5',
                    'nvidia/NV-Embed-v2'
                ],
                'selected_embedding_model':'sentence-transformers/all-mpnet-base-v2',
                'use_embedding_model_for_reranking':True,
                'knowledge_domain_list':[
                    'General',
                    'Technical',
                    'Legal',
                    'Financial',
                    'Medical',
                    'Business',
                    'Education',
                    'Casual'
                ],
                'selected_knowledge_domain':'General',
                'knowledge_domain_base_directory': base_directory + '/knowledge_domains'
            }.get(key, 'undefined') # "implicit string concatenation" used for keys with large-string values!

            if default_value == 'undefined':
                raise KeyError(f"Key \'{key}\' not found in config.json and no default value has been defined either.\n")
            
            return_dict[key] = default_value
            update_config_dict[key] = default_value

    if update_config_dict:
        # Write Defaults
        try:
            write_config(update_config_dict)
        except Exception as e:
            handle_error_no_return("Could not write defaults to config.json. Encountered error: ", e)
    
    ##print(f"return_dict: {return_dict}")

    return return_dict


def read_hf_config(keys, default_value=None, filename='hf_config.json'):
    
    # Open hf_config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            hf_config = json.load(file)
    except Exception as e:
        handle_error_no_return("Could not read hf_config.json, encountered error: ", e)
        return {key: default_value for key in keys}     #because a read scenario wherein hf_config.json does not exist shouldn't occur!
    
    return_dict = {}

    for key in keys:
        if key in hf_config:
            return_dict[key] = hf_config[key]
        else:
            return_dict[key] = default_value

    return return_dict


# Method for API route to read from config.json
# Deviates from typical RESTful principals to use a POST call to fetch values but practical & justifyable because we:
# 1. Do not want to make the URL huge with a ever-growing list of query-params 2. Do not wish to expose values via query-params
@app.route('/config_reader_api', methods=['POST'])
def config_reader_api():
    # keys = request.args.getlist('keys') # Assuming keys are passed as query parameters
    
    try:
        keys = request.json.get('keys', []) # Could also do keys = request.json['keys'] but this way we can provide a default list should 'keys' be missing!
    except Exception as e:
        return handle_api_error("Server-side error - could not read keys for config_reader_api request. Encountered error:", e)

    try:
        values = read_config(keys)  # send list of keys, get dict of key:values
    except Exception as e:
        return handle_api_error("Server-side error - could not read keys from config.json. Encountered error: ", e)
    
    return jsonify(success=True, values=values)


# Method for API route to write to config.json
@app.route('/config_writer_api', methods=['POST'])
def config_writer_api():

    try:
        config_updates = request.json['config_updates']
        print(f"config_updates for config_writer_api: {config_updates}")
    except Exception as e:
        return handle_api_error("Server-side error - could not read values for config_writer_api request. Encountered error: ", e)
    
    try:
        write_return = write_config(config_updates)
    except Exception as e:
        return handle_api_error("Server-side error - could not write keys to config.json. Encountered error: ", e)
    
    return jsonify({"success": write_return['success'], "restart_required": write_return['restart_required']})



#########################------------Setup Directories-------------###############################
BASE_DIRECTORY = ""

if platform.system() == 'Windows':
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    import azure.ai.vision as sdk
    
    #BASE_DIRECTORY = 'C:/lars_storage'
    try:
        read_return = read_config(['windows_base_directory'])   #passing list of values to read
        BASE_DIRECTORY = str(read_return['windows_base_directory']) #received dict of key:values
    except Exception as e:
        handle_local_error("Could not read windows_base_directory on boot, encountered error: ", e)

elif platform.system() == 'Linux':
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    import azure.ai.vision as sdk
    
    #BASE_DIRECTORY = '/app/lars_storage'
    try:
        read_return = read_config(['unix_and_docker_base_directory'])
        BASE_DIRECTORY = str(read_return['unix_and_docker_base_directory'])
    except Exception as e:
        handle_local_error("Could not read unix_and_docker_base_directory on boot, encountered error: ", e)

else:   #Likely 'Darwin' and hence MacOS
    #BASE_DIRECTORY = 'app'
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
    os.makedirs(BASE_DIRECTORY, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create Base App Directory, encountered error: ", e)
        
try:
    read_return = read_config(['model_dir', 'highlighted_docs', 'upload_folder', 'ocr_pdfs', 'pdfs_to_txts', 'docs_to_knowledge_graph_dir'])
    model_dir = read_return['model_dir']
    highlighted_docs = read_return['highlighted_docs']
    upload_folder = read_return['upload_folder']
    ocr_pdfs = read_return['ocr_pdfs']
    pdfs_to_txts = read_return['pdfs_to_txts']
    docs_to_knowledge_graph_dir = read_return['docs_to_knowledge_graph_dir']
except Exception as e:
    handle_local_error("Could not read paths for app directories (model_dir, highlighted_docs, upload_folder, etc.) from config.json on boot, encountered error: ", e)

try:
    os.makedirs(model_dir, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create Model Directory (model_dir), encountered error: ", e)

try:
    os.makedirs(highlighted_docs, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create Highlighted Docs Directory (highlighted_docs), encountered error: ", e)

try:
    os.makedirs(upload_folder, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create Uploaded Docs Directory (upload_folder), encountered error: ", e)
    
try:
    os.makedirs(ocr_pdfs, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create OCR'ed Docs Directory (ocr_pdfs), encountered error: ", e)

try:
    os.makedirs(pdfs_to_txts, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create txt-docs Directory (pdfs_to_txts), encountered error: ", e)

try:
    os.makedirs(docs_to_knowledge_graph_dir, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create docs_to_knowledge_graph_dir, encountered error: ", e)


app.config['UPLOAD_FOLDER'] = upload_folder
app.config['DOWNLOAD_FOLDER'] = highlighted_docs


def clean_text_string(text_to_be_cleaned):
    
    # Clean text
    # text_to_be_cleaned = text_to_be_cleaned.replace("►", "").replace("■", "").replace("▼", "")
    # text_to_be_cleaned = text_to_be_cleaned.replace("Confidential Copy \n            for \n         DKPPU", "")
    #clean_text = re.sub(r'\n(?=[a-z.])', ' ', text)     # replaces newline chars immediately followed by a small-letter or dot with a space as they're likely to be the same sentence split-up across lines.
    clean_text = re.sub(r'\n+', '\n', text_to_be_cleaned)

    # This regex substitutes anything that is not a word character or whitespace with an empty string.
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)

    # This regex substitutes any sequence of whitespace characters with a single space.
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


def load_json_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            return handle_local_error("Could not load JSON file, encountered error: ", e)
    else:
        return {}


def save_json_file(data, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        return handle_local_error("Could not save JSON file, encountered error: ", e)

    return True


def get_path_to_knowledge_domain():
    print("Getting path to knowledge domain")
    try:
        read_return = read_config(['selected_knowledge_domain', 'knowledge_domain_base_directory'])
        selected_knowledge_domain = read_return['selected_knowledge_domain']
        knowledge_domain_base_directory = read_return['knowledge_domain_base_directory']
    except Exception as e:
        handle_local_error("Missing values in config.json, could not get_path_to_knowledge_domain. Error: ", e)

    try:
        path_to_knowledge_domain = os.path.join(knowledge_domain_base_directory, selected_knowledge_domain)
        if not os.path.exists(path_to_knowledge_domain):
            os.makedirs(path_to_knowledge_domain, exist_ok=True)
            print(f"\n\nCreated knowledge domain directory: {path_to_knowledge_domain}\n\n")
        return path_to_knowledge_domain
    except Exception as e:
        return handle_local_error("Could not create knowledge domain folder, encountered error: ", e)


def determine_whoosh_index_folder():
    print("Determining Whoosh Index Folder")

    path_to_knowledge_domain = get_path_to_knowledge_domain()

    try:
        selected_embedding_model = read_config(['selected_embedding_model'])['selected_embedding_model']
    except Exception as e:
        handle_local_error("Could not determine selected embedding model, encountered error: ", e)

    try:
        whoosh_index_folder = os.path.join(path_to_knowledge_domain, "vector_db_and_whoosh_index", selected_embedding_model, "whoosh_index")
    except Exception as e:
        handle_local_error("Could not determine whoosh index folder, encountered error: ", e)

    return whoosh_index_folder


def create_whoosh_index_in_folder(whoosh_index_folder):

    print(f"Creating Whoosh Index in folder: {whoosh_index_folder}")
    
     # Define the Index schema: what fields it contains
    schema = Schema(
        content=TEXT(stored=True),
        source=ID(stored=True),
        page_number=ID(stored=True)
    )

    # Create a directory for persistent storage of the index to disk
    try:
        os.makedirs(whoosh_index_folder, exist_ok=True)
    except Exception as e:
        handle_local_error("Failed to create directory for the Whoosh Index, encountered error: ", e)
    # Create the index based on the schema definted above
    try:
        ix = create_in(whoosh_index_folder, schema)
    except Exception as e:
        handle_local_error("Failed to create Whoosh Index, encountered error: ", e)

    return ix


def get_whoosh_index_object_for_folder(whoosh_index_folder):

    print(f"Getting Whoosh Index Object for folder: {whoosh_index_folder}")

    if not os.path.exists(whoosh_index_folder):
        try:
            ix = create_whoosh_index_in_folder(whoosh_index_folder)
        except Exception as e:
            handle_local_error("Failed to create Whoosh Index, encountered error: ", e)
    else:
        try:
            ix = open_dir(whoosh_index_folder)
        except Exception as e:
            handle_local_error("Failed to open Whoosh Index, encountered error: ", e)

    return ix


def whoosh_indexer(new_chunks):

    print("\n\nWhoosh Indexing Chunks\n\n")
    
    try:
        whoosh_index_folder = determine_whoosh_index_folder()
    except Exception as e:
        handle_local_error("Failed to determine Whoosh Index Folder, encountered error: ", e)
    
    try:
        ix = get_whoosh_index_object_for_folder(whoosh_index_folder)
    except Exception as e:
        handle_local_error("Failed to get Whoosh Index Object, encountered error: ", e)
        
    # init writer and write to the index:
    try:
        writer = ix.writer()
        for chunk in new_chunks:
            writer.add_document(
                content=chunk['content'], 
                source=chunk['source'], 
                page_number=str(chunk['page_number'])
            )
        
        writer.commit()
        print(f"Added {len(new_chunks)} chunks to Whoosh Index")
        
    except Exception as e:
        handle_local_error("Failed to write to Whoosh Index, encountered error: ", e)


def search_whoosh_index(query):

    print("Searching Whoosh Index")
    
    try:
        read_return = read_config(['fetch_top_k_results_from_whoosh', 'whoosh_search_weighting', 'min_lexical_similarity_threshold'])
        fetch_top_k_results_from_whoosh = read_return['fetch_top_k_results_from_whoosh']
        whoosh_search_weighting = read_return['whoosh_search_weighting']
        min_lexical_similarity_threshold = read_return['min_lexical_similarity_threshold']  # Like semantic search with ChromaDB, higher scores indicate better matches but the range with Whoosh is different!
    except Exception as e:
        handle_local_error("Missing whoosh_index_folder in config.json for method search_whoosh_index. Error: ", e)

    try:
        whoosh_index_folder = determine_whoosh_index_folder()
    except Exception as e:
        handle_local_error("Failed to determine Whoosh Index Folder, encountered error: ", e)

    try:
        ix = get_whoosh_index_object_for_folder(whoosh_index_folder)
    except Exception as e:
        handle_local_error("Failed to get Whoosh Index Object, encountered error: ", e)

    whoosh_weighting = scoring.BM25F()  # Rough ranges: 0.0: No Match; 1-2: Weak Match; 3-5: Moderate Match; 6+: Strong Match
    if whoosh_search_weighting == "TF-IDF":
        whoosh_weighting = scoring.TF_IDF()  # Rough ranges: 0.0: No Match; 1-4: Weak Match; 5-10: Moderate Match; 10+: Strong Match
    
    try:
        with ix.searcher(weighting=whoosh_weighting) as searcher:
            query_parser = QueryParser("content", schema=ix.schema, group=OrGroup)
            parsed_query = query_parser.parse(query)

            results = searcher.search(parsed_query, limit=fetch_top_k_results_from_whoosh)
            print(f"Whoosh Results: Number of results: {len(results)}")

            # Filter by score threshold
            filtered_results = [
                {
                    'content': result['content'],
                    'source': result['source'],
                    'page_number': result['page_number'],
                    'score': result.score
                }
                for result in results
                if result.score >= min_lexical_similarity_threshold
            ]
            print(f"Whoosh Results:Number of results after filtering by score threshold {min_lexical_similarity_threshold}: {len(filtered_results)}")

            # If no results, let's try a more lenient search:
            if len(filtered_results) == 0:
                print("No lexical results found after filtering by score threshold, trying a more lenient search...")
                terms = [Term("content", word) for word in query.lower().split()]
                or_query = Or(terms)
                lenient_results = searcher.search(or_query, limit=fetch_top_k_results_from_whoosh)
                print(f"number of results after very lenient search: {len(lenient_results)}")

                filtered_results = [
                    {
                        'content': result['content'],
                        'source': result['source'],
                        'page_number': result['page_number'],
                        'score': result.score
                    }
                    for result in lenient_results
                    if result.score >= min_lexical_similarity_threshold
                ]
                print(f"Whoosh Results: Number of results after filtering by score threshold {min_lexical_similarity_threshold} in very lenient search: {len(filtered_results)}")

            return filtered_results
            
            # return [{'content': result['content'], 'source': result['source'], 'page_number': result['page_number']} for result in results]

    except Exception as e:
        handle_error_no_return("Failed to search Whoosh Index, encountered error: ", e)
        return []


def PDFtoAzureDocAiTXT(input_filepath):

    print("\n\nProcessing Document - PDF to Azure DocAI TXT\n\n")
    
    try:
        read_return = read_config(['azure_doc_ai_endpoint', 'azure_doc_ai_subscription_key', 'ocr_pdfs', 'force_extract_previously_extracted_text'])
        azure_doc_ai_endpoint = read_return['azure_doc_ai_endpoint']
        azure_doc_ai_subscription_key = read_return['azure_doc_ai_subscription_key']
        ocr_pdfs = read_return['ocr_pdfs']
        force_extract_previously_extracted_text = str(read_return['force_extract_previously_extracted_text']).lower() == 'true'
    except Exception as e:
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for PDFtoAzureDocAiTXT, please provide required API config. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")   # Using this instead of os.path.splitext() in case filename contains a period
    output_text_file_path = os.path.join(ocr_pdfs, output_text_file_name).replace("\\","/")

    if os.path.exists(output_text_file_path) and not force_extract_previously_extracted_text:
        if os.path.getsize(output_text_file_path) > 0:
            print("Azure-OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("Azure-OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    try:
        docai_client = DocumentAnalysisClient(azure_doc_ai_endpoint, AzureKeyCredential(azure_doc_ai_subscription_key))
    except Exception as e:
        handle_local_error("Could not create ComputerVisionClient for Azure DocAI, encountered error: ", e)

    try:
        with open(input_filepath, "rb") as pdf_file:
            # 1 - Get page count:
            try:
                pypdf_reader = PyPDF2.PdfReader(pdf_file)
                page_count = len(pypdf_reader.pages)
                page_range = f"1-{page_count}" if page_count > 1 else "1"
                print(f"page_range: {page_range}")
            except Exception as e:
                handle_local_error("Could not get page count for call to Azure DocAI, encountered error: ", e)

            # 2 - Reset file-read stream's internal pointer, which has now been set to the end of the file due to the above read operation!
            pdf_file.seek(0)

            # 3 - Call Azure DocAI:
            try:
                poller = docai_client.begin_analyze_document("prebuilt-layout", pdf_file, pages=page_range)
                result = poller.result()
            except Exception as e:
                handle_local_error("Could not get results for begin_analyze_document for Azure DocAI, encountered error: ", e)

        # print(f"result: \n{result}")

        used_regions = set()   # set will avoid duplicates

        if hasattr(result, 'tables'):
            for table in result.tables:
                #print("Found table")
                if table.cells:     # Check if there are cells in the table 
                    for cell in table.cells:
                        #print(f"Row {cell.row_index}, Column {cell.column_index}, Text: {cell.content}")
                        cell_text = f'Row {cell.row_index}, Column {cell.column_index}: {cell.content}'

                        # Get page number
                        page_number = ""
                        if cell.bounding_regions:   # Check if there are bounding regions
                            for region in cell.bounding_regions:
                                page_number = region.page_number
                                cell_polygon = region.polygon
                                cell_polygon_tuple = tuple((point.x, point.y) for point in cell_polygon)    # lists aren't hashable to cast to a tuple
                                used_regions.add(cell_polygon_tuple)

                        try:
                            output_text_file.write(f"[PAGE:{page_number}]\n{cell_text}\n")
                        except Exception as e:
                            handle_local_error("could not write to output text file, encountered error: ", e)

        # Get paragraphs
        if hasattr(result, 'paragraphs'):
            for paragraph in result.paragraphs:
                para_page_number = paragraph.bounding_regions[0].page_number
                para_polygon = paragraph.bounding_regions[0].polygon
                para_polygon_tuple = tuple((point.x, point.y) for point in para_polygon)
                
                if para_polygon_tuple in used_regions:
                    continue
                
                para_content = paragraph.content
                #print(f"\n---Processing Page: {para_page_number}---\n")
                #print(f"paragraph: {para_content}")

                # write the extracted text to the file:
                try:
                    output_text_file.write(f"[PAGE:{para_page_number}]\n{para_content}\n")
                    used_regions.add(para_polygon_tuple)
                except Exception as e:
                    handle_local_error("could not write to output text file, encountered error: ", e)

    except Exception as e:
        handle_local_error("Error processing document with azure DocAI: ", e)

    # Close all files
    output_text_file.close()

    return output_text_file_path


def PDFtoAzureOCRTXT(input_filepath):
    
    print("\n\nProcessing Document - PDF to Azure OCR TXT\n\n")
    
    try:
        read_return = read_config(['azure_ocr_endpoint', 'azure_ocr_subscription_key', 'ocr_pdfs', 'azure_cv_free_tier', 'force_extract_previously_extracted_text'])
        azure_ocr_endpoint = read_return['azure_ocr_endpoint']
        azure_ocr_subscription_key = read_return['azure_ocr_subscription_key']
        ocr_pdfs = read_return['ocr_pdfs']
        azure_cv_free_tier = read_return['azure_cv_free_tier']
        force_extract_previously_extracted_text = str(read_return['force_extract_previously_extracted_text']).lower() == 'true'
    except Exception as e:
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for PDFtoAzureOCRTXT, please provide required API config. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")   # Using this instead of os.path.splitext() in case filename contains a period
    output_text_file_path = os.path.join(ocr_pdfs, output_text_file_name).replace("\\","/")

    if os.path.exists(output_text_file_path) and not force_extract_previously_extracted_text:
        if os.path.getsize(output_text_file_path) > 0:
            print("OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_filepath, 300) # The convert_from_path() function from pdf2image lib intertnally uses Poppler to convert PDF pages to images, and then creates PIP Image objects from them. 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    try:
        computervision_client = ComputerVisionClient(azure_ocr_endpoint, CognitiveServicesCredentials(azure_ocr_subscription_key))
    except Exception as e:
        handle_local_error("Could not create ComputerVisionClient for Azure OCR, encountered error: ", e)
    
    calls_made = 0

    # Iterate over each page and apply OCR:
    print("\n\nBeginning image to Text OCR\n\n")
    for page_number, image in enumerate(pages, start = 1): # start=1 to match the page numbers in the PDF
        
        # Convert to bytes and create a stream
        try:
            img_stream = io.BytesIO()
            image.save(img_stream, format='PNG')
            img_stream.seek(0)  # Reset the stream position to the beginning
        except Exception as e:
            handle_local_error("Could not convert image to Byte Stream for Azure OCR, encountered error: ", e)
            continue

        # Send to Azure OCR
        try:
            if azure_cv_free_tier:
                if calls_made < 20:
                    print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                    result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
                    #analyze_result = computervision_client.begin_analyze_document("prebuilt-layout", img_stream).result()
                    calls_made += 1
                else:
                    print("Sleeping for 60secs due to AzureOCR free-tier restrictions!")
                    time.sleep(63)  #free tier restrictions!
                    print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                    result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
                    #analyze_result = computervision_client.begin_analyze_document("prebuilt-layout", img_stream).result()
                    calls_made = 1  #reset counter
            else:
                print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
        except HttpResponseError as e:
            print(f"\n\nHttpResponseError e: {e}\n\n")
            if e.status_code == 429:
                print("Exceeded free-tier usage limits, waiting for one-minute and retrying")
                time.sleep(63)  #free tier restrictions!
                img_stream.seek(0)
                print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
                calls_made = 1  #reset counter
        except Exception as e:
            if "(429)" in str(e):
                print("Exceeded free-tier usage limits, waiting for one-minute and retrying")
                time.sleep(63)  #free tier restrictions!
                img_stream.seek(0)
                print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
                calls_made = 1  #reset counter\
            else:
                handle_local_error("Could not convert image to Byte Stream for Azure OCR, encountered error: ", e)

        for region in result.regions:
            for line in region.lines:
                #print(" ".join([word.text for word in line.words]))

                try:
                    clean_text = str(" ".join([word.text for word in line.words]))
                except Exception as e:
                    handle_error_no_return("Could not obtain line from Azure OCR result, encountered error: ", e)
                    continue

                # Write the extracted text to the file:
                try:
                    output_text_file.write(f"[PAGE:{page_number}]\n{clean_text}\n")
                except Exception as e:
                    handle_local_error("Could not write to output text file, encountered error: ", e)

    # Close all files
    output_text_file.close()

    return output_text_file_path


def get_vision_llm_request_params():
    try:
        read_return = read_config(['vision_llm_local_url', 'vision_ocr_prompt'])
        vision_llm_local_url = read_return['vision_llm_local_url']
        vision_ocr_prompt = read_return['vision_ocr_prompt']
    except Exception as e:
        handle_local_error("Missing Vision LLM URL or Vision OCR Prompt for get_vision_llm_request_params, please provide required API config. Error: ", e)

    vision_request_payload = {
        'messages': json.dumps([
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": vision_ocr_prompt}
                ]
            }
        ])
    }

    headers = {
        'X-DPI': '300',
        'X-Max-New-Tokens': '5000'
    }

    return vision_llm_local_url, vision_request_payload, headers


def PDFtoVisionLLMOCRTXT(input_filepath):
    
    print("\n\nProcessing Document - PDF to Vision LLM OCR TXT\n\n")

    try:
        print("\n\nChecking if HF-Waitress Server is Online\n\n")
        if not is_local_server_online('hf-waitress')['server_available']:
            return handle_local_error("HF-Waitress Server is Offline! Please start the HF-Waitress Server and try again.")
        else:
            print("\n\nHF-Waitress Server is Online\n\n")
    except Exception as e:
        handle_error_no_return("Could not check if HF-Waitress Server is Online, presuming online and proceeding. Encountered error: ", e)

    try:
        read_return = read_config(['ocr_pdfs', 'force_extract_previously_extracted_text'])
        ocr_pdfs = read_return['ocr_pdfs']
        force_extract_previously_extracted_text = str(read_return['force_extract_previously_extracted_text']).lower() == 'true'
    except Exception as e:
        handle_local_error("Missing OCR PDFs directory for PDFtoVisionLLMOCRTXT, please provide required API config. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")   # Using this instead of os.path.splitext() in case filename contains a period
    output_text_file_path = os.path.join(ocr_pdfs, output_text_file_name).replace("\\","/")

    if os.path.exists(output_text_file_path) and not force_extract_previously_extracted_text:
        if os.path.getsize(output_text_file_path) > 0:
            print("Vision LLM OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("Vision LLM OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")

    # Convert PDF to  a list of images
    pil_image_object_list = []
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pil_image_object_list = convert_from_path(input_filepath, 300) # The convert_from_path() function from pdf2image lib intertnally uses Poppler to convert PDF pages to images, and then creates PIP Image objects from them. 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    try:
        vision_llm_local_url, vision_request_payload, headers = get_vision_llm_request_params()
    except Exception as e:
        return handle_local_error("Could not get Vision LLM request parameters, encountered error: ", e)

    for page_number, image in enumerate(pil_image_object_list, start=1): # start=1 to match the page numbers in the PDF

        print(f"\n\nProcessing Page: {page_number} from file: {source_filename}\n\n")

        try:
            print("\n\nConverting PIL-image object to Byte-Stream\n\n")
            img_stream = io.BytesIO()
            image.save(img_stream, format='PNG')
            img_stream.seek(0)
        except Exception as e:
            handle_error_no_return("Could not convert PIL-image object to Byte-Stream, encountered error: ", e)
            continue

        try:
            print("\n\nPreparing file payload for Vision LLM\n\n")
            file_payload = [
                ('file', ('page-image.png', img_stream, 'image/png'))
            ]
        except Exception as e:
            handle_error_no_return("Could not prepare file payload for Vision LLM, encountered error: ", e)
            continue

        try:
            print("\n\nSending request to Vision LLM\n\n")
            response = requests.post(vision_llm_local_url, headers=headers, data=vision_request_payload, files=file_payload)    # requests.post() is a convenience function specifically for sending POST requests with form-encoded or multipart data. It automatically sets the Content-Type header to multipart/form-data. It's more readable than the generic requests.request() function.
            print(f"\n\nVision LLM response: {response.json()}\n\n")
        except Exception as e:
            handle_error_no_return("Could not send request to Vision LLM, encountered error: ", e)
            continue

        try:
            print("\n\nExtracting inference output-text from Vision LLM Response\n\n")
            vision_output = response.json()['response']
        except Exception as e:
            handle_error_no_return("Could not obtain inference output from Vision LLM, encountered error: ", e)
            continue

        try:
            print(f"\n\nWriting output to output text file\n\n")
            output_text_file.write(f"[PAGE:{page_number}]\n{vision_output}\n")
        except Exception as e:
            handle_error_no_return("Could not write to output text file, encountered error: ", e)
            continue
    
     # Close all files
    output_text_file.close()

    return output_text_file_path


def get_kosmos_request_params():
    try:
        read_return = read_config(['kosmos_local_url', 'kosmos_task', 'kosmos_threshold'])
        kosmos_local_url = read_return['kosmos_local_url']
        kosmos_task = read_return['kosmos_task']
        kosmos_threshold = read_return['kosmos_threshold']
    except Exception as e:
        handle_local_error("Missing Kosmos API config, please provide required API config. Error: ", e)

    payload = {
        'task': kosmos_task,
        'threshold': kosmos_threshold
    }

    headers = {}

    return kosmos_local_url, payload, headers


def PDFtoKosmosOCRTXT(input_filepath):

    print("\n\nProcessing Document - PDF to Kosmos OCR TXT\n\n")

    try:
        read_return = read_config(['ocr_pdfs', 'force_extract_previously_extracted_text'])
        ocr_pdfs = read_return['ocr_pdfs']
        force_extract_previously_extracted_text = str(read_return['force_extract_previously_extracted_text']).lower() == 'true'
    except Exception as e:
        handle_local_error("Missing OCR PDFs directory for PDFtoKosmosOCRTXT, please provide required API config. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")   # Using this instead of os.path.splitext() in case filename contains a period
    output_text_file_path = os.path.join(ocr_pdfs, output_text_file_name).replace("\\","/")

    if os.path.exists(output_text_file_path) and not force_extract_previously_extracted_text:
        if os.path.getsize(output_text_file_path) > 0:
            print("Kosmos OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("Kosmos OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")
    
    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    try:
        kosmos_local_url, payload, headers = get_kosmos_request_params()
    except Exception as e:
        return handle_local_error("Could not get Kosmos request parameters, encountered error: ", e)

    try:
        print("\nPreparing file payload for Kosmos\n")
        file_payload = [
            ('file', (source_filename, open(input_filepath,'rb'),'application/pdf'))
        ]
    except Exception as e:
        handle_error_no_return("Could not prepare file payload for Kosmos, encountered error: ", e)

    # Send request to Kosmos and open an event stream to receive the response
    page_number = 0
    try:
        print("\nSending request to Kosmos\n")
        with requests.post(kosmos_local_url, headers=headers, data=payload, files=file_payload, stream=True) as response:
            response.raise_for_status() # Raise an exception for bad 4xx or 5xx status codes

            print("\nReceiving event-streaming response from Kosmos\n")
            for event in response.iter_lines(decode_unicode=True):
                if event:
                    if event.startswith('data:'):
                        event_data = event[5:].strip()
                        try:
                            json_data = json.loads(event_data)
                            if 'full_parsed_text' in json_data:
                                page_number += 1
                                full_parsed_text = json_data['full_parsed_text']
                                print(f"\n\nWriting full_parsed_text to output text file: {full_parsed_text}\n\n")
                                output_text_file.write(f"[PAGE:{page_number}]\n{full_parsed_text}\n")
                            else:
                                print(f"\n\nReceived plain-text event from Kosmos: {event_data}\n\n")
                        except json.JSONDecodeError as e:
                            print(f"\n\nCould not parse event from Kosmos as JSON dictionary, encountered error: {e}\n\n")
                            print(f"\n\nReceived plain-text event from Kosmos: {event}\n\n")
                        except Exception as e:
                            handle_error_no_return("Could not process event from Kosmos, encountered error: ", e)
    
    except requests.exceptions.RequestException as e:
        handle_local_error("Could not send request to Kosmos, encountered error: ", e)
    except Exception as e:
        handle_error_no_return("Could not send request to Kosmos, encountered error: ", e)

    # Close all files
    output_text_file.close()

    return output_text_file_path


def PDFtoTXT(input_file):

    print("\n\nProcessing Document - PDF to TXT\n\n")

    try:
        read_return = read_config(['pdfs_to_txts', 'force_extract_previously_extracted_text'])
        pdfs_to_txts = read_return['pdfs_to_txts']
        force_extract_previously_extracted_text = str(read_return['force_extract_previously_extracted_text']).lower() == 'true'
    except Exception as e:
        handle_local_error("Missing pdfs_to_txts directory for PDFtoTXT in config.json, encountered error: ", e)
    
    # Initialize PDF file reader
    try:
        pdf_file = open(input_file, 'rb')
    except Exception as e:
        handle_local_error("Could not open PDF file, encountered error: ", e)

    try:
        source_filename = os.path.basename(input_file)
    except Exception as e:
        handle_local_error("Could not open PDF file, encountered error: ", e)

    # Initialize PDF reader
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
    except Exception as e:
        handle_local_error("Could not initialize PDF reader, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")   # Using this instead of os.path.splitext() in case filename contains a period
    output_text_file_path = os.path.join(pdfs_to_txts, output_text_file_name).replace("\\","/")

    if os.path.exists(output_text_file_path) and not force_extract_previously_extracted_text:
        if os.path.getsize(output_text_file_path) > 0:
            print("PyPDF2-extracted .txt already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("PyPDF2-extracted .txt already exists but is empty! Overwriting with new .txt file.")

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    # Loop through all the pages and extract text
    for page_num in range(len(pdf_reader.pages)):
        
        try:
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
        except Exception as e:
            handle_error_no_return("Could not extract text from page, encountered error: ", e)

        #clean_text = text
        # Clean text
        clean_text = clean_text_string(text)
        page_number = int(page_num) + 1
        
        # Optionally, you can include page numbers in the text file
        # output_text_file.write(f'\n\n--- Page {page_num + 1} ---\n\n')
        
        # Write the extracted text to the file
        try:
            output_text_file.write(f"[PAGE:{page_number}]\n{clean_text}\n")
        except Exception as e:
            handle_local_error("Could not write to output text file, encountered error: ", e)

    # Close all files
    pdf_file.close()
    output_text_file.close()

    return output_text_file_path


def add_column_if_not_exists(cursor, table_name, column_name, column_type):
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        if column_name not in column_names:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    except Exception as e:
        handle_error_no_return(f"Error adding column {column_name} to {table_name}: ", e)


def init_and_connect_to_docs_loaded_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    try:
        read_return = read_config(['sqlite_docs_loaded_db'])
        sqlite_docs_loaded_db = read_return['sqlite_docs_loaded_db']
    except Exception as e:
        return handle_local_error("Missing sqlite_docs_loaded_db in config.json for method init_and_get_cursor_fordocs_loaded_db. Error: ", e)

    try:
        conn = sqlite3.connect(sqlite_docs_loaded_db)
        cursor = conn.cursor()
    except Exception as e:
        return handle_local_error("Could not establish connection to sqlite_docs_loaded_db, encountered error: ", e)

    # If the database does not currently exist...
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_records (
                    id INTEGER PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    vectordb_used TEXT,
                    chunk_size INTEGER,
                    chunk_overlap INTEGER,
                    knowledge_domain TEXT
            )
        ''')

        conn.commit()   # Auto-incrementing primary key 'id'
    except Exception as e:
        return handle_local_error("Could not create document_records DB, encountered error: ", e)
    
    try:
        add_column_if_not_exists(cursor, 'document_records', 'document_name', 'TEXT')
        add_column_if_not_exists(cursor, 'document_records', 'embedding_model', 'TEXT')
        add_column_if_not_exists(cursor, 'document_records', 'vectordb_used', 'TEXT')
        add_column_if_not_exists(cursor, 'document_records', 'chunk_size', 'INTEGER')
        add_column_if_not_exists(cursor, 'document_records', 'chunk_overlap', 'INTEGER')
        add_column_if_not_exists(cursor, 'document_records', 'knowledge_domain', 'TEXT')
    except Exception as e:
        return handle_local_error("Could not add necessary columns to document_records db, encountered error: ", e)

    return conn, cursor


def is_doc_already_loaded_to_db(document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain):
    print(f"Checking if {document_name} already exists in the appropriate records DB")

    try:
        conn, cursor = init_and_connect_to_docs_loaded_db()
    except Exception as e:
        handle_local_error("Could not initialize and connect to docs_loaded_db to check_if_doc_already_loaded_to_db, encountered error: ", e)

    try:
        cursor.execute("""
            SELECT id FROM document_records
            WHERE document_name = ?
            AND embedding_model = ?
            AND chunk_size = ?
            AND chunk_overlap = ?
            AND knowledge_domain = ?
        """, (document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain))
        existing_record = cursor.fetchone()
        if existing_record is not None:
            print(f"Document '{document_name}' already exists in records DB as loaded into {knowledge_domain} with embedding model '{embedding_model}' and chunk size {chunk_size} and chunk overlap {chunk_overlap}.")
            conn.close()
            return True
        else:
            print("Document does not exist in records DB.")
            conn.close()
            return False
    except Exception as e:
        handle_error_no_return("Could not check if document exists in records DB, returning False. Encountered error: ", e)
        conn.close()
        return False


def record_doc_loaded_to_db(document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain):

    print("\n\nRecording document loading to records DB\n\n")

    if is_doc_already_loaded_to_db(document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain):
        print("Document already exists in records DB, skipping insertion.")
        return True

    try:
        conn, cursor = init_and_connect_to_docs_loaded_db()
    except Exception as e:
        return handle_local_error("Could not initialize and connect to docs_loaded_db to record_doc_loaded_to_db, encountered error: ", e)
    
    try:
        cursor.execute("INSERT INTO document_records (document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain) VALUES (?, ?, ?, ?, ?)", (document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain))
        conn.commit()
        conn.close()
        print("\nDocument record added to document_records DB\n")
        return True
    except Exception as e:
        return handle_local_error("Could not update document_records DB, encountered error: ", e)
    


# List-splitter function for a large number of embeddings!
def split_embeddings_list(all_splits, max_emmbeddings_list_size):
    for i in range(0, len(all_splits), max_emmbeddings_list_size):  # Step through the large list in steps of max size
        yield all_splits[i:i + max_emmbeddings_list_size]   # Yield a slice of all_splits from index i upto but NOT including i+max_size. 
        '''
        While memory efficient, there is a key limitation: cannot use len() on the yielded object as if it were a list! This is because the yielded object is a generator object created via iteration, not a list!
        Also, issues may arise if attmepting to enumerate() as there's an implicit split_docs[i] indexing, especially if anywhere in the loop you then try a len() on the yielded object!
        This is because you would be using the generator multiple times (once for length, then again for iteration): Generators are "single-use" iterators - once you iterate through them, 
        they're exhausted and can't be used again without recreating them. Plus, they don't support random access or length checking because they generate values on-the-fly!
        Hence this method is not used by core_embedder(), but the code retained here for future reference and debugging/documentation purposes!
        '''


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self): #to provide string-representation of an object
        # return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"    # Does not truncate the actual page_content or even str(doc.page_content), rather it only comes into play for display purposes when we print the entire object as a string!
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

# Consider turning this into a generator function in the future for efficiency when dealing with large files!
def chunk_docs_with_page_numbers(input_file, chunk_size=250):
    print("\nGenerating Document Chunks\n")
    documents = []
    current_chunk = ""
    current_page = 1

    def add_chunk(chunk, page):
        if chunk.strip():   #if chunk is not empty!
            #print(f"\n\nAdding chunk from page {page}: {chunk.strip()}\n\n")
            documents.append({
                'content': chunk.strip(),
                'source': input_file,
                'page_number': page
            })
    
    def provide_chunk_space(chunk_size, current_chunk_for_eval):
        available_chunk_space = chunk_size - len(current_chunk_for_eval)
                    
        if available_chunk_space < 0:   # negative space means the current chunk is too big for the chunk_size, so we need to split it, add the first part to the documents list, and keep the rest in current_chunk
            space = current_chunk_for_eval[:chunk_size].rfind(" ")
            if space == -1 or space == 0: # if by any chance no space is found or it's at the beginning of the chunk, we'll force a split at chunk_size
                space = chunk_size    
            add_chunk(current_chunk_for_eval[:space], current_page)
            current_chunk_for_eval = current_chunk_for_eval[space:].lstrip()
        
        elif available_chunk_space == 0:   # zero space means the current chunk is exactly the chunk_size, so we add it and reset the current chunk
            add_chunk(current_chunk_for_eval, current_page)
            current_chunk_for_eval = ""

        return current_chunk_for_eval
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('[PAGE:'):   # Our text-extraction method adds '[PAGE:' to the beginning of each page, and these are on a separate line, so this line will only contain the added page number. Example: [PAGE:1]
                    new_page = int(line.strip()[6:-1])  #strip() removes leading and trailing whitespace, [6:-1] removes the [PAGE: and ]
                    if new_page != current_page:    # update page_number if necessary
                        current_page = new_page
                        if current_chunk.strip():   # if the current chunk is not empty, add it to the documents list as we're starting a new page
                            add_chunk(current_chunk, current_page)
                            current_chunk = ""
                else:   # non-page-number, normal line of text
                    current_chunk += line
                    while len(current_chunk) >= chunk_size:
                        current_chunk = provide_chunk_space(chunk_size, current_chunk)

        # Add any remaining content
        if current_chunk.strip():
            add_chunk(current_chunk, current_page)

    except Exception as e:
        handle_local_error("Could not chunk document, encountered error: ", e)

    print(f"\n\nGenerated {len(documents)} document chunks\n\n")
    return documents


def create_vector_db_directory(path_to_knowledge_domain, embedding_function):
    try:
        vector_db_path = os.path.join(path_to_knowledge_domain, "vector_db_and_whoosh_index", embedding_function)
        if not os.path.exists(vector_db_path):
            os.makedirs(vector_db_path, exist_ok=True)
            print(f"\n\nCreated vector_db directory: {vector_db_path}\n\n")
        else:
            print(f"\n\nVector_db directory already exists, returning path: {vector_db_path}\n\n")
        return vector_db_path
    except Exception as e:
        return handle_local_error("Could not create vector_db directory, encountered error: ", e)


def core_embedder(chunks, selected_embedding_model, path_to_knowledge_domain):
    # Convert Chunks to Document objects:
    try:
        numbered_splits = [Document(page_content=chunk['content'], metadata={'page_number': chunk['page_number'], 'source': chunk['source']}) for chunk in chunks]  # Generates a list of Document objects, each containing a 'page_content' string, and a 'metadata' dictionary with 'source' and 'page_number' keys
    except Exception as e:
        handle_local_error("Failed to convert chunks to Document objects for storage to VectorDB, encountered error: ", e)

    # Load Embedding Model
    embedding_model = None
    try:
        embedding_model = SentenceTransformer(selected_embedding_model, trust_remote_code=True)
    except Exception as e:
        handle_local_error("Could not load embedding model, encountered error: ", e)

    # Generate Embeddings for just the page_content
    try:
        texts_to_embed = [doc.page_content for doc in numbered_splits]
        print("\n\nGenerating embeddings...\n\n")
        embeddings = embedding_model.encode(texts_to_embed)    # By default, convert_to_tensor=False and this is what we want because ChromaDB expects numpy arrays, not PyTorch tensors!
    except Exception as e:
        handle_local_error("Could not generate embeddings, encountered error: ", e)
    finally:
        if embedding_model is not None:
            del embedding_model
            if torch.cuda.is_available():
                print("Emptying CUDA cache")
                torch.cuda.empty_cache()
            print("Collecting garbage")
            gc.collect()

    # Get VectorDB Directory
    vector_db_path = create_vector_db_directory(path_to_knowledge_domain, selected_embedding_model)

    # Store Chunks in VectorDB
    print("Storing to VectorDB: ChromaDB")
    try:
        # Initialize Chroma Client and collection
        chroma_client = chromadb.PersistentClient(path=vector_db_path, settings=chromadb.Settings(allow_reset=True))
        collection = chroma_client.get_or_create_collection(name="knowledge_domain", metadata={"hnsw:space": "cosine"}) # By default, ChromaDB returns the L2 distance (lower is better), but we want cosine distance (higher is better)

        batch_size = 5000
        total_batches = (len(numbered_splits) + batch_size - 1) // batch_size # Eg: batch_size = (10000 + 5000 - 1) // 5000 = 3 batches. Floor division will result in an integer, but always rounds down, so adding (batch_size - 1) ensures we round up instead of down.
        
        if total_batches > 1:
            print(f"\n\nLarge number of document embeddings detected, splitting into {total_batches} batches of {batch_size} each for storage to VectorDB...\n\n")

        for i in range(0, len(numbered_splits), batch_size):
            print(f"\n\Storing embeddings batch {(i // batch_size) + 1} of total {total_batches} batches into ChromaDB...\n\n")
            # Get the current batch of docs
            batch_docs = numbered_splits[i:i+batch_size]

            # Get the corresponding embeddings for the current batch
            batch_embeddings = embeddings[i:i+batch_size]

            # Prepare the data for ChromaDB format
            documents = [chunk.page_content for chunk in batch_docs]
            metadatas = [chunk.metadata for chunk in batch_docs]
            ids = [str(uuid.uuid4()) for _ in batch_docs]

            # Add the data to the collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=batch_embeddings.tolist()  # Convert embeddings from NumPy arrays to list of lists
            )
    except Exception as e:
        handle_local_error("Could not store to VectorDB, encountered error: ", e)

    return True


def get_graphing_request_params():
    try:
        read_return = read_config(['exl2_quantize_graph_model', 'graph_model_access_url', 'graph_model_server_port', 'graph_model_max_new_tokens', 'graph_model_temperature', 'graph_model_do_sample', 'graph_model_top_k', 'graph_model_top_p', 'graph_model_min_p'])
    except Exception as e:
        return handle_local_error("Could not get graphing request params, encountered error: ", e)

    exl2_quantize_graph_model = str(read_return['exl2_quantize_graph_model']).lower() == 'true'
    graph_model_access_url = read_return['graph_model_access_url']
    graph_model_server_port = read_return['graph_model_server_port']
    graph_model_max_new_tokens = read_return['graph_model_max_new_tokens']
    graph_model_temperature = read_return['graph_model_temperature']
    graph_model_do_sample = read_return['graph_model_do_sample']
    graph_model_top_k = read_return['graph_model_top_k']
    graph_model_top_p = read_return['graph_model_top_p']
    graph_model_min_p = read_return['graph_model_min_p']

    headers = {
        'Content-Type': 'application/json',
        'X-Max-New-Tokens': str(graph_model_max_new_tokens),
        'X-Temperature': str(graph_model_temperature),
        'X-Top-K': str(graph_model_top_k),
        'X-Top-P': str(graph_model_top_p),
        'X-Min-P': str(graph_model_min_p)
    }

    grapher_url = f"http://{graph_model_access_url}:{graph_model_server_port}"

    if not exl2_quantize_graph_model:
        headers['X-Return-Full-Text'] = 'False'
        headers['X-Do-Sample'] = str(graph_model_do_sample)
        grapher_url += "/completions"
    else:
        headers['Connection'] = 'keep-alive'
        grapher_url += "/exl2_grapher"

    return grapher_url, headers, exl2_quantize_graph_model


def get_graphing_request_payload(chunk, exl2_quantize_graph_model):
    payload_content = str(chunk) + "\n<knowledge_graph>"

    if exl2_quantize_graph_model:
        payload = json.dumps(f"<start_of_turn>user\n{payload_content}<end_of_turn>\n<start_of_turn>model\n")
    else:
        payload = f'''
            {{
                "messages": [
                    {{"role": "user", "content": {json.dumps(payload_content)}}}
                ]
            }}
        '''

    return payload


def hf_waitress_non_streaming_request_response_handler(endpoint_url, headers, payload):
    print(f"\nHF-Waitress Non-Streaming Request Response Handler Invoked\n")
    try:
        response = requests.post(endpoint_url, headers=headers, data=payload)
        print("\nCompleted, returning response\n")
        return (response.json()['response'])
    except Exception as e:
        return handle_local_error("Failed /completions request to extract entities and relationships from chunk, encountered error: ", e)


def hf_waitress_bulk_stream_request_response_handler(endpoint_url, headers, payload):
    print(f"\nHF-Waitress Bulk-Stream Request Response Handler Invoked\n")
    try:
        response = requests.post(endpoint_url, headers=headers, data=payload, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes so we can catch them in the except block

        full_response = {}
        chunk_number = 1
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data:"):
                    event_data = line[6:].strip()
                    try:
                        received_data = json.loads(event_data)  # json.loads() converts the JSON string into a Python object (dict, list, etc.)
                        if not isinstance(received_data, dict):
                            handle_error_no_return("Received data is not a dict, and is of type: ", type(received_data), ". Skipping. For Debug - Received data: ", received_data)
                            continue
                        full_response[chunk_number] = received_data
                        
                        try:
                            if 'entities_and_relationships' in full_response[chunk_number] and isinstance(full_response[chunk_number]['entities_and_relationships'], str):
                                '''
                                While the received data is a dict, the value of the entities_and_relationships key itself may be returned as a string by exl2_grapher's extraction_task(), 
                                so we need to convert it to a dict via ast.literal_eval().
                                The summary_generation_task() on the other hand always returns a dict so we don't need to do anything.

                                NOTE: ast.literal_eval() works only on string inputs! If we were to print:

                                print(f"Type of entities_and_relationships: {type(full_response[chunk_number]['entities_and_relationships'])}")

                                We'd get str for extraction mode, dict for summary generation mode, thus failing in the latter as ast.literal_eval() works only on string inputs!
                                We only want to literal_eval() if the entities_and_relationships is a string, otherwise it's already a dict and we can skip the literal_eval.
                                '''
                                full_response[chunk_number]['entities_and_relationships'] = ast.literal_eval(full_response[chunk_number]['entities_and_relationships'])
                        except Exception as e:
                            handle_error_no_return("Failed to literal_eval entities_and_relationships, skipping. Encountered error: ", e)
                        
                        chunk_number += 1
                    
                    except json.JSONDecodeError as e:
                        handle_error_no_return(f"Failed to parse event data: {event_data}, encountered error: ", e)
                elif line.startswith("event: END"):
                    break
                else:
                    print(f"\nUnexpected Line Format: {line}\n")

        if not full_response:
            print("\nWarning: No response from exl2_stream / exl2_grapher request\n")
            return None

        print("\nCompleted, returning response\n")
        return full_response
        
    except Exception as e:
        return handle_local_error("Failed request to /exl2_stream or /exl2_grapher APIs, encountered error: ", e)


def graphing_request_response_handler(grapher_url, headers, payload):
    print(f"\nHF-Waitress Non-Streaming Graphing-Request Response Handler Invoked\n")
    try:
        response = hf_waitress_non_streaming_request_response_handler(grapher_url, headers, payload)
        print(f"\nResponse (Entities and Relationships): {response}\n")
        return ast.literal_eval(response)
    except Exception as e:
        return handle_local_error("Failed /completions request to extract entities and relationships from chunk, encountered error: ", e)


def extract_all_entities_and_relationships(chunk_entities: dict, rag_response_mode: bool = False) -> dict:
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

    print("\nExtracting all entities and relationships for the entire document\n")

    try:
        grapher_url, headers, exl2_quantize_graph_model = get_graphing_request_params()
    except Exception as e:
        return handle_local_error("Could not get graphing request params, encountered error: ", e)

    if exl2_quantize_graph_model:   # invoke /exl2_grapher
        try:
            payload = json.dumps({"chunk_entities": chunk_entities, "extraction_mode": True, "rag_response_mode": rag_response_mode})
            full_response = hf_waitress_bulk_stream_request_response_handler(grapher_url, headers, payload)
            # print(f"\nExl2 Bulk-Graphing Response (Entities and Relationships):\n\n{full_response}\n")
            return full_response
        except Exception as e:
            return handle_local_error("Error with request to /exl2_grapher, encountered error: ", e)

    else:   # invoke /completions
        # TODO: Implement non-exl2 bulk-summary generation with /completions

        try:
            for chunk_number, chunk_data in chunk_entities.items(): # chunk_entities.items() returns a dict_items object which is iterable. chunk_data is a dict for each chunk while chunk_number is a string ranging from 0 to len(chunk_entities) - 1
                try:
                    print(f"\nExtracting entities and relationships for chunk {chunk_number} of total {len(chunk_entities)} chunks...\n")
                    payload = get_graphing_request_payload(chunk_data['chunk_text'], exl2_quantize_graph_model)
                    response = graphing_request_response_handler(grapher_url, headers, payload)

                    if response is None or response == {}:
                        print(f"\nNo entities or relationships found for chunk {chunk_number}...\n")
                        response = {}
                    else:
                        print(f"\nSuccessfully extracted entities and relationships for chunk {chunk_number}\n")

                    chunk_entities[chunk_number]['entities_and_relationships'] = response  # Not using chunk_data['entities_and_relationships'] as that would not update the original dict because dicts are passed by reference in Python
                except Exception as e:
                    handle_error_no_return(f"\nCould not extract entities and relationships for graph chunk {chunk_number} of total {len(chunk_entities)} chunks, encountered error: ", e)
            
            return chunk_entities
                    
        except Exception as e:
            return handle_local_error("\nCould not iterate over chunk entities, encountered error: ", e)


def get_graph_db_client():
    print("\nObtaining Graph DB Client\n")

    read_return = read_config(['graph_db_server_host', 'assign_host_port_to_graph_db_server'])
    graph_db_server_host = read_return['graph_db_server_host']
    assign_host_port_to_graph_db_server = read_return['assign_host_port_to_graph_db_server']

    try:
        client = FalkorDB(host=graph_db_server_host, port=assign_host_port_to_graph_db_server)
        print(f"\nGraph DB Client obtained successfully!\n")
        return client
    except Exception as e:
        return handle_local_error("Could not obtain Graph DB Client, encountered error: ", e)


def sanitize_names(name):
    name_str = str(name).lower()
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name_str)  # Matches all non-alphanumeric characters and replaces them with an underscore as OpenCypher spec disallows them in node names
    if sanitized[0].isdigit():
        sanitized = 'n_' + sanitized    # OpenCypher spec disallows digits at the beginning of a node name, even if they're strings eg "2025"!
    return sanitized


def get_request_params_for_local_llm_server(formatted_prompt=""):
    try:
        read_return = read_config(['local_llm_server', 'hf_waitress_access_url', 'hf_waitress_server_port', 'llama_cpp_access_url', 'llama_cpp_server_port', 'local_llm_temperature', 'local_llm_top_k', 'local_llm_top_p', 'local_llm_min_p'])
    except Exception as e:
        return handle_local_error("Could not read request params from config.json, encountered error: ", e)

    headers = {'Content-Type': 'application/json'}

    if read_return['local_llm_server'] == 'hf-waitress':

        try:
            read_hf_return = read_hf_config(['exl2', 'max_new_tokens', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p'])
        except Exception as e:
            return handle_local_error("Could not read hf-waitress config, encountered error: ", e)

        exl2 = str(read_hf_return['exl2']).lower() == 'true'
        
        headers['X-Max-New-Tokens'] = str(read_hf_return['max_new_tokens'])
        headers['X-Temperature'] = str(read_hf_return['temperature'])
        headers['X-Top-K'] = str(read_hf_return['top_k'])
        headers['X-Top-P'] = str(read_hf_return['top_p'])
        headers['X-Min-P'] = str(read_hf_return['min_p'])
        
        base_url = f"http://{read_return['hf_waitress_access_url']}:{read_return['hf_waitress_server_port']}"
        
        payload = formatted_prompt
        
        if not exl2:
            headers['X-Return-Full-Text'] = 'False'
            headers['X-Do-Sample'] = str(read_hf_return['do_sample'])
            endpoint_url = f"{base_url}/completions"
        else:
            payload = json.dumps(formatted_prompt)
            headers['Connection'] = 'keep-alive'
            endpoint_url = f"{base_url}/exl2_grapher"
    
    else:
        payload = {
            'prompt': formatted_prompt,
            'temperature': read_return['local_llm_temperature'],
            'top_k': read_return['local_llm_top_k'],
            'top_p': read_return['local_llm_top_p'],
            'min_p': read_return['local_llm_min_p']
        }

        endpoint_url = f"http://{read_return['llama_cpp_access_url']}:{read_return['llama_cpp_server_port']}/completion"

    return endpoint_url, headers, payload, exl2


def summary_generator(chunk_entities=None):
    print("\nGenerating summary...\n")

    if chunk_entities is None:
        return handle_local_error("Chunk entities are required to generate summaries")

    local_llm_server = read_config(['local_llm_server'])['local_llm_server']
    exl2 = str(read_hf_config(['exl2'])['exl2']).lower() == 'true'

    if local_llm_server == 'hf-waitress' and exl2:

        exl2_prompt_template_format = read_config(['exl2_prompt_template_format'])['exl2_prompt_template_format']

        try:
            endpoint_url, headers, _, _ = get_request_params_for_local_llm_server()
        except Exception as e:
            return handle_local_error("Could not get graphing request params, encountered error: ", e)

        try:
            payload = json.dumps({"chunk_entities": chunk_entities, "exl2_prompt_template_format": exl2_prompt_template_format, "summary_generation_mode": True})
            full_response = hf_waitress_bulk_stream_request_response_handler(endpoint_url, headers, payload)
            # print(f"\nExl2 Bulk-Summary Generation Response:\n\n{full_response}\n")
            return full_response
        except Exception as e:
            return handle_local_error("Error with request to /exl2_grapher, encountered error: ", e)

    elif local_llm_server == 'hf-waitress' and not exl2:
        # TODO: Implement non-exl2 bulk-summary generation with /completions
        pass

    elif local_llm_server == 'llama-cpp':
        # TODO: Implement llama-cpp summary generation
        pass


def add_nodes_to_graph(selected_knowledge_domain: str, nodes: list, graph: FalkorDB, source_document: str = '', page_number: list = None):
    '''
    We can define a default blank string as strings are immutable in Python, but cannot define a default blank list as lists are mutable!

    # BAD: Using mutable default argument
    def add_to_list(item, my_list=[]):
        my_list.append(item)
        return my_list

    print(add_to_list(1))  # Output: [1]
    print(add_to_list(2))  # Output: [1, 2] -- Surprise! Not [2]
    print(add_to_list(3))  # Output: [1, 2, 3] -- The list keeps growing!

    # GOOD: Using None as default and creating new list in function body
    def add_to_list_safe(item, my_list=None):
        if my_list is None:
            my_list = []  # Creates a new list each time when None is passed
        my_list.append(item)
        return my_list

    print(add_to_list_safe(1))  # Output: [1]
    print(add_to_list_safe(2))  # Output: [2]
    print(add_to_list_safe(3))  # Output: [3]
    '''

    print(f"\nStoring entities(nodes) to {selected_knowledge_domain} graph DB\n")

    processed_nodes = {}    # Format: {(name, node_type): True}
    page_number = [] if page_number is None else page_number   

    for node in nodes:
        try:
            name = str(node['name'])
            node_type = str(node['type'])
            updated_summary = list(node.get('summary', [])) if node.get('summary') else []   # dict .get() method is safer than `if node['summary']` because it provides a default value if the key doesn't exist and handles NoneType errors gracefully!
            
            node_key = (name, node_type)

            if node_key in processed_nodes:
                print(f"Skipping duplicate node {name} of type {node_type} in {selected_knowledge_domain} graph DB")
                continue

            node_name = sanitize_names(name)

            # MERGE on stable properties to prevent duplicates, then SET the summary and add source_document as per the case:
            graph.query(f"""
                MERGE (n:{node_name} {{name:$name, type:$type}})
                SET n.summary = CASE
                    WHEN $summary = [] THEN n.summary
                    WHEN n.summary IS NULL THEN $summary
                    ELSE n.summary + $summary
                END
                SET n.source_documents = CASE
                    WHEN $source_document = '' THEN n.source_documents
                    WHEN n.source_documents IS NULL THEN [$source_document]
                    WHEN NOT $source_document IN n.source_documents THEN n.source_documents + [$source_document]
                    ELSE n.source_documents
                END
                SET n.page_number = CASE
                    WHEN $page_number = [] THEN n.page_number
                    WHEN n.page_number IS NULL THEN $page_number
                    WHEN NOT $page_number IN n.page_number THEN n.page_number + $page_number
                    ELSE n.page_number
                END
            """, {
                'name': name.replace("'", ""),
                'type': node_type.replace("'", ""),
                'summary': updated_summary,
                'source_document': source_document.replace("'", ""),
                'page_number': page_number
            })
            

            '''
            For source_docs:
            CASE 1: If the `source_documents` list property doesn't exist yet: create a new list with just this document.
            CASE 2: If the `source_documents` list property exists but doesn't contain this document: add this document to the list.
            CASE 3: If the `source_documents` list property already contains this document: leave the list unchanged!

            For summary:
            CASE 1: If the new summary list is empty: leave the list unchanged!
            CASE 2: If the existing list in the graph is NULL, might as well set to the new summary list even if it's empty!
            CASE 3: If neither are blank.null, append the new summary to the existing list.

            Not passing exisiting summaries to the summary generator as it overwhlems the LLMs context window, so must append to the existing list here!
            '''
            
            # Mark node as processed:
            processed_nodes[node_key] = True
            
            print(f"\nCreated node - name: {name}, type: {node_type} - in {selected_knowledge_domain} graph DB\n")
        except Exception as e:
            handle_error_no_return(f"Could not create node - name: {name}, type: {node_type} - in {selected_knowledge_domain} graph DB, skipping. Encountered error: ", e)


def add_relationships_to_graph(selected_knowledge_domain: str, relationships: list, graph: FalkorDB, source_document: str = '', page_number: list = None):
    print(f"\nStoring relationships to {selected_knowledge_domain} graph DB\n")

    processed_relationships = {}    # Format: {(source, target, relationship): True}
    page_number = [] if page_number is None else page_number

    for relationship in relationships:
        try:
            source = str(relationship['source'])
            target = str(relationship['target'])
            relationship_type = str(relationship['relationship'])
            relationship_key = (source, target, relationship_type)
            
            updated_summary = list(relationship.get('summary', [])) if relationship.get('summary') else []   # dict .get() method is safer than `if relationship['summary']` because it provides a default value if the key doesn't exist and handles NoneType errors gracefully!
            
            if relationship_key in processed_relationships:
                print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type}) in {selected_knowledge_domain} graph DB")
                continue

            source = sanitize_names(source)
            target = sanitize_names(target)
            relationship_type = sanitize_names(relationship_type).upper()

            '''
            MERGE on stable properties to prevent duplicates, then SET the summary and add source_document as per the case.
            Track weights for relationships to improve clustering by tracking how many times each relationship is detected:
            Microsoft GraphRAG paper's emphasis on "normalized counts of detected relationship instances" is quite deliberate - 
            it's a way to let the data itself tell you which relationships are more significant in your knowledge graph!
            '''
            graph.query(f"""
                MERGE (s:{source} {{name:$source_name}})
                MERGE (t:{target} {{name:$target_name}})
                MERGE (s)-[r:{relationship_type}]->(t)
                SET r.weight = CASE
                    WHEN r.weight IS NULL THEN 1
                    ELSE r.weight + 1
                END
                SET r.summary = CASE
                    WHEN $summary = [] THEN r.summary
                    WHEN r.summary IS NULL THEN $summary
                    ELSE r.summary + $summary
                END
                SET r.source_documents = CASE
                    WHEN $source_document = '' THEN r.source_documents
                    WHEN r.source_documents IS NULL THEN [$source_document]
                    WHEN NOT $source_document IN r.source_documents THEN r.source_documents + [$source_document]
                    ELSE r.source_documents
                END
                SET r.page_number = CASE
                    WHEN $page_number = [] THEN r.page_number
                    WHEN r.page_number IS NULL THEN $page_number
                    WHEN NOT $page_number IN r.page_number THEN r.page_number + $page_number
                    ELSE r.page_number
                END
            """, {
                'source_name': relationship['source'].replace("'", ""),
                'target_name': relationship['target'].replace("'", ""),
                'summary': updated_summary,
                'source_document': source_document.replace("'", ""),
                'page_number': page_number
            })

                # We don't want to use the sanitized names as both labels and property values

            # Mark relationship as processed:
            processed_relationships[relationship_key] = True
            
            print(f"\nCreated relationship - source: {source}, target: {target}, relationship: {relationship} - in {selected_knowledge_domain} graph DB\n")
        except Exception as e:
            handle_error_no_return(f"Could not create relationship from data: {relationship} in {selected_knowledge_domain} graph DB, skipping. Encountered error: ", e)


def get_summary_and_source_documents_for_node(graph, name, node_type):
    print(f"\nChecking if summary for node {name} of type {node_type} exists in graph\n")

    try:
        node_name = sanitize_names(name)

        query = f"""
            MATCH (n:{node_name} {{name: '%s', type: '%s'}})
            RETURN n.summary AS summary, n.source_documents AS source_documents
        """ % (name.replace("'", ""), node_type.replace("'", ""))

        result = graph.query(query)
        
        if hasattr(result, 'result_set') and result.result_set:
            print(f"\nExisting summary for node found: {result.result_set[0][0]}\n")
            summary_list = list(result.result_set[0][0]) if result.result_set[0][0] else []
            source_documents_list = list(result.result_set[0][1]) if result.result_set[0][1] else []
            return summary_list, source_documents_list
        else:
            print(f"\nNo existing summary for node found...\n")
            return [], []   # If no summary is found, return an empty list:

    except Exception as e:
        handle_error_no_return(f"Could not check if node {name} of type {node_type} exists in graph, returning empty list. Encountered error: ", e)
        return [], []


def get_summary_and_source_documents_for_relationship(graph, source, target, relationship_type):
    print(f"\nChecking if summary for relationship {source} -> {target} ({relationship_type}) exists in graph\n")

    source_label = sanitize_names(source)
    target_label = sanitize_names(target)

    try:
        query = f"""
            MATCH (s:{source_label} {{name: '{source}'}})-[r:{relationship_type}]->(t:{target_label} {{name: '{target}'}})
            RETURN r.summary AS summary, r.source_documents AS source_documents
        """

        result = graph.query(query)

        if hasattr(result, 'result_set') and result.result_set:
            print(f"\nExisting summary for relationship found: {result.result_set[0][0]}\n")
            summary_list = list(result.result_set[0][0]) if result.result_set[0][0] else []
            source_documents_list = list(result.result_set[0][1]) if result.result_set[0][1] else []
            return summary_list, source_documents_list
        else:
            print(f"\nNo existing summary for relationship found...\n")
            return [], []   # If no summary is found, return an empty list:

    except Exception as e:
        handle_error_no_return(f"Could not check if summary for relationship {source} -> {target} ({relationship_type}) exists in graph, returning empty list. Encountered error: ", e)
        return [], []


def get_summaries_for_all_nodes(nodes: list, graph: FalkorDB, print_string: str = "", get_source_documents: bool = False):
    nodes_with_existing_summaries = []
    processed_nodes = {}    # Will de-duplicate nodes!

    for count, node in enumerate(nodes):
        print(f"Checking for existing summary for node {count+1} of {len(nodes)} {print_string}...")
        try:
            name = str(node['name'])
            node_type = str(node['type'])

            node_key = (name, node_type)

            if node_key in processed_nodes:
                print(f"Skipping duplicate node {name} of type {node_type} when checking for existing summaries in graph DB")
                continue

            try:
                existing_summary, existing_source_documents = get_summary_and_source_documents_for_node(graph, name, node_type)
            except Exception as e:
                existing_summary = []
                existing_source_documents = []
                handle_error_no_return(f"Could not check existing summary for node {name} of type {node_type}, skipping. Encountered error: ", e)

            # update node in chunk_entities dict:
            if get_source_documents:
                nodes_with_existing_summaries.append({
                    'name': name,
                    'type': node_type,
                    'summary': existing_summary,
                    'source_documents': existing_source_documents
                })
            else:
                nodes_with_existing_summaries.append({
                    'name': name,
                    'type': node_type,
                    'summary': existing_summary
                })

            processed_nodes[node_key] = True

        except Exception as e:
            handle_error_no_return(f"Could not get summary for node {name} of type {node_type}, skipping. Encountered error: ", e)
                    
    return nodes_with_existing_summaries


def get_summaries_for_all_relationships(relationships: list, graph: FalkorDB, print_string: str = "", get_source_documents: bool = False):
    relationships_with_existing_summaries = []
    processed_relationships = {}    # Will de-duplicate relationships!

    for count, relationship in enumerate(relationships):
        print(f"Checking for existing summary for relationship {count+1} of {len(relationships)} {print_string}...")
        try:
            source = str(relationship['source'])
            target = str(relationship['target'])
            relationship_type = sanitize_names(str(relationship['relationship']).upper())   # Added as sanitize_names().upper() hence formatting here too!

            relationship_key = (source, target, relationship_type)

            if relationship_key in processed_relationships:
                print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type}) when checking for existing summaries in graph DB")
                continue

            try:
                existing_summary, existing_source_documents = get_summary_and_source_documents_for_relationship(graph, source.replace("'", ""), target.replace("'", ""), relationship_type)
            except Exception as e:
                existing_summary = []
                existing_source_documents = []
                handle_error_no_return(f"Could not check existing summary for relationship {source} -> {target} ({relationship_type}), skipping. Encountered error: ", e)

            if get_source_documents:
                relationships_with_existing_summaries.append({
                    'source': source,
                    'target': target,
                    'relationship': str(relationship['relationship']),
                    'summary': existing_summary,
                    'source_documents': existing_source_documents
                })
            else:
                relationships_with_existing_summaries.append({
                    'source': source,
                    'target': target,
                    'relationship': str(relationship['relationship']),
                    'summary': existing_summary
                })

            processed_relationships[relationship_key] = True

        except Exception as e:
            handle_error_no_return(f"Could not get summary for relationship {source} -> {target} ({relationship_type}), skipping. Encountered error: ", e)

    return relationships_with_existing_summaries


def store_entities_and_relationships_in_graph_db(chunk_entities: dict, selected_knowledge_domain: str, graph: FalkorDB, skip_summary_generation: bool = False, summaries_filepath: str = None,):
    '''
    Receives a complete chunk_entities dict:

    chunk_entities = {
        '<graph_chunk_number>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>'
        }
    }

    And basis the 'chunk_text' and 'entities_and_relationships' for each `graph_chunk_number`:
        1. Handles Summaries if applicable:
            - Retrieves existing summaries for each node and relationship
            - Generates a summary for each node and relationship. This sequence ensures highest throughput of summary-generation requests to the LLM as both the lookup for existing summaries and the storage to the GraphDB are handled completely seperately!
        2. Stores the nodes and relationships in the graph DB
    '''

    print(f"\nStoring entities and relationships in {selected_knowledge_domain} graph DB\n")

    try:
        # First, Get/Generate summaries for each node and relationship if applicable:
        if not skip_summary_generation:

            try:
                chunks = summary_generator(chunk_entities=chunk_entities)
                try:
                    save_json_file(chunks, summaries_filepath)
                except Exception as e:
                    handle_error_no_return(f"Error saving summaries to file, they will have to be re-generated in the future if this document is re-processed. Encountered error: ", e)
            except Exception as e:
                handle_error_no_return(f"Error generating summaries for nodes and relationships, proceeding without new summaries. Encountered error: ", e)
                chunks = chunk_entities
        
        else:
            print(f"\nSkipping summary generation for all nodes and relationships in {selected_knowledge_domain} graph DB\n")
            chunks = chunk_entities

        # Store the nodes and relationships in the graph DB
        for chunk_number, chunk_data in chunks.items():
            try:
                add_nodes_to_graph(selected_knowledge_domain, chunk_data['entities_and_relationships']['nodes'], graph, chunk_data['source_doc_name'], chunk_data['page_number'])
                add_relationships_to_graph(selected_knowledge_domain, chunk_data['entities_and_relationships']['relationships'], graph, chunk_data['source_doc_name'], chunk_data['page_number'])
            except Exception as e:
                handle_error_no_return(f"Could not store entities and relationships for chunk {chunk_number} in {selected_knowledge_domain} graph DB, encountered error: ", e)
    except Exception as e:
        return handle_local_error("Could not iterate over chunk entities, encountered error: ", e)

    print(f"\nSuccessfully stored {len(chunk_entities)} chunks in {selected_knowledge_domain} graph DB\n")
    return True


def graph_db_docker_container_is_running(container_name):
    print("\nChecking if FalkorDB Docker container is running...\n")

    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name={container_name}' , '--format', '{{.ID}}'], # get container ID
            capture_output=True,    # captures the command's output and error, while suppressing the print to the terminal
            text=True,  # Get output as string and not bytes
            check=True
        )
        container_id = result.stdout.strip()
        print(f"\n{container_name} Docker container ID: {container_id}\n")
        return container_id is not None and container_id != ""
    except Exception as e:
        handle_error_no_return(f"Could not check if {container_name} Docker container is running, encountered error: ", e)
        return False


def bring_graph_db_online():    # launch FalkorDB Docker container
    print(f"\nLaunching FalkorDB Docker container...\n")

    read_return = read_config(['launch_graph_db_with_ui', 'assign_host_port_to_graph_db_server', 'assign_host_port_to_graph_db_ui'])
    launch_graph_db_with_ui = read_return['launch_graph_db_with_ui']
    assign_host_port_to_graph_db_server = read_return['assign_host_port_to_graph_db_server']
    assign_host_port_to_graph_db_ui = read_return['assign_host_port_to_graph_db_ui']

    # Check if Docker Engine is running
    try:
        subprocess.run(['docker', 'info'], capture_output=True, check=True)  # check=True will raise an exception if the command returns a non-zero exit code
    except Exception as e:
        return handle_local_error("Docker Engine is not running, encountered error: ", e)

    print("\nDocker Engine is running, proceeding with FalkorDB Docker container launch...\n")

    if graph_db_docker_container_is_running('falkor-db'):
        print("\nFalkorDB Docker container is already running, skipping launch...\n")
        return True

    command = [
        'docker', 'run', '-p', f'{assign_host_port_to_graph_db_server}:6379',
        *(['-p', f'{assign_host_port_to_graph_db_ui}:3000'] if launch_graph_db_with_ui else []),
        '--name', 'falkor-db',
        '-it', '--rm', '-v', './data:/data', 'falkordb/falkordb:edge'
    ]   # Using conditional list-unpacking with * to handle optional arguments!

    try:
        subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE) if platform.system() == 'Windows' else subprocess.Popen(command, shell=True)
        # Check if the container is running
        container_name = 'falkor-db'
        timeout = 2
        attempts = 50
        for _ in range(attempts):
            if graph_db_docker_container_is_running(container_name):
                print(f"\nFalkorDB Docker container launched successfully!\n")
                return True
            else:
                print(f"FalkorDB Docker container not yet running, waiting {timeout} seconds before retrying...")
                time.sleep(timeout)

    except Exception as e:
        return handle_local_error("Could not launch FalkorDB Docker container, encountered error: ", e)

    return True


def graphing_model_server_is_online(graph_model_access_url, graph_model_server_port):
    print("\nChecking if graphing model server is online...\n")

    try:
        response = requests.get(f"http://{graph_model_access_url}:{graph_model_server_port}/health")
        if response:
            print(f"\nKB-Generator model launched successfully!\n")
            return True
    except Exception as e:
        handle_error_no_return(f"KB-Generator model is not online, status: ", e)
        return False


def read_graph_model_config():
    try:
        read_return = read_config(['graph_model_access_url', 'graph_model_server_port', 'quantize_graph_model', 'quantize_graph_model_bits', 'graph_generator_model', 'exl2_quantize_graph_model', 'exl2_quantize_graph_model_bpw', 'graph_model_max_new_tokens', 'graph_model_max_seq_len'])
    except Exception as e:
        handle_error_no_return("Could not read graph model config, encountered error: ", e)
    
    graph_model_access_url = read_return['graph_model_access_url']
    graph_model_server_port = read_return['graph_model_server_port']
    quantize_graph_model = read_return['quantize_graph_model']
    quantize_graph_model_bits = read_return['quantize_graph_model_bits']
    graph_generator_model = read_return['graph_generator_model']
    exl2_quantize_graph_model = str(read_return['exl2_quantize_graph_model']).lower() == 'true'
    exl2_quantize_graph_model_bpw = str(read_return['exl2_quantize_graph_model_bpw'])
    graph_model_max_new_tokens = read_return['graph_model_max_new_tokens']
    graph_model_max_seq_len = read_return['graph_model_max_seq_len']

    return graph_model_access_url, graph_model_server_port, quantize_graph_model, quantize_graph_model_bits, graph_generator_model, exl2_quantize_graph_model, exl2_quantize_graph_model_bpw, graph_model_max_new_tokens, graph_model_max_seq_len


def bring_graphing_model_online():  # Launch HF-Waitress instance with kb-generator model
    print(f"\nLaunching HF-Waitress instance with kb-generator model...\n")

    graph_model_access_url, graph_model_server_port, quantize_graph_model, quantize_graph_model_bits, graph_generator_model, exl2_quantize_graph_model, exl2_quantize_graph_model_bpw, graph_model_max_new_tokens, graph_model_max_seq_len = read_graph_model_config()

    if graphing_model_server_is_online(graph_model_access_url, graph_model_server_port):
        print("\nGraphing model server is already online, skipping launch...\n")
        return True

    hf_waitress_kb_generator_server_path = os.path.normpath(os.path.join(os.getcwd(), "knowledge-graph-model-server"))    # normpath() is used to "normalize" i.e. convert to a path that is appropriate for the current OS
    print(f"\nLaunching HF-Waitress instance with kb-generator model at path: {hf_waitress_kb_generator_server_path}\n")

    # Need to format this way because the f"""<>""" multiline way will maintain newlines in the command, which will cause the command to fail!
    # Also cannot format this as we have in hf_waitress.py's exllama_bpw_quantize_model() because of the command seperators (&& and ;), which would be incorrectly treated as parameters to the cd command in that list format!
    # We cd first because the command should execute from within that directory otherwise the main hf_config.json gets incorrectly modified! 
    command = (
        f"cd {hf_waitress_kb_generator_server_path} "
        f"{'&&' if platform.system() == 'Windows' else ';'} "
        f"{'python' if platform.system() == 'Windows' else 'python3'} hf_waitress.py "
        f"--port {str(graph_model_server_port)} "
        f"--model_id {str(graph_generator_model)} "
        f"--max_new_tokens {str(graph_model_max_new_tokens)} "
    )
    if exl2_quantize_graph_model:
        command += f" --exl2 --exl2_bpw {str(exl2_quantize_graph_model_bpw)} --exl2_max_seq_len {str(graph_model_max_seq_len)}"
    else:
        command += f" --quantize {str(quantize_graph_model)} --quant_level {str(quantize_graph_model_bits)}"

    try:
        if platform.system() == 'Windows':
            windows_command = f'start cmd /k "{command}"'   # /k tells cmd to keep the window open even after the command has finished, versus /c which closes the window after the command has finished
            subprocess.Popen(windows_command, shell=True)   # Popen is used to launch the command in a new process in a new terminal, while subprocess.run() is used to simply run the command and wait for it to finish
            # Using `start` in this manner explicitly instructs Windows to start a new command window to run the command, while CREATE_NEW_CONSOLE combined with shell=True might not work correctly as the shell itself might capture the console!
        else:
            subprocess.Popen(command, shell=True)
        # The shell=True lets the system's shell interpret the command string, including special operators like && (Windows) or ; (Unix) that chain commands together.
        # This is exactly what you need when you want to change directory before running a script!

        timeout = 5
        attempts = 25
        for _ in range(attempts):
            if graphing_model_server_is_online(graph_model_access_url, graph_model_server_port):
                print(f"\nKB-Generator model launched successfully!\n")
                return True
            else:
                print(f"KB-Generator model not yet running, waiting {timeout} seconds before retrying...")
                time.sleep(timeout)
            
    except Exception as e:
        return handle_local_error("Could not launch HF-Waitress instance with kb-generator model, encountered error: ", e)
    
    return True


def is_graph_blank_or_newly_created(graph):
    try:
        result = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        return result == 0
    except Exception as e:
        handle_error_no_return("Could not check if graph is blank or newly created, encountered error: ", e)
        return False    # Default to False means the check will occur just to be safe!


def apply_leiden_clustering_to_graph(selected_knowledge_domain):
    print(f"\n\nApplying Leiden clustering to the graph for {selected_knowledge_domain}...\n\n")

    try:
        client = get_graph_db_client()
        clustering_success, status_message = apply_leiden_clustering(client, str(selected_knowledge_domain))
        if not clustering_success:
            raise Exception(status_message)
        return True
    except Exception as e:
        return handle_local_error(f"Could not apply Leiden clustering to the graph for {selected_knowledge_domain}, encountered error: ", e)


@app.route('/generate_graph_communities', methods=['POST'])
def generate_graph_communities():
    print("\n\nGenerating graph communities...\n\n")

    try:
        knowledge_domain = request.json.get('knowledge_domain')
    except Exception as e:
        return handle_api_error("Could not get selected knowledge domain from request, encountered error: ", e)

    try:
        apply_leiden_clustering_to_graph(knowledge_domain)
    except Exception as e:
        return handle_api_error("Could not generate graph communities, encountered error: ", e)

    return jsonify({"message": "Graph communities generated successfully"}), 200


def assemble_chunks_for_graph_db(chunks, user_query=None):
    '''
    Assembles document chunks into a dictionary of entities for storage to the GraphDB:
    '''
    try:
        read_return = read_config(['graph_chunk_size', 'graph_chunk_overlap'])
        graph_chunk_size = read_return['graph_chunk_size']
        graph_chunk_overlap = read_return.get('graph_chunk_overlap', 300)
        
        graphing_chunk = ""
        chunks_in_storage_queue = []    # chunks will eb combined upto `graph_chunk_size` and stored here
        source_filename = ""
        page_number_list = []
        chunk_entities = {}
        graph_chunk_count = 1
        overlap_text = ""

        if user_query is not None:  # For GraphRAG response query-pipeline, we need to add the user query as a chunk
            user_query = user_query.replace("'", "").replace("<br>", "").replace("?", "")
            user_query_chunk_text = f"Do not attempt to answer any query that follows, simply proceed to extract nodes and relationships from the following text:\n{user_query}"
            chunk_entities[0] = {
                'chunk_text': user_query_chunk_text,
                'source_chunks': [],
                'source_doc_name': 'user_query'
            }   # Graph chunk numbering starts from 1, so we can add the user query as chunk 0!

        print("\nGenerating Graphing Chunks Dictionary...\n")
        
        for count, chunk in enumerate(chunks):

            try:
                chunk_source_filepath = chunk['source']
                source_filename = os.path.basename(chunk_source_filepath)

                try:
                    page_number_list.append(int(chunk['page_number']))
                    page_number_list = list(set(page_number_list))
                except Exception as e:
                    handle_error_no_return(f"Could not obtain page number from chunk {count} of {len(chunks)}, encountered error: ", e)

                graphing_chunk += str(chunk['content'])
                chunks_in_storage_queue.append(count)

                if len(graphing_chunk) >= graph_chunk_size:
                    chunk_entities[graph_chunk_count] = {
                        'chunk_text': graphing_chunk,
                        'source_chunks': chunks_in_storage_queue,
                        'source_doc_name': source_filename,
                        'page_number': page_number_list
                    }

                    # Add overlap text to the next chunk
                    overlap_text = graphing_chunk[-graph_chunk_overlap:] if len(graphing_chunk) > graph_chunk_size else graphing_chunk  # The [-<number>:] syntax is used to get the last <number> of characters from the string

                    graph_chunk_count += 1
                    graphing_chunk = overlap_text # Start the next chunk with the overlap text
                    chunks_in_storage_queue = []    # Reset the storage queue for the next chunk
                    page_number_list = []
            except Exception as e:
                handle_error_no_return(f"Error processing chunk {count} of {len(chunks)} in assemble_chunks_for_grapd_db(), encountered error: ", e)

        if graphing_chunk: # Add final batch to chunk_entities dict
            try:
                chunk_entities[graph_chunk_count] = {
                    'chunk_text': graphing_chunk,
                    'source_chunks': chunks_in_storage_queue,
                    'source_doc_name': source_filename,
                    'page_number': page_number_list
                }   # Clean-up left to the garbage collector as we're at the end of the loop!
            except Exception as e:
                handle_error_no_return(f"Error processing final batch of chunks in assemble_chunks_for_graph_db(), encountered error: ", e)
    except Exception as e:
        return handle_local_error(f"Error assembling chunks for graph DB, encountered error: ", e)

    return chunk_entities


def determine_graph_cache_reuse(entities_and_relationships_filepath, summaries_filepath):
    try:
        read_return = read_config(['reuse_previously_extracted_graph_entities_and_relationships', 'reuse_previously_generated_graph_summaries', 'skip_summary_generation'])
        reuse_previously_extracted_graph_entities_and_relationships = str(read_return['reuse_previously_extracted_graph_entities_and_relationships']).lower() == 'true'
        reuse_previously_generated_graph_summaries = str(read_return['reuse_previously_generated_graph_summaries']).lower() == 'true'
        skip_summary_generation = str(read_return['skip_summary_generation']).lower() == 'true'
        complete_chunk_entities = None
        reuse_previous_extract = False
        loaded_entities_and_relationships = False
    except Exception as e:
        return handle_local_error("Could not determine graph cache config, encountered error: ", e)
    
    if reuse_previously_extracted_graph_entities_and_relationships and os.path.exists(entities_and_relationships_filepath):
        try:
            candidate = load_json_file(entities_and_relationships_filepath)
            if isinstance(candidate, dict): # TODO: validation of the JSON file format
                print(f"Previously extracted graph entities and relationships found, reusing from: {entities_and_relationships_filepath}")
                complete_chunk_entities = candidate
                loaded_entities_and_relationships = True
            else:
                raise ValueError("Invalid JSON file format for previously extracted graph entities and relationships")
        except Exception as e:
            complete_chunk_entities = None
            loaded_entities_and_relationships = False
            handle_error_no_return("Could not load previously extracted graph entities and relationships, encountered error: ", e)
    
    if reuse_previously_generated_graph_summaries and os.path.exists(summaries_filepath):
        try:
            candidate = load_json_file(summaries_filepath)
            if isinstance(candidate, dict): # TODO: validation of the JSON file format
                complete_chunk_entities = candidate
                if complete_chunk_entities is not None:
                    skip_summary_generation = True  # Else return the value from config.json
                    print(f"Previously generated graph summaries found, reusing from: {summaries_filepath}")
            else:
                raise ValueError("Invalid JSON file format for previously generated graph summaries")
        except Exception as e:
            if not loaded_entities_and_relationships:
                complete_chunk_entities = None
            handle_error_no_return("Could not load previously generated graph summaries, encountered error: ", e)
    
    if complete_chunk_entities is not None:
        reuse_previous_extract = True

    return reuse_previous_extract, skip_summary_generation, complete_chunk_entities # skip_summary_generation flag is used to determine if we should skip summary generation


def get_graph_cache_filepaths(input_file, docs_to_knowledge_graph_dir):
    try:
        input_filename = os.path.basename(input_file)
        input_filename_no_ext = str(os.path.splitext(input_filename)[0]) # os.path.splitext() returns a tuple containing the path's name and extension separately
        
        entities_and_relationships_filename = input_filename_no_ext + '_entities_and_relationships.json'
        entities_and_relationships_filepath = os.path.join(docs_to_knowledge_graph_dir, entities_and_relationships_filename)

        summaries_filename = input_filename_no_ext + '_summaries.json'
        summaries_filepath = os.path.join(docs_to_knowledge_graph_dir, summaries_filename)

        return entities_and_relationships_filepath, summaries_filepath
    
    except Exception as e:
        return handle_local_error("Could not set graph generator file names and paths, encountered error: ", e)


def read_graph_generator_config():
    try:
        read_return = read_config(['docs_to_knowledge_graph_dir', 'selected_knowledge_domain'])
        docs_to_knowledge_graph_dir = read_return['docs_to_knowledge_graph_dir']
        selected_knowledge_domain = read_return['selected_knowledge_domain']
    except Exception as e:
        handle_error_no_return("Could not read graph generator config, encountered error: ", e)
    
    return docs_to_knowledge_graph_dir, selected_knowledge_domain


def graph_generator(chunks, input_file):
    '''
    Assembles document chunks (via assemble_chunks_for_graph_db()) into a dictionary of entities for storage to the GraphDB:

    chunk_entities = {
        '<graph_chunk_number>': {
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>'
        }
    }

    This is then passed to the graphing model which will process each graph_chunk and append the `entities_and_relationships` key to each chunk_entities dict:
        
        '<entities_and_relationships>': '<node_relationships_dict>'
        # eg: {"nodes": [{"type": "organization","name": "Intel"},{"type": "object","name": "Intel Products"}], "relationships": [{"source": "Intel","target": "Intel Products","relationship": "business unit"}]},

    The final `chunk_entities` dict is then passed to the `store_entities_and_relationships_in_graph_db()` function which will store the entities and relationships in the GraphDB.

    Future TODO: Launch Graphing Model and Summarizer LLM as independent threads. Requires minimum two GPUs.
    '''
    
    print(f"\n\nGraph Generator Invoked. Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    try:
        bring_graph_db_online()
    except Exception as e:
        return handle_local_error("Could not bring graph DB online, encountered error: ", e)

    try:
        docs_to_knowledge_graph_dir, selected_knowledge_domain = read_graph_generator_config()
        complete_chunk_entities = None
    except Exception as e:
        return handle_local_error("Could not read graph generator config, encountered error: ", e)

    try:
        entities_and_relationships_filepath, summaries_filepath = get_graph_cache_filepaths(input_file, docs_to_knowledge_graph_dir)
    except Exception as e:
        return handle_local_error("Could not set graph generator file names and paths, encountered error: ", e)

    try:
        reuse_previous_extract, skip_summary_generation, complete_chunk_entities = determine_graph_cache_reuse(entities_and_relationships_filepath, summaries_filepath)
    except Exception as e:
        return handle_local_error("Could not determine graph cache reuse, encountered error: ", e)
    
    if not reuse_previous_extract:
        try:
            bring_graphing_model_online()
        except Exception as e:
            return handle_local_error("Could not bring graphing model online, encountered error: ", e)

        try:
            chunk_entities = assemble_chunks_for_graph_db(chunks)
        except Exception as e:
            return handle_local_error("Could not assemble chunks for graph DB, encountered error: ", e)

        try:
            complete_chunk_entities = extract_all_entities_and_relationships(chunk_entities)
            try:
                save_json_file(complete_chunk_entities, entities_and_relationships_filepath)
            except Exception as e:
                handle_error_no_return("Could not save entities and relationships to file, encountered error: ", e)
        except Exception as e:
            return handle_local_error("Failed to extract entities and relationships from chunk entities, encountered error: ", e)
    
    try:
        client = get_graph_db_client()
        graph = client.select_graph(selected_knowledge_domain)  # Will create the graph if it doesn't exist
    except Exception as e:
        return handle_local_error(f"Could not connect to / initialize graph for '{selected_knowledge_domain}' domain in graph DB, encountered error: ", e)
    
    try:
        store_entities_and_relationships_in_graph_db(complete_chunk_entities, selected_knowledge_domain, graph, skip_summary_generation, summaries_filepath)
    except Exception as e:
        return handle_local_error("Could not store entities and relationships in graph DB, encountered error: ", e)

    try:
        apply_leiden_clustering_to_graph(selected_knowledge_domain)
    except Exception as e:
        handle_error_no_return("Could not apply Leiden clustering to the graph, encountered error: ", e)

    print(f"\n\nCompleted Graph Generator at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return True


def read_embeddings_config() -> tuple[str, str]:
    print("\n\nReading embeddings config\n\n")
    try:
        read_return = read_config(['selected_embedding_model', 'chunk_size', 'chunk_overlap', 'perform_graph_rag', 'perform_only_graph_rag'])
        selected_embedding_model = read_return['selected_embedding_model']
        chunk_sz = read_return['chunk_size']
        chunk_olp = read_return['chunk_overlap']
        perform_graph_rag = str(read_return['perform_graph_rag']).lower() == 'true'
        perform_only_graph_rag = str(read_return['perform_only_graph_rag']).lower() == 'true'
    except Exception as e:
        handle_local_error("Missing values in config.json, could not read_embeddings_config. Error: ", e)

    path_to_knowledge_domain = get_path_to_knowledge_domain()

    return selected_embedding_model, path_to_knowledge_domain, chunk_sz, chunk_olp, perform_graph_rag, perform_only_graph_rag


# Document vectorization and chunking
def whoosh_embed_and_graph_doc_chunks(input_file):
    print("\n\nCore Document Vectorization and Chunking Function Invoked\n\n")

    # Read Embeddings Config
    try:
        selected_embedding_model, path_to_knowledge_domain, chunk_sz, chunk_olp, perform_graph_rag, perform_only_graph_rag = read_embeddings_config()
    except Exception as e:
        handle_local_error("Could not read embeddings config, encountered error: ", e)

    # Chunk Source Data
    print("Chunking Doc")
    try:
        chunks = chunk_docs_with_page_numbers(input_file, chunk_sz) # Generates a list of dictionaries, each containing 'content', 'source', and 'page_number' as keys
        if len(chunks) > 0:
            if not perform_only_graph_rag:
                
                try:
                    whoosh_indexer(chunks)
                except Exception as e:
                    handle_error_no_return("Could not index chunks, skipping and attempting vector embedding. Encountered error: ", e)
                
                try:
                    core_embedder(chunks, selected_embedding_model, path_to_knowledge_domain)
                except Exception as e:
                    handle_error_no_return("Could not embed chunks, skipping and attempting graph generation. Encountered error: ", e)
            
            else:
                print("Performing only graph RAG")
            
            if perform_graph_rag:
                try:
                    graph_generator(chunks, input_file)
                except Exception as e:
                    handle_error_no_return("Could not graph chunks, skipping. Encountered error: ", e)
            print("Document added to knowledge domain.")
        else:
            print("No chunks generated, skipping indexing and embedding")
    except Exception as e:
        handle_local_error("Failed to chunk document for storage to VectorDB, encountered error: ", e)
    finally:
        print("\nCompleted Document-Upload Processing.\n")
        return chunk_sz, chunk_olp


def highlight_text_on_page(highlight_list, stream_session_id):

    print("\nHighlighting Document\n")
    threshold = 80

    try:
        read_return = read_config(['upload_folder', 'highlighted_docs'])
        upload_folder = read_return['upload_folder']
        highlighted_pdfs_path = read_return['highlighted_docs']
    except Exception as e:
        handle_local_error("Missing upload_folder in config.json for method highlight_text_on_page. Error: ", e)
    
    for index, doc in enumerate(highlight_list, start=1):

        try:
            pdf_path = os.path.join(upload_folder, doc).replace("\\","/")
            output_file_extension = "_" + stream_session_id + '.pdf'
            output_file_name = doc.replace(".pdf",output_file_extension) 
            output_pdf_path = os.path.join(highlighted_pdfs_path, output_file_name).replace("\\","/")
            highlight_doc = fitz.open(pdf_path)
        except Exception as e:
            handle_error_no_return("Could not open doc for highlighting, encountered error: ", e)
            continue
        
        for target in highlight_list[doc]:
            try:
                text_to_highlight = str(target[1])
                text_to_highlight = re.sub(r'Row \d+, Column \d+: ', '', text_to_highlight)
                page_number = int(target[0])
                page = highlight_doc.load_page(page_number-1)
                page_text = page.get_text("text")

                # Split the page text into overlapping phrases
                words = page_text.split()
                phrases = [' '.join(words[i:i+len(text_to_highlight.split())]) for i in range(len(words))]

                # Find fuzzy matches
                good_matches = []
                for phrase in phrases:
                    score = fuzz.partial_ratio(text_to_highlight.lower(), phrase.lower())
                    if score >= threshold:
                        good_matches.append(phrase)

                for match in good_matches:
                    if len(str(match)) > 3:
                        text_instances = page.search_for(match)
                        for inst in text_instances:
                            try:
                                #print(f"HIGHLIGHTING inst {inst} in document {doc}")
                                page.add_highlight_annot(inst)
                            except Exception as e:
                                handle_error_no_return("Could not highlight text instance, encountered error: ", e)
                                continue

            except Exception as e:
                handle_error_no_return("Error loading page or searching for text to highlight, encountered error: ", e)
                continue
            
        try:
            highlight_doc.save(output_pdf_path, garbage=0, deflate=False, clean=False)
        except Exception as e:
            handle_error_no_return("Could not save highlighted doc, encountered error: ", e)
            continue

    return True


def highlighter_interface(reference_pages, stream_session_id):

    user_should_refer_pages_in_doc = {}
    highlight_list = {}
    docs_have_relevant_info = False

    # print(f"\n\nreference_pages: {reference_pages}\n\n")

    for file_path, content in reference_pages.items():
        source_filename = os.path.basename(file_path)
        print(f"\nsource_filename basename: {source_filename}\n")
        output_file_extension = "_" + stream_session_id + '.pdf'
        output_file_name = source_filename.replace(".pdf",output_file_extension) 
        page_numbers = set()
        highlight_strings = set()

        for item in content:
            # Each item in the list has two elements
            page_text, page_number = item
            page_numbers.add(int(page_number))
            highlight_strings.add((int(page_number), str(page_text[:50])))

        if page_numbers:
            user_should_refer_pages_in_doc[output_file_name] = page_numbers
            docs_have_relevant_info = True

        if highlight_strings:
            highlight_list[source_filename] = list(highlight_strings)

    if docs_have_relevant_info:
        try:
            highlight_text_on_page(highlight_list, str(stream_session_id))
        except Exception as e:
            handle_error_no_return("Could not highlight text, encountered error: ", e)

    return docs_have_relevant_info, user_should_refer_pages_in_doc


def determine_sequence_id_for_chat(chat_id):

    print(f"\n\nDetermining sequence ID for chat with chat_id: {chat_id}")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing keys in config.json for method determine_sequence_id_for_chat(). Error: ", e)

    # Connect to or create the DB
    try:
        conn = sqlite3.connect(sqlite_history_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to DB for chat history storage, encountered error: ", e)

    try:
        # Determine sequence_id
        cursor.execute("SELECT COALESCE(MAX(sequence_id), 0) FROM chat_history WHERE chat_id = ?", (int(chat_id),))
        # "The COALESCE function accepts two or more arguments and returns the first non-null argument."
        # This query returns 0 for a new chat, and the last sequence_id for an existing chat.
        # Note that trailing comma! Without it, the simple select query will produce an error: "parameters are of unsupported type" !!
        # This is because the SQLite3 module can have trouble recognizing single-item tuples as tuples, so a trailing comma helps alleviate this! 

        result = cursor.fetchone()
        current_sequence_id = result[0]     # 'result' will be a list, so extract the first value
        
    except Exception as e:
        handle_local_error("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)

    return int(current_sequence_id) # returning current max sequence_id, this will be incremented by 1 when a new response is stored to the db


def store_local_llm_chat_history_to_db(chat_id, sequence_id, stream_session_id, user_query, model_response_for_history_db, fully_formatted_prompt, local_llm_server, local_llm_chat_template_format):

    print(f"\n\nStoring chat history for chat with chat_id: {chat_id} and sequence_id: {sequence_id}")

    try:
        read_return = read_config(['sqlite_history_db', 'local_llm_server', 'model_choice'])
        sqlite_history_db = read_return['sqlite_history_db']
        local_llm_server = read_return['local_llm_server']
        model_choice = read_return['model_choice']
    except Exception as e:
        handle_local_error("Missing keys in config.json for method store_local_llm_chat_history_to_db. Error: ", e)

    if local_llm_server == "hf-waitress":
        model_choice = read_hf_config(['model_id'])['model_id']

    # Connect to or create the DB
    try:
        conn = sqlite3.connect(sqlite_history_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to DB for chat history storage, encountered error: ", e)

    if int(sequence_id) == 1:
        try:
            chat_id = determine_latest_chat_id(cursor)
        except Exception as e:
            handle_local_error("Could not determine latest chat ID, encountered error: ", e)
    
    try:
        prev_sequence_id = determine_sequence_id_for_chat(chat_id)
        sequence_id = prev_sequence_id + 1
    except Exception as e:
        handle_local_error("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)

    try:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime('%d %b %Y - %I:%M %p %Z')
    except Exception as e:
        return handle_api_error("Could not obtain timestamp in store_local_llm_chat_history_to_db, encountered error: ", e)

    try:
        # Store conversation history into DB
        cursor.execute("INSERT INTO chat_history (chat_id, sequence_id, stream_session_id, user_query, llm_response, llm_model, prompt_template, local_llm_server, prompt_template_format, date_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (int(chat_id), int(sequence_id), str(stream_session_id), user_query, model_response_for_history_db, model_choice, str(fully_formatted_prompt), str(local_llm_server), str(local_llm_chat_template_format), str(formatted_datetime)))
        conn.commit()
        print(f"\n\nInserted chat history into DB with chat_id: {chat_id}\n\n")
    except Exception as e:
        handle_local_error("Could not insert chat history into DB, encountered error: ", e)

    return formatted_datetime, chat_id


@app.route('/check_gdrive_auth')
def check_gdrive_auth():
    global GDRIVE_CREDS

    try:
        if not GDRIVE_CREDS:
            if os.path.exists("gdrive_token.json"):
                try:
                    GDRIVE_CREDS = Credentials.from_authorized_user_file("gdrive_token.json", GDRIVE_SCOPES)

                    # Check if we have a refresh token
                    if not hasattr(GDRIVE_CREDS, 'refresh_token') or not GDRIVE_CREDS.refresh_token:
                        print("No refresh token found in credentials")
                        return jsonify(is_authenticated=False)
                    
                    if GDRIVE_CREDS.valid:
                        return jsonify(is_authenticated=True)
                    elif GDRIVE_CREDS.expired:
                        GDRIVE_CREDS.refresh(Request())
                        # Save refreshed credentials
                        with open("gdrive_token.json", "w") as token:
                            token.write(GDRIVE_CREDS.to_json())
                        return jsonify(is_authenticated=True)
                    else:
                        return jsonify(is_authenticated=False)
                except Exception as e:
                    print(f"Error loading credentials from file: {str(e)}")
                    # If there's an error with the token file, consider deleting it
                    os.remove("gdrive_token.json")
                    return jsonify(is_authenticated=False)
            else:
                return jsonify(is_authenticated=False)
            
        # Check if credentials exist and are valid
        if GDRIVE_CREDS and GDRIVE_CREDS.valid:
            return jsonify(is_authenticated=True)
        
        # Check if credentials exist but need refresh
        if (GDRIVE_CREDS and GDRIVE_CREDS.expired and hasattr(GDRIVE_CREDS, 'refresh_token') and GDRIVE_CREDS.refresh_token):
            GDRIVE_CREDS.refresh(Request())
            # Save refreshed credentials
            with open("gdrive_token.json", "w") as token:
                token.write(GDRIVE_CREDS.to_json())
            return jsonify(is_authenticated=True)
            
        return jsonify(is_authenticated=False)
    except Exception as e:
        # Log the error but return false for authentication status
        print(f"Error checking Google Drive auth status: {str(e)}")
        return jsonify(is_authenticated=False)


@app.route('/web_login_to_google_drive')
def web_login_to_google_drive():
    # Get the redirect URL from query parameters, default to "/"
    redirect_url = request.args.get('redirect', '/')
    
    # Create the OAuth flow
    flow = Flow.from_client_secrets_file(
        "gdrive_webapp_creds.json",
        GDRIVE_SCOPES,
        redirect_uri=url_for('google_drive_callback', _external=True)
    )

    # Generate the Google OAuth URL
    auth_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )

    # Add redirect URL to the URL (but not state, as it's already included)
    auth_url += f"&redirect_url={redirect_url}"

    # Redirect the user to the Google OAuth URL
    return redirect(auth_url)


@app.route('/google_drive_callback')
def google_drive_callback():
    global GDRIVE_CREDS

    # Get the redirect URL from query parameters
    redirect_url = request.args.get('redirect_url', '/')
    state = request.args.get('state')   # Get the state from the query parameters

    # Create the OAuth flow with the state and redirect URL
    flow = Flow.from_client_secrets_file(
        "gdrive_webapp_creds.json",
        scopes=GDRIVE_SCOPES,
        state=state,
        redirect_uri=url_for('google_drive_callback', _external=True)
    )

    # Fetch the token from the authorization response
    flow.fetch_token(authorization_response=request.url)
    GDRIVE_CREDS = flow.credentials

    # Save the token to a file
    with open("gdrive_token.json", "w") as token:
        token.write(GDRIVE_CREDS.to_json())
    
    return redirect(redirect_url)


@app.route('/get_google_drive_user')
def get_google_drive_user():
    try:
        service = build('drive', 'v3', credentials=GDRIVE_CREDS)
        about_result = service.about().get(fields="user").execute()
        user_name = about_result.get('user', {}).get('emailAddress', 'Unknown User')
        return jsonify(success=True, user_name=user_name)
    except Exception as e:
        handle_error_no_return("Could not get user name from Google Drive, encountered error: ", e)
        return jsonify(success=True)


@app.route('/logout_from_google_drive')
def logout_from_google_drive():
    global GDRIVE_CREDS
    try:
        if GDRIVE_CREDS:
            # Revoke the credentials on Google's side
            revoke_url = "https://oauth2.googleapis.com/revoke"
            headers = {'content-type': 'application/x-www-form-urlencoded'}
            
            # If we have a refresh token, revoke that (it will revoke access token too)
            if hasattr(GDRIVE_CREDS, 'refresh_token') and GDRIVE_CREDS.refresh_token:
                token_to_revoke = GDRIVE_CREDS.refresh_token
            else:
                # Otherwise revoke the access token
                token_to_revoke = GDRIVE_CREDS.token

            # Make the revocation request
            response = requests.post(
                revoke_url,
                params={'token': token_to_revoke},
                headers=headers
            )

            if response.status_code != 200:
                print(f"Failed to revoke token: {response.text}")
                # Continue with local cleanup even if revocation fails

        # Remove token file if it exists
        if os.path.exists("gdrive_token.json"):
            os.remove("gdrive_token.json")
        
        # Reset credentials
        GDRIVE_CREDS = None
        
        return jsonify(success=True, message="Successfully logged out from Google Drive")
    except Exception as e:
        print(f"Error during logout: {str(e)}")
        # Still try to clean up locally even if there was an error
        try:
            if os.path.exists("gdrive_token.json"):
                os.remove("gdrive_token.json")
            GDRIVE_CREDS = None
        except:
            pass
        return handle_api_error("Failed to logout from Google Drive: ", e)


def categorize_mimetype(mimetype):
    mimetype = mimetype.lower()

    word_mimetypes = {
        # Microsoft Word (modern formats)
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-word.document.macroEnabled.12",
        "application/vnd.ms-word.template.macroEnabled.12",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",

        # Microsoft Word (legacy format)
        "application/msword",

        # Google Docs
        "application/vnd.google-apps.document",

        # OpenDocument Presentation
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.text-template",
        "application/vnd.oasis.opendocument.text-web"
    }

    excel_mimetypes = {
        # Microsoft Excel (modern formats)
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel.sheet.macroEnabled.12",
        "application/vnd.ms-excel.template.macroEnabled.12",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.template",

        # Microsoft Excel (legacy format)
        "application/vnd.ms-excel",

        # Google Sheets
        "application/vnd.google-apps.spreadsheet",

        # OpenDocument Presentation
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.spreadsheet-template",
        "text/csv",
        "text/tab-separated-values"
    }

    presentation_mimetypes = {
        # Microsoft PowerPoint (modern formats)
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
        "application/vnd.ms-powerpoint.presentation.macroEnabled.12",  # .pptm
        "application/vnd.openxmlformats-officedocument.presentationml.template",  # .potx
        "application/vnd.ms-powerpoint.template.macroEnabled.12",  # .potm
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",  # .ppsx
        "application/vnd.ms-powerpoint.slideshow.macroEnabled.12",  # .ppsm

        # Microsoft PowerPoint (legacy format)
        "application/vnd.ms-powerpoint",  # .ppt, .pot, .pps

        # Google Slides
        "application/vnd.google-apps.presentation",

        # OpenDocument Presentation
        "application/vnd.oasis.opendocument.presentation",  # .odp
        "application/vnd.oasis.opendocument.presentation-template"  # .otp
    }

    pdf_mimetypes = {
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "application/vnd.pdf",
        "text/pdf",
        "text/x-pdf"
    }

    text_mimetypes = {
        "application/rtf",
        "text/rtf",
        "text/plain"
    }

    image_mimetypes = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/webp",
        "image/svg+xml",
        "image/tiff",
        "image/x-icon",
        "image/vnd.microsoft.icon",
        "image/heic",
        "image/heif"
    }

    video_mimetypes = {
        "video/mp4",
        "video/mpeg",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-ms-wmv",
        "video/x-flv",
        "video/webm",
        "video/3gpp",
        "video/3gpp2",
        "video/x-matroska"
    }

    audio_mimetypes = {
        "audio/mpeg",
        "audio/x-wav",
        "audio/wav",
        "audio/x-m4a",
        "audio/aac",
        "audio/ogg",
        "audio/webm",
        "audio/flac",
        "audio/x-ms-wma",
        "audio/x-aiff"
    }

    folder_mimetypes = {
        "application/vnd.google-apps.folder",  # Google Drive folder
        "application/x-directory",             # Generic directory mime type
        "inode/directory",                     # Often used in Unix-like systems
        "folder",                              # Some systems might use this
    }

    # Word file
    if mimetype in word_mimetypes:
        return "word"
    # Excel files
    elif mimetype in excel_mimetypes:
        return "excel"
    # Presentation files
    elif mimetype in presentation_mimetypes:
        return "presentation"
    # Text files
    elif mimetype in text_mimetypes:
        return "text"
    # PDFs
    elif mimetype in pdf_mimetypes:
        return "pdf"
    # Images
    elif mimetype in image_mimetypes:
        return "image"
    # Videos
    elif mimetype in video_mimetypes:
        return "video"
    # Audio files
    elif mimetype in audio_mimetypes:
        return "audio"
    # Folders
    elif mimetype in folder_mimetypes:
        return "folder"
    # Other
    else:
        return "other"

@app.route('/fetch_file_list_from_google_drive')
def fetch_file_list_from_google_drive():
    gdrive_files = []
    try:
        service = build("drive", "v3", credentials=GDRIVE_CREDS)

        about_result = service.about().get(fields="storageQuota,user").execute()
        print(f"about_result: {about_result}")

        items = []
        page_token = None

        while True:
            results = (
                service.files().list(
                    q="trashed=false",  # If working with specific file types or folders, consider enhancing the q parameter (e.g., q="'<folder_id>' in parents and trashed=false")
                    pageSize=1000,
                    fields="nextPageToken, files(id, name, mimeType, version)",
                    supportsAllDrives=True,  #  Required to include Shared Drive Files: Without it, the API assumes the files are only in the user's private Drive!
                    includeItemsFromAllDrives=True,  # Search and listing operations will include items from Shared Drives in the results if this is set to True.
                    pageToken=page_token    # Handle pagination
                ).execute()
            )
            
            files = results.get("files", [])
            items.extend(files)
            page_token = results.get("nextPageToken", None) # If users report missing files, log the nextPageToken and current page count for debugging
            if not page_token:
                break
        
        print(f"All files retrieved, total count: {len(items)}")

        if not items:
            print("No files found.")
            return jsonify(success=True, gdrive_files=gdrive_files)
        
        # Categorize files and format the output        
        for item in items:
            category = categorize_mimetype(item['mimeType'])
            gdrive_files.append({
                'name': item['name'],
                'id': item['id'],
                'mimeType': item['mimeType'],
                'version': item['version'],
                'type': category
            })

    except Exception as e:
        return handle_api_error("Could not fetch GDrive files, encountered error: ", e)
    
    return jsonify({'success': True, 'gdrive_files': gdrive_files})


def download_folder(service, folder_id, path, data_queue=None, folder_name=None, indent=''):

    print(f"\n\nDownloading GoogleDrive Folder with id: {folder_id}\n\n")

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        data_queue.put(f"Could not create local directory to save GDrive folder: '{folder_name}' | failure")
        return handle_local_error("Server-side error - could not create nested directory in the download_folder() method: ", e)

    query = f"'{folder_id}' in parents"
    fields = "files(id, name, mimeType)"
    
    gdrive_folder_contents = service.files().list(q=query, fields=fields).execute()
    items = gdrive_folder_contents.get('files', [])

    try:
        for item in items:
            file_id = item['id']
            filename = item['name']
            mime_type = item['mimeType']
            print(f"folder item mime_type f{mime_type}")

            if "folder" in str(mime_type):
                sub_folder_path = os.path.join(path, filename)    # in this case, filename will be the folder name

                try:
                    data_queue.put(f"Downloading nested folder: '{filename}'")
                    download_folder(service, file_id, sub_folder_path, data_queue)  # in this case, file_or_folder_id will be the folder id
                    data_queue.put(f"Folder '{filename}' downloaded successfully")
                except Exception as e:
                    data_queue.put(f"Could not download nested folder: '{filename}' | failure")
                    handle_error_no_return("Could not download_folder in the download_folder() method, encountered error: ", e)
            else:
                try:
                    data_queue.put(f"Downloading file: '{filename}'")
                    filename_with_extension, file_content = download_gdrive_file(service, file_id, filename, mime_type)
                    if filename_with_extension is None or file_content is None:
                        raise Exception(f"Server-side error - Could not download {filename} from Google Drive folder {folder_name} in the download_folder() method")
                    data_queue.put(f"File '{filename_with_extension}' downloaded successfully")
                except Exception as e:
                    data_queue.put(f"Could not download file: '{filename}' | failure")
                    handle_error_no_return("Server-side error - could not get_file_content in the download_folder() method: ", e)

                try:
                    # filepath = os.path.join(path, secure_filename(filename_with_extension))
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename_with_extension))
                    print(f"Saving {filename_with_extension} to {filepath}")
                    with open(filepath, 'wb') as f:
                        f.write(file_content)

                    data_queue.put(f"Vector Embedding & Indexing Document: '{filename_with_extension}'")
                    document_extractor_and_loader(filename_with_extension, filepath)
                    data_queue.put(f"Document '{filename_with_extension}' processed & uploaded successfully! | success")

                except Exception as e:
                    data_queue.put(f"Server-side error - could not save file: '{filename}' downloaded from Google Drive | failure")
                    handle_error_no_return("Server-side error - could not save Google Drive file: '{filename}' in the download_folder() method: ", e)
    except Exception as e:
        data_queue.put(f"Error downloading Google Drive folder: '{folder_name}' | failure")
        return handle_local_error("Error downloading Google Drive folder: '{folder_name}' in the download_folder() method, encountered error: ", e)

    return True


def download_gdrive_file(service, file_id, filename, mime_type):

    print(f"\n\nDownloading GoogleDrive File with mime_type: {mime_type}\n\n")

    file_mime_category = categorize_mimetype(mime_type)

    filename_with_extension = filename

    try:
        if file_mime_category in ["word", "excel", "presentation"]:
            # Handle Google Docs Editors files
            if 'google-apps' in mime_type:
                if file_mime_category == "word":
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    file_extension = '.docx'
                elif file_mime_category == "excel":
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    file_extension = '.xlsx'
                elif file_mime_category == "presentation":
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                    file_extension = '.pptx'
                
                print(f"Downloading google-apps file with mimeType {file_mime_category}")
                gdrive_request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
                if not filename.endswith(file_extension):
                    filename_with_extension += file_extension
            
            else:
                # For non-Google formats, use get_media
                print(f"Downloading office file with non google-apps mimeType {file_mime_category}")
                gdrive_request = service.files().get_media(fileId=file_id)
        else:
            print(f"Downloading non-office file with mimeType {file_mime_category}")
            gdrive_request = service.files().get_media(fileId=file_id)
        
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, gdrive_request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
    
    except Exception as e:
        handle_error_no_return("Error downloading file from Google Drive in the download_gdrive_file() method: ", e)
        return None, None
    
    try:
        file_content = file.getvalue()
    except Exception as e:
        handle_error_no_return("Server-side error - could not getValue() for downloaded Google Drive file  in the download_gdrive_file() method: ", e)
        return None, None

    return filename_with_extension, file_content


def gdrive_downloader(service, file_or_folder_id, filename, mime_type, data_queue=None, path=app.config['UPLOAD_FOLDER']):
    file_mime_category = categorize_mimetype(mime_type)

    try:
        if file_mime_category == "folder":
            download_path = os.path.join(path, secure_filename(filename))    # in this case, filename will be the folder name
            data_queue.put(f"Downloading folder: '{filename}'")
            download_folder(service, file_or_folder_id, download_path, data_queue, filename)
            data_queue.put(f"Folder '{filename}' processed | success")
            return filename, None    # Return None for file_content as it's a folder
        else:
            data_queue.put(f"Downloading file: '{filename}'")
            filename_with_extension, file_content = download_gdrive_file(service, file_or_folder_id, filename, mime_type)
            if file_content is None or filename_with_extension is None:
                raise Exception("Server-side error - Could not download individual file from Google Drive in the gdrive_downloader() method")
            data_queue.put(f"File '{filename_with_extension}' downloaded successfully")
            return filename_with_extension, file_content
    except Exception as e:
        return handle_local_error("Error downloading file from Google Drive in the gdrive_downloader() method, encountered error: ", e)


@app.route('/google_drive_loader', methods=['POST'])
def google_drive_loader():

    try:
        gdrive_file_id = str(request.form['file_id'])
        gdrive_file_mimeType = str(request.form['file_mimeType'])
    except Exception as e:
        return handle_api_error("Server-side error reading Google Drive file details for download in the google_drive_loader() method: ", e)
    
    try:
        service = build("drive", "v3", credentials=GDRIVE_CREDS)
    except Exception as e:
        return handle_api_error("Could not create Google service handler in the google_drive_loader() method, check credentials and re-try: ", e)

    data_queue = queue.Queue()
    stop_event = threading.Event()

    def sync_task():

        try:

            try:
                file_metadata = service.files().get(fileId=gdrive_file_id, fields='name, mimeType', supportsAllDrives=True).execute()   # includeItemsFromAllDrives=True not need here because it's a search and listing operation!
                original_filename = file_metadata.get('name', 'untitled')
                mime_type = file_metadata.get('mimeType', gdrive_file_mimeType)
            except Exception as e:
                data_queue.put(f"Error fetching metadata for '{original_filename}' | failure")
                return handle_api_error("Could not read GoogleDrive file metadata for file: '{original_filename}' in the google_drive_loader() method, encountered error: ", e)
            
            data_queue.put(f"Fetched metadata for '{original_filename}', proceeding to download...")
            
            try:
                filename_with_extension, file_content = gdrive_downloader(service, gdrive_file_id, original_filename, mime_type, data_queue)
            except Exception as e:
                data_queue.put(f"Error downloading {original_filename} from Google Drive | failure")
                return handle_api_error("Server-side error - could not download file: '{original_filename}' from Google Drive in the google_drive_loader() method: ", e)

            if file_content is not None:    # gdrive_downloader downloaded & returned a single file
                try:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename_with_extension))

                    print(f"Saving {filename_with_extension} to {filepath}")
                    with open(filepath, 'wb') as f:
                        f.write(file_content)

                    data_queue.put(f"Vector Embedding & Indexing Document: '{filename_with_extension}'")
                    document_extractor_and_loader(filename_with_extension, filepath)
                    data_queue.put(f"Document '{filename_with_extension}' processed & uploaded successfully! | success")

                except Exception as e:
                    data_queue.put(f"Server-side error - could not process file: '{original_filename}' from Google Drive | failure")
                    return handle_api_error("Server-side error - could not process file: '{original_filename}' from Google Drive in the google_drive_loader() method: ", e)
        
        finally:
            data_queue.put(None)
            print("\n\nGDrive-Sync stream done, breaking thread\n\n")
    
    def load_gdrive():

        try:
            thread = threading.Thread(target=sync_task)
            thread.start()
        except Exception as e:
            data_queue.put(f"Could not start sync process for Google Drive | failure")
            return handle_api_error("Could not start thread in the google_drive_loader() method, encountered error: ", e)

        while True:
            if stop_event.is_set(): #TODO: Add Cancel-Sync button to UI! Logic here will be simialr to STOP_GENERATION in hf_waitress.py
                print("\n\nStopping GDrive-Sync stream as requested by stop_event\n\n")
                thread.join()
                break
            output = data_queue.get()
            if output is None:
                print("\n\nNone read, breaking and stopping thread\n\n")
                thread.join()
                break
            yield f"data: {json.dumps(output)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"
    
    print("\n\nGoogle Drive Sync Begins!\n\n")
    return Response(load_gdrive(), content_type='text/event-stream')


# Route for loading all models from model dir
@app.route('/load_local_models')
def load_local_models():

    try:
        read_return = read_config(['model_dir'])
        model_dir = read_return['model_dir']
    except Exception as e:
        return handle_api_error("Missing model_dir in config.json for method load_local_models. Error: ", e)
    
    try:
        models = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    except Exception as e:
        return handle_api_error("Could not load list of local models, encountered error: ", e)
        
    #print(f"locally available models: {models}")
    return jsonify({'success': True, 'models': models})


@app.route('/upload_new_llm', methods=['POST'])
def upload_new_llm():
    print("\n\nUploading new LLM\n\n")

    try:
        read_return = read_config(['model_dir'])
        model_dir = read_return['model_dir']
    except Exception as e:
        return handle_api_error("Could not determine model_dir in upload_new_llm. Error: ", e)

    try:
        input_file = request.files['file']
    except Exception as e:
        return handle_api_error("Server-side error recieving LLM file: ", e)

    # Ensure the filename is secure
    filename = secure_filename(input_file.filename)

    try:
        filepath = os.path.join(model_dir, filename)

        print("Loading new LLM - filename: ", filename)
        print("Loading new LLM - filepath: ", filepath)

        # Save the uploaded file to the specified path
        input_file.save(filepath)
    except Exception as e:
        return handle_api_error("Failed to save LLM to model_dir, encountered error: ", e)

    return jsonify(success=True)


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


def check_if_converted_file_exists(filepath):
    try:
        source_filename = os.path.basename(filepath)
        check_file_name = os.path.splitext(source_filename)[0] + ".pdf"   # os.path.splitext() returns a tuple containing the path's name and extension separately
        check_file_path = os.path.join(app.config['UPLOAD_FOLDER'], check_file_name).replace("\\","/")
        return os.path.exists(check_file_path), check_file_path
    except Exception as e:
        handle_error_no_return("Could not determine if converted file already exists, proceeding to convert file regardless. Encountered error: ", e)
        return False, None

def document_extractor_and_loader(filename, filepath):
    print("Vector Embedding Document")

    if not filename.lower().endswith('.pdf'):
        converted_file_exists, converted_pdf_file_path = check_if_converted_file_exists(filepath)
        if not converted_file_exists:
            _, filepath = convert_non_pdf_to_pdf_with_unoconv(filename, filepath)
        else:
            filepath = converted_pdf_file_path

    use_ocr = False
    try:
        read_return = read_config(['use_ocr', 'ocr_service_choice', 'selected_embedding_model', 'selected_knowledge_domain'])
        use_ocr = read_return['use_ocr']
        ocr_service_choice = read_return['ocr_service_choice']
        selected_embedding_model = read_return['selected_embedding_model']
        selected_knowledge_domain = read_return['selected_knowledge_domain']
    except Exception as e:
        return handle_local_error("Could not determine use_ocr in config.json for process_new_file. Disabling OCR and proceeding. Error: ", e)
    
    print("Processing PDF file")
    
    if use_ocr:
        try:
            if ocr_service_choice == 'AzureVision':
                input_file = PDFtoAzureOCRTXT(filepath)
            elif ocr_service_choice == 'AzureDocAi':
                input_file = PDFtoAzureDocAiTXT(filepath)
            elif ocr_service_choice == 'LocalVisionLLM':
                input_file = PDFtoVisionLLMOCRTXT(filepath)
            elif ocr_service_choice == 'Kosmos':
                input_file = PDFtoKosmosOCRTXT(filepath)
        except Exception as e:
            handle_error_no_return("Failed to OCR text from PDF. Will now attempt to extract text via PyPDF2. Encountered error: ", e)
            try:
                input_file = PDFtoTXT(filepath)
            except Exception as e:
                return handle_local_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)
    else:
        try:
            input_file = PDFtoTXT(filepath)
        except Exception as e:
            return handle_local_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)
    
    try:
        if os.path.getsize(input_file) > 0:
            chunk_size, chunk_overlap = whoosh_embed_and_graph_doc_chunks(input_file)
        else:
            print("Extracted document is empty! Skipping vector embedding & whoosh indexing.")
    except Exception as e:
        return handle_local_error("Failed to embed & index document: ", e)

    try:
        if os.path.getsize(input_file) > 0:
            record_doc_loaded_to_db(filename, selected_embedding_model, chunk_size, chunk_overlap, selected_knowledge_domain)
        else:
            print("Extracted document is empty! Not saving to records DB.")
    except Exception as e:
        handle_error_no_return("Unable to record document loading to records DB, encountered error: ", e)

    return True


# Route to handle the submission of the second form (file loading)
@app.route('/process_new_file', methods=['POST'])
def process_new_file():

    try:
        input_file = request.files['file']
    except Exception as e:
        return handle_api_error("Server-side error recieving file: ", e)

    # Ensure the filename is secure
    filename = secure_filename(input_file.filename)
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
        document_extractor_and_loader(filename, filepath)
    except Exception as e:
        return handle_api_error("Could not document_extractor_and_loader() in the process_new_file() method, encountered error: ", e)

    return jsonify(success=True)


# Route to store user rating: 
# ATTN: comment out print() statements, as users may elect to leave a rating as a response is being generated, which is when the stdout is redirected to the event stream! 
@app.route('/store_user_rating', methods=['POST'])
def store_user_rating():
    
    # print("Stroing user rating")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json for method store_user_rating. Error: ", e)
    
    try:
        user_rating = request.form['rating']
        chat_id_for_rating = request.form['chat_id']
        sequence_id_for_rating = request.form['sequence_id']
    except Exception as e:
        return handle_api_error("Server-side error, could not read user rating or failed to obtain chat/sequence ID, encountered error: ", e)

    # print("user_rating: ", user_rating)
    # print("chat_id_for_rating: ", chat_id_for_rating)
    # print("sequence_id_for_rating: ", sequence_id_for_rating)

    try:
        conn = sqlite3.connect(sqlite_history_db)
        cursor = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to chat history DB for storage of user-rating, encountered error: ", e)

    try:
        cursor.execute(
            '''
            UPDATE chat_history
            SET user_rating = ?
            WHERE chat_id = ? AND sequence_id = ?
            ''',
            (user_rating, chat_id_for_rating, sequence_id_for_rating)
        )
        conn.commit()
    except Exception as e:
        return handle_api_error("Could not store user-rating to chat history db, encountered error: ", e)

    conn.close()

    return jsonify(success=True)


def get_url_for_server(server_to_check):
    if server_to_check == 'llama-cpp':
        try:
            read_return = read_config(['llama_cpp_access_url', 'llama_cpp_server_port'])
            return f'http://{read_return["llama_cpp_access_url"]}:{read_return["llama_cpp_server_port"]}'
        except Exception as e:
            handle_error_no_return("Could not read llama_cpp_access_url and llama_cpp_server_port from config.json in method is_local_server_online(), using default localhost:8080 instead. Encountered error: ", e)
            return 'http://localhost:8080'
    elif server_to_check == 'hf-waitress':
        try:
            read_return = read_config(['hf_waitress_access_url', 'hf_waitress_server_port'])
            return f'http://{read_return["hf_waitress_access_url"]}:{read_return["hf_waitress_server_port"]}'
        except Exception as e:
            handle_error_no_return("Could not read hf_waitress_access_url and hf_waitress_server_port from config.json in method is_local_server_online(), using default localhost:9069 instead. Encountered error: ", e)
            return 'http://localhost:9069'
    else:
        return handle_api_error("Invalid server_to_check in method get_url_for_server, encountered error: ", e)


def is_local_server_online(server_to_check):
    server_health_url = get_url_for_server(server_to_check) + '/health'
    print(f"\n\nChecking {server_to_check} server status - URL: {server_health_url}\n\n")

    try:
        response = requests.get(server_health_url)
        
        if response.status_code == 200:
            data = response.json()  # parse the JSON response to determine the server status
            if data['status'] == 'ok' and server_to_check == 'llama-cpp':
                print(f"llama.cpp Server ready and online.")
                return {"server_available":True, "loading_model":False, "status_code":200}
            elif data['status'] == 'ok' and server_to_check == 'hf-waitress':
                print(f"hf-waitress Server ready and online")
                return {"server_available":True, "loading_model":False, "status_code":200}
            elif data['status'] == 'no slot available':
                print("No slots available. Server is running but cannot handle more requests.")
                return {"server_available":False, "loading_model":False, "status_code":200}
            
        elif response.status_code == 503:   # model still loading or no slots
            data = response.json()
            if data['status'] == 'loading model':
                print("Server is loading the selected LLM, please wait")
                return {"server_available":False, "loading_model":True, "status_code":503}
            else:
                print("No slots available. Server is running but cannot handle more requests.")
                return {"server_available":False, "loading_model":False, "status_code":503}
        
        elif response.status_code == 500:
            print("Server error: Failed to load LLM.")
            logger.error("Local LLM Server - 500 event")
            return {"server_available":False, "loading_model":False, "status_code":500}
        
        else:
            return {"server_available":False, "loading_model":False, "status_code":500}
    
    except requests.exceptions.ConnectionError as e:
        error_message = "\n\nECONNREFUSED event\n\n"
        if logger:
            logger.error(error_message)
            print(error_message)
        else:
            print(error_message)
        return {"server_available":False, "loading_model":True, "status_code":500}
    except Exception as e:
        error_message = f"\n\nCould not check local LLM Server health, encountered error: {e}\n\n"
        if logger:
            logger.error(error_message)
            print(error_message)
        else:
            print(error_message)
        return {"server_available":False, "loading_model":False, "status_code":500}
    

def send_ctrl_c_to_process(process):
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
                print("\n\nProcess still running after force kill attempt.\n\n")


def terminate_local_llm_server_process(process):
    try:
        # process.terminate() sends 'SIGTERM' on Unix-like systems / 'TerminateProcess' on Windows, allows for graceful termination
        # process.wait()
        send_ctrl_c_to_process(process)
        if process.poll() is not None:  # process has indeed terminated
            print("\n\nProcess terminated gracefully.\n\n")
    except Exception as e:
        handle_local_error("Failed to terminate local LLM server process, encountered error: ", e)


@app.route('/llama_cpp_server_starter')
def llama_cpp_server_starter():
    print("\n\nStarting llama.cpp Server\n\n")

    global LLM_CHANGE_RELOAD_TRIGGER_SET
    global LLAMA_CPP_PROCESS
    global HF_WAITRESS_PROCESS
    global LLM_LOADED_UP

    other_server_running = False

    # Before attempting to start the llama.cpp server, check if HF-Waitress is running and if so, shut it down:
    try:
        if HF_WAITRESS_PROCESS is not None and is_local_server_online('hf-waitress')['server_available']:
            print("\n\nThe HF-Waitress server is running. Attempting to shut it down before starting the llama.cpp server.\n\n")
            try:
                terminate_local_llm_server_process(HF_WAITRESS_PROCESS)
                HF_WAITRESS_PROCESS = None
                LLM_LOADED_UP = False
            except Exception as e:
                LLM_LOADED_UP = True    # We know the HF-Waitress server is running, which means `hf_waitress.py` is available, so we set LLM_LOADED_UP to True
                other_server_running = True # Set to True as we've determined the other server is running and we failed to terminate it
                handle_error_no_return("Warning: Failed to terminate running HF-Waitress process before launching llama.cpp. It was likely launched by a previous session or external process. Consider manually shutting down this server to conserve memory. Technical error-details follow: ", e)
    except Exception as e:
        handle_error_no_return("Warning: Could not check if HF-Waitress server is running. Proceeding to launch llama.cpp server. Encountered error: ", e)   

    is_llama_cpp_running = False
    try:
        is_llama_cpp_running = is_local_server_online('llama-cpp')['server_available']
    except Exception as e:
        handle_error_no_return("Warning: Could not check if llama.cpp server is running. Proceeding to launch llama.cpp server. Encountered error: ", e)

    model_choice = 'undefined'
    try:
        read_return = read_config(['model_choice'])
        model_choice = read_return['model_choice']
    except Exception as e:
        handle_error_no_return("Missing model_choice in config.json in method llama_cpp_server_starter. Printing error and proceeding with model_choice: 'undefined' ", e)

    if is_llama_cpp_running and not LLM_CHANGE_RELOAD_TRIGGER_SET:
        LLM_LOADED_UP = True
        print(f'\n\nThe llama.cpp server is already loaded and the reload trigger is not set. Simply returning with model choice: {model_choice}\n\n')
        return jsonify({'success': True, 'llm_model': model_choice, 'other_server_running': other_server_running})
    
    elif is_llama_cpp_running and LLM_CHANGE_RELOAD_TRIGGER_SET:
        print("\n\nllama.cpp server online and LLM_CHANGE_RELOAD_TRIGGER_SET is set. Attempting to terminate and reload from config.json\n\n")
        try:
            terminate_local_llm_server_process(LLAMA_CPP_PROCESS)
            LLAMA_CPP_PROCESS = None
            LLM_CHANGE_RELOAD_TRIGGER_SET = False
        except Exception as e:
            LLM_LOADED_UP = True  # We know the llama.cpp server is running but there was an error terminating it, so we set LLM_LOADED_UP to True while leaving LLM_CHANGE_RELOAD_TRIGGER_SET to True as we know the server needs to be re-loaded.
            handle_error_no_return("Failed to terminate running llama.cpp process, server was likely launched by a previous session. Returning with the currently loaded LLM. To change, shutdown the previously launched server manually and reload this page. Technical error-details follow: ", e)
            return jsonify({'success': True, 'llm_model': 'undefined', 'other_server_running': other_server_running})   # We still return success:True as we've at least determined llama.cpp is running and loaded with a model, even if we cannot reload it.
                 
    elif LLM_CHANGE_RELOAD_TRIGGER_SET: # llama.cpp is not running, set flags to false and new settings will be loaded on next llama.cpp launch
        print("\n\nResetting the LLM_CHANGE_RELOAD_TRIGGER_SET flag and attemping to launch the server with the currently selected LLM.\n\n")
        LLM_CHANGE_RELOAD_TRIGGER_SET = False
        LLM_LOADED_UP = False


    try:
        read_return = read_config(['model_dir', 'model_choice', 'local_llm_context_length', 'local_llm_max_new_tokens', 'local_llm_gpu_layers', 'server_timeout_seconds', 'server_retry_attempts', 'use_gpu'])
        model_dir = read_return['model_dir']
        model_choice = read_return['model_choice']
        local_llm_context_length = read_return['local_llm_context_length']
        local_llm_max_new_tokens = read_return['local_llm_max_new_tokens']
        local_llm_gpu_layers = read_return['local_llm_gpu_layers']
        server_timeout_seconds = read_return['server_timeout_seconds']
        server_retry_attempts = read_return['server_retry_attempts']
        use_gpu = read_return['use_gpu']
    except Exception as e:
        return handle_api_error("Missing values in config.json when preparing to launch llama.cpp server, encountered error: ", e)


    try:
        cpp_model = os.path.join(model_dir, model_choice)
    except Exception as e:
        return handle_api_error("Could not os.join path to model file to launch llama.cpp server, encountered error: ", e)

    if not use_gpu:
        local_llm_gpu_layers = 0

    try:
        cpp_app = ['llama-server', '-m', cpp_model, '-ngl', str(local_llm_gpu_layers), '-c', str(local_llm_context_length), '-n', str(local_llm_max_new_tokens), '--host', '0.0.0.0']

        if platform.system() == 'Windows':
            LLAMA_CPP_PROCESS = subprocess.Popen(cpp_app, creationflags=subprocess.CREATE_NEW_CONSOLE)  # Windows only! Comment when containerizing or deploying to Linux/MacOS!
        else:           
            # Platform & container agnostic:
            with open('llama_cpp_server_output_log.txt', 'w') as f:
                LLAMA_CPP_PROCESS = subprocess.Popen(cpp_app, stdout=f, stderr=subprocess.STDOUT, text=True)    #stdout has already been redirected to the file, so simply direct stderr to stdout!

    except Exception as e:
        return handle_api_error("Could not launch llama.cpp process, encountered error: ", e)


    timeout = server_timeout_seconds   
    attempts = server_retry_attempts

    try:
        for _ in range(attempts):
            if is_local_server_online('llama-cpp')['server_available']:
                print("\n\nllama.cpp server launched succesfully! Returning.\n\n")
                LLM_LOADED_UP = True
                return jsonify({'success': True, 'llm_model': model_choice, 'other_server_running': other_server_running})
            time.sleep(timeout)
    except Exception as e:
        handle_error_no_return("Could not check server status after launch attempt, printing error and retrying: ", e)

    return handle_api_error("Failed to start llama.cpp local-server")


def get_hf_waitress_serving_host_and_port():
    try:
        read_return = read_config(['hf_waitress_serving_url', 'hf_waitress_server_port'])
        return read_return['hf_waitress_serving_url'], read_return['hf_waitress_server_port']
    except Exception as e:
        handle_error_no_return("Could not read hf_waitress_serving_url and hf_waitress_server_port from config.json in method get_hf_waitress_serving_host_and_port(), using default localhost:9069 instead. Encountered error: ", e)
        return '0.0.0.0', 9069


def hf_waitress_server_starter(hard_reboot_required = False):
    print("\n\nStarting HF-Waitress Server\n\n")

    global LLM_CHANGE_RELOAD_TRIGGER_SET    # This is only set by LARS basis llama.cpp, so we simply handle it here!
    global LLM_LOADED_UP
    global HF_WAITRESS_PROCESS
    global LLAMA_CPP_PROCESS

    other_server_running = False
    if hard_reboot_required:
        print("\nHard-Reboot of HF-Waitress server requested.\n")
        if is_local_server_online('hf-waitress')['server_available']:
            print("\nHF-Waitress server is running, terminating it before hard-reboot.\n")
            try:
                terminate_local_llm_server_process(HF_WAITRESS_PROCESS)
            except Exception as e:
                other_server_running = True
                handle_error_no_return("Could not terminate running HF-Waitress process before hard-reboot. It was likely launched by a previous session or external process. Consider manually shutting down this server to conserve memory. Technical error-details follow: ", e)
            finally:
                HF_WAITRESS_PROCESS = None
                LLM_LOADED_UP = False
        else:
            print("\nHF-Waitress server is not running, proceeding to hard-reboot.\n")

    # Before attempting to start the HF-Waitress server, check if llama.cpp is running and if so, shut it down:
    try:
        if LLAMA_CPP_PROCESS is not None and is_local_server_online('llama-cpp')['server_available']:
            print("\n\nThe llama.cpp server is running. Attempting to shut it down before starting the HF-Waitress server.\n\n")
            try:
                terminate_local_llm_server_process(LLAMA_CPP_PROCESS)
                LLAMA_CPP_PROCESS = None
                LLM_LOADED_UP = False
            except Exception as e:
                LLM_LOADED_UP = True    # We know the llama.cpp server is running, which means `llama-server` is available, so we set LLM_LOADED_UP to True
                other_server_running = True # Set to True as we've determined the other server is running and we failed to terminate it
                handle_error_no_return("Warning: Failed to terminate running llama.cpp process before launching HF-Waitress. It was likely launched by a previous session or external process. Consider manually shutting down this server to conserve memory. Technical error-details follow: ", e)
    except Exception as e:
        handle_error_no_return("Could not check if llama.cpp server is running. Proceeding to launch HF-Waitress server. Encountered error: ", e)

    model_choice = 'microsoft/Phi-3-mini-4k-instruct'   # match default in hf_waitress.py as this will only be used in the very first run, as the hf_config.json file is created in the first run!
    try:
        hf_read_return = read_hf_config(['model_id', 'awq', 'use_flash_attention_2', 'flux_diffusers', 'flux_low_vram_optimizations', 'load_quantized_flux', 'vision', 'exl2'])
        model_choice = hf_read_return['model_id']
        is_awq = str(hf_read_return['awq']).lower() == 'true'
        use_flash_attention_2 = str(hf_read_return['use_flash_attention_2']).lower() == 'true'
        flux_diffusers = str(hf_read_return['flux_diffusers']).lower() == 'true'
        flux_low_vram_optimizations = str(hf_read_return['flux_low_vram_optimizations']).lower() == 'true'
        load_quantized_flux = str(hf_read_return['load_quantized_flux']).lower() == 'true'
        vision = str(hf_read_return['vision']).lower() == 'true'
        exl2 = str(hf_read_return['exl2']).lower() == 'true'
    except Exception as e:
        return handle_api_error("Could not read hf_config.json in method hf_waitress_server_starter, encountered error: ", e)

    if is_local_server_online('hf-waitress')['server_available']:
        print("\n\nHF-Waitress server already running. Resetting LLM_CHANGE_RELOAD_TRIGGER_SET and simply returning!\n\n")
        LLM_LOADED_UP = True
        LLM_CHANGE_RELOAD_TRIGGER_SET = False   # The only instance where we're in this method and LLM_CHANGE_RELOAD_TRIGGER_SET is set while the HF-Waitress server is running is when we're trying to switch back to it after running llama.cpp. So we simply reset the flag and return.
        return jsonify({'success': True, 'llm_model': model_choice, 'other_server_running': other_server_running})
    elif LLM_CHANGE_RELOAD_TRIGGER_SET:  # Switching to HF-Waitress and it's offline, so we set LLM_CHANGE_RELOAD_TRIGGER_SET to False and proceed to launch the server.
        print('\n\nProceeding to reload the LLM & resetting the LLM_CHANGE_RELOAD_TRIGGER_SET flag.\n\n')
        LLM_CHANGE_RELOAD_TRIGGER_SET = False
        LLM_LOADED_UP = False
    
    print("\n\nProceeding to launch HF-Waitress server\n\n")
    
    hf_waitress_host, hf_waitress_port = get_hf_waitress_serving_host_and_port()
    launch_args = f'--host={hf_waitress_host} --port={hf_waitress_port} '
    if is_awq:
        launch_args += '--awq '
    if use_flash_attention_2:
        launch_args += '--use_flash_attention_2 '
    if flux_diffusers:
        launch_args += '--flux_diffusers '
    if flux_low_vram_optimizations:
        launch_args += '--flux_low_vram_optimizations '
    if load_quantized_flux:
        launch_args += '--load_quantized_flux '
    if vision:
        launch_args += '--vision '
    if exl2:
        launch_args += '--exl2 '
    launch_args = launch_args.strip()
    base_command = 'python' if platform.system() == 'Windows' else 'python3'
    full_command = f"{base_command} hf_waitress.py {launch_args}"

    try:
        if platform.system() == 'Windows':
            HF_WAITRESS_PROCESS = subprocess.Popen(full_command, creationflags=subprocess.CREATE_NEW_CONSOLE)   #Popen is non-blocking, so the server will keep running in the background
        else:
            # Platform & container agnostic - On Linux/Unix, you need to explicitly provide the arguments as a list to avoid shell interpretation issues:
            command_list = [base_command]  # 'python3'
            # Add script name
            command_list.append('hf_waitress.py')
            # Add any additional arguments
            if launch_args.strip():  # Only add if there are actual arguments
                command_list.extend(launch_args.split())
            
            with open('hf_waitress_output_log.txt', 'w') as f:
                # HF_WAITRESS_PROCESS = subprocess.Popen(command_list, stdout=f, stderr=subprocess.STDOUT, text=True)
                HF_WAITRESS_PROCESS = subprocess.Popen(command_list)

    except Exception as e:
        return handle_api_error(f"Could not launch HF-Waitress process in directory: {os.getcwd()}, encountered error: ", e)

    timeout = 5   # seconds
    attempts = 120  # 120 * 5 = 600 seconds = 10 minutes

    try:
        for _ in range(attempts):
            if is_local_server_online('hf-waitress')['server_available']:
                print("\n\nHF-Waitress server launched succesfully! Returning.\n\n")
                LLM_LOADED_UP = True
                return jsonify({'success': True, 'llm_model': model_choice, 'other_server_running': other_server_running})
            time.sleep(timeout)
    except Exception as e:
        handle_error_no_return("Could not check server status after launch attempt, printing error and retrying: ", e)

    return handle_api_error("Failed to start HF-Waitress Server. It may be taking a while to download the model, try refreshing LARS in a few minutes.")


@app.route('/hf_waitress_server_starter_endpoint', methods=['POST'])
def hf_waitress_server_starter_endpoint():
    print("\n\nHF-Waitress server starter endpoint called\n\n")
    try:
        data = request.get_json()
        hard_reboot_required = data.get('hard_reboot_required', 'false')
        print(f"\n\nhard_reboot_required: {hard_reboot_required}\n\n")
    except Exception as e:
        return handle_api_error("Could not read hard_reboot_required from the POST request in method hf_waitress_server_starter_endpoint, encountered error: ", e)

    try:
        return hf_waitress_server_starter(hard_reboot_required)
    except Exception as e:
        return handle_api_error("Could not start HF-Waitress server in method hf_waitress_server_starter_endpoint, encountered error: ", e)


@app.route('/check_local_llm_server_status', methods=['POST'])
def check_local_llm_server_status():
    
    server_online = False
    
    try:
        server_to_check = request.form['server_to_check']
    except Exception as e:
        return handle_api_error("Server-side error, could not read server_to_check from the POST request in method check_local_llm_server_status, encountered error: ", e)
    
    try:
        server_online = is_local_server_online(server_to_check)['server_available']
    except Exception as e:
        return handle_api_error(f"Error checking {server_to_check} server status in method check_local_llm_server_status, encountered error: ", e)

    return jsonify({'success': True, 'server_online': server_online})


@app.route('/local_llm_server_starter')
def local_llm_server_starter():
    print("\n\nStarting Local LLM Server\n\n")

    try:
        read_return = read_config(['local_llm_server'])
        server_to_start = read_return['local_llm_server']
    except Exception as e:
        return handle_api_error("Server-side error, could not read local_llm_server from config.json in method local_llm_server_starter, encountered error: ", e)

    try:
        if server_to_start == 'hf-waitress':
            return hf_waitress_server_starter()
        elif server_to_start == 'llama-cpp':
            return llama_cpp_server_starter()
        else:
            return handle_api_error(f"Invalid local LLM server choice in method local_llm_server_starter: {server_to_start}")
    except Exception as e:
        return handle_api_error("Server-side error, could not start local LLM server in method local_llm_server_starter, encountered error: ", e)

    # return jsonify({'success': True})


@app.route('/set_prompt_template', methods=['POST'])
def set_prompt_template():

    base_template = ""

    try:
        base_template = request.form['prompt_template']
    except Exception as e:
        return handle_api_error("Server-side error, could not read prompt_template from the POST request in method set_prompt_template, encountered error: ", e)
    
    try:
        write_config({'base_template':base_template})
    except Exception as e:
        return handle_api_error("Could not update base_template in method set_prompt_template, encountered error: ", e)

    return jsonify({'success':True})


@app.route('/fetch_file_list_for_vector_db', methods=['POST'])
def fetch_file_list_for_vector_db():

    try:
        selected_embedding_model_choice = request.form['selected_embedding_model']
        knowledge_domain = request.form['selected_knowledge_domain']
    except Exception as e:
        return handle_api_error("Server-side error, could not read selected_embedding_model or selected_knowledge_domain from the POST request in method fetch_file_list_for_vector_db, encountered error: ", e)

    print(f"\nFetching file list for documents in knowledge domain: {knowledge_domain} embedded with: {selected_embedding_model_choice}...\n")
    
    try:
        conn, cursor = init_and_connect_to_docs_loaded_db()
    except Exception as e:
        return handle_api_error("Could not initialize and connect to docs_loaded_db to fetch_file_list_for_vector_db, encountered error: ", e)

    file_row_list = []

    try:
        cursor.execute("SELECT document_name, knowledge_domain, embedding_model, chunk_size FROM document_records where embedding_model LIKE ? AND knowledge_domain LIKE ?", (selected_embedding_model_choice, knowledge_domain))
    except Exception as e:
        return handle_api_error("Could not get document list from document_records db, encountered error: ", e)
    
    try:
        result = cursor.fetchall()

        for list_item in result:
            file_row_list.append(list(list_item))
    except Exception as e:
        return handle_api_error("Could not parse document list from document_records db, encountered error: ", e)

    try:
        cursor.close()
        conn.close()
    except Exception as e:
        handle_error_no_return("Could not close connections to document_records db, encountered error: ", e)
    
    return jsonify({'success': True, 'file_row_list': file_row_list})


def shell_delete_folder(folder_path, delete_vector_db = False):
    print(f"Deleting folder / resetting vector_db at path: {folder_path}")
    try:
        if os.path.exists(folder_path):

            if delete_vector_db:
                try:
                    print(f"Attempting to close connection to VectorDB at path: {folder_path} before deleting it...")
                    client = chromadb.PersistentClient(path=folder_path, settings=chromadb.Settings(allow_reset=True))
                    client.reset()  # Specifically mentioned in the chromadb docs as the way to cleanup and remove vector_dbs
                    print(f"Successfully reset vectorDB at path: {folder_path}")
                    return True
                except Exception as e:
                    handle_error_no_return("Could not reset vector_db, encountered error: ", e)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(folder_path)
                    print(f"Removed existing folder: {folder_path}")
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"Permission denied. Retrying in 1 second... (Attempt {attempt + 1} of {max_retries})")
                        time.sleep(1)
                    else:
                        print(f"Failed to remove folder after {max_retries} attempts.")
                        break
        else:
            print(f"No existing folder found at: {folder_path}")
    except Exception as e:
        return handle_api_error("Could not remove existing folder, encountered error: ", e)


def clean_up_docs_loaded_db(selected_embedding_model_choice, knowledge_domain):
    print(f"Cleaning up docs_loaded_db for embedding model: {selected_embedding_model_choice} and knowledge domain: {knowledge_domain}")
    try:
        conn, cursor = init_and_connect_to_docs_loaded_db()
    except Exception as e:
        return handle_api_error("Could not initialize and connect to docs_loaded_db to clean up, encountered error: ", e)

    try:
        with conn:  # Handles commit/rollback automatically
            cursor.execute("DELETE FROM document_records WHERE embedding_model LIKE ? AND knowledge_domain LIKE ?", (selected_embedding_model_choice, knowledge_domain))
    except Exception as e:
        return handle_api_error("Could not delete document records from docs_loaded_db, encountered error: ", e)
    finally:
        cursor.close()


def delete_knowledge_domain_graph(knowledge_domain):
    print(f"Deleting knowledge domain graph for: {knowledge_domain}")
    
    try:
        client = get_graph_db_client()
        graph = client.select_graph(knowledge_domain)
        graph.delete()  # client.delete_graph() is unsupported by FalkorDB. Delete individual nodes with Cypher: `MATCH (n) DETACH DELETE n`
        print(f"Successfully deleted graph for {knowledge_domain} domain in graph DB")
    except Exception as e:
        handle_error_no_return("Could not delete graph for {knowledge_domain} domain in graph DB, encountered error: ", e)


@app.route('/reset_vector_db_on_disk', methods=['POST'])
def reset_vector_db_on_disk():

    print("Resetting selected VectorDB")

    try:
        selected_embedding_model_choice = request.form['selected_embedding_model']
        knowledge_domain = request.form['selected_knowledge_domain']
    except Exception as e:
        return handle_api_error("Server-side error, could not read selected_embedding_model or selected_knowledge_domain from the POST request in method reset_vector_db_on_disk, encountered error: ", e)

    try:
        knowledge_domain_base_directory = read_config(['knowledge_domain_base_directory'])['knowledge_domain_base_directory']
    except Exception as e:
        handle_local_error("Missing values in config.json, could not determine knowledge_domain_base_directory. Error: ", e)

    try:
        path_to_knowledge_domain = os.path.join(knowledge_domain_base_directory, knowledge_domain)
        if not os.path.exists(path_to_knowledge_domain):
            os.makedirs(path_to_knowledge_domain, exist_ok=True)
            print(f"\n\nCreated knowledge domain directory: {path_to_knowledge_domain}\n\n")
    except Exception as e:
        handle_api_error("Could not determine path to knowledge domain, encountered error: ", e)

    try:
        vector_db_path = os.path.join(path_to_knowledge_domain, "vector_db_and_whoosh_index", selected_embedding_model_choice)
        whoosh_index_path = os.path.join(path_to_knowledge_domain, "vector_db_and_whoosh_index", selected_embedding_model_choice, "whoosh_index")
    except Exception as e:
        handle_api_error("Could not determine path to vector_db or whoosh_index, encountered error: ", e)
    
    shell_delete_folder(vector_db_path, delete_vector_db=True)
    shell_delete_folder(whoosh_index_path, delete_vector_db=False)
    clean_up_docs_loaded_db(selected_embedding_model_choice, knowledge_domain)
    delete_knowledge_domain_graph(knowledge_domain)
    
    return jsonify({'success': True})


@app.route('/delete_chat', methods=['POST'])
def delete_chat():

    print("\nDeleting chat\n")

    try:
        chat_id = request.form['chat_id']
    except Exception as e:
        return handle_api_error("Could not read chat_id from request form, encountered error: ", e)

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_api_error("Missing sqlite_history_db in config.json in method delete_chat. Error: ", e)
    
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to sqlite_history_db database to delete chat, encountered error: ", e)
    
    try:
        c.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        return handle_api_error("Could not delete chat from chat history db, encountered error: ", e)

    return jsonify({'success': True})


@app.route('/rename_chat', methods=['POST'])
def rename_chat():

    print("\nRenaming chat\n")

    try:
        chat_id = request.form['chat_id']
        new_chat_name = request.form['new_chat_name']
    except Exception as e:
        return handle_api_error("Could not read chat_id or new_chat_name from request form, encountered error: ", e)

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_api_error("Missing sqlite_history_db in config.json in method rename_chat. Error: ", e)

    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to sqlite_history_db database to rename chat, encountered error: ", e)

    try:
        c.execute("UPDATE chat_history SET chat_name = ? WHERE chat_id = ?", (new_chat_name, chat_id))
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        return handle_api_error("Could not rename chat in chat history db, encountered error: ", e)

    return jsonify({'success': True})


@app.route('/load_chat_history_list')
def load_chat_history_list():

    print("loading chat history list for sidebar")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_api_error("Missing sqlite_history_db in config.json in method load_chat_history_list. Error: ", e)

    history_list = []
    
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to sqlite_history_db database to load chat history list, encountered error: ", e)

    try:
        c.execute("""
            WITH RankedChats AS (
                SELECT
                    chat_id,
                    local_llm_server,
                    chat_name,
                    date_time,
                    prompt_template_format,
                    ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY date_time ASC) AS rn
                FROM chat_history
            )
            SELECT chat_id, local_llm_server, chat_name, date_time, prompt_template_format
            FROM RankedChats
            WHERE rn = 1
            ORDER BY date_time DESC;
        """)
    except Exception as e:
        return handle_api_error("Could not get list from chat history db, encountered error: ", e)
    
    try:
        result = c.fetchall()

        for row in result:
            history_list.append({
                'chat_id': row[0],
                'local_llm_server': row[1],
                'chat_name': row[2],
                'date_time': row[3],
                'prompt_template_format': row[4]
            })
    except Exception as e:
        return handle_api_error("Could not parse chat history list from db, encountered error: ", e)

    #print(f'returning chat hsitory list: {history_id_list}')

    return jsonify({'success': True, 'history_list': history_list})


@app.route('/load_chat_history', methods=['POST'])
def load_chat_history():

    print("loading chat history")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json in method load_chat_history. Error: ", e)

    try:
        chat_id_for_history_search = request.form['chat_id']
        chat_id = request.form['chat_id']
    except Exception as e:
        return handle_api_error("Could not retrieve Chat ID from request form, encountered error: ", e)

    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to chat history database, encountered error: ", e)

    sequence_id_for_history_search = 1
    retrieve_history = True
    chat_history = []
    old_chat_model = ""

    while(retrieve_history):

        try:
            c.execute("SELECT user_query FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()
            user_message = str(result[0])

            c.execute("SELECT stream_session_id FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            stream_session_id = c.fetchone()
            stream_session_id = str(stream_session_id[0])

            user_message = user_message.strip('\n')

            user_message = f'''
            <div class="user-message glassmorphism" data-stream-session-id="{stream_session_id}" data-chat-id="{chat_id_for_history_search}" data-sequence-id="{sequence_id_for_history_search}">
                {user_message}
                <div class="regenerate-menu">
                    <i class="fas fa-ellipsis-v"></i>
                    <div class="regenerate-menu-options">
                        <span class="regenerate-menu-option regenerate-option">Regenerate Response</span>
                        <span class="regenerate-menu-option regenerate-with-citations-enabled-option">Regenerate Response with Citations Force Enabled</span>
                        <span class="regenerate-menu-option regenerate-with-citations-disabled-option">Regenerate Response with Citations Force Disabled</span>
                        <span class="regenerate-menu-option delete-option">Delete</span>
                    </div>
                </div>
            </div>
            '''

            chat_history.append(user_message)

            c.execute("SELECT llm_response FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()
            result = str(result[0])

            result_parts = result.split("pdf_pane_data=",1)
            llm_response = f'<div class="response-and-viewer-container" data-stream-session-id="{stream_session_id}"><div class="llm-wrapper"> <div class="llm-response">' + result_parts[0]

        except Exception as e:
            return handle_api_error("Could not retrieve chat history, encountered error: ", e)
        
        llm_response = llm_response.strip('\n')
        llm_response = llm_response.replace('\n\n', '<br><br>')
        llm_response = llm_response.replace('\n', '<br>')
        
        try:
            c.execute("SELECT user_rating FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()
        except Exception as e:
            handle_error_no_return("Could not fetch user rating, encountered error: ", e)

        response_rated = False
        user_rating_for_history_chat = None

        if result[0]:
            response_rated = True
            try:
                user_rating_for_history_chat = int(result[0])
                #print(f'rating exists: {user_rating_for_history_chat}')
            except Exception as e:
                handle_error_no_return("Could not retrieve integer value of user rating, encountered error: ", e)


        llm_rating = f'''<div class="star-rating" data-rated={response_rated} rating-chat-id={chat_id_for_history_search} rating-sequence-id={sequence_id_for_history_search}>
        <i class="far fa-star" data-rate="1"></i>
        <i class="far fa-star" data-rate="2"></i>
        <i class="far fa-star" data-rate="3"></i>
        <i class="far fa-star" data-rate="4"></i>
        <i class="far fa-star" data-rate="5"></i>
        </div>
        </div>
        </div>'''


        if user_rating_for_history_chat:
            rating_parts = llm_rating.split("far", user_rating_for_history_chat)
            if len(rating_parts) <= user_rating_for_history_chat:
                llm_rating = "fas".join(rating_parts)
            else:
                llm_rating = "fas".join(rating_parts[:-1]) + "fas" + "far".join(rating_parts[-1:])

        llm_response += llm_rating

        if len(result_parts) > 1:
            llm_response += result_parts[1]
            llm_response += "</div>"
            llm_response = llm_response.strip('\n')
            llm_response = llm_response.replace('\n\n', '<br><br>')
            llm_response = llm_response.replace('\n', '<br>')

        chat_history.append(llm_response)

        # Increment sequence ID for next iteration:
        sequence_id_for_history_search += 1

        # But first, check to see if next sequence exists!
        try:
            c.execute("SELECT EXISTS(SELECT 1 FROM chat_history WHERE chat_id = ? AND sequence_id = ?)", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            exists = c.fetchone()[0]
        except Exception as e:
            return handle_api_error("Could not determine if next sequence exists in chat history DB, encountered error: ", e)
            
        if not exists:  # Fetch last used LLM
            sequence_id = sequence_id_for_history_search - 1
            retrieve_history = False
            try:
                c.execute("SELECT llm_model FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (chat_id, sequence_id))
                result = c.fetchone()
                old_chat_model = str(result[0])
            except Exception as e:
                handle_error_no_return("Could not determine previously used LLM in chat, encountered error: ", e)
            c.close()

    print(f'\n\nChat history loaded for chat with model: {old_chat_model}\n\n')

    try:
        sequence_id = determine_sequence_id_for_chat(chat_id)
        print(f"Sequence ID determined: {sequence_id}")
    except Exception as e:
        return handle_api_error("Could not determine sequence_id, encountered error: ", e)  

    return jsonify({'success': True, 'chat_history': chat_history, 'old_chat_model': old_chat_model, 'sequence_id': sequence_id})


def determine_latest_chat_id(c):
    print("Determining chat ID")
    c.execute("SELECT COALESCE(MAX(chat_id), 0) FROM chat_history") # "The COALESCE function accepts two or more arguments and returns the first non-null argument."
    result = c.fetchone()
    max_chat_id = result[0]
    new_chat_id = max_chat_id + 1
    print(f"Chat ID determined: {new_chat_id}")
    return new_chat_id


@app.route('/init_chat_history_db')
def init_chat_history_db():

    print("Initializing chat history DB")
    
    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_api_error("Missing sqlite_history_db in config.json in method init_chat_history_db. Error: ", e)

    # Connect to chat_history.db to determine appropriate chat_id
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to chat history database, encountered error: ", e)

    # If the database does not currently exist...
    try:
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY,
                        chat_id INTEGER,
                        sequence_id INTEGER,
                        stream_session_id TEXT,
                        user_query TEXT,
                        llm_response TEXT,
                        user_rating INTEGER,
                        llm_model TEXT, 
                        prompt_template TEXT,
                        history_summary TEXT,
                        local_llm_server TEXT,
                        chat_name TEXT,
                        date_time TEXT,
                        prompt_template_format TEXT
            )
        ''')
        conn.commit()
    except Exception as e:
        return handle_api_error("Could not create new chat history db, encountered error: ", e)

    try:
        add_column_if_not_exists(c, 'chat_history', 'chat_id', 'INTEGER')
        add_column_if_not_exists(c, 'chat_history', 'sequence_id', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'stream_session_id', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'user_query', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'llm_response', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'user_rating', 'INTEGER')
        add_column_if_not_exists(c, 'chat_history', 'llm_model', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'prompt_template', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'history_summary', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'local_llm_server', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'chat_name', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'date_time', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'prompt_template_format', 'TEXT')
    except Exception as e:
        return handle_api_error("Could not add necessary columns to chat history db, encountered error: ", e)

    try:
        chat_id = determine_latest_chat_id(c)
    except Exception as e:
        return handle_api_error("Could not set chat_id, encountered error: ", e)

    conn.close()

    return jsonify({'success': True, 'chat_id': chat_id})


def delete_chat_history_for_chat_id_from_sequence_id(c: sqlite3.Cursor, conn: sqlite3.Connection, chat_id: int, sequence_id: int) -> bool:
    print(f"Deleting chat history for chat with chat_id: {chat_id} and sequence_id greater than: {sequence_id}")

    # Delete all chat hsitory for the given chat_id where sequence_id is greater than the given sequence_id, if it exists
    try:
        c.execute("DELETE FROM chat_history WHERE chat_id = ? AND sequence_id > ?", (chat_id, sequence_id))
        deleted_count = c.rowcount
        conn.commit()
        print(f"Deleted {deleted_count} rows of chat history for chat with chat_id: {chat_id} and sequence_id greater than: {sequence_id}")
        return True
    except Exception as e:
        return handle_local_error(f"Could not delete chat history for chat with chat_id: {chat_id} and sequence_id greater than: {sequence_id}, encountered error: ", e)


@app.route('/delete_messages', methods=['POST'])
def delete_messages():
    print("delete_messages route triggered")

    try:
        chat_id = request.form['chat_id']
        sequence_id = request.form['sequence_id']
    except Exception as e:
        return handle_local_error("Could not read chat_id or sequence_id from request, encountered error: ", e)
    
    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_local_error("Missing sqlite_history_db in config.json in method delete_messages(). Error: ", e)

    # Connect to chat_history.db to determine appropriate chat_id
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_local_error("Could not connect to chat history database, encountered error: ", e)

    try:
        c.execute("DELETE FROM chat_history WHERE chat_id = ? AND sequence_id >= ?", (chat_id, sequence_id))
        deleted_count = c.rowcount
        conn.commit()
        print(f"Deleted {deleted_count} rows of chat history for chat with chat_id: {chat_id} beginning with sequence_id: {sequence_id}")
    except Exception as e:
        return handle_local_error(f"Could not delete chat history for chat with chat_id: {chat_id} beginning with sequence_id: {sequence_id}, encountered error: ", e)

    conn.close()
    return jsonify({'success': True})


def update_llm_response_in_history_db(chat_id: int, stream_session_id: str, user_query: str, llm_response: str) -> tuple[datetime.datetime, int]:

    print(f"Updating LLM response in chat history DB for chat_id: {chat_id} and stream_session_id: {stream_session_id}")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_local_error("Missing sqlite_history_db in config.json in method update_llm_response_in_history_db. Error: ", e)

    # Connect to chat_history.db to determine appropriate chat_id
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_local_error("Could not connect to chat history database, encountered error: ", e)

    try:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime('%d %b %Y - %I:%M %p %Z')
    except Exception as e:
        return handle_api_error("Could not obtain timestamp in update_llm_response_in_history_db, encountered error: ", e)
    
    # Update the LLM response in the chat history DB for the given stream_session_id:
    try:
        c.execute("UPDATE chat_history SET user_query = ?, llm_response = ?, date_time = ? WHERE stream_session_id = ?", (user_query, llm_response, formatted_datetime, stream_session_id))
        conn.commit()
    except Exception as e:
        return handle_local_error("Could not update LLM response in chat history DB, encountered error: ", e)
    
    try:
        c.execute("SELECT sequence_id FROM chat_history WHERE chat_id = ? AND stream_session_id = ?", (chat_id, stream_session_id))
        result = c.fetchone()
        sequence_id = result[0]
        delete_chat_history_for_chat_id_from_sequence_id(c, conn, chat_id, sequence_id)
    except Exception as e:
        return handle_local_error("Could not determine sequence_id / delete chat history in update_llm_response_in_history_db, encountered error: ", e)
    
    conn.close()

    return formatted_datetime, chat_id


def extract_significant_phrases(query):
    print("Extracting significant phrases")

    if not query:
        print("No query to extract significant phrases from")
        return []

    try:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        custom_stop_words = {"you", "me", "anything", "tell", "can", "could", "would", "should", "write", "writes", "wrote", "written", "read", "reads"}
        stop_words.update(custom_stop_words)
    except Exception as e:
        handle_error_no_return("Failed to download & set stopwords, encountered error: ", e)
    
    try:
        tokens = [token for token in query.lower().split() if token.isalnum() and token not in stop_words]  # isalnum() to remove punctuation and non-alphanumeric characters
    except Exception as e:
        handle_local_error("Could not extract significant tokens, encountered error: ", e)

    print(f"\nReturning tokens: {tokens}\n")
    return tokens


def calculate_relevance_score(phrases, document_content):
    #print("calculating relevance score")
    
    try:
        content_lower = document_content.lower()
    except Exception as e:
        handle_local_error("Could not read document_content in calculate_relevance_score(), encountered error: ", e)
    
    #print(f"document content: {content_lower}")
    
    #score = sum(1 for phrase in phrases if phrase in content_lower)
    
    score = 0
    try:
        for phrase in phrases:
            if phrase in content_lower:
                print(f"Match found to enable RAG: {phrase}")
                score += 1
    except Exception as e:
        handle_local_error("Could not compare phrases in calculate_relevance_score(), encountered error: ", e)
    
    return score


def filter_relevant_documents(query, search_results, threshold=1):

    print("Checking relevant docs to determin if RAG is required")

    do_rag = False
    page_contents = []

    try:
        significant_phrases = extract_significant_phrases(query)
    except Exception as e:
        handle_local_error("Could not extract significant phrases, encountered error: ", e)
    
    print(f"significant tokens: {significant_phrases}")
    #relevant_documents = []

    try:
        for document in search_results:
            # check for non-empty source field
            if document.page_content:
                page_contents.append(document.page_content)

            if not do_rag:  # if do_rag has already been set to true, why look?
                if document.metadata.get('source'):
                    score = calculate_relevance_score(significant_phrases, document.page_content)
                    if score >= threshold:
                        #relevant_documents.append(document)
                        print("Must do RAG!")
                        do_rag = True
    except Exception as e:
        handle_local_error("Could not read calculate relevance score, encountered error: ", e)

    #return relevant_documents
    return page_contents, do_rag


def rerank_results_ml(query, documents, top_n=5):
    print("\n\nReranking Invoked\n\n")

    try:
        read_return = read_config(['use_embedding_model_for_reranking', 'selected_embedding_model'])
        use_embedding_model_for_reranking = str(read_return['use_embedding_model_for_reranking']).lower() == 'true'
        selected_embedding_model = str(read_return['selected_embedding_model'])
    except Exception as e:
        use_embedding_model_for_reranking = True
        handle_error_no_return("Could not read use_embedding_model_for_reranking or selected_embedding_model from config.json proceeding to re-use embedding model for reranking. Encountered error: ", e)

    if not use_embedding_model_for_reranking:
        selected_embedding_model = 'all-MiniLM-L6-v2'

    print(f"\n\nSelected embedding model for reranking: {selected_embedding_model}\n\n")

    model = None
    try:
        # Load pre-trained SBERT model
        model = SentenceTransformer(selected_embedding_model)
        
        # Encode the query
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Encode the documents
        doc_embeddings = model.encode([doc.page_content for doc in documents], convert_to_tensor=True)
    except Exception as e:
        handle_local_error("Could not rerank results with SBERT, encountered error: ", e)
        return [doc.page_content for doc in documents]
    finally:
        if model is not None:
            del model
            if torch.cuda.is_available():
                print("Emptying CUDA cache")
                torch.cuda.empty_cache()
            print("Collecting garbage")
            gc.collect()

    try:
        # Compute cosine similarities
        cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    except Exception as e:
        handle_local_error("Could not compute cosine similarities, encountered error: ", e)
        return [doc.page_content for doc in documents]
    
    try:
        # Create a list of (index, score) tuples
        indexed_scores = list(enumerate(cosine_scores))
        
        # Sort by score in descending order
        sorted_indexes = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        
        # Reorder the original documents based on the sorted indexes
        ranked_documents = [documents[idx] for idx, _ in sorted_indexes[:top_n]]

        # print(f"\n\nReturning Top {len(ranked_documents)} Ranked Documents: {ranked_documents}\n\n")
    
        return ranked_documents[::-1]   #Slice to reverse the list, as `.reverse()` would return None because it creates an inplace change on the original list without returning anything
    except Exception as e:
        handle_local_error("Could not reorder documents, encountered error: ", e)
        return [doc.page_content for doc in documents]
    

def determine_do_rag(query, docs, force_enable_rag, force_disable_rag):

    print("\n\nDetermining do_rag \n\n")

    do_rag = False
    
    # We do not modify the force_enable_rag or force_disable_rag flags in this method, we simply respond to them here. UI updates should handle those flags.
    if force_enable_rag:
        print("\n\nFORCE_ENABLE_RAG True, force enabling RAG and returning\n\n")
        try:
            do_rag = True
        except Exception as e:
            do_rag = False
            handle_error_no_return("Error force-enabling RAG, disabling RAG and continuing: could not filter_relevant_documents during setup_for_streaming_response, encountered error: ", e)
    elif force_disable_rag:
        print("\n\nFORCE_DISABLE_RAG True, force disabling RAG and returning\n\n")
        do_rag = False
    else:
        try:
            _, do_rag = filter_relevant_documents(query, docs)
        except Exception as e:
            do_rag = True
            handle_error_no_return("Error determining if RAG is required, default enabling RAG and continuing: could not filter_relevant_documents during setup_for_streaming_response, encountered error: ", e)

    return do_rag



def get_formatted_prompt_from_history_db(chat_id, sequence_id):

    print(f"\n\nFormatting prompt from history for chat with chat_id: {chat_id} and sequence_id: {sequence_id}\n\n")

    formatted_prompt = ""

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_error_no_return("Missing keys in config.json for method get_formatted_prompt_from_history_db(). Error: ", e)

    # Connect to or create the DB
    try:
        conn = sqlite3.connect(sqlite_history_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_error_no_return("Could not establish connection to DB for chat history storage, encountered error: ", e)

    try:
        cursor.execute("SELECT prompt_template FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id), int(sequence_id)))
        result = cursor.fetchone()
        formatted_prompt = str(result[0])
        
    except Exception as e:
        handle_error_no_return("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)

    return formatted_prompt



def read_config_for_hf_waitress_prompt_formatting() -> tuple[bool, str, bool, bool]:
    try:
        exl2 = read_hf_config(['exl2'])['exl2']
        exl2_prompt_template_format = read_config(['exl2_prompt_template_format'])['exl2_prompt_template_format']
        vision = read_hf_config(['vision'])['vision']
        flux_diffusers = read_hf_config(['flux_diffusers'])['flux_diffusers']
        return exl2, exl2_prompt_template_format, vision, flux_diffusers
    except Exception as e:
        handle_error_no_return("Could not read exl2 details from config.json / hf-config.json, encountered error: ", e)


def format_prompt_for_hf_waitress(formatted_prompt:str, user_query:str, current_sequence_id:int, base_template:str, skip_system_prompt:bool) -> str:

    print("\n\nFormatting prompt for hf-waitress\n\n")

    try:
        exl2, exl2_prompt_template_format, vision, flux_diffusers = read_config_for_hf_waitress_prompt_formatting()
    except Exception as e:
        handle_error_no_return("Could not read exl2 details from config.json / hf-config.json, encountered error: ", e)

    if exl2:
        return(manually_format_prompt_with_prompt_template(formatted_prompt, user_query, current_sequence_id, base_template, exl2_prompt_template_format, skip_system_prompt))

    try:
    
        # double curly braces necessitated by Python's f-string syntax, to escape the inner curly braces in the JSON string
        if flux_diffusers:
            formatted_prompt = f'''
            {{
                "messages": [
                    {{"prompt": {json.dumps(user_query)}}}
                ]
            }}
            '''
        else:
            if current_sequence_id > 0:
                history_prompt_json = json.loads(formatted_prompt)
                new_message = {"role":"user", "content":user_query}
                history_prompt_json['messages'].append(new_message)
                updated_history_prompt_json = json.dumps(history_prompt_json, indent=4)
                if vision:  
                    formatted_prompt = updated_history_prompt_json  # return json object
                else:
                    formatted_prompt = str(updated_history_prompt_json)
            else:   # first message in chat
                if vision:
                    formatted_prompt = {
                        "messages": [
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": user_query}
                                ]
                            }
                        ]
                    }
                    formatted_prompt = json.dumps(formatted_prompt) # Convert to a JSON string
                else:
                    if skip_system_prompt:
                        first_prompt_json = f'''
                        {{
                                "messages": [
                                    {{"role": "user", "content": {json.dumps(user_query)}}}
                                ]
                            }}
                        '''
                    else:
                        first_prompt_json = f'''
                        {{
                                "messages": [
                                    {{"role": "system", "content": {json.dumps(base_template)}}},
                                    {{"role": "user", "content": {json.dumps(user_query)}}}
                                ]
                            }}
                        '''                    

                    formatted_prompt = str(first_prompt_json)
    except Exception as e:
        handle_error_no_return("Could not format prompt for hf-waitress in method format_prompt_for_hf_waitress, encountered error: ", e)

    return formatted_prompt


def combine_whoosh_and_vector_results(whoosh_results, vector_results):
    print("\n\nCombining whoosh and vector results\n\n")

    combined_results = []

    # Convert whoosh results to Document objects
    for result in whoosh_results:
        combined_results.append(Document(
            page_content=result['content'].strip().replace('\n', ' '),
            metadata={
                'source': result['source'],
                'page_number': result['page_number']
            }
        ))

    # Add the vector results to the combined results
    combined_results.extend(vector_results)

    # Filter out any duplicate documents based on page_content
    try:
        seen = {}
        unique_results = []
        for doc in combined_results:
            if doc.page_content not in seen:
                seen[doc.page_content] = True
                unique_results.append(doc)

        combined_results = unique_results
    except Exception as e:
        handle_error_no_return("Could not filter out duplicate documents in method combine_whoosh_and_vector_results. Returning all results. Encountered error: ", e)
    
    return combined_results


def get_session_id_and_vector_key() -> tuple[str, str]:
    '''
    # Generate a unique session ID using universally Unique Identifier via the uuid4() method, wherein the randomness of the result is dependent on the randomness of the underlying operating system's random number generator
    # UUI is a standard used for creating unique strings that have a very high likelihood of being unique across all time and space, for ex: f47ac10b-58cc-4372-a567-0e02b2c3d479
    '''
    stream_session_id = str(uuid.uuid4())
    key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
    return stream_session_id, key_for_vector_results


def read_config_for_setup_for_local_llm_response() -> dict:
    read_return = read_config([
        'local_llm_server',
        'force_enable_rag', 
        'force_disable_rag', 
        'local_llm_chat_template_format', 
        'base_template',
        'fetch_top_k_results_from_vectordb', 
        'filter_top_k_results_by_reranking', 
        'skip_system_prompt'
    ])
    return {
        'force_enable_rag': read_return['force_enable_rag'],
        'force_disable_rag': read_return['force_disable_rag'],
        'local_llm_chat_template_format': read_return['local_llm_chat_template_format'],
        'base_template': read_return['base_template'],
        'local_llm_server': read_return['local_llm_server'],
        'fetch_top_k_results_from_vectordb': read_return['fetch_top_k_results_from_vectordb'],
        'filter_top_k_results_by_reranking': read_return['filter_top_k_results_by_reranking'],
        'skip_system_prompt': str(read_return['skip_system_prompt']).lower() == 'true'
    }


def read_request_data_for_setup_for_local_llm_response(request: Request) -> tuple[str, str, bool]:
    user_query = request.json['user_query']
    chat_id = request.json['chat_id']
    file_attached = request.json['file_attached']
    return user_query, chat_id, file_attached


def determine_special_model_type_for_hf_waitress() -> tuple[bool, bool]:
    try:
        hf_read_return = read_hf_config(['flux_diffusers', 'vision'])
        return (
            str(hf_read_return['flux_diffusers']).lower() == 'true',
            str(hf_read_return['vision']).lower() == 'true'
        )
    except Exception as e:
        handle_error_no_return("Could not determine if flux_diffusers or vision model in method determine_special_model_type_for_hf_waitress, encountered error: ", e)
        return False, False


def prepare_special_model_response(formatted_prompt:str, user_query:str, current_sequence_id:int, new_sequence_id:int, stream_session_id:str, local_llm_server:str) -> dict:
    print("\n\nPreparing special model response\n\n")
    try:
        is_diffusers = local_llm_server == 'hfw-diffusers'
        formatted_prompt = format_prompt_for_hf_waitress(
            formatted_prompt="" if is_diffusers else formatted_prompt, 
            user_query=user_query, 
            current_sequence_id=0 if is_diffusers else current_sequence_id, 
            base_template="",  
            skip_system_prompt=True
        )
        return {
            "success": True, 
            "stream_session_id": stream_session_id,
            "do_rag": False, 
            "formatted_user_prompt": formatted_prompt, 
            "sequence_id":new_sequence_id, 
            "server_type":local_llm_server
        }
    except Exception as e:
        handle_local_error("Could not prepare special model response in method prepare_special_model_response, encountered error: ", e)


def reject_rag() -> dict:
    try:
        write_config({'do_rag':False})
        return {"success": True}
    except Exception as e:
        handle_error_no_return("Could not default do_rag to False in method reject_rag, encountered error: ", e)
        return {"success": False}


def prepare_for_quick_response(current_sequence_id:int, regeneration_request:bool) -> int:
    print("Invoking quick-return route for hfw-diffusers or hfw-vision(file_attached)")
    if not regeneration_request or current_sequence_id == 0: current_sequence_id = int(current_sequence_id) + 1
    reject_rag()
    return current_sequence_id


def get_formatted_prompt_for_setup_for_local_llm_response(chat_id:int, current_sequence_id:int) -> str:
    print("\n\nGetting formatted prompt for setup_for_local_llm_response\n\n")
    formatted_prompt = ""
    if current_sequence_id > 0:    # get the last prompt so we can continue the completions
        formatted_prompt = get_formatted_prompt_from_history_db(chat_id, current_sequence_id)
    return formatted_prompt


def read_request_data_for_response_setup(request: Request) -> tuple[str, str, str, int, bool, bool, bool, bool]:
    stream_session_id = request.json.get('stream_session_id')
    user_query = request.json.get('user_query')
    chat_id = request.json.get('chat_id')
    sequence_id = request.json.get('sequence_id')
    file_attached = request.json.get('file_attached')
    regeneration_request = request.json.get('regeneration_request')
    regenerate_with_citations_force_enabled = request.json.get('regenerate_with_citations_force_enabled')
    regenerate_with_citations_force_disabled = request.json.get('regenerate_with_citations_force_disabled')
    return stream_session_id, user_query, chat_id, sequence_id, file_attached, regeneration_request, regenerate_with_citations_force_enabled, regenerate_with_citations_force_disabled


def get_full_prompt_for_server(local_llm_server: str, formatted_history_prompt: str, user_query: str, current_sequence_id: int, base_template: str, local_llm_chat_template_format: str, skip_system_prompt: bool) -> str:
    if local_llm_server == 'llama-cpp':
        formatted_updated_prompt = manually_format_prompt_with_prompt_template(formatted_history_prompt, user_query, current_sequence_id, base_template, local_llm_chat_template_format, skip_system_prompt)
    elif local_llm_server == 'hf-waitress':
        formatted_updated_prompt = format_prompt_for_hf_waitress(formatted_history_prompt, user_query, current_sequence_id, base_template, skip_system_prompt)
    elif local_llm_server == 'hfw-vision':
        formatted_updated_prompt = format_prompt_for_hf_waitress(formatted_history_prompt, user_query, current_sequence_id, "", True)  # No base_template for hfw-vision
    # print("Returning formatted_prompt: ", formatted_updated_prompt)
    return formatted_updated_prompt


def get_base_values_for_setup_for_local_llm_response(stream_session_id:str, chat_id:str, sequence_id:str, regeneration_request:bool) -> tuple[str, str, int]:
    if regeneration_request:
        print(f"\nSetting defaults for regeneration for request ID {stream_session_id}\n")
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        current_sequence_id = int(sequence_id)
        return stream_session_id, key_for_vector_results, current_sequence_id
    
    try:
        current_sequence_id = determine_sequence_id_for_chat(chat_id)
        new_stream_session_id, key_for_vector_results = get_session_id_and_vector_key()
        print(f"Current Chat ID: {chat_id} & Sequence ID: {current_sequence_id}")
        return new_stream_session_id, key_for_vector_results, current_sequence_id
    except Exception as e:
        return handle_api_error("Error determining sequence_id and/or getting session_id and vector_key in get_base_values_for_setup_for_local_llm_response, encountered error: ", e)


def handle_special_model_case(local_llm_server:str, current_sequence_id:int, file_attached:bool, stream_session_id:str, user_query:str, formatted_history_prompt:str, regeneration_request:bool) -> tuple[str, Response]:
    print("\n\nHandling special model case\n\n")
    if local_llm_server != 'hf-waitress':   #if llama.cpp
        return local_llm_server, None
    
    flux_diffusers, vision = determine_special_model_type_for_hf_waitress()

    if flux_diffusers or file_attached:
        new_sequence_id = prepare_for_quick_response(current_sequence_id, regeneration_request)
        new_local_llm_server='hfw-diffusers' if flux_diffusers else 'hfw-vision'
        try:
            response = prepare_special_model_response(
                local_llm_server=new_local_llm_server,
                stream_session_id=stream_session_id,
                user_query=user_query,
                current_sequence_id=current_sequence_id,
                new_sequence_id=new_sequence_id,
                formatted_prompt=formatted_history_prompt
            )
            print(f"Returning quick-return formatted_user_prompt: {response['formatted_user_prompt']}")
            return new_local_llm_server, jsonify(response)
        except Exception as e:
            return handle_api_error("Could not prepare special model response in method setup_for_local_llm_response, encountered error: ", e)
    
    if vision: 
        return 'hfw-vision', None
    
    return local_llm_server, None   #if hf-waitress but not hfw-diffusers or hfw-vision


def handle_force_disabled_rag(local_llm_server:str, formatted_history_prompt:str, user_query:str, current_sequence_id:int, stream_session_id:str, regeneration_request:bool, base_template:str, local_llm_chat_template_format:str, skip_system_prompt:bool) -> Response:
    print(f"\nForce disabling RAG for request ID {stream_session_id}\n")
    reject_rag()
    try:
        formatted_updated_prompt = get_full_prompt_for_server(local_llm_server, formatted_history_prompt, user_query, current_sequence_id, base_template, local_llm_chat_template_format, skip_system_prompt)
    except Exception as e:
        return handle_api_error("Could not get formatted_updated_prompt in method setup_for_streaming_response, encountered error: ", e)
    if not regeneration_request or current_sequence_id == 0: current_sequence_id = int(current_sequence_id) + 1
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": False, "formatted_user_prompt": formatted_updated_prompt, "sequence_id":current_sequence_id, "server_type":local_llm_server})


def search_vector_db(user_query:str, embedding_function:str, fetch_top_k_results_from_vectordb: int):
    print("Searching vectorDB")

    min_semantic_similarity_threshold = read_config(['min_semantic_similarity_threshold'])['min_semantic_similarity_threshold']

    path_to_knowledge_domain = get_path_to_knowledge_domain()
    vector_db_path = create_vector_db_directory(path_to_knowledge_domain, embedding_function)

    print(f"Searching Knowledge Domain: {path_to_knowledge_domain} with embedding function: {embedding_function}")

     # Load Embedding Model
    embedding_model = None
    try:
        embedding_model = SentenceTransformer(embedding_function, trust_remote_code=True)
    except Exception as e:
        handle_local_error("Could not load embedding model, encountered error: ", e)

    try:
        # Initialize Chroma Client and collection
        chroma_client = chromadb.PersistentClient(path=vector_db_path, settings=chromadb.Settings(allow_reset=True))
        collection = chroma_client.get_or_create_collection(name="knowledge_domain", metadata={"hnsw:space": "cosine"}) # By default, ChromaDB returns the L2 distance (lower is better), but we want cosine distance (higher is better)

        query_embedding = embedding_model.encode(user_query)

        # Perform the semantic search - 'results' is a dictionary with keys 'documents', 'metadatas', 'distances', whose values are lists of length = fetch_top_k_results_from_vectordb
        results = collection.query(
            query_embeddings=query_embedding.tolist(),  # Convert embeddings from NumPy arrays to list of lists
            n_results=fetch_top_k_results_from_vectordb,    # top-k here implies the top from the matched set, regardless of the actual similarity score!
            include=["documents", "metadatas", "distances"]
        )

        # Format similar to LangChain's Output so as to maintain consistency
        docs_list_with_cosine_distance = [
            (
                Document(
                    page_content=doc,
                    metadata=metadata
                ),
                distance
            )
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
            if distance >= min_semantic_similarity_threshold    # ChromaDB's score ranges from -1 (perfect dissimilarity) to 1 (perfect similarity), with 0.0 meaning no similarity.
        ]   # The zip() function combines multiple iterables (lists, tuples, etc.) element by element and helps iterate over multiple lists simultaneously

        print(f"Result of Semantic Search: Found {len(docs_list_with_cosine_distance)} documents of {len(results['documents'][0])} with a minimum semantic similarity threshold of {min_semantic_similarity_threshold}")
        return docs_list_with_cosine_distance
    except Exception as e:
        handle_error_no_return("Could not perform similarity_search to determine do_rag when attempting to setup_for_streaming_response, encountered error: ", e)
        return []
    finally:
        if embedding_model is not None:
            del embedding_model
            if torch.cuda.is_available():
                print("Emptying CUDA cache")
                torch.cuda.empty_cache()
            print("Collecting garbage")
            gc.collect()


def get_summary_report(summarized_chunk_entities: dict) -> str:
    print(f"\n\nGetting summary report\n\n")
    
    summary_report = set()
    try:
        
        for _, chunk_data in summarized_chunk_entities.items():
            source_doc_name = chunk_data['source_doc_name']
            
            if source_doc_name == 'user_query':
                print("\nSkipping user query chunk\n")
                continue
            
            for node in chunk_data['entities_and_relationships']['nodes']:

                if not node.get('summary'):
                    continue    # Skip nodes with no summaries
                
                source_docs = node.get('source_documents')
                node_source_doc_name = set(source_docs) if source_docs else set(source_doc_name.split(' '))

                for summary in node.get('summary', []):
                    if summary is not None and summary != '':
                        entry = (
                            f"Summary for node '{node['name']}' of type '{node['type']}' - {summary} \n"
                            f"Source Document(s): {node_source_doc_name}\n\n"
                        )
                        summary_report.add(entry)
            
            for relationship in chunk_data['entities_and_relationships']['relationships']:
                if not relationship.get('summary'):
                    continue    # Skip relationships with no summaries
                
                source_docs = relationship.get('source_documents')
                relationship_source_doc_name = set(source_docs) if source_docs else set(source_doc_name.split(' '))

                for summary in relationship.get('summary', []):
                    if summary is not None and summary != '':
                        entry = (
                            f"Summary for relationship '{relationship['relationship']}' between '{relationship['source']}' and '{relationship['target']}' - {summary} \n"
                            f"Source Document(s): {relationship_source_doc_name}\n\n"
                        )
                        summary_report.add(entry)
    
    except Exception as e:
        handle_error_no_return("Could not add to summary report, skipping chunk {count}. Encountered error: ", e)
    return ''.join(summary_report)


def get_summaries_from_graph_db(chunk_entities: dict, selected_knowledge_domain: str, graph: FalkorDB):
    '''
    Receives a complete chunk_entities dict:

    chunk_entities = {
        '<graph_chunk_number>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>'
        }
    }

    And basis the 'chunk_text' and 'entities_and_relationships' for each `graph_chunk_number`:
        1. Handles Summaries if applicable:
            - Retrieves existing summaries for each node and relationship
            - Generates a summary for each node and relationship. This sequence ensures highest throughput of summary-generation requests to the LLM as both the lookup for existing summaries and the storage to the GraphDB are handled completely seperately!
        2. Stores the nodes and relationships in the graph DB
    '''

    print(f"\nStoring entities and relationships in {selected_knowledge_domain} graph DB\n")

    try:
        # a. Get summaries for all nodes and relationships:
        for chunk_number, chunk_data in chunk_entities.items():
            print_string = f" in chunk {chunk_number} of total {len(chunk_entities)} chunks"
            print(f"\nChecking for existing summaries for all nodes and relationships {print_string}...\n")

            try:
                nodes_with_existing_summaries = get_summaries_for_all_nodes(nodes=chunk_data['entities_and_relationships']['nodes'], graph=graph, print_string=print_string, get_source_documents=True)
                chunk_entities[chunk_number]['entities_and_relationships']['nodes'] = nodes_with_existing_summaries
            except Exception as e:
                handle_error_no_return(f"Error checking for existing summaries for nodes, skipping chunk {chunk_number}. Encountered error: ", e)

            try:
                relationships_with_existing_summaries = get_summaries_for_all_relationships(relationships=chunk_data['entities_and_relationships']['relationships'], graph=graph, print_string=print_string, get_source_documents=True)
                chunk_entities[chunk_number]['entities_and_relationships']['relationships'] = relationships_with_existing_summaries
            except Exception as e:
                handle_error_no_return(f"Error checking for existing summaries for relationships, skipping chunk {chunk_number}. Encountered error: ", e)
    
    except Exception as e:
        handle_error_no_return("Could not get summaries from graph DB, encountered error: ", e)

    return chunk_entities


def merge_chunk_entities_for_graph_rag(chunk_entities: dict) -> dict:
    '''
    Receives a complete chunk_entities dict:

    chunk_entities = {
        '<graph_chunk_number_1>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>'
        },
        '<graph_chunk_number_2>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>'
        },
        ...
    }

    And returns a merged chunk_entities dict, because all entities and relationships are extracted from RAG context, and 
    merging will allow for de-duplication of nodes and relationships in the get_summary step.
    '''

    print(f"\n\nMerging chunk entities for graph RAG. Received chunk_entities: \n {chunk_entities}\n\n")

    try:
        chunk_entities_merged = {0: {
            'chunk_text': '',
            'entities_and_relationships': {
                'nodes': [],
                'relationships': []
            },
            'source_chunks': [],
            'source_doc_name': ''
        }}
        for _, chunk_data in chunk_entities.items():
            chunk_entities_merged[0]['chunk_text'] += chunk_data['chunk_text'] if chunk_entities_merged[0]['chunk_text'] == '' else f" {chunk_data['chunk_text']}"
            chunk_entities_merged[0]['entities_and_relationships']['nodes'].extend(chunk_data['entities_and_relationships']['nodes'])
            chunk_entities_merged[0]['entities_and_relationships']['relationships'].extend(chunk_data['entities_and_relationships']['relationships'])
            chunk_entities_merged[0]['source_chunks'].extend(chunk_data['source_chunks'])
            chunk_entities_merged[0]['source_doc_name'] += chunk_data['source_doc_name'] if chunk_entities_merged[0]['source_doc_name'] == '' else f" {chunk_data['source_doc_name']}"
        
        print(f"\n\nMerged chunk entities for graph RAG. Resulting chunk_entities_merged: \n {chunk_entities_merged}\n\n")
        return chunk_entities_merged
    except Exception as e:
        return handle_local_error("Could not merge chunk entities for graph RAG, encountered error: ", e)


def get_doc_object_dict(docs: list[Document]) -> dict:
    parsed_documents = []
    
    for doc in docs:
        
        try:
            relevant_page_text = str(doc.page_content)
            relevant_page_number = str(doc.metadata.get('page_number'))
            source_filepath_full = str(doc.metadata.get('source'))
            source_filepath = os.path.basename(source_filepath_full)
        
            relevant_page_text = relevant_page_text.replace('\n', ' ')
            parsed_documents.append({
                'content': relevant_page_text.strip().replace("'", ""),
                'source': source_filepath,
                'page_number': relevant_page_number
            })
            
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue

    return parsed_documents


def execute_graph_rag(user_query:str, docs: list[Document]) -> str:
    '''
    Assembles document chunks (via assemble_chunks_for_graph_db()) into a dictionary of entities for storage to the GraphDB:

    chunk_entities = {
        '<graph_chunk_number>': {
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>'
        }
    }

    This is then passed to the graphing model which will process each graph_chunk and append the `entities_and_relationships` key to each chunk_entities dict:
        
        '<entities_and_relationships>': '<node_relationships_dict>'
        # eg: {"nodes": [{"type": "organization","name": "Intel"},{"type": "object","name": "Intel Products"}], "relationships": [{"source": "Intel","target": "Intel Products","relationship": "business unit"}]},

    The final `chunk_entities` dict is then passed to the `get_summaries_from_graph_db()` function which will store the entities and relationships in the GraphDB.
    '''
    
    print(f"\n\nExecuting GraphRAG. Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    try:
        bring_graph_db_online()
        bring_graphing_model_online()
    except Exception as e:
        return handle_local_error("Could not bring graph DB or graphing model online, encountered error: ", e)

    try:
        doc_object_dict = get_doc_object_dict(docs)
    except Exception as e:
        return handle_local_error("Could not get doc object dict, encountered error: ", e)

    try:
        chunk_entities = assemble_chunks_for_graph_db(doc_object_dict, user_query)
    except Exception as e:
        return handle_local_error("Could not assemble chunks for graph DB, encountered error: ", e)
    
    try:
        selected_knowledge_domain = read_config(['selected_knowledge_domain'])['selected_knowledge_domain']    
        client = get_graph_db_client()
        graph = client.select_graph(selected_knowledge_domain)  # Will create the graph if it doesn't exist
    except Exception as e:
        return handle_local_error(f"Could not connect to / initialize graph for '{selected_knowledge_domain}' domain in graph DB, encountered error: ", e)

    try:
        complete_chunk_entities = extract_all_entities_and_relationships(chunk_entities, True)  # RAG response mode = True
    except Exception as e:
        return handle_local_error("Failed to extract entities and relationships from chunk entities, encountered error: ", e)

    try:
        merged_graph_rag_entities_and_relationships_dict = merge_chunk_entities_for_graph_rag(complete_chunk_entities)
    except Exception as e:
        handle_error_no_return("Could not merge chunk entities for graph RAG, proceeding with original chunk_entities dict. Encountered error: ", e)
        merged_graph_rag_entities_and_relationships_dict = complete_chunk_entities

    try:
        summarized_chunk_entities = get_summaries_from_graph_db(merged_graph_rag_entities_and_relationships_dict, selected_knowledge_domain, graph)
    except Exception as e:
        return handle_local_error("Could not store entities and relationships in graph DB, encountered error: ", e)

    try:
        summary_report = get_summary_report(summarized_chunk_entities)
    except Exception as e:
        return handle_local_error("Could not get summary report, encountered error: ", e)

    return summary_report


def search_knowledge_base(user_query:str, embedding_function:str, perform_graph_rag:bool, force_enable_rag:bool, force_disable_rag:bool, filter_top_k_results_by_reranking:int, fetch_top_k_results_from_vectordb:int, ) -> tuple[list[Document], bool]:
    print("Searching knowledge base")

    filtered_docs = []
    try:
        docs_list_with_cosine_distance = search_vector_db(user_query, embedding_function, fetch_top_k_results_from_vectordb)
        filtered_docs = [doc for doc, score in docs_list_with_cosine_distance]  # the `doc,score` is crucial, as it ensure we select only the Document object, and not a tuple comprising of a Document object and a float score!
    except Exception as e:
        handle_error_no_return("Could not perform vector search to determine do_rag when attempting to setup_for_streaming_response, encountered error: ", e)

    whoosh_results = []
    try:
        whoosh_results = search_whoosh_index(user_query)
    except Exception as e:
        handle_error_no_return("Could not perform whoosh search to determine do_rag when attempting to setup_for_streaming_response, encountered error: ", e)

    if whoosh_results:  # Combine the whoosh and vector results
        combined_docs = combine_whoosh_and_vector_results(whoosh_results, filtered_docs)
    else:
        combined_docs = filtered_docs

    if not combined_docs:   # i.e. if blank
        print("No documents for citations, setting do_rag to False")
        do_rag = False
        return [], do_rag, None

    print(f"\n\nContext docs prior to reranking: {combined_docs}\n\n")
    docs = rerank_results_ml(user_query, combined_docs, top_n=filter_top_k_results_by_reranking)
    print(f"\n\nContext docs after reranking: {docs}\n\n")
    do_rag = determine_do_rag(user_query, docs, force_enable_rag, force_disable_rag)

    graph_rag_context = None
    if perform_graph_rag:
        try:
            graph_rag_context = execute_graph_rag(user_query, docs)
        except Exception as e:
            handle_error_no_return("Could not execute graph RAG, encountered error: ", e)


    return docs, do_rag, graph_rag_context


@app.route('/setup_for_local_llm_response', methods=['POST'])
def setup_for_local_llm_response():
    print("\n\nSetting up for local LLM response\n\n")

    global QUERIES
    do_rag = True

    try:    # Read config and request data, determine base values while handling regeneration case
        config = read_config_for_setup_for_local_llm_response()
        stream_session_id, user_query, chat_id, sequence_id, file_attached, regeneration_request, regenerate_with_citations_force_enabled, regenerate_with_citations_force_disabled = read_request_data_for_response_setup(request)
        QUERIES[stream_session_id] = user_query     # Store the query associated with the ID
        stream_session_id, key_for_vector_results, current_sequence_id = get_base_values_for_setup_for_local_llm_response(stream_session_id, chat_id, sequence_id, regeneration_request)
    except Exception as e:
        return handle_api_error("Error getting base values for setup_for_streaming_response, encountered error: ", e)

    try:    # Get formatted prompt from history db
        formatted_history_prompt = get_formatted_prompt_for_setup_for_local_llm_response(int(chat_id), int(current_sequence_id) if not regeneration_request else int(current_sequence_id) - 1) # if regeneration_request, we must act as if the current sequence id does not exist in the history db when formatting the prompt!
        if formatted_history_prompt == "": current_sequence_id = 0
    except Exception as e:
        return handle_api_error("Could not get formatted_history_prompt from history db in method setup_for_local_llm_response, encountered error: ", e)
    
    local_llm_server, special_response = handle_special_model_case(config['local_llm_server'], current_sequence_id, file_attached, stream_session_id, user_query, formatted_history_prompt, regeneration_request)
    if special_response is not None:    # If a special model response is returned, quick-return here
        print(f"Returning special model response: {special_response}")
        return special_response
    
    if config['force_disable_rag'] or regenerate_with_citations_force_disabled:
        return handle_force_disabled_rag(local_llm_server, formatted_history_prompt, user_query, current_sequence_id, stream_session_id, regeneration_request, config['base_template'], config['local_llm_chat_template_format'], config['skip_system_prompt'])
            
    try:    # RAG Routine Begins: Perform semantic search on the vector DB, lexical search on the whoosh index, combine and rerank results and determine if RAG is necessary
        print("\n\nRAG Routine Begins: Performing semantic search on VectorDB, lexical search on Whoosh index, combining and reranking results and determining if RAG is necessary\n\n") 
        selected_embedding_function = read_config(['selected_embedding_model'])['selected_embedding_model']
        perform_graph_rag = read_config(['perform_graph_rag'])['perform_graph_rag']
        docs, do_rag, graph_rag_context = search_knowledge_base(user_query, selected_embedding_function, perform_graph_rag,
        (config['force_enable_rag'] or regenerate_with_citations_force_enabled), 
        (config['force_disable_rag'] or regenerate_with_citations_force_disabled), 
        config['filter_top_k_results_by_reranking'], config['fetch_top_k_results_from_vectordb'])
    except Exception as e:
        return handle_error_no_return("Could not process vector search in method setup_for_local_llm_response, encountered error: ", e)

    try:    # Write do_rag to config and prepare RAG context if necessary
        print(f'Do RAG? {do_rag}')
        write_config({'do_rag':do_rag})
        if do_rag:    # Add similarity search results for RAG if necessary!
            QUERIES[key_for_vector_results] = docs
            if graph_rag_context is not None:
                user_query += f"\n\nThe following context might be helpful in answering the user query above. If so, please reference useful documents by name in your response:\n{graph_rag_context}"
            else:
                user_query += f"\n\nThe following context might be helpful in answering the user query above. If so, please reference it in your response by name and page number:\n{docs}"
    except Exception as e:
        reject_rag()
        handle_error_no_return("Could not write do_rag or prepare RAG context during setup_for_streaming_response, encountered error: ", e)

    try:    # Get full prompt for server
        formatted_updated_prompt = get_full_prompt_for_server(local_llm_server, formatted_history_prompt, user_query, current_sequence_id, config['base_template'], config['local_llm_chat_template_format'], config['skip_system_prompt'])
    except Exception as e:
        return handle_api_error("Could not get formatted_updated_prompt in method setup_for_streaming_response, encountered error: ", e)

    if not regeneration_request or current_sequence_id == 0: current_sequence_id = int(current_sequence_id) + 1
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag, "formatted_user_prompt": formatted_updated_prompt, "sequence_id":current_sequence_id, "server_type":local_llm_server})


def is_fuzzy_subset(string1: str, string2: str, threshold: int) -> bool:
    score = fuzz.partial_ratio(string1, string2)
    return score >= threshold


def is_citation_relevant(llm_response: str, source_filename: str) -> bool:
    print(f"Checking citation relevance: {source_filename} in LLM response?")
    try:
        if not llm_response or not source_filename:
            print("LLM response or source filename is empty, returning False")
            return False
        
        # Normalize inputs:
        llm_response = llm_response.lower().strip()
        source_filename = source_filename.lower().strip()

        # Variations of the filename:
        source_filename_no_extension, _ = os.path.splitext(source_filename) # os.path.splitext() returns a tuple containing the path's name and extension. It handles edge cases and is platform-independent.
        source_filename_cleaned = re.sub(r'[-_+]', ' ', source_filename_no_extension)
        source_filename_cleaned = re.sub(r' +', ' ', source_filename_cleaned)

        llm_response_cleaned = re.sub(r'[-_+]', ' ', llm_response)
        llm_response_cleaned = re.sub(r' +', ' ', llm_response_cleaned)

        # Regex patterns for matching:
        """
        re.escape() is used to escape special characters in the source filename, ensuring they are treated as literal characters in the regex pattern.
        \b is a word boundary, ensuring the pattern is a whole word. 
        rf'' is a raw f-string, allowing for the use of \b without it being interpreted as an escape character. This prevents partial matches, eg "doc1" matching on "doc123".
        """
        patterns = [
            rf'\b{re.escape(source_filename)}\b', # Exact filename match with extension
            rf'\b{re.escape(source_filename_cleaned)}\b', # Filename with dashes or underscores replaced by spaces
            rf'\b{re.escape(source_filename_no_extension)}\b', # Filename without extension
        ]

        responses_to_check = [
            llm_response,
            llm_response_cleaned
        ]

        is_relevant = any(
            re.search(pattern, response) 
            for pattern in patterns
            for response in responses_to_check
        )

        threshold = 80
        if not is_relevant: # No exact matches found, LLM may have mentioned the filename just differently enough, so time to check if a Fuzzy match is found
            print(f"\nNo exact matches found, checking for fuzzy match with a {threshold}% or higher threshold\n")
            is_relevant = is_fuzzy_subset(llm_response_cleaned, source_filename_cleaned, threshold)
            print(f"Fuzzy match result: {is_relevant} for {source_filename}\n")

        print(f"Citation relevance check result: {is_relevant} for {source_filename}")
        return is_relevant
    
    except Exception as e:
        handle_error_no_return("Could not determine if citation is relevant in is_citation_relevant(), encountered error: ", e)
        return False


def filter_all_citations(docs: list[Document], llm_response: str, return_top_k: bool, user_query: str) -> list[Document]:
    print(f"Pre-filtering citations to determine if any are relevant to the LLM response")
    all_docs = []
    for doc in docs:
        
        try:
            relevant_page_text = str(doc.page_content)
            source_filepath = str(doc.metadata.get('source'))
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue
        
        relevant_page_text = relevant_page_text.replace('\n', ' ')
        
        try:
            source_filename = os.path.basename(source_filepath)
        except Exception as e:
            handle_error_no_return("Could not parse path with OS lib, encountered error: ", e)
            continue
        
        try:
            if is_citation_relevant(llm_response, source_filename):
                all_docs.append(doc)
            else:
                print(f"Citation {source_filename} is not relevant, skipping")
                continue
        except Exception as e:
            handle_error_no_return("Could not determine if citation is relevant in filter_all_citations(), encountered error: ", e)
            continue

    if all_docs == [] and return_top_k:
        print("No relevant citations found but top K requested, reranking all docs")
        all_docs = rerank_results_ml(user_query, docs, top_n=3)

    return all_docs


def read_config_for_get_references() -> tuple[str, str, str, bool, bool, str]:
    try:
        read_return = read_config(['local_llm_server', 'upload_folder', 'local_llm_chat_template_format', 'llm_filter_citations', 'force_enable_rag', 'exl2_prompt_template_format'])
        local_llm_server = read_return['local_llm_server']
        upload_folder = read_return['upload_folder']
        local_llm_chat_template_format = read_return['local_llm_chat_template_format']
        llm_filter_citations = read_return['llm_filter_citations']
        force_enable_rag = read_return['force_enable_rag']
        exl2_prompt_template_format = read_return['exl2_prompt_template_format']
        return local_llm_server, upload_folder, local_llm_chat_template_format, llm_filter_citations, force_enable_rag, exl2_prompt_template_format
    except Exception as e:
        return handle_local_error("Could not read config.json in method read_config_for_get_references(), encountered error: ", e)


def get_request_parameters_for_get_references(request: Request) -> tuple[str, str, str, str, str, str, bool, bool]:
    try:
        stream_session_id = request.json['stream_session_id']
        user_query = request.json['user_query']
        llm_response = request.json['llm_response']
        formatted_user_prompt = request.json['formatted_user_prompt']
        chat_id = request.json['chat_id']
        sequence_id = request.json['sequence_id']
        regeneration_request = request.json['regeneration_request']
        regenerate_with_citations_force_enabled = request.json['regenerate_with_citations_force_enabled']
        return stream_session_id, user_query, llm_response, formatted_user_prompt, chat_id, sequence_id, regeneration_request, regenerate_with_citations_force_enabled
    except Exception as e:
        return handle_local_error("Could not read request content in method get_request_parameters_for_get_references(), encountered error: ", e)


def get_vector_results_for_get_references(stream_session_id: str) -> tuple[list[Document], bool]:
    try:
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        docs = QUERIES.pop(key_for_vector_results, None)
        return docs, docs is not None
    except Exception as e:
        return handle_local_error("Could not get vector results for stream_session_id in method get_vector_results_for_get_references(), encountered error: ", e)


def get_llama_cpp_formatted_user_prompt(local_llm_chat_template_format: str, llm_response: str) -> str:
    if local_llm_chat_template_format == 'llama3':
        return f"{llm_response}<|eot_id|>"
    elif local_llm_chat_template_format == 'llama2':
        return f"{llm_response} </s>\n"
    elif local_llm_chat_template_format == 'chatml':
        return f"{llm_response}<|im_end|>\n"
    elif local_llm_chat_template_format == 'qwen-chatml':
        return f"{llm_response}<|im_end|>\n"
    elif local_llm_chat_template_format == 'phi3':
        return f"{llm_response}<|end|>\n"
    elif local_llm_chat_template_format == 'phi4':
        return f"{llm_response}<|im_end|>\n"
    elif local_llm_chat_template_format == 'command-r':
        return f"{llm_response}<|END_OF_TURN_TOKEN|>"
    elif local_llm_chat_template_format == 'deepseek':
        return f"{llm_response}\n\n"
    elif local_llm_chat_template_format == 'deepseek-coder-v2':
        return f"{llm_response}<|end_of_sentence|>"
    elif local_llm_chat_template_format == 'vicuna':
        return f"{llm_response} </s>\n"
    elif local_llm_chat_template_format == 'openchat':
        return f"{llm_response}<|end_of_turn|>"
    elif local_llm_chat_template_format == 'gemma2':
        return f"{llm_response}<end_of_turn>\n"
    elif local_llm_chat_template_format == 'mistral-small-v7':
        return f"{llm_response}</s>"
    elif local_llm_chat_template_format == 'mistral-large-v7':
        return f"{llm_response}</s>"
    elif local_llm_chat_template_format == 'raw':
        return f"{llm_response}\n"
    else:
        return False


def determine_if_flux_diffusers_is_enabled() -> bool:
    try:
        hf_read_return = read_hf_config(['flux_diffusers'])
        flux_diffusers = str(hf_read_return['flux_diffusers']).lower() == 'true'
        return flux_diffusers
    except Exception as e:
        return False


def get_hf_waitress_formatted_user_prompt(formatted_user_prompt: str, llm_response: str) -> str:
    history_prompt_json = json.loads(formatted_user_prompt)
    new_response = {"role":"assistant", "content":llm_response}
    history_prompt_json['messages'].append(new_response)
    updated_history_prompt_json = json.dumps(history_prompt_json, indent=4)
    return str(updated_history_prompt_json)


def get_sources_and_pages_for_get_references(docs: list[Document], llm_response: str, llm_filter_citations: bool, upload_folder: str, force_enable_rag: bool, user_query: str) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    if llm_filter_citations:
        try:
            docs = filter_all_citations(docs=docs, llm_response=llm_response, return_top_k=force_enable_rag, user_query=user_query)
        except Exception as e:
            handle_error_no_return("Could not pre-filter citations in get_sources_and_pages_for_get_references(), proceeding without pre-filtering. Encountered error: ", e)
    
    all_sources = {}
    reference_pages = {}
    for doc in docs:
        
        try:
            relevant_page_text = str(doc.page_content)
            relevant_page_number = str(doc.metadata.get('page_number'))
            source_filepath = str(doc.metadata.get('source'))
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue
        
        relevant_page_text = relevant_page_text.replace('\n', ' ')
        
        try:
            source_filename = os.path.basename(source_filepath)
            _, file_extension = os.path.splitext(source_filepath)
        except Exception as e:
            handle_error_no_return("Could not parse path with OS lib, encountered error: ", e)
            continue

        pdf_version_path = os.path.join(upload_folder, os.path.basename(source_filepath).replace('.txt', '.pdf'))   # not catching an error here as os.path.basename(source_filepath) has already been caught just above! Construct the path to the potential PDF version.
        if os.path.exists(pdf_version_path):
            #print("\n\pdf exists\n\n")
            source_filename = source_filename.replace('.txt', '.pdf')
            
            if pdf_version_path in reference_pages:
                reference_pages[pdf_version_path].extend([[relevant_page_text,relevant_page_number]])
            else:
                reference_pages[pdf_version_path] = [[relevant_page_text,relevant_page_number]]

            if source_filename not in all_sources:  # Add this file to our sources dictionary if it's not already present
                source_filepath = pdf_version_path
                all_sources.update({source_filename: source_filepath})

        else:
            print("\n\nNo PDF source doc found (TXT Source) in the 'uploaded_pdfs' dir, RAG ACTIVE BUT REFERENCING WILL NOT DISPLAY!\n\n")
            if source_filename not in all_sources: # Do not duplicate if the TXT file is already in the sources dict
                try:
                    source_filepath = os.path.join(upload_folder, source_filename) # reconstructed path using the OS module just to be safe
                    all_sources.update({source_filename: source_filepath})
                except Exception as e:
                    handle_error_no_return("Could not construct filepath for TXT file, encountered error: ", e)

    return all_sources, reference_pages


def get_refer_pages_and_download_link_html(user_should_refer_pages_in_doc: dict[str, list[list[str]]], stream_session_id: str) -> tuple[str, str]:
    refer_pages_string = "<br><h6>Additional data may be found in the following documents & pages:</h6>"
    
    for index, doc in enumerate(user_should_refer_pages_in_doc, start=1):
        pdf_iframe_id = f"stream{stream_session_id}PdfViewer{str(index)}"
        tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
        frame_doc_path = f"/pdf/{doc}"
        try:
            stream_id_string_to_remove = f"_{stream_session_id}"
            doc_name_without_stream_id = str(doc).replace(stream_id_string_to_remove, "")
            refer_pages_string += f"<br><h6>{doc_name_without_stream_id}: "
            for page in user_should_refer_pages_in_doc[doc]:
                frame_doc_path += f"#page={str(page)}" 
                refer_pages_string += f'<a href="javascript:void(0)" onclick="goToPageAndSwitchTab(\'{pdf_iframe_id}\', \'{frame_doc_path}\', \'tab{tab_name_string}\', \'{stream_session_id}\')">Page {page}</a>, '
                frame_doc_path = f"/pdf/{doc}"
            refer_pages_string = refer_pages_string.strip(', ') + "</h6>"
        except Exception as e:
            handle_error_no_return("Could not construct refer_pages_string, encountered error: ", e)

    pdf_right_pane_id = f"stream{stream_session_id}PdfPane"
    download_link_html = f'<div class="pdf-viewer-container" id="{pdf_right_pane_id}">'

    # Add tab buttons
    download_link_html += '<div class="tab-buttons">'
    for index, source in enumerate(user_should_refer_pages_in_doc, start=1):
        tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
        stream_id_string_to_remove = f"_{stream_session_id}"
        doc_name_without_stream_id = str(source).replace(stream_id_string_to_remove, "")
        default_open = ' defaultTabs' if index == 1 else ''
        download_link_html += f'<button class="tab-button{default_open}" stream-session-id="{stream_session_id}" onclick="openTab(event, \'tab{tab_name_string}\', \'{stream_session_id}\')">{doc_name_without_stream_id}</button>'
    download_link_html += '</div>'

    # Add tab content
    for index, source in enumerate(user_should_refer_pages_in_doc, start=1):
        try:
            download_link_url = url_for('download_file', filename=source)
            pdf_iframe_id = f"stream{stream_session_id}PdfViewer{str(index)}"
            tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
            download_link_html += f'<div id="tab{tab_name_string}" class="tab-content" stream-session-id="{stream_session_id}">'
            download_link_html += f'<iframe id="{pdf_iframe_id}" src="{download_link_url}" width="100%" height="600"></iframe>'
            download_link_html += "</div>"
        except Exception as e:
            handle_error_no_return("Could not construct download_link_html, encountered error: ", e)

    download_link_html += "</div>"

    return refer_pages_string, download_link_html


def get_model_response_for_history_db_for_get_references(download_link_html: str, llm_response: str, reference_response: str) -> str:
    model_response_for_history_db = str(llm_response)
    model_response_for_history_db += f"\n\n{reference_response}"
    model_response_for_history_db += f"\n\npdf_pane_data={download_link_html}"
    model_response_for_history_db = model_response_for_history_db.strip('\n')
    return model_response_for_history_db


@app.route('/get_references', methods=['POST'])
def get_references():

    print("\n\nStoring History Post-Response -- Determining if Citations are Necessary\n\n")

    try:
        local_llm_server, upload_folder, local_llm_chat_template_format, llm_filter_citations, force_enable_rag, exl2_prompt_template_format = read_config_for_get_references()
        exl2 = read_hf_config(['exl2'])['exl2']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to get_references. Error: ", e)

    try:
        stream_session_id, user_query, llm_response, formatted_user_prompt, chat_id, sequence_id, regeneration_request, regenerate_with_citations_force_enabled = get_request_parameters_for_get_references(request)
    except Exception as e:
        return handle_api_error("Could not read request content in method get_references, encountered error: ", e)

    do_rag = False
    try:
        docs, do_rag = get_vector_results_for_get_references(stream_session_id)
    except Exception as e:
        handle_error_no_return("Error determining if RAG was used in method get_references - Could not check the QUERIES dict. Proceeding without RAG. Encountered error: ", e)

    if local_llm_server == 'llama-cpp':
        formatted_user_prompt += get_llama_cpp_formatted_user_prompt(local_llm_chat_template_format, llm_response)
    elif local_llm_server == 'hf-waitress':
        local_llm_chat_template_format = "hf-transformers"
        flux_diffusers = determine_if_flux_diffusers_is_enabled()
        if flux_diffusers:
            do_rag = False
        else:
            if exl2:
                formatted_user_prompt += get_llama_cpp_formatted_user_prompt(exl2_prompt_template_format, llm_response)
            else:
                formatted_user_prompt = get_hf_waitress_formatted_user_prompt(formatted_user_prompt, llm_response)

    if not do_rag:
        print("\n\nRAG Citations unnecessary, storing chat history and returning\n\n")
        try:
            if not regeneration_request:
                stored_datetime, chat_id = store_local_llm_chat_history_to_db(chat_id, sequence_id, stream_session_id, user_query, llm_response, formatted_user_prompt, local_llm_server, local_llm_chat_template_format)
            else:
                stored_datetime, chat_id = update_llm_response_in_history_db(chat_id, stream_session_id, user_query, llm_response)
        except Exception as e:
            handle_error_no_return("Could not store or update chat history DB in get_references(), encountered error: ", e)
        return jsonify({'success': True, 'stored_datetime':stored_datetime, 'local_llm_server':local_llm_server, 'local_llm_chat_template_format':local_llm_chat_template_format, 'chat_id':chat_id})
    

    print("\n\nFetching Citations\n\n")

    all_sources = {}
    reference_pages = {}
    try:
        all_sources, reference_pages = get_sources_and_pages_for_get_references(docs, llm_response, llm_filter_citations, upload_folder, (force_enable_rag or regenerate_with_citations_force_enabled), user_query)
    except Exception as e:
        return handle_api_error("Could not get sources and pages for get_references(), encountered error: ", e)
    
    try:
        docs_have_relevant_info, user_should_refer_pages_in_doc = highlighter_interface(reference_pages, stream_session_id)
    except Exception as e:
        handle_error_no_return("Could not complete highlighter_interface, encountered error: ", e)
    
    reference_response = ""
    download_link_html = ""
    if docs_have_relevant_info:
        try:
            reference_response, download_link_html = get_refer_pages_and_download_link_html(user_should_refer_pages_in_doc, stream_session_id)
        except Exception as e:
            handle_error_no_return("Could not get refer_pages_string and download_link_html, encountered error: ", e)
    
    try:
        model_response_for_history_db = get_model_response_for_history_db_for_get_references(download_link_html, llm_response, reference_response)
    except Exception as e:
        handle_error_no_return("Could not prep data to store to chat history DB in get_references(), encountered error: ", e)

    try:
        if not regeneration_request:
            stored_datetime, chat_id = store_local_llm_chat_history_to_db(chat_id, sequence_id, stream_session_id, user_query, model_response_for_history_db, formatted_user_prompt, local_llm_server, local_llm_chat_template_format)
        else:
            stored_datetime, chat_id = update_llm_response_in_history_db(chat_id, stream_session_id, user_query, model_response_for_history_db)
    except Exception as e:
        handle_error_no_return("Could not store or update chat history DB in get_references(), encountered error: ", e)

    return jsonify({'success': True, 'response': reference_response, 'pdf_frame':download_link_html, 'stored_datetime':stored_datetime, 'local_llm_server':local_llm_server, 'local_llm_chat_template_format':local_llm_chat_template_format, 'chat_id':chat_id})



def parse_arguments():

    try:
        parser = argparse.ArgumentParser(description="Server for HuggingFace Transformers models")
    except Exception as e:
        handle_local_error("Could not create parser to parse_arguments(), proceeding with defaults. Encountered error: ", e)

    # Even if a parser object could not be created, a read_request will write & return defaults
    try:
        read_return = read_config(['lars_host', 'lars_port', 'hf_waitress_access_url', 'hf_waitress_serving_url', 'hf_waitress_server_port', 'llama_cpp_access_url', 'llama_cpp_serving_url', 'llama_cpp_server_port'])
        lars_host = str(read_return['lars_host'])
        lars_port = int(read_return['lars_port'])
        hf_waitress_access_url = str(read_return['hf_waitress_access_url'])
        hf_waitress_serving_url = str(read_return['hf_waitress_serving_url'])
        hf_waitress_server_port = int(read_return['hf_waitress_server_port'])
        llama_cpp_access_url = str(read_return['llama_cpp_access_url'])
        llama_cpp_serving_url = str(read_return['llama_cpp_serving_url'])
        llama_cpp_server_port = int(read_return['llama_cpp_server_port'])
    except Exception as e:
        handle_error_no_return("Could not get host and port from hf_config.json, encountered error: ", e)

    if parser:
        parser.add_argument("--reset_to_defaults", action="store_true", default=False, help="Use default settings")
        parser.add_argument("--lars_host", type=str, default=lars_host, help="Specify the host to be used by the server. Remembers previously set value and falls-back to 0.0.0.0 as a default.")
        parser.add_argument("--lars_port", type=int, default=lars_port, help="Specify the port to be used by the server. Remembers previously set value and falls-back to 5000 as a default.")
        parser.add_argument("--hf_waitress_access_url", type=str, default=hf_waitress_access_url, help="Specify the access URL to be used by the HF-Waitress server. Remembers previously set value and falls-back to localhost as a default.")
        parser.add_argument("--hf_waitress_serving_url", type=str, default=hf_waitress_serving_url, help="Specify the serving URL to be used by the HF-Waitress server. Remembers previously set value and falls-back to 0.0.0.0 as a default.")
        parser.add_argument("--hf_waitress_server_port", type=int, default=hf_waitress_server_port, help="Specify the port to be used by the HF-Waitress server. Remembers previously set value and falls-back to 9069 as a default.")
        parser.add_argument("--llama_cpp_access_url", type=str, default=llama_cpp_access_url, help="Specify the access URL to be used by the Llama-CPP server. Remembers previously set value and falls-back to localhost as a default.")
        parser.add_argument("--llama_cpp_serving_url", type=str, default=llama_cpp_serving_url, help="Specify the serving URL to be used by the Llama-CPP server. Remembers previously set value and falls-back to 0.0.0.0 as a default.")
        parser.add_argument("--llama_cpp_server_port", type=int, default=llama_cpp_server_port, help="Specify the port to be used by the Llama-CPP server. Remembers previously set value and falls-back to 8080 as a default.")

        args = parser.parse_args()
        # print(f"\n\nparser.parse_args():\n\n{args}\n\n")

        if args.reset_to_defaults:
            print("\n\nLoading Server with Safe Defaults\n\n")
            try:
                # Empty hf_config.json
                with open('hf_config.json', 'w') as file:
                    json.dump({}, file, indent=4)
                
                # Set defaults
                read_config([
                    'lars_host',
                    'lars_port',
                    'hf_waitress_access_url',
                    'hf_waitress_serving_url',
                    'hf_waitress_server_port',
                    'llama_cpp_access_url',
                    'llama_cpp_serving_url',
                    'llama_cpp_server_port'
                ])
            except Exception as e:
                handle_local_error("Could not reset hosts and ports in config.json, encountered error: ", e)

        else:
            try:
                write_config({
                    'lars_host':args.lars_host,
                    'lars_port':args.lars_port,
                    'hf_waitress_access_url':args.hf_waitress_access_url,
                    'hf_waitress_serving_url':args.hf_waitress_serving_url,
                    'hf_waitress_server_port':args.hf_waitress_server_port,
                    'llama_cpp_access_url':args.llama_cpp_access_url,
                    'llama_cpp_serving_url':args.llama_cpp_serving_url,
                    'llama_cpp_server_port':args.llama_cpp_server_port
                })
            except Exception as e:
                handle_local_error("Could not write hosts and ports to config.json, encountered error: ", e)

        return args

    # Return None if parser was not created
    return None


def get_host_and_port():
    try:
        read_return = read_config(['lars_host', 'lars_port'])
        lars_host = str(read_return['lars_host'])
        lars_port = int(read_return['lars_port'])
        return lars_host, lars_port
    except Exception as e:
        handle_error_no_return("Could not get host and port from hf_config.json, encountered error: ", e)


if __name__ == '__main__':
    _ = parse_arguments()
    lars_host, lars_port = get_host_and_port()
    print(f"\n\nServing LARS-Enterprise on {lars_host} port {lars_port}\n\n")
    # app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000)
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024 * 1024  # 100GB in bytes upload limit
    serve(app, host=lars_host, port=lars_port, max_request_body_size=MAX_UPLOAD_SIZE)