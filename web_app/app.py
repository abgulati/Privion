from flask import Flask, render_template, request, redirect, url_for, Response
from flask import send_from_directory
from flask_cors import CORS
from flask import jsonify

from sentence_transformers import SentenceTransformer, util
import torch

from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from falkordb import FalkorDB
import chromadb

from pdf2image import convert_from_path

from urllib.parse import urlparse, parse_qs
from urlextract import URLExtract

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
import ast
import os
import io
import re
import gc
from logging.handlers import RotatingFileHandler

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, Or
from whoosh import scoring

from waitress import serve

import fitz # PyMuPDF

try:
    # Standard Pipeline
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, 
        TableFormerMode,
        EasyOcrOptions,
        TesseractOcrOptions,
        TesseractCliOcrOptions,
        OcrMacOptions,
        RapidOcrOptions,
    )

    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # VLM Pipeline
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel import vlm_model_specs
    from docling.pipeline.vlm_pipeline import VlmPipeline
except Exception as e:
    print(f"Could not import Docling OCR, skipping. If not installed, please run `pip install docling`. Encountered error: {e}")

try:
    from graph_clustering import apply_leiden_clustering
except Exception as e:
    print(f"Could not import graph_clustering (likely not installed), skipping. Encountered error: {e}")

try:
    import prompt_formatting as prompt_formatting_module
except ImportError:
    print("WARNING: Prompt Formatter module `prompt_formatting.py` is not present. Skipping import. Exl2 and llama.cpp will not work!")

try:
    import utils
except ImportError:
    print("WARNING: utils.py is not present. Skipping import.")

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
LLM = None
LOADED_UP = False
LLM_LOADED_UP = False
LLM_CHANGE_RELOAD_TRIGGER_SET = False
DOCLING_CONVERTER = None

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
    LOGGER = logging.getLogger('my_logger')
    LOGGER.setLevel(logging.ERROR)

    # 2 - Create a RotatingFileHandler
    # maxBytes: max file size of log file after which a new file is created; set to 1024 * 1024 * 5 for 5MB: 1024x1024 is 1MB, then a multiplyer for the number of MB
    # backupCount: number of backup files to keep specifying how many old log files to keep
    handler = RotatingFileHandler('lars_server_log.log', maxBytes=1024*1024*5, backupCount=2)
    handler.setLevel(logging.ERROR)

    # 3 - Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)

    # 4 - Add the handler to the logger
    LOGGER.addHandler(handler)
    # Logger ready! Usage: logger.error(f"This is an error message with error {e}")
except Exception as e:
    print(f"\n\nCould not establish logger, encountered error: {e}")


def central_error_logging(message, exception=None):
    error_message = f"{message} {str(exception) if exception else '; No exception info.'}".strip()
    #traceback_details = traceback.format_exc()
    #full_message = f"\n\n{error_message}\n\nTraceback: {traceback_details}\n\n"
    full_message = f"\n\n{error_message}\n\n"
    if LOGGER:
        LOGGER.error(full_message)
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



############################------------configuration manager-------------###############################

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
                'upload_staging_folder':base_directory + '/upload_staging',
                'upload_staging_db':base_directory + '/upload_staging.db',
                'knowledge_domain_base_directory': base_directory + '/knowledge_domains',
                'graph_db_data_directory': base_directory + '/graph_db_data',
                'graph_models_base_directory_name': 'graph-model-servers',
                'graph_extraction_model_directory_name': 'graph-extraction-model-server',
                'graph_summary_generator_directory_name': 'graph-summary-generator-server',
                'local_llm_server':'hf-waitress',
                'model_choice':'Meta-Llama-3-8B-Instruct.f16.gguf',
                'vision_llm_local_url':"http://localhost:9069/completions",
                'kosmos_local_url':"http://localhost:25000",
                'kosmos_task':'ocr',
                'kosmos_threshold':30,
                'kosmos_offload_vram':True,
                'kosmos_container_name':'kosmos-2.5',
                'min_char_threshold_for_backup_ocr':1000,
                'minimum_free_vram_for_kosmos_ocr':10240,
                'ocr_service_choice':'Docling',
                'backup_ocr_service_choice':'Backup-Docling',
                'docling_pipeline':'standard',
                'docling_vlm_model':'phi4_transformers',
                'docling_ocr_model':'easyocr',
                'docling_do_ocr':True,
                'docling_do_code_enrichment':False,
                'docling_do_formula_enrichment':False,
                'docling_do_table_structure':True,
                'docling_do_picture_classification':False,
                'docling_do_picture_description':False,
                'docling_table_structure_mode':'accurate',
                'docling_do_cell_matching':True,
                'docling_cuda_use_flash_attention_2':False,
                'docling_force_full_page_ocr':False,
                'docling_num_threads':4,
                'force_ocr':False,
                'lars_host':'0.0.0.0',
                'lars_port':5000,
                'hf_waitress_serving_url':'0.0.0.0',    # the serving URL is where the HF-Waitress server is listening for requests, and is specified in the serve() launch command of the Flask/Waitress WSGI server. 0.0.0.0 means all interfaces.
                'hf_waitress_access_url':'localhost',   # the access URL is the URL that the HF-Waitress server is accessible to clients for API calls, localhost means only from the local machine.
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
                'fetch_top_k_results_from_vectordb':50,
                'filter_top_k_results_by_reranking':11,
                'min_semantic_similarity_threshold':0.5,
                'min_lexical_similarity_threshold':3.0,
                'chunk_size':250,
                'chunk_overlap':0,
                'perform_graph_rag':True,
                'perform_only_graph_rag':False,
                'upload_doc_to_graph_db':True,
                'graph_chunk_size':1500,    # Larger chunks are better because they provide more context for the model to identify meaningful entities and relationships
                'graph_chunk_overlap':300,  # 20% overlap for a 1500 char chunk: provides very reasonable overlap while not being too redundant.
                'graph_generator_model':'Metin/Gemma-2-2B-TR-Knowledge-Graph',
                'graph_model_server_port':9070,
                'graph_model_access_url':'localhost',
                'quantize_graph_model':'n',
                'quantize_graph_model_bits':'int8',
                'exl2_quantize_graph_model':True,
                'exl2_quantize_graph_model_bpw':8.0,
                'graph_summarizer_model':'google/gemma-2-2b-it',
                'graph_summarizer_model_prompt_template_format':'gemma2',
                'graph_summarizer_server_port':9071,
                'graph_summarizer_access_url':'localhost',
                'quantize_graph_summarizer_model':'n',
                'quantize_graph_summarizer_model_bits':'int8',
                'exl2_quantize_graph_summarizer_model':True,
                'exl2_quantize_graph_summarizer_model_bpw':8.0,
                'graph_db_server_host':'localhost',
                'assign_host_port_to_graph_db_server':6379,
                'assign_host_port_to_graph_db_ui':3000,
                'launch_graph_db_with_ui':True,
                'apply_clustering_to_graph_db_on_doc_load': False,
                'graph_model_max_new_tokens':4096,
                'graph_model_max_seq_len':15360,
                'graph_model_temperature':0.1,
                'graph_model_do_sample':True,
                'graph_model_top_k':40,
                'graph_model_top_p':0.95,
                'graph_model_min_p':0.05,
                'minimum_free_vram_for_graph_extraction_model':7168,
                'graph_summarizer_max_new_tokens':8192,
                'graph_summarizer_max_seq_len':15360,
                'graph_summarizer_temperature':0.15,
                'graph_summarizer_do_sample':True,
                'graph_summarizer_top_k':40,
                'graph_summarizer_top_p':0.95,
                'graph_summarizer_min_p':0.05,
                'minimum_free_vram_for_graph_summarizer_model':7168,
                'skip_summary_generation':False,
                'reuse_previously_extracted_graph_entities_and_relationships':True,
                'reuse_previously_generated_graph_summaries':True,
                'graph_rag_context_length_limit_chars':25000,
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
                'selected_knowledge_domain':'General'
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

############################----------------------------------------------###############################


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
    read_return = read_config([
        'model_dir', 'highlighted_docs', 'upload_folder', 'ocr_pdfs', 'pdfs_to_txts', 
        'docs_to_knowledge_graph_dir', 'upload_staging_folder', 'graph_db_data_directory'])
    model_dir = read_return['model_dir']
    highlighted_docs = read_return['highlighted_docs']
    upload_folder = read_return['upload_folder']
    ocr_pdfs = read_return['ocr_pdfs']
    pdfs_to_txts = read_return['pdfs_to_txts']
    docs_to_knowledge_graph_dir = read_return['docs_to_knowledge_graph_dir']
    upload_staging_folder = read_return['upload_staging_folder']
    graph_db_data_directory = read_return['graph_db_data_directory']
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

try:
    os.makedirs(upload_staging_folder, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create upload_staging_folder, encountered error: ", e)

try:
    os.makedirs(graph_db_data_directory, exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create graph_db_data_directory, encountered error: ", e)

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['DOWNLOAD_FOLDER'] = highlighted_docs
app.config['UPLOAD_STAGING_FOLDER'] = upload_staging_folder

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

############################----------------------------------------------###############################



############################------------File & Folder Management-------------###############################

def load_json_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            return handle_local_error("Could not load JSON file, encountered error: ", e)
    else:
        return None   # NOTE: isinstance({}, dict) will return True so better to return None!


def update_and_save_json_file(data, file_path):
    current_cache = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                current_cache = json.load(file)
        except Exception as e:
            return handle_local_error("Could not save JSON file, encountered error: ", e)
    
    try:
        current_cache.update(data)
        with open(file_path, 'w') as file:
            json.dump(current_cache, file, indent=4)
    except Exception as e:
        return handle_local_error("Could not save JSON file, encountered error: ", e)

    return True


def overwrite_json_file(data, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        return handle_local_error("Could not save JSON file, encountered error: ", e)

    return True


def remove_file_from_filepath(filepath):
    print(f"\n\nRemoving file from filepath: {filepath}\n\n")
    try:
        os.remove(filepath)
        print(f"Successfully deleted file: {filepath}")
    except Exception as e:
        return handle_local_error(f"Could not remove file from filepath: {filepath}, encountered error: ", e)


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
        return handle_local_error(f"Could not remove folder from filepath: {folderpath}, encountered error: ", e)
    

def safe_remove_folder_from_filepath(folderpath):
    try:
        remove_folder_from_filepath(folderpath)
    except Exception as e:
        handle_error_no_return(f"Could not remove folder from filepath: {folderpath}, encountered error: ", e)

############################----------------------------------------------###############################

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
        source_link=ID(stored=True),
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
                source_link=chunk['source_link'],
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
                    'source_link': result['source_link'],
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
                        'source_link': result['source_link'],
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
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for Azure Document Intelligence OCR, please provide required API config. Error: ", e)

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
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for Azure OCR, please provide required API config. Error: ", e)

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
        hf_waitress_base_url = get_url_for_server('hf-waitress')
        if not utils.is_local_server_online(hf_waitress_base_url)['server_available']:
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
        handle_local_error("Missing OCR PDFs directory for Vision LLM-based OCR, please provide required API config. Error: ", e)

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


def get_container_id_by_container_name(container_name):
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
        return handle_local_error(f"Could not check if {container_name} Docker container is running, encountered error: ", e)
    

def check_if_container_is_running(container_name):
    try:
        container_id = get_container_id_by_container_name(container_name)
        return container_id is not None and container_id != ""
    except Exception as e:
        handle_error_no_return(f"Could not check if {container_name} Docker container is running, encountered error: ", e)
        return False


def start_kosmos_container():
    try:
        read_return = read_config(['kosmos_container_name', 'minimum_free_vram_for_kosmos_ocr'])
        kosmos_container_name = read_return['kosmos_container_name']
        minimum_free_vram_for_kosmos_ocr = read_return['minimum_free_vram_for_kosmos_ocr']
    except Exception as e:
        return handle_local_error("Could not read Kosmos container name from config.json, encountered error: ", e)
    
    try:
        hf_waitress_base_url = get_url_for_server('hf-waitress')
        url_config = read_config(['graph_model_access_url', 'graph_model_server_port', 'graph_summarizer_access_url', 'graph_summarizer_server_port'])
        graph_model_base_url = f"http://{url_config['graph_model_access_url']}:{url_config['graph_model_server_port']}"
        graph_summarizer_base_url = f"http://{url_config['graph_summarizer_access_url']}:{url_config['graph_summarizer_server_port']}"
        utils.ensure_minimum_free_vram(minimum_free_vram_for_kosmos_ocr, [graph_summarizer_base_url, graph_model_base_url, hf_waitress_base_url])   
        # Graph-Extraction model is used for GraphRAG responses, so we try shutting down the summarizer model first!
    except Exception as e:
        return handle_local_error(f"Could not ensure minimum free GPU memory ({minimum_free_vram_for_kosmos_ocr}MB) required for Kosmos OCR, encountered error: ", e)

    # Check if Docker Engine is running
    try:
        subprocess.run(['docker', 'info'], capture_output=True, check=True)  # check=True will raise an exception if the command returns a non-zero exit code
    except Exception as e:
        return handle_local_error("Docker Engine is not running, encountered error: ", e)

    print("\nDocker Engine is running, proceeding with FalkorDB Docker container launch...\n")

    if check_if_container_is_running(kosmos_container_name):
        print("\nKosmos Docker container is already running, skipping launch...\n")
        return True

    command = ['docker', 'start', f'{kosmos_container_name}']

    try:
        subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE) if platform.system() == 'Windows' else subprocess.Popen(command, shell=True)
        # Check if the container is running
        timeout = 5
        attempts = 50
        for _ in range(attempts):
            if check_if_container_is_running(kosmos_container_name):
                print(f"\nKosmos Docker container launched successfully! Returning in {timeout} seconds...\n")
                time.sleep(timeout) # small delay to ensure the container is fully started
                return True
            else:
                print(f"Kosmos Docker container not yet running, waiting {timeout} seconds before retrying...")
                time.sleep(timeout)

    except Exception as e:
        return handle_local_error("Could not launch Kosmos Docker container, encountered error: ", e)

    return True


def get_kosmos_request_params():
    try:
        read_return = read_config(['kosmos_local_url', 'kosmos_task', 'kosmos_threshold', 'kosmos_offload_vram'])
        kosmos_local_url = read_return['kosmos_local_url'] + '/infer_file_stream'
        kosmos_task = read_return['kosmos_task']
        kosmos_threshold = read_return['kosmos_threshold']
        kosmos_offload_vram = str(read_return['kosmos_offload_vram']).lower() == 'true'
    except Exception as e:
        handle_local_error("Missing Kosmos API config, please provide required API config. Error: ", e)

    payload = {
        'task': kosmos_task,
        'threshold': kosmos_threshold,
        'offload_vram': kosmos_offload_vram
    }

    headers = {}

    return kosmos_local_url, payload, headers


def kosmos_ocr_page(page_as_pil_image_object, retry_count=0):
    print(f"\n\nProcessing Page - Kosmos OCR\n\n")
    try:
        kosmos_local_url, payload, headers = get_kosmos_request_params()
    except Exception as e:
        return handle_local_error("Could not get Kosmos request parameters, encountered error: ", e)
    
    try:
        print("\n\nConverting PIL-image object to Byte-Stream\n\n")
        img_stream = io.BytesIO()
        page_as_pil_image_object.save(img_stream, format='PNG')
        img_stream.seek(0)
    except Exception as e:
        return handle_local_error("Could not convert PIL-image object to Byte-Stream, encountered error: ", e)
    
    try:
        print("Preparing file payload for Kosmos\n")
        file_payload = [
            ('file', ('page-image.png', img_stream, 'image/png'))
        ]
    except Exception as e:
        return handle_local_error("Could not prepare file payload for Kosmos, encountered error: ", e)
    
    try:
        start_kosmos_container()    # Will do nothing if the container is already running!
    except Exception as e:
        return handle_local_error("Could not start Kosmos Docker container, encountered error: ", e)
    
    try:
        print("\nSending request to Kosmos\n")
        '''
        # 100 seconds is more than enough for Kosmos to be loaded and process the request! 
        # NOTE: Does NOT account for model download time! Ensure the service has been previously run so the models have been downloaded.
        '''
        with requests.post(kosmos_local_url, headers=headers, data=payload, files=file_payload, stream=True, timeout=100) as response:
            response.raise_for_status() # Raise an exception for bad 4xx or 5xx status codes

            print("\nReceiving event-streaming response from Kosmos\n")
            for event in response.iter_lines(decode_unicode=True):
                if event:
                    if event.startswith('data:'):
                        event_data = event[5:].strip()
                        try:
                            json_data = json.loads(event_data)
                            if 'full_parsed_text' in json_data:
                                return json_data['full_parsed_text']
                            else:
                                print(f"\n\nReceived plain-text event from Kosmos: {event_data}\n\n")
                        except json.JSONDecodeError as e:
                            print(f"\n\nCould not parse event from Kosmos as JSON dictionary, encountered error: {e}\n\n")
                            print(f"\n\nReceived plain-text event from Kosmos: {event}\n\n")
                        except Exception as e:
                            return handle_local_error("Could not process event from Kosmos, encountered error: ", e)
    except requests.exceptions.RequestException as e:
        handle_error_no_return(f"Could not send request to Kosmos, checking if Kosmos is running and attempting to start the service if not. Retry attempt {retry_count+1} of 3. Encountered error: ", e)
        if retry_count >= 3:
            return handle_local_error("Failed to receive a proper response from the Kosmos OCR service even after 3 retries, stopping execution. Encountered error: ", e)
        else:
            try:
                start_kosmos_container()    # Will do nothing if the container is already running!
                return kosmos_ocr_page(page_as_pil_image_object, retry_count=retry_count+1)
            except Exception as e:
                return handle_local_error("Could not start Kosmos Docker container, encountered error: ", e)
    except Exception as e:
        return handle_local_error("Could not receive a complete response from the Kosmos OCR service, encountered error: ", e)
    finally:
        if 'img_stream' in locals() and img_stream:
            img_stream.close()  # io.BytesIO objects are in-memory streams, and while the GC will eventually close them, it's best to do it explicitly here.


def PDFtoKosmosOCRTXT(input_pdf_filepath):
    print("\n\nProcessing Document - PDF to Kosmos OCR TXT\n\n")

    try:
        read_return = read_config(['ocr_pdfs', 'force_extract_previously_extracted_text'])
        ocr_pdfs = read_return['ocr_pdfs']
        force_extract_previously_extracted_text = str(read_return['force_extract_previously_extracted_text']).lower() == 'true'
    except Exception as e:
        return handle_local_error("Missing OCR PDFs directory for Kosmos OCR, please provide required API config. Error: ", e)

    try:
        source_filename = os.path.basename(input_pdf_filepath)
    except Exception as e:
        return handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")   # Using this instead of os.path.splitext() in case filename contains a period
    output_text_file_path = os.path.join(ocr_pdfs, output_text_file_name)

    if os.path.exists(output_text_file_path) and not force_extract_previously_extracted_text:
        if os.path.getsize(output_text_file_path) > 0:
            print("Kosmos OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("Kosmos OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")

    # Convert PDF to  a list of images
    pil_image_object_list = []
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pil_image_object_list = convert_from_path(input_pdf_filepath, 300) # The convert_from_path() function from pdf2image lib intertnally uses Poppler to convert PDF pages to images, and then creates PIP Image objects from them. 300dpi - good balance between quality and performance
    except Exception as e:
        return handle_local_error("Could not image PDF file, encountered error: ", e)
    
    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        return handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    # Initialize page number
    page_number = 0
    for count, image in enumerate(pil_image_object_list, start=1):
        try:
            page_number = count
            full_parsed_text = kosmos_ocr_page(image)
            output_text_file.write(f"[PAGE:{page_number}]\n{full_parsed_text}\n")
        except Exception as e:
            handle_error_no_return("Could not process page, encountered error: ", e)
            continue
    
    # Close & return
    output_text_file.close()
    return output_text_file_path


def get_docling_ocr_model(model_name_string):
    try:
        if model_name_string == 'easyocr':
            return EasyOcrOptions()
        
        if model_name_string == 'tesseract':
            return TesseractOcrOptions()
        
        if model_name_string == 'tesseract_cli':
            return TesseractCliOcrOptions()
        
        if model_name_string == 'ocrmac':
            return OcrMacOptions()
        
        if model_name_string == 'rapidocr':
            return RapidOcrOptions()
        
    except Exception as e:
        return handle_local_error("Could not get Docling OCR model, encountered error: ", e)
        

def get_docling_vlm_model(model_name_string):
    try:
        if model_name_string == 'smoldocling_mlx':
            return vlm_model_specs.SMOLDOCLING_MLX
        
        if model_name_string == 'smoldocling_transformers':
            return vlm_model_specs.SMOLDOCLING_TRANSFORMERS
        
        if model_name_string == 'granite_vision_transformers':
            return vlm_model_specs.GRANITE_VISION_TRANSFORMERS
        
        if model_name_string == 'granite_vision_ollama':
            return vlm_model_specs.GRANITE_VISION_OLLAMA

        if model_name_string == 'pixtral_12b_transformers':
            return vlm_model_specs.PIXTRAL_12B_TRANSFORMERS
        
        if model_name_string == 'pixtral_12b_mlx':
            return vlm_model_specs.PIXTRAL_12B_MLX
        
        if model_name_string == 'phi4_transformers':
            return vlm_model_specs.PHI4_TRANSFORMERS
        
        if model_name_string == 'qwen25_vl_3b_mlx':
            return vlm_model_specs.QWEN25_VL_3B_MLX
        
        if model_name_string == 'gemma3_12b_mlx':
            return vlm_model_specs.GEMMA3_12B_MLX
        
        if model_name_string == 'gemma3_27b_mlx':
            return vlm_model_specs.GEMMA3_27B_MLX
        
    except Exception as e:
        return handle_local_error("Could not get Docling VLM model, encountered error: ", e)


def get_docling_config():
    try:
        return read_config(
            [
                'docling_pipeline',
                'docling_vlm_model',
                'docling_ocr_model',
                'docling_do_ocr',
                'docling_do_code_enrichment',
                'docling_do_formula_enrichment',
                'docling_do_table_structure',
                'docling_do_picture_classification',
                'docling_do_picture_description',
                'docling_table_structure_mode',
                'docling_do_cell_matching',
                'docling_cuda_use_flash_attention_2',
                'docling_num_threads',
                'docling_force_full_page_ocr'
            ]
        )
    except Exception as e:
        return handle_local_error("Could not read Docling config, encountered error: ", e)


def get_docling_converter(docling_config):
    try:

        if docling_config['docling_pipeline'] == 'vlm':
            
            # a. Set VLM Pipeline Options
            global VLM_PIPELINE_OPTIONS
            VLM_PIPELINE_OPTIONS = None
            VLM_PIPELINE_OPTIONS = VlmPipelineOptions()
            VLM_PIPELINE_OPTIONS.vlm_options = get_docling_vlm_model(docling_config['docling_vlm_model'])

            # b. VLM Converter
            vlm_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=VLM_PIPELINE_OPTIONS,
                    ),
                }
            )

            return vlm_converter

        # Standard Pipeline
        # a. Set PDF Pipeline Options
        global PDF_PIPELINE_OPTIONS
        PDF_PIPELINE_OPTIONS = None
        PDF_PIPELINE_OPTIONS = PdfPipelineOptions()
        PDF_PIPELINE_OPTIONS.do_ocr = str(docling_config['docling_do_ocr']).lower() == 'true'
        PDF_PIPELINE_OPTIONS.do_code_enrichment = str(docling_config['docling_do_code_enrichment']).lower() == 'true'
        PDF_PIPELINE_OPTIONS.do_formula_enrichment = str(docling_config['docling_do_formula_enrichment']).lower() == 'true'
        PDF_PIPELINE_OPTIONS.do_table_structure = str(docling_config['docling_do_table_structure']).lower() == 'true'
        PDF_PIPELINE_OPTIONS.do_picture_classification = str(docling_config['docling_do_picture_classification']).lower() == 'true'
        PDF_PIPELINE_OPTIONS.do_picture_description = str(docling_config['docling_do_picture_description']).lower() == 'true'
        PDF_PIPELINE_OPTIONS.table_structure_options.mode = TableFormerMode.ACCURATE if str(docling_config['docling_table_structure_mode']) == 'accurate' else TableFormerMode.FAST
        PDF_PIPELINE_OPTIONS.table_structure_options.do_cell_matching = str(docling_config['docling_do_cell_matching']).lower() == 'true'
        PDF_PIPELINE_OPTIONS.accelerator_options = AcceleratorOptions(
            num_threads = int(docling_config['docling_num_threads']),
            device = AcceleratorDevice.AUTO,
            # cuda_use_flash_attention_2 = str(docling_config['docling_cuda_use_flash_attention_2']).lower() == 'true'
        )

        # b. Set OCR Options
        ocr_options = get_docling_ocr_model(str(docling_config['docling_ocr_model']))
        ocr_options.force_full_page_ocr = str(docling_config['docling_force_full_page_ocr']).lower() == 'true'
        PDF_PIPELINE_OPTIONS.ocr_options = ocr_options

        # c. Initialize converter and process
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PDF_PIPELINE_OPTIONS,
                )
            }
        )
        
        return converter

    except Exception as e:
        return handle_local_error("Could not get Docling converter, encountered error: ", e)


def docling_ocr_page(page_as_pdf_bytes, page_number, retry_count=0):
    '''
    OCR a single page using Docling
    '''
    global DOCLING_CONVERTER
    try:
        # Create Document-Stream object from bytes
        buf = io.BytesIO(page_as_pdf_bytes)
        source = DocumentStream(name=f"page_{page_number}.pdf", stream=buf)

        # Get Docling converter
        docling_config = get_docling_config()
        DOCLING_CONVERTER = get_docling_converter(docling_config) if DOCLING_CONVERTER is None else DOCLING_CONVERTER

        # Extract text and return the result
        result = DOCLING_CONVERTER.convert(source=source)
        return str(result.document.export_to_markdown())

    except Exception as e:
        if retry_count < 3:
            print(f"Retrying Docling OCR for page {page_number}, retry attempt {retry_count+1} of 3. Encountered error: {e}")
            return docling_ocr_page(page_as_pdf_bytes, page_number, retry_count + 1)
        else:
            return handle_local_error("Failed to receive a proper response from the Docling OCR service even after 3 retries, stopping execution. Encountered error: ", e)


def PDFtoDoclingOCRTXT(input_pdf_filepath):
    '''
    OCR PDFs using Docling by iterating through each page, converting to a binary stream and then invoking `dolcing_ocr_page()`
    '''

    print("\n\nProcessing Document - PDF to Docling OCR TXT\n\n")

    try:
        read_return = read_config(['force_extract_previously_extracted_text', 'ocr_pdfs'])
        ocr_pdfs = read_return['ocr_pdfs']
        force_extract_previously_extracted_text = str(read_return['force_extract_previously_extracted_text']).lower() == 'true'
    except Exception as e:
        return handle_local_error("Could not read required values from config.json when attempting to convert PDF to TXT, encountered error: ", e)
    
    try:
        source_filename = os.path.basename(input_pdf_filepath)
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

    # Open with PyMuPDF for conversion to binary byte stream
    try:
        pdf_document = fitz.open(input_pdf_filepath)
        pdf_document_length = len(pdf_document)
    except Exception as e:
        return handle_local_error("Could not open PDF file, encountered error: ", e)
    
    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        return handle_local_error("Could not initialize/access output text file, encountered error: ", e)
    
    # Iterate through each page and OCR
    for page_number in range(pdf_document_length):
        try:
            print(f"\n\nProcessing Page: {page_number + 1} of {pdf_document_length} from file: {source_filename}\n\n")

            # Extract single page as a new PDF
            single_page_pdf = fitz.open()
            single_page_pdf.insert_pdf(pdf_document, from_page=page_number, to_page=page_number)

            # Convert to bytes
            single_page_pdf_bytes = single_page_pdf.tobytes()
            single_page_pdf.close()

            # Process with Docling
            full_parsed_text = docling_ocr_page(single_page_pdf_bytes, page_number + 1)

            # Write to output file
            output_text_file.write(f"[PAGE:{page_number + 1}]\n{full_parsed_text}\n")

        except Exception as e:
            handle_error_no_return(f"Could not process page {page_number+1} of {pdf_document_length}, encountered error: ", e)
            continue
    
    # Close & return
    output_text_file.close()
    return output_text_file_path


def UseBackupOcrOnPage(input_pdf_filepath, current_page_num):
    try:
        read_return = read_config(['backup_ocr_service_choice'])
        backup_ocr_service_choice = read_return['backup_ocr_service_choice']
    except Exception as e:
        return handle_local_error("Could not read required values from config.json when attempting to use backup OCR on page, encountered error: ", e)
    
    if backup_ocr_service_choice == 'Kosmos':
        try:
            current_page_as_pil_image_object = convert_from_path(
                input_pdf_filepath,
                dpi=300,
                first_page=current_page_num,
                last_page=current_page_num
            )[0]
       
            if current_page_as_pil_image_object:
                ocr_result = kosmos_ocr_page(current_page_as_pil_image_object)
                return ocr_result if ocr_result is not None else ""
            else:
                raise Exception(f"Imaged page is empty. Skipping page {current_page_num}.")
        
        except Exception as e:
            return handle_local_error("Could not OCR page with Kosmos, skipping. Encountered error: ", e)
        
    elif backup_ocr_service_choice == 'Docling':
        try:
            pdf_document = fitz.open(input_pdf_filepath)
            # Extract single page as a new PDF
            single_page_pdf = fitz.open()
            single_page_pdf.insert_pdf(pdf_document, from_page=current_page_num, to_page=current_page_num)

            # Convert to bytes
            single_page_pdf_bytes = single_page_pdf.tobytes()
            single_page_pdf.close()

            # Process with Docling
            full_parsed_text = docling_ocr_page(single_page_pdf_bytes, current_page_num)
            return full_parsed_text if full_parsed_text is not None else ""
        except Exception as e:
            return handle_local_error("Could not OCR page with Docling, skipping. Encountered error: ", e)

    else:
        return handle_local_error("Invalid backup OCR service choice, skipping. Encountered error: ", backup_ocr_service_choice)
    

def PDFtoTXT(input_pdf_filepath):

    print("\n\nProcessing Document - PDF to TXT\n\n")

    try:
        read_return = read_config(['pdfs_to_txts', 'force_extract_previously_extracted_text', 'ocr_pdfs', 'min_char_threshold_for_backup_ocr'])
        pdfs_to_txts = read_return['pdfs_to_txts']
        ocr_pdfs = read_return['ocr_pdfs']
        force_extract_previously_extracted_text = str(read_return['force_extract_previously_extracted_text']).lower() == 'true'
        min_char_threshold_for_backup_ocr = int(read_return['min_char_threshold_for_backup_ocr'])
    except Exception as e:
        return handle_local_error("Could not read required values from config.json when attempting to convert PDF to TXT, encountered error: ", e)

    try:
        source_filename = os.path.basename(input_pdf_filepath)

        # Set output path
        output_text_file_name = source_filename.replace(".pdf",".txt")   # Using this instead of os.path.splitext() in case filename contains a period
        output_text_file_path = os.path.join(pdfs_to_txts, output_text_file_name)
        ocr_pdf_file_path = os.path.join(ocr_pdfs, output_text_file_name)

        if not force_extract_previously_extracted_text:
            if os.path.exists(output_text_file_path) and os.path.getsize(output_text_file_path) > 0:
                print("PyPDF2-extracted .txt already exists and is not empty! Returning existing file.")
                return output_text_file_path
            elif os.path.exists(ocr_pdf_file_path) and os.path.getsize(ocr_pdf_file_path) > 0:
                print("Kosmos OCR'ed .txt already exists and is not empty! Returning existing file.")
                return ocr_pdf_file_path
            else:
                print("PyPDF2-extracted .txt already exists but is empty! Overwriting with new .txt file.")
    
    except Exception as e:
        return handle_local_error("Could not complete necessary file system operations when attempting to convert PDF to TXT, encountered error: ", e)
    
    try:
        with open(input_pdf_filepath, 'rb') as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            num_pages = len(pdf_reader.pages)

            with open(output_text_file_path, 'w', encoding='utf-8') as output_text_file:

                # Loop through all the pages and extract text
                for page_num in range(num_pages):
                    print(f"Processing page {page_num+1} of {num_pages}")
                    current_page_num = int(page_num) + 1
                    use_backup_ocr = False
                    pypdf2_text = ""
                    ocr_text = ""
                    
                    try:
                        page = pdf_reader.pages[page_num]
                        pypdf2_text = page.extract_text()
                    except Exception as e:
                        handle_error_no_return("Could not extract text from page, attempting to OCR. Encountered error: ", e)
                        use_backup_ocr = True
                        pypdf2_text = ""

                    if use_backup_ocr or pypdf2_text is None or len(pypdf2_text) < min_char_threshold_for_backup_ocr:
                        print(f"OCR Necessary - Attempting to OCR page {current_page_num} of {num_pages} using backup OCR method.")
                        try:
                            ocr_text = UseBackupOcrOnPage(input_pdf_filepath, current_page_num)
                        except Exception as e:
                            handle_error_no_return("Could not OCR page with backup OCR method, skipping. Encountered error: ", e)
                            ocr_text = ""
                        
                    try:
                        #clean_text = clean_text_string(text)
                        text_to_write = pypdf2_text if len(pypdf2_text) >= len(ocr_text) else ocr_text
                        output_text_file.write(f"[PAGE:{current_page_num}]\n{text_to_write}\n")
                    except Exception as e:
                        handle_error_no_return("Could not write to output text file, encountered error: ", e)
                        continue
    except Exception as e:
        return handle_local_error("Could not process PDF file, encountered error: ", e)

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


def record_doc_loaded_to_db(document_name, chunk_size, chunk_overlap):

    print("\n\nRecording document loading to records DB\n\n")

    try:
        read_return = read_config(['selected_embedding_model', 'selected_knowledge_domain'])
        embedding_model = read_return['selected_embedding_model']
        knowledge_domain = read_return['selected_knowledge_domain']
    except Exception as e:
        return handle_local_error("Could not determine embedding model and knowledge domain in config.json. Error: ", e)

    if is_doc_already_loaded_to_db(document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain):
        print("Document already exists in records DB, skipping insertion.")
        return True

    try:
        conn, cursor = init_and_connect_to_docs_loaded_db()
    except Exception as e:
        return handle_local_error("Could not connect to docs_loaded_db, encountered error: ", e)
    
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
            input_file_name = os.path.basename(input_file)
            source_link = f"http://llm-citations-database.net/source?doc_name={input_file_name}&page_number={page}"
            documents.append({
                'content': chunk.strip(),
                'source_link': source_link,
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
    try:    # To generate a list of Document objects, each containing a 'page_content' string, and a 'metadata' dictionary with 'source_link', 'page_number', and 'source' keys
        numbered_splits = [Document(page_content=chunk['content'], metadata={'source_link':chunk['source_link'], 'page_number': chunk['page_number'], 'source': chunk['source']}) for chunk in chunks]
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


def get_request_params_for_graph_entity_extraction_model(rag_response_mode: bool = False):
    try:
        config_data = read_config([
            'exl2_quantize_graph_model', 'graph_model_access_url',
            'graph_model_server_port', 'graph_model_max_new_tokens',
            'graph_model_temperature', 'graph_model_do_sample', 'graph_model_top_k',
            'graph_model_top_p', 'graph_model_min_p', 'graph_chunk_overlap'
        ])
        exl2_quantize_graph_model = str(config_data['exl2_quantize_graph_model']).lower() == 'true'
    except Exception as e:
        return handle_local_error("Could not get graphing request params, encountered error: ", e)

    headers = {
        'Content-Type': 'application/json',
        'X-Max-New-Tokens': str(config_data['graph_model_max_new_tokens']),
        'X-Temperature': str(config_data['graph_model_temperature']),
        'X-Top-K': str(config_data['graph_model_top_k']),
        'X-Top-P': str(config_data['graph_model_top_p']),
        'X-Min-P': str(config_data['graph_model_min_p']),
        'X-Chunk-Overlap': str(config_data['graph_chunk_overlap']) if not rag_response_mode else '0'
    }

    grapher_url = f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}"

    if not exl2_quantize_graph_model:
        headers['X-Return-Full-Text'] = 'False'
        headers['X-Do-Sample'] = str(config_data['graph_model_do_sample'])
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
                    event_data = line[6:].strip()   # 6 because there's a space after the colon in 'data: '
                    try:
                        received_data = json.loads(event_data)  # json.loads() converts the JSON string into a Python object (dict, list, etc.)
                        if not isinstance(received_data, dict):
                            handle_error_no_return("Received data is not a dict, and is of type: ", type(received_data), ". Skipping. For Debug - Received data: ", received_data)
                            continue
                        full_response[chunk_number] = received_data
                        
                        try:
                            if 'entities_and_relationships' in full_response[chunk_number] and isinstance(full_response[chunk_number]['entities_and_relationships'], str):
                                '''
                                While the received data is a dict, the value of the `entities_and_relationships` key itself may be returned as a string by exl2_grapher's extraction_task(), 
                                so we need to convert it to a dict via ast.literal_eval().
                                The summary_generation_task() on the other hand always returns a dict so we don't need to do anything.

                                NOTE: ast.literal_eval() works only on string inputs! If we were to print:

                                print(f"Type of entities_and_relationships: {type(full_response[chunk_number]['entities_and_relationships'])}")

                                We'd get str for extraction mode, dict for summary generation mode, thus failing in the latter as ast.literal_eval() works only on string inputs!
                                We only want to literal_eval() if the value of the `entities_and_relationships` key is a string, otherwise it's already a dict and we can skip the literal_eval.
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
        return handle_local_error("Failed request to exl2_stream or exl2_grapher APIs, encountered error: ", e)


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
        grapher_url, headers, exl2_quantize_graph_model = get_request_params_for_graph_entity_extraction_model(rag_response_mode=rag_response_mode)
    except Exception as e:
        return handle_local_error("Could not get graphing request params, encountered error: ", e)

    if exl2_quantize_graph_model:   # invoke exl2_grapher API
        try:
            payload = json.dumps({"chunk_entities": chunk_entities, "extraction_mode": True, "rag_response_mode": rag_response_mode})
            full_response = hf_waitress_bulk_stream_request_response_handler(grapher_url, headers, payload)
            # print(f"\nExl2 Bulk-Graphing Response (Entities and Relationships):\n\n{full_response}\n")
            return full_response
        except Exception as e:
            return handle_local_error("Error with request to exl2_grapher API, encountered error: ", e)

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

    try:
        read_return = read_config(['graph_db_server_host', 'assign_host_port_to_graph_db_server'])
        graph_db_server_host = read_return['graph_db_server_host']
        assign_host_port_to_graph_db_server = read_return['assign_host_port_to_graph_db_server']

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


# For general purpose use with HF-Waitress or llama.cpp LLMs:
def get_request_params_for_local_llm_server(formatted_prompt=""):
    try:
        read_return = read_config([
            'local_llm_server', 'hf_waitress_access_url', 'hf_waitress_server_port', 
            'llama_cpp_access_url', 'llama_cpp_server_port', 'local_llm_temperature', 
            'local_llm_top_k', 'local_llm_top_p', 'local_llm_min_p', 'graph_chunk_overlap'
        ])
    except Exception as e:
        return handle_local_error("Could not read request params from config.json, encountered error: ", e)

    headers = {'Content-Type': 'application/json'}

    if read_return['local_llm_server'] == 'hf-waitress':

        payload = formatted_prompt
        base_url = f"http://{read_return['hf_waitress_access_url']}:{read_return['hf_waitress_server_port']}"

        try:
            read_hf_return = read_hf_config(['exl2', 'max_new_tokens', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p'])
            exl2 = str(read_hf_return['exl2']).lower() == 'true'
        except Exception as e:
            return handle_local_error("Could not read hf-waitress config, encountered error: ", e)
        
        headers['X-Max-New-Tokens'] = str(read_hf_return['max_new_tokens'])
        headers['X-Temperature'] = str(read_hf_return['temperature'])
        headers['X-Top-K'] = str(read_hf_return['top_k'])
        headers['X-Top-P'] = str(read_hf_return['top_p'])
        
        if not exl2:
            headers['X-Return-Full-Text'] = 'False'
            headers['X-Min-P'] = str(read_hf_return['min_p'])
            headers['X-Do-Sample'] = str(read_hf_return['do_sample'])
            endpoint_url = f"{base_url}/completions"
        else:
            payload = json.dumps(formatted_prompt)
            headers['Connection'] = 'keep-alive'
            endpoint_url = f"{base_url}/exl2_grapher"
    
    else:   # llama.cpp LLMs
        payload = {
            'prompt': formatted_prompt,
            'temperature': read_return['local_llm_temperature'],
            'top_k': read_return['local_llm_top_k'],
            'top_p': read_return['local_llm_top_p'],
            'min_p': read_return['local_llm_min_p'],
            'chunk_overlap': read_return['graph_chunk_overlap']
        }

        endpoint_url = f"http://{read_return['llama_cpp_access_url']}:{read_return['llama_cpp_server_port']}/completion"

    return endpoint_url, headers, payload, exl2


def get_request_params_for_graph_summarizer_model():
    try:
        config_data = read_config([
            'exl2_quantize_graph_summarizer_model', 'graph_summarizer_access_url',
            'graph_summarizer_server_port', 'graph_summarizer_max_new_tokens',
            'graph_summarizer_temperature', 'graph_summarizer_do_sample', 'graph_summarizer_top_k',
            'graph_summarizer_top_p', 'graph_summarizer_min_p'
        ])
        exl2_quantize_graph_summarizer_model = str(config_data['exl2_quantize_graph_summarizer_model']).lower() == 'true'
    except Exception as e:
        return handle_local_error("Could not get graphing request params, encountered error: ", e)

    graph_summarizer_url = f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}"

    headers = {
        'Content-Type': 'application/json',
        'X-Max-New-Tokens': str(config_data['graph_summarizer_max_new_tokens']),
        'X-Temperature': str(config_data['graph_summarizer_temperature']),
        'X-Top-K': str(config_data['graph_summarizer_top_k']),
        'X-Top-P': str(config_data['graph_summarizer_top_p'])
    }

    if not exl2_quantize_graph_summarizer_model:
        headers['X-Return-Full-Text'] = 'False'
        headers['X-Min-P'] = str(config_data['graph_summarizer_min_p'])
        headers['X-Do-Sample'] = str(config_data['graph_summarizer_do_sample'])
        graph_summarizer_url += "/completions"
    else:
        headers['Connection'] = 'keep-alive'
        graph_summarizer_url += "/exl2_grapher"

    return graph_summarizer_url, headers, exl2_quantize_graph_summarizer_model


def bring_graph_summarizer_model_online():  # Launch HF-Waitress instance with graph summarizer model
    try:
        config_data = read_config([
            'graph_summarizer_model', 'graph_summarizer_access_url', 'graph_summarizer_server_port',
            'quantize_graph_summarizer_model', 'quantize_graph_summarizer_model_bits',
            'exl2_quantize_graph_summarizer_model', 'exl2_quantize_graph_summarizer_model_bpw',
            'graph_summarizer_max_new_tokens', 'graph_summarizer_max_seq_len', 
            'graph_models_base_directory_name', 'graph_summary_generator_directory_name',
            'minimum_free_vram_for_graph_summarizer_model', 'graph_model_access_url', 'graph_model_server_port'
        ])
        hf_waitress_graph_summarizer_server_path = os.path.normpath(os.path.join(os.getcwd(), config_data['graph_models_base_directory_name'], config_data['graph_summary_generator_directory_name']))    # normpath() is used to "normalize" i.e. convert to a path that is appropriate for the current OS
        print(f"\nLaunching HF-Waitress instance with graph summarizer model at path: {hf_waitress_graph_summarizer_server_path}\n")
    except Exception as e:
        return handle_local_error("Could not read graph model config, encountered error: ", e)

    if utils.is_local_server_online(f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}")['server_online']:
        print("\nGraph summarizer model server is already online, skipping launch...\n")
        return True
    
    # Get free GPU memory and shutdown the main Waitress LLM chat server if necessary:
    try:
        hf_waitress_base_url = get_url_for_server('hf-waitress')    # shutdown general-chat LLM server at this URL in case free VRAM is insufficient
        graph_model_base_url = f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}"
        utils.ensure_minimum_free_vram(config_data['minimum_free_vram_for_graph_summarizer_model'], [graph_model_base_url, hf_waitress_base_url])
    except Exception as e:
        return handle_local_error(f"Could not reserve minimum GPU memory ({config_data['minimum_free_vram_for_graph_summarizer_model']}MB) required for Graph-Summarizer model, encountered error: ", e)

    # Need to format this way because the f"""<>""" multiline way will maintain newlines in the command, which will cause the command to fail!
    # Also cannot format this as we have in hf_waitress.py's exllama_bpw_quantize_model() because of the command seperators (&& and ;), which would be incorrectly treated as parameters to the cd command in that list format!
    # We cd first because the command should execute from within that directory otherwise the main hf_config.json gets incorrectly modified! 
    command = (
        f"cd {hf_waitress_graph_summarizer_server_path} "
        f"{'&&' if platform.system() == 'Windows' else ';'} "
        f"{'python' if platform.system() == 'Windows' else 'python3'} hf_waitress.py "
        f"--port {str(config_data['graph_summarizer_server_port'])} "
        f"--model_id {str(config_data['graph_summarizer_model'])} "
        f"--max_new_tokens {str(config_data['graph_summarizer_max_new_tokens'])} "
    )
    if config_data['exl2_quantize_graph_summarizer_model']:
        command += f" --exl2 --exl2_bpw {str(config_data['exl2_quantize_graph_summarizer_model_bpw'])} --exl2_max_seq_len {str(config_data['graph_summarizer_max_seq_len'])}"
    else:
        command += f" --quantize {str(config_data['quantize_graph_summarizer_model'])} --quant_level {str(config_data['quantize_graph_summarizer_model_bits'])}"

    try:
        if platform.system() == 'Windows':
            windows_command = f'start cmd /c "{command}"'   # /k tells cmd to keep the window open even after the command has finished, which is useful for debugging, versus /c which closes the window after the command has finished.
            subprocess.Popen(windows_command, shell=True)   # Popen is used to launch the command in a new process in a new terminal, while subprocess.run() is used to simply run the command and wait for it to finish.
            # Using `start` in this manner explicitly instructs Windows to start a new command window to run the command, while CREATE_NEW_CONSOLE combined with shell=True might not work correctly as the shell itself might capture the console!
        else:
            subprocess.Popen(command, shell=True)
        # The shell=True lets the system's shell interpret the command string, including special operators like && (Windows) or ; (Unix) that chain commands together.
        # This is exactly what you need when you want to change directory before running a script!

        timeout = 5
        attempts = 25
        for _ in range(attempts):
            if utils.is_local_server_online(f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}")['server_online']:
                print(f"\nGraph summarizer model launched successfully!\n")
                return True
            else:
                print(f"Graph summarizer model not yet running, waiting {timeout} seconds before retrying...")
                time.sleep(timeout)
            
    except Exception as e:
        return handle_local_error("Could not launch HF-Waitress instance with kb-generator model, encountered error: ", e)
    
    return False


def summary_generator(chunk_entities=None):
    print("\nGenerating summaries...\n")

    if chunk_entities is None:
        return handle_local_error("Chunk entities are required to generate summaries")

    try:
        bring_graph_summarizer_model_online()
    except Exception as e:
        return handle_local_error("Could not bring graph summarizer model online, encountered error: ", e)

    local_llm_server = read_config(['local_llm_server'])['local_llm_server']
    exl2 = str(read_hf_config(['exl2'])['exl2']).lower() == 'true'

    if local_llm_server == 'hf-waitress':
        
        summarizer_url = ""
        if exl2:

            graph_summarizer_model_prompt_template_format = read_config(['graph_summarizer_model_prompt_template_format'])['graph_summarizer_model_prompt_template_format']

            try:
                summarizer_url, headers, _ = get_request_params_for_graph_summarizer_model()
            except Exception as e:
                return handle_local_error("Could not get graphing request params, encountered error: ", e)

            try:
                payload = json.dumps({"chunk_entities": chunk_entities, "graph_summarizer_model_prompt_template_format": graph_summarizer_model_prompt_template_format, "summary_generation_mode": True})
                full_response = hf_waitress_bulk_stream_request_response_handler(summarizer_url, headers, payload)
                # print(f"\nExl2 Bulk-Summary Generation Response:\n\n{full_response}\n")
                return full_response
            except Exception as e:
                return handle_local_error("Error with request to exl2_grapher API, encountered error: ", e)

        else:
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

    # print(f"\nStoring entities(nodes) to {selected_knowledge_domain} graph DB\n")

    processed_nodes = {}    # Format: {(name, node_type): True}
    page_number = [] if page_number is None else page_number   

    for node in nodes:
        try:
            name = str(node['name'])
            node_type = str(node['type'])
            updated_summary = list(node.get('summary', []))   # dict .get() method is safer than `if node['summary']` because it provides a default value if the key doesn't exist and handles NoneType errors gracefully!
            
            node_key = (name, node_type)
            if node_key in processed_nodes:
                # print(f"Skipping duplicate node {name} of type {node_type} in {selected_knowledge_domain} graph DB")
                continue

            node_name = sanitize_names(name)

            # MERGE on stable properties to prevent duplicates, then SET the summary and add source_documents and page_numbers as per the case:
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
            
            # Mark node as processed:
            processed_nodes[node_key] = True
            
            # print(f"\nCreated node - name: {name}, type: {node_type} - in {selected_knowledge_domain} graph DB\n")
        except Exception as e:
            handle_error_no_return(f"Could NOT create node - name: {name}, type: {node_type} - in {selected_knowledge_domain} graph DB, skipping. Encountered error: ", e)


def add_relationships_to_graph(selected_knowledge_domain: str, relationships: list, graph: FalkorDB, source_document: str = '', page_number: list = None):
    # print(f"\nStoring relationships to {selected_knowledge_domain} graph DB\n")

    processed_relationships = {}    # Format: {(source, target, relationship): True}
    page_number = [] if page_number is None else page_number

    for relationship in relationships:
        try:
            source = str(relationship['source'])
            target = str(relationship['target'])
            relationship_type = str(relationship['relationship'])            
            updated_summary = list(relationship.get('summary', []))   # dict .get() method is safer than `if relationship['summary']` because it provides a default value if the key doesn't exist and handles NoneType errors gracefully!
            
            relationship_key = (source, target, relationship_type)
            if relationship_key in processed_relationships:
                # print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type}) in {selected_knowledge_domain} graph DB")
                continue

            source = sanitize_names(source)
            target = sanitize_names(target)
            relationship_type = sanitize_names(relationship_type).upper()

            '''
            MERGE on stable properties to prevent duplicates, then SET the summary and add source_documents and page_numbers as per the case.
            Track weights for relationships to improve clustering by tracking how many times each relationship is detected:
            Microsoft's GraphRAG papers emphasis on "normalized counts of detected relationship instances" is quite deliberate - 
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

            # Mark relationship as processed:
            processed_relationships[relationship_key] = True
            
            # print(f"\nCreated relationship - source: {source}, target: {target}, relationship: {relationship} - in {selected_knowledge_domain} graph DB\n")
        except Exception as e:
            handle_error_no_return(f"Could NOT create relationship from data: {relationship} in {selected_knowledge_domain} graph DB, skipping. Encountered error: ", e)


def store_entities_and_relationships_in_graph_db(chunk_entities: dict, selected_knowledge_domain: str, graph: FalkorDB, skip_summary_generation: bool = False, summaries_filepath: str = None):
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
        1. Handles summary generation if applicable: 
            - Summaries are generated for a given chunk basis the entities and relationships identified within it. 
            - The updated chunk_entities dict is then stored to the `summaries_filepath` for persistence and re-use before being stored in the graph DB.
        2. Stores the nodes and relationships in the graph DB.
    '''

    try:
        # First, Get/Generate summaries for each node and relationship if applicable:
        if not skip_summary_generation:

            try:
                chunks = summary_generator(chunk_entities=chunk_entities)
                try:
                    overwrite_json_file(chunks, summaries_filepath)
                except Exception as e:
                    handle_error_no_return(f"Error saving summaries to file, they will have to be re-generated in the future if this document is re-processed. Encountered error: ", e)
            except Exception as e:
                handle_error_no_return(f"Error generating summaries for nodes and relationships, proceeding without new summaries. Encountered error: ", e)
                chunks = chunk_entities
        
        else:
            print(f"\nSkipping summary generation for all nodes and relationships in {selected_knowledge_domain} graph DB\n")
            chunks = chunk_entities

        # Store the nodes and relationships in the graph DB
        print(f"\nStoring entities and relationships in {selected_knowledge_domain} graph DB\n")
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


def bring_graph_db_online():    # launch FalkorDB Docker container
    print(f"\nLaunching FalkorDB Docker container...\n")

    try:
        read_return = read_config([
            'launch_graph_db_with_ui', 'assign_host_port_to_graph_db_server', 
            'assign_host_port_to_graph_db_ui', 'graph_db_data_directory'])
        launch_graph_db_with_ui = read_return['launch_graph_db_with_ui']
        assign_host_port_to_graph_db_server = read_return['assign_host_port_to_graph_db_server']
        assign_host_port_to_graph_db_ui = read_return['assign_host_port_to_graph_db_ui']
        graph_db_data_directory = read_return['graph_db_data_directory']
    except Exception as e:
        return handle_local_error("Could not read graph DB config when attempting to bring FalkorDB online, encountered error: ", e)

    # Check if Docker Engine is running
    try:
        subprocess.run(['docker', 'info'], capture_output=True, check=True)  # check=True will raise an exception if the command returns a non-zero exit code
    except Exception as e:
        return handle_local_error("Docker Engine is not running, encountered error: ", e)

    print("\nDocker Engine is running, proceeding with FalkorDB Docker container launch...\n")

    if check_if_container_is_running('falkor-db'):
        print("\nFalkorDB Docker container is already running, skipping launch...\n")
        return True

    command = [
        'docker', 'run', '-p', f'{assign_host_port_to_graph_db_server}:6379',
        *(['-p', f'{assign_host_port_to_graph_db_ui}:3000'] if launch_graph_db_with_ui else []),
        '--name', 'falkor-db',
        '-it', '--rm', '-v', f'{graph_db_data_directory}:/var/lib/falkordb/data', 'falkordb/falkordb:edge'
    ]   # Using conditional list-unpacking with * to handle optional arguments!

    try:
        subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE) if platform.system() == 'Windows' else subprocess.Popen(command, shell=True)
        # Check if the container is running
        container_name = 'falkor-db'
        timeout = 2
        attempts = 50
        for _ in range(attempts):
            if check_if_container_is_running(container_name):
                print(f"\nFalkorDB Docker container launched successfully!\n")
                return True
            else:
                print(f"FalkorDB Docker container not yet running, waiting {timeout} seconds before retrying...")
                time.sleep(timeout)

    except Exception as e:
        return handle_local_error("Could not launch FalkorDB Docker container, encountered error: ", e)

    return True


@app.route('/start_falkordb')
def start_falkordb():
    print("\nStarting FalkorDB Docker container...\n")
    try:
        bring_graph_db_online()
        return jsonify({"message": "FalkorDB Docker container started successfully"}), 200
    except Exception as e:
        return handle_api_error("Could not start FalkorDB Docker container, encountered error: ", e)


def bring_graph_extraction_model_online():  # Launch HF-Waitress instance with kb-generator model

    try:
        config_data = read_config([
            'graph_model_access_url', 'graph_model_server_port',
            'quantize_graph_model', 'quantize_graph_model_bits',
            'graph_generator_model', 'exl2_quantize_graph_model',
            'exl2_quantize_graph_model_bpw', 'graph_model_max_new_tokens',
            'graph_model_max_seq_len', 'graph_models_base_directory_name',
            'graph_extraction_model_directory_name', 'graph_summary_generator_directory_name',
            'minimum_free_vram_for_graph_extraction_model', 'graph_summarizer_access_url', 'graph_summarizer_server_port'
        ])
        hf_waitress_kb_generator_server_path = os.path.normpath(os.path.join(os.getcwd(), config_data['graph_models_base_directory_name'], config_data['graph_extraction_model_directory_name']))    # normpath() is used to "normalize" i.e. convert to a path that is appropriate for the current OS
        print(f"\nLaunching HF-Waitress instance with kb-generator model at path: {hf_waitress_kb_generator_server_path}\n")
    except Exception as e:
        return handle_local_error("Could not read graph model config, encountered error: ", e)

    if utils.is_local_server_online(f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}")['server_online']:
        print("\nGraphing model server is already online, skipping launch...\n")
        return True
    
    # Get free GPU memory and shutdown the main Waitress LLM chat server if necessary:
    try:
        hf_waitress_base_url = get_url_for_server('hf-waitress')
        graph_summarizer_base_url = f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}"
        utils.ensure_minimum_free_vram(config_data['minimum_free_vram_for_graph_extraction_model'], [graph_summarizer_base_url, hf_waitress_base_url])
    except Exception as e:
        return handle_local_error(f"Could not reserve minimum GPU memory ({config_data['minimum_free_vram_for_graph_extraction_model']}MB) required for Graph-Extraction model, encountered error: ", e)

    # Need to format this way because the f"""<>""" multiline way will maintain newlines in the command, which will cause the command to fail!
    # Also cannot format this as we have in hf_waitress.py's exllama_bpw_quantize_model() because of the command seperators (&& and ;), which would be incorrectly treated as parameters to the cd command in that list format!
    # We cd first because the command should execute from within that directory otherwise the main hf_config.json gets incorrectly modified! 
    command = (
        f"cd {hf_waitress_kb_generator_server_path} "
        f"{'&&' if platform.system() == 'Windows' else ';'} "
        f"{'python' if platform.system() == 'Windows' else 'python3'} hf_waitress.py "
        f"--port {str(config_data['graph_model_server_port'])} "
        f"--model_id {str(config_data['graph_generator_model'])} "
        f"--max_new_tokens {str(config_data['graph_model_max_new_tokens'])} "
    )
    if config_data['exl2_quantize_graph_model']:
        command += f" --exl2 --exl2_bpw {str(config_data['exl2_quantize_graph_model_bpw'])} --exl2_max_seq_len {str(config_data['graph_model_max_seq_len'])}"
    else:
        command += f" --quantize {str(config_data['quantize_graph_model'])} --quant_level {str(config_data['quantize_graph_model_bits'])}"

    try:
        if platform.system() == 'Windows':
            windows_command = f'start cmd /c "{command}"'   # /k tells cmd to keep the window open even after the command has finished, which is useful for debugging, versus /c which closes the window after the command has finished.
            subprocess.Popen(windows_command, shell=True)   # Popen is used to launch the command in a new process in a new terminal, while subprocess.run() is used to simply run the command and wait for it to finish.
            # Using `start` in this manner explicitly instructs Windows to start a new command window to run the command, while CREATE_NEW_CONSOLE combined with shell=True might not work correctly as the shell itself might capture the console!
        else:
            subprocess.Popen(command, shell=True)
        # The shell=True lets the system's shell interpret the command string, including special operators like && (Windows) or ; (Unix) that chain commands together.
        # This is exactly what you need when you want to change directory before running a script!

        timeout = 5
        attempts = 25
        for _ in range(attempts):
            if utils.is_local_server_online(f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}")['server_online']:
                print(f"\nKB-Generator model launched successfully!\n")
                return True
            else:
                print(f"KB-Generator model not yet running, waiting {timeout} seconds before retrying...")
                time.sleep(timeout)
            
    except Exception as e:
        return handle_local_error("Could not launch HF-Waitress instance with kb-generator model, encountered error: ", e)
    
    return False


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


def assemble_chunks_for_graph_db(chunks):
    '''
    Assembles document chunks into a dictionary of entities for storage to the GraphDB:
    '''
    try:
        read_return = read_config(['graph_chunk_size', 'graph_chunk_overlap'])
        graph_chunk_size = read_return['graph_chunk_size']
        graph_chunk_overlap = read_return.get('graph_chunk_overlap', 300)
        
        graphing_chunk = ""
        chunks_in_storage_queue = []    # chunks will be combined upto `graph_chunk_size` and stored here
        source_filename = ""
        page_number_list = []
        chunk_entities = {}
        graph_chunk_count = 1
        overlap_text = ""

        print("\nGenerating Graphing Chunks Dictionary...\n")
        
        for count, chunk in enumerate(chunks):

            try:
                chunk_source_filepath = chunk['source']
                source_filename = os.path.basename(chunk_source_filepath)

                try:    # page numbers while useful are non-essential which is why I'm wrapping in a dedicated try-except block that does not raise an error!
                    page_number_list.append(int(chunk['page_number']))
                    page_number_list = list(set(page_number_list))   # Remove duplicates
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
            if isinstance(candidate, dict) and candidate != {}: # TODO: validation of the JSON file format
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
            if isinstance(candidate, dict) and candidate != {}: # TODO: validation of the JSON file format
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
        read_return = read_config(['docs_to_knowledge_graph_dir', 'selected_knowledge_domain', 'apply_clustering_to_graph_db_on_doc_load'])
        docs_to_knowledge_graph_dir = read_return['docs_to_knowledge_graph_dir']
        selected_knowledge_domain = read_return['selected_knowledge_domain']
        apply_clustering_to_graph_db_on_doc_load = str(read_return['apply_clustering_to_graph_db_on_doc_load']).lower() == 'true'
    except Exception as e:
        handle_error_no_return("Could not read graph generator config, encountered error: ", e)
    
    return docs_to_knowledge_graph_dir, selected_knowledge_domain, apply_clustering_to_graph_db_on_doc_load


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
        
        '<entities_and_relationships>': {"nodes": [{"type": "organization","name": "Intel"},{"type": "object","name": "Intel Products"},...], "relationships": [{"source": "Intel","target": "Intel Products","relationship": "business unit"},...]}

    The final `chunk_entities` dict is then passed to the `store_entities_and_relationships_in_graph_db()` function which will invoke the summary generator if applicable, and then store the entities and relationships in the GraphDB.

    Future TODO: Split work across multiple instances of the Graphing Model and Summarizer LLM. Requires minimum two GPUs.
    '''
    
    print(f"\n\nGraph Generator Invoked. Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    try:
        bring_graph_db_online()
    except Exception as e:
        return handle_local_error("Could not bring graph DB online, encountered error: ", e)

    try:
        docs_to_knowledge_graph_dir, selected_knowledge_domain, apply_clustering_to_graph_db_on_doc_load = read_graph_generator_config()
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
            bring_graph_extraction_model_online()
        except Exception as e:
            return handle_local_error("Could not bring graphing model online, encountered error: ", e)

        try:
            chunk_entities = assemble_chunks_for_graph_db(chunks)
        except Exception as e:
            return handle_local_error("Could not assemble chunks for graph DB, encountered error: ", e)

        try:
            complete_chunk_entities = extract_all_entities_and_relationships(chunk_entities)
            try:
                overwrite_json_file(complete_chunk_entities, entities_and_relationships_filepath)
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

    if apply_clustering_to_graph_db_on_doc_load:
        try:
            apply_leiden_clustering_to_graph(selected_knowledge_domain)
        except Exception as e:
            handle_error_no_return("Could not apply Leiden clustering to the graph, encountered error: ", e)

    print(f"\n\nCompleted Graph Generator at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return True


def read_embeddings_config() -> tuple[str, str]:
    print("\n\nReading embeddings config\n\n")
    try:
        read_return = read_config(['selected_embedding_model', 'chunk_size', 'chunk_overlap', 'upload_doc_to_graph_db', 'perform_only_graph_rag'])
        selected_embedding_model = read_return['selected_embedding_model']
        chunk_sz = read_return['chunk_size']
        chunk_olp = read_return['chunk_overlap']
        upload_doc_to_graph_db = str(read_return['upload_doc_to_graph_db']).lower() == 'true'
        perform_only_graph_rag = str(read_return['perform_only_graph_rag']).lower() == 'true'
    except Exception as e:
        handle_local_error("Missing values in config.json, could not read_embeddings_config. Error: ", e)

    path_to_knowledge_domain = get_path_to_knowledge_domain()

    return selected_embedding_model, path_to_knowledge_domain, chunk_sz, chunk_olp, upload_doc_to_graph_db, perform_only_graph_rag


# Document vectorization and chunking
def whoosh_embed_and_graph_doc_chunks(input_file):
    print("\n\nCore Document Vectorization and Chunking Function Invoked\n\n")

    # Read Embeddings Config
    try:
        selected_embedding_model, path_to_knowledge_domain, chunk_sz, chunk_olp, upload_doc_to_graph_db, perform_only_graph_rag = read_embeddings_config()
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
            
            if upload_doc_to_graph_db:
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


def stage_gdrive_file(filename_with_extension, file_content, mime_type, lars_user_id):
    staging_info_record = {}
    try:
        staging_info_record = prepare_basic_staging_info_record(filename_with_extension, lars_user_id, 'Google Drive')
    except Exception as e:
        return handle_local_error(f"Server-side error - could not prepare staging info record for Google Drive file: '{filename_with_extension}', encountered error: ", e)
    
    try:
        staged_file_info = check_if_file_already_staged(staging_info_record)
        if staged_file_info is not None:
            return staged_file_info
    except Exception as e:
        handle_error_no_return(f"Could not check if GDrive file '{filename_with_extension}' is already staged, proceeding afresh. Encountered error: ", e)

    # New file, proceed with fresh upload
    try:    # convert file_content to FileStorage object and pass to save_file_to_staging_dir
        file_to_stage = FileStorage(
            stream=io.BytesIO(file_content),
            filename=filename_with_extension,
            content_type=mime_type
        )
    except Exception as e:
        return handle_local_error("Server-side error recieving file: ", e)
    
    try:
        staged_filename, staged_filepath = save_file_to_staging_dir(file_to_stage)
    except Exception as e:
        return handle_local_error("Server-side error, could not save file to staging area, encountered error: ", e)
    
    try:
        staging_info_record['document_name_and_extension'] = staged_filename # secure_filename version of original filename
        staging_info_record['staged_datetime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        staging_info_record['staged_filepath'] = staged_filepath
        staging_info_record['txt_filepath'] = ''
        staging_info_record['status'] = 'Staged - File Saved to Staging Area'
        insert_into_staging_db(staging_info_record, skip_check=True)  # `file_already_staged` check has already been performed, so this will insert directly into the DB barring some error.
    except Exception as e:
        return handle_local_error("Server-side error, could not insert file into staging DB, encountered error: ", e)
    
    return staging_info_record


def download_gdrive_file(service, file_id, filename, mime_type, lars_user_id):

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
                
                base_filename, _ = os.path.splitext(filename) # Remove existing extension if any
                filename_with_extension = base_filename + file_extension
            
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
        return handle_local_error("Error downloading file from Google Drive: ", e)
    
    try:
        file_content = file.getvalue()
        if file_content is None:
            raise Exception("Server-side error - Could not download file from Google Drive")
    except Exception as e:
        return handle_local_error("Server-side error downloading file from Google Drive: ", e)
    
    try:
        return stage_gdrive_file(filename_with_extension, file_content, mime_type, lars_user_id)
    except Exception as e:
        return handle_local_error("Server-side error staging file from Google Drive: ", e)


def download_folder(service, folder_id, folder_name=None, lars_user_id=None):
    print(f"\n\nDownloading GoogleDrive Folder: {folder_name}\n\n")

    items = []
    page_token = None
    query = f"'{folder_id}' in parents"
    fields = "nextPageToken, files(id, name, mimeType)"

    while True:
        response = service.files().list(
            q=query, 
            fields=fields, 
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        items.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if not page_token:
            break

    staged_file_records = []

    try:
        for item in items:
            file_id = item['id']
            filename = item['name']
            mime_type = item['mimeType']
            print(f"folder item mime_type f{mime_type}")

            if "folder" in str(mime_type):
                try:
                    staged_file_records.extend(download_folder(service, file_id, filename, lars_user_id))  # in this case, file_id will be the folder id!
                    print(f"\nNested GDrive Folder '{filename}' downloaded successfully\n")
                except Exception as e:
                    handle_error_no_return("Could not download nested folder from GDrive, encountered error: ", e)
            else:
                try:
                    staged_file_records.append(download_gdrive_file(service, file_id, filename, mime_type, lars_user_id))
                    print(f"\nGDrive File '{filename}' downloaded successfully\n")
                except Exception as e:
                    handle_error_no_return("Server-side error downloading GDrive file, encountered error: ", e)
            
        print(f"\n\nGDrive folder '{folder_name}' downloaded successfully\n\n")
    except Exception as e:
        return handle_local_error(f"Error downloading Google Drive folder: '{folder_name}' in the GDrive download folder method, encountered error: ", e)

    return staged_file_records


def gdrive_downloader(service, file_or_folder_id, filename, mime_type, mime_type_category, lars_user_id):
    try:
        if mime_type_category == "folder":
            return download_folder(service, file_or_folder_id, filename, lars_user_id)    # filename is the folder name in this case
        else:
            return download_gdrive_file(service, file_or_folder_id, filename, mime_type, lars_user_id)
    except Exception as e:
        return handle_local_error("Error downloading file from Google Drive in the gdrive-downloader() method, encountered error: ", e)


@app.route('/gdrive_file_transfer_to_staging', methods=['POST'])
def gdrive_file_transfer_to_staging():
    try:
        gdrive_file_id = str(request.form['file_id'])
        gdrive_file_mimeType = str(request.form['file_mimeType'])
        lars_user_id = str(request.form['user_id'])
    except Exception as e:
        return handle_api_error("Server-side error reading Google Drive file details for download: ", e)
    
    try:
        service = build("drive", "v3", credentials=GDRIVE_CREDS)
    except Exception as e:
        return handle_api_error("Could not create Google service handler, check credentials and re-try: ", e)
    
    try:
        file_metadata = service.files().get(fileId=gdrive_file_id, fields='name, mimeType', supportsAllDrives=True).execute()   # includeItemsFromAllDrives=True not needed here because it's a search and listing operation, whereas here we're retrieving a specific item by ID!
        original_filename = file_metadata.get('name', 'untitled')
        mime_type = file_metadata.get('mimeType', gdrive_file_mimeType)
        mime_type_category = categorize_mimetype(mime_type)
    except Exception as e:
        return handle_api_error(f"Could not read GoogleDrive file metadata for file: '{original_filename}', encountered error: ", e)

    try:
        staged_info = gdrive_downloader(service, gdrive_file_id, original_filename, mime_type, mime_type_category, lars_user_id)
        staged_file_info_list = staged_info if isinstance(staged_info, list) else [staged_info]
        return jsonify(success=True, staged_file_info_list=staged_file_info_list)
    except Exception as e:
        return handle_api_error(f"Server-side error - could not download file: '{original_filename}' from Google Drive: ", e)


def get_text_extract_from_pdf(pdf_filepath):
    print("\nGetting text extract from PDF\n")

    try:
        read_return = read_config(['force_ocr', 'ocr_service_choice'])
        force_ocr = read_return.get('force_ocr', False)
        ocr_service_choice = read_return.get('ocr_service_choice', None)
    except Exception as e:
        return handle_local_error("Could not determine force_ocr in config.json. Disabling OCR and proceeding. Error: ", e)
    
    if force_ocr:
        try:
            if ocr_service_choice == 'Docling':
                txt_filepath = PDFtoDoclingOCRTXT(pdf_filepath)
            elif ocr_service_choice == 'AzureVision':
                txt_filepath = PDFtoAzureOCRTXT(pdf_filepath)
            elif ocr_service_choice == 'AzureDocAi':
                txt_filepath = PDFtoAzureDocAiTXT(pdf_filepath)
            elif ocr_service_choice == 'LocalVisionLLM':
                txt_filepath = PDFtoVisionLLMOCRTXT(pdf_filepath)
            elif ocr_service_choice == 'Kosmos':
                txt_filepath = PDFtoKosmosOCRTXT(pdf_filepath)
            else:
                raise Exception(f"Invalid OCR service choice: {ocr_service_choice}")
        except Exception as e:
            handle_error_no_return("Failed to OCR text from PDF. Will now attempt to extract text via PyPDF2. Encountered error: ", e)
            try:
                txt_filepath = PDFtoTXT(pdf_filepath)
            except Exception as e:
                return handle_local_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)
    else:
        try:
            txt_filepath = PDFtoTXT(pdf_filepath)
        except Exception as e:
            return handle_local_error("Failed to extract text from the PDF document via PyPDF2, encountered error: ", e)
    
    return txt_filepath


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
        pdf_filename = os.path.splitext(source_filename)[0] + ".pdf"   # os.path.splitext() returns a tuple containing the path's name and extension separately
        pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        return os.path.exists(pdf_filepath), pdf_filepath
    except Exception as e:
        handle_error_no_return("Could not determine if converted file already exists, proceeding to convert file regardless. Encountered error: ", e)
        return False, None


def get_pdf_filepath_for_upload(filename, filepath):
    try:
        if not filename.lower().endswith('.pdf'):
            converted_file_exists, converted_pdf_file_path = check_if_converted_file_exists(filepath)
            if not converted_file_exists:
                _, pdf_filepath = convert_non_pdf_to_pdf_with_unoconv(filename, filepath)
            else:
                pdf_filepath = converted_pdf_file_path
            return pdf_filepath
        else:
            return filepath
    except Exception as e:
        return handle_local_error("Could not get PDF filepath for upload, encountered error: ", e)


def upload_to_rag_and_records_databases(original_filename, txt_filepath):
    try:
        if os.path.getsize(txt_filepath) > 0:
            chunk_size, chunk_overlap = whoosh_embed_and_graph_doc_chunks(txt_filepath)
        else:
            print("Extracted document is empty! Skipping upload to RAG databases.")
    except Exception as e:
        return handle_local_error("Failed to embed & index document: ", e)

    try:
        if os.path.getsize(txt_filepath) > 0:
            record_doc_loaded_to_db(original_filename, chunk_size, chunk_overlap)
        else:
            print("Extracted document is empty! Not saving to Records DB.")
    except Exception as e:
        handle_error_no_return("Unable to record document loading to Records DB, encountered error: ", e)

    return True


def prepare_basic_staging_info_record(filename_with_extension:str, user_id:str, source:str = 'Local Drive'):
    '''
    user_id: str
    upload_id: str
    staged_filepath: str
    txt_filepath: str
    document_name_and_extension: str
    embedding_model: str
    knowledge_domain: str
    source: str
    text_extraction_method: str
    upload_initiated_datetime: str
    staged_datetime: str
    status: str
    '''
    try:
        read_return = read_config(['selected_embedding_model', 'selected_knowledge_domain', 'force_ocr', 'ocr_service_choice'])
        selected_embedding_model = read_return['selected_embedding_model']
        selected_knowledge_domain = read_return['selected_knowledge_domain']
        force_ocr = str(read_return['force_ocr']).lower() == 'true'
        ocr_service_choice = str(read_return['ocr_service_choice'])
    except Exception as e:
        return handle_local_error("Could not read config when preparing basic staging info record, encountered error: ", e)

    try:
        upload_id = str(uuid.uuid4())
    except Exception as e:
        return handle_local_error("Could not generate unique ID for staging info record, encountered error: ", e)
    
    try:
        staging_info_record = {
            'user_id': user_id,
            'upload_id': upload_id,
            'document_name_and_extension': filename_with_extension,
            'embedding_model': selected_embedding_model,
            'knowledge_domain': selected_knowledge_domain,
            'source': source,
            'text_extraction_method': ocr_service_choice if force_ocr else 'default',
            'upload_initiated_datetime': datetime.datetime.now().isoformat(),
            'status': 'Basic Ticket Created'
        }
    except Exception as e:
        return handle_local_error("Could not prepare basic staging info record, encountered error: ", e)
    
    return staging_info_record
    

def init_and_connect_to_upload_staging_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    try:
        read_return = read_config(['upload_staging_db'])
        upload_staging_db = read_return['upload_staging_db']
    except Exception as e:
        return handle_local_error("Missing upload_staging_db in config.json for method init_and_connect_to_upload_staging_db. Error: ", e)
    
    try:
        conn = sqlite3.connect(upload_staging_db)
        cursor = conn.cursor()
    except Exception as e:
        return handle_local_error("Could not establish connection to upload_staging_db, encountered error: ", e)
    
    try:    # 1. Create table if it doesn't already exist - not adding the UNIQUE constraint here as it will only apply to new table creations, not future schema changes!
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS upload_staging (
                id INTEGER PRIMARY KEY,
                upload_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                document_name_and_extension TEXT NOT NULL,
                txt_filepath TEXT NOT NULL,
                staged_filepath TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                knowledge_domain TEXT NOT NULL,
                source TEXT NOT NULL,
                text_extraction_method TEXT NOT NULL,
                upload_initiated_datetime TEXT NOT NULL,
                staged_datetime TEXT NOT NULL,
                status TEXT NOT NULL
            )
        ''')

        conn.commit()   # Auto-incrementing primary key 'id'
    except Exception as e:
        return handle_local_error("Could not create upload_staging table, encountered error: ", e)
    
    try:    # 2. Add columns if they don't already exist
        add_column_if_not_exists(cursor, 'upload_staging', 'upload_id', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'user_id', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'document_name_and_extension', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'txt_filepath', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'staged_filepath', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'embedding_model', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'knowledge_domain', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'source', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'text_extraction_method', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'upload_initiated_datetime', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'staged_datetime', 'TEXT')
        add_column_if_not_exists(cursor, 'upload_staging', 'status', 'TEXT')
    except Exception as e:
        return handle_local_error("Could not add necessary columns to upload_staging table, encountered error: ", e)
    
    return conn, cursor


def update_existing_entry_in_staging_db(doc_info: dict, id: int):
    try:
        conn, cursor = init_and_connect_to_upload_staging_db()
    except Exception as e:
        return handle_local_error("Could not connect to upload staging DB, encountered error: ", e)
    
    try:
        cursor.execute('''
            UPDATE upload_staging
            SET txt_filepath = ?, staged_filepath = ?, upload_initiated_datetime = ?, staged_datetime = ?, status = ?
            WHERE id = ?
            ''',
            (doc_info['txt_filepath'], doc_info['staged_filepath'], doc_info['upload_initiated_datetime'], doc_info['staged_datetime'], doc_info['status'], id)
        )
        conn.commit()
    except Exception as e:
        return handle_local_error("Could not update existing entry in upload staging DB, encountered error: ", e)
    finally:
        cursor.close()
        conn.close()
    return True


def check_for_staged_file_info_in_staging_db(file_transfer_info: dict):
    try:
        conn, cursor = init_and_connect_to_upload_staging_db()
    except Exception as e:
        return handle_local_error("Could not connect to upload staging DB, encountered error: ", e)
    
    try:
        cursor.execute('''
            SELECT * FROM upload_staging
            WHERE user_id = ?
            AND document_name_and_extension = ?
            AND embedding_model = ?
            AND knowledge_domain = ?
            AND source = ?
            AND text_extraction_method = ?
            ''',
            (
                file_transfer_info['user_id'],
                file_transfer_info['document_name_and_extension'],
                file_transfer_info['embedding_model'],
                file_transfer_info['knowledge_domain'],
                file_transfer_info['source'],
                file_transfer_info['text_extraction_method']
            )
        )

        column_names = [desc[0] for desc in cursor.description]
        result = cursor.fetchone()
        if result:
            row_dict = dict(zip(column_names, result))  # Convert tuple to dictionary
            return row_dict
        else:
            return None
    except Exception as e:
        return handle_local_error("Could not check if files are already staged, encountered error: ", e)
    finally:
        cursor.close()
        conn.close()


def validate_file_transfer_info(file_transfer_info: dict):
    try:
        keys_to_validate = ['user_id', 'upload_id', 'document_name_and_extension', 'embedding_model', 'knowledge_domain', 'source', 'text_extraction_method', 'upload_initiated_datetime']
        for key in keys_to_validate:
            if key not in file_transfer_info:
                raise Exception(f"Invalid request, missing key information for bulk download request: {key}")
        return True
    except Exception as e:
        return handle_local_error("File transfer info validation failed, encountered error: ", e)


def check_if_file_already_staged(file_transfer_info: dict):
    try:
        staged_file_info = check_for_staged_file_info_in_staging_db(file_transfer_info)
        # print(f"\n\nStaged file info: {staged_file_info}\n\n")
        if staged_file_info is not None:
            print(f"\n\nStaged file info is not None! Validating...\n\n")
            try:    # validate file transfer data
                validate_file_transfer_info(staged_file_info) # exception will be raised if invalid
                print(f"\n\nStaged file info validated successfully!\n\n")
                if os.path.exists(os.path.join(app.config['UPLOAD_STAGING_FOLDER'], staged_file_info['document_name_and_extension'])):  # `document_name_and_extension` key will exist because it passed validation
                    print(f"\n\nStaged file info is valid and file exists in staging area!\n\n")
                    staged_file_info.pop('id', None)    # 'id' primary key is for internal DB use only
                    return staged_file_info
                else:
                    print(f"\n\nStaged file info is valid but file does not exist in staging area! Deleting entry and proceeding with fresh upload.\n\n")
                    delete_entry_from_staging_db(None, staged_file_info['id'])  #TODO: Will likely cause an issue with the resume function, redo when resume is implemented
                    return None
            except Exception as e:
                handle_error_no_return("File already staged for uploading but error validating staged file data. Deleting entry and proceeding with fresh upload. Error Log: ", e)
                safe_delete_entry_from_staging_db(staged_file_info)
        else:
            print(f"\n\nFile not previously staged.\n\n")
            return None
    except Exception as e:
        return handle_local_error("Could not check if file is already staged, encountered error: ", e)


def delete_entry_from_staging_db(doc: dict, id: int = None):
    try:
        result = check_for_staged_file_info_in_staging_db(doc) if id is None else {'id': id}
        if result is not None:
            try:
                conn, cursor = init_and_connect_to_upload_staging_db()
                cursor.execute('''
                    DELETE FROM upload_staging
                    WHERE id = ?
                    ''',
                    (result['id'],)
                )
                conn.commit()
            except Exception as e:
                return handle_local_error("Could not delete entry from upload staging DB, encountered error: ", e)
            finally:
                cursor.close()
                conn.close()
        return True
    except Exception as e:
        handle_error_no_return("Error deleting entry from staging records DB. Encountered error: ", e)


def safe_delete_entry_from_staging_db(doc: dict):
    try:
        delete_entry_from_staging_db(doc)
    except Exception as e:
        handle_error_no_return("Could not delete entry from staging records DB, skipping. Encountered error: ", e)


def insert_into_staging_db(doc_info: dict, skip_check: bool = False):
    if not skip_check:
        try:
            result = check_for_staged_file_info_in_staging_db(doc_info)
            if result is not None:  # this call is likely made by the bulk_text_extract method to update the txt_filepath!
                try:
                    update_existing_entry_in_staging_db(doc_info, result['id'])
                    return True
                except Exception as e:
                    handle_error_no_return("Could not update existing entry in upload staging DB. Adding fresh entry. Encountered error: ", e)
        except Exception as e:
            handle_error_no_return("Could not check if file is already staged when attempting to avoid duplicate entries in upload staging DB. Adding fresh entry. Encountered error: ", e)
    
    try:
        conn, cursor = init_and_connect_to_upload_staging_db()
    except Exception as e:
        return handle_local_error("Could not connect to upload staging DB, encountered error: ", e)
    
    try:    # Queue in staging DB
        cursor.execute('''
            INSERT INTO upload_staging (
                upload_id,
                user_id,
                document_name_and_extension,
                txt_filepath,
                staged_filepath,
                embedding_model,
                knowledge_domain,
                source, 
                text_extraction_method,
                upload_initiated_datetime,
                staged_datetime,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                doc_info['upload_id'],
                doc_info['user_id'],
                doc_info['document_name_and_extension'],
                doc_info['txt_filepath'],
                doc_info['staged_filepath'],
                doc_info['embedding_model'],
                doc_info['knowledge_domain'],
                doc_info['source'],
                doc_info['text_extraction_method'],
                doc_info['upload_initiated_datetime'],
                doc_info['staged_datetime'],
                doc_info['status']
            )
        )
        conn.commit()
    except Exception as e:
        return handle_local_error("Could not add new entry to upload staging DB, encountered error: ", e)
    finally:
        cursor.close()
        conn.close()
    
    return True


def perform_post_bulk_upload_cleanup():
    # 1. Shutdown the graph summarizer model - while the summarizer model is being shutdown after use, we don't terminate the graph-extraction model as it's used for GraphRAG responses!
    try:
        config_data = read_config(['graph_summarizer_access_url', 'graph_summarizer_server_port', 'graph_model_access_url', 'graph_model_server_port'])
        summarizer_url = f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}"
        response = utils.shutdown_waitress_server(summarizer_url)
        if isinstance(response, dict) and response.get('success', False):
            print(f"\nSuccessfully shut down graph summarizer model at URL {summarizer_url}\n")
        else:
            handle_error_no_return(f"\nCould not shut down graph summarizer model at URL {summarizer_url}, proceeding regardless...\n")
    except Exception as e:
        handle_error_no_return("Could not shutdown graph summarizer model, encountered error: ", e)

    # 2. Ensure chat-LLM server is online
    try:
        read_return = read_config(['local_llm_server'])
        server_to_start = read_return['local_llm_server']
    except Exception as e:
        handle_error_no_return("Server-side error, could not read local_llm_server from config.json, encountered error: ", e)
    
    try:
        if server_to_start == 'hf-waitress':
            hf_waitress_base_url = get_url_for_server('hf-waitress')
            if not utils.is_local_server_online(hf_waitress_base_url)['server_online']: # This means the main chat server was likely shut due to insufficient VRAM, so we should also shut down the graph-extraction model

                print("\nMain LLM server offline, shutting down graph-extraction model and attempting restart of chat server...\n")
                graph_model_base_url = f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}"
                response = utils.shutdown_waitress_server(graph_model_base_url)
                if isinstance(response, dict) and response.get('success'):
                    print(f"\nSuccessfully shut down graph model at URL {graph_model_base_url}\n")
                else:
                    handle_error_no_return(f"\nCould not shut down graph model at URL {graph_model_base_url}, proceeding regardless...\n")

                server_starter_response = hf_waitress_server_starter()
                if server_starter_response is not None and server_starter_response.get('success'):
                    print("\nSuccessfully started HF-Waitress Chat server\n")
                    return True
                else:
                    handle_error_no_return(f"\nCould not start HF-Waitress server, proceeding regardless...\n")

            else:
                print("\nHF-Waitress chat server already online\n")
                return True
        
        elif server_to_start == 'llama-cpp':
            server_starter_response = llama_cpp_server_starter()
            if server_starter_response is not None and server_starter_response.get('success'):
                print("\nSuccessfully started llama-cpp Chat server\n")
                return True
            else:
                handle_error_no_return(f"\nCould not start llama-cpp server, proceeding regardless...\n")
        else:
            handle_error_no_return(f"Invalid local LLM server choice: {server_to_start}")
    except Exception as e:
        handle_error_no_return("Could not start local LLM server post-document upload, encountered error: ", e)

    return True


def bulk_upload_files_to_rag_and_records_databases(docs_to_upload: list[dict], data_queue: queue.Queue = None):
    try:
        for count, doc in enumerate(docs_to_upload):
            if data_queue is not None: data_queue.put(f"Performing Step 2 of 2 for document: {doc.get('document_name_and_extension', 'Unknown document name')} - Uploading Document to RAG & Records Databases. Progress: {count + 1} of {len(docs_to_upload)}... | waiting")
            try:
                doc['status'] = 'Processing - Uploading to RAG & Records Databases'
                insert_into_staging_db(doc)
            except Exception as e:
                if data_queue is not None: data_queue.put(f"Error staging file for upload | failure")
                handle_error_no_return(f"Could not update the `status` key in the staging DB. Encountered error: ", e)

            try:
                upload_to_rag_and_records_databases(doc['document_name_and_extension'], doc['txt_filepath'])
                if data_queue is not None: data_queue.put(f"Successfully Uploaded Document: {doc.get('document_name_and_extension', 'Unknown document name')} | success")
                safe_delete_entry_from_staging_db(doc)
            except Exception as e:
                if data_queue is not None: data_queue.put(f"Error uploading file to RAG & Records databases | failure")
                handle_error_no_return(f"Could not upload {doc.get('document_name_and_extension', 'Unknown document name')} to RAG & Records databases, skipping this document. Encountered error: ", e)
            if data_queue is not None: data_queue.put(f"Completed processing file {count + 1} of {len(docs_to_upload)}! | success")
    finally:    # Cleanup!
        perform_post_bulk_upload_cleanup()

    return True


def move_file_to_upload_dir(staged_filename: str, staged_filepath: str):
    
    try:    # copy from staging to upload folder
        final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], staged_filename)   # staged_filename is the clean, secure_filename version of the original filename
        shutil.copy2(staged_filepath, final_filepath)   # copy2 preserves more metadata than copy()
        safe_remove_file_from_filepath(staged_filepath)
        return staged_filename, final_filepath
    except FileNotFoundError as f:
        return handle_local_error(f"File not found: {staged_filepath}", f)
    except Exception as e:
        return handle_local_error("Could not move file to upload folder, encountered error: ", e)


def enable_kosmos_vram_offloading():
    try:
        write_config({'kosmos_offload_vram': True})
        return True
    except Exception as e:
        handle_error_no_return("Could not enable Kosmos VRAM offloading, encountered error: ", e)


def disable_kosmos_vram_offloading():
    try:
        kosmos_offload_vram_enabled = str(read_config(['kosmos_offload_vram'])['kosmos_offload_vram']).lower() == 'true'
        if kosmos_offload_vram_enabled: write_config({'kosmos_offload_vram': False})
        return kosmos_offload_vram_enabled
    except Exception as e:
        handle_error_no_return("Could not disable Kosmos VRAM offloading, encountered error: ", e)


def invoke_offload_kosmos_vram_endpoint():
    kosmos_local_url = read_config(['kosmos_local_url'])['kosmos_local_url']
    url = f"{kosmos_local_url}/reclaim_kosmos_vram"
    response = requests.request("GET", url, headers={}, data={})
    print(response.text)


def bulk_text_extract_from_staging_area(staged_docs_to_upload: list[dict], data_queue: queue.Queue = None):
    '''
    Receives a list of dictionaries from bulk downloader methods, each containing the following keys:

        staged_docs_to_upload = [
            {
                'upload_id': str,
                'user_id': str,
                'document_name_and_extension': str,
                'staged_filepath': str,
                'txt_filepath': str,    # will be added here!
                'embedding_model': str,
                'knowledge_domain': str,
                'source': str,
                'text_extraction_method': str,
                'upload_initiated_datetime': str,
                'staged_datetime': str,
                'status': str
            },...
        ]
    
    And iterates through the list, adding documents to LARS
    '''
    print("\nBulk uploading documents from staging area\n")

    try:

        must_enable_kosmos_vram_offloading_after_bulk_upload_completes = disable_kosmos_vram_offloading()

        for count, doc in enumerate(staged_docs_to_upload):
            if data_queue is not None: data_queue.put(f"Performing Step 1 of 2 for document: {doc.get('document_name_and_extension', 'Unknown document name')} - Data Extraction. Progress: {count + 1} of {len(staged_docs_to_upload)}... | waiting")
            try:    # Move to upload dir
                filename, filepath = move_file_to_upload_dir(doc['document_name_and_extension'], doc['staged_filepath']) # will also delete from staging dir
            except Exception as e:
                if data_queue is not None: data_queue.put(f"Error processing file - Could not move file to upload directory | failure")
                handle_error_no_return(f"Could not move file to upload directory. WARNING: {doc.get('document_name_and_extension', 'Unknown document name')} will not be uploaded to RAG & Records databases. Encountered error: ", e)
                continue
            
            try:
                doc['status'] = 'Extracting Text - Moved to Upload Dir'
                insert_into_staging_db(doc)
            except Exception as e:
                if data_queue is not None: data_queue.put(f"Error processing file - Could not update file status in staging DB | failure")
                handle_error_no_return(f"Could not update the `status` key in the staging DB. WARNING: {doc.get('document_name_and_extension', 'Unknown document name')} will not be uploaded to RAG & Records databases. Encountered error: ", e)
            
            try:    # Get PDF filepath for upload
                pdf_filepath = get_pdf_filepath_for_upload(filename, filepath)
            except Exception as e:
                if data_queue is not None: data_queue.put(f"Error processing file - Could not get filepath for upload | failure")
                handle_error_no_return(f"Could not get PDF filepath for upload. WARNING: {doc.get('document_name_and_extension', 'Unknown document name')} will not be uploaded to RAG & Records databases. Encountered error: ", e)
                continue
            
            try:    # Get text from PDF
                txt_filepath = get_text_extract_from_pdf(pdf_filepath)
                if data_queue is not None: data_queue.put(f"Successfully extracted text from document: {doc.get('document_name_and_extension', 'Unknown document name')} | success")
            except Exception as e:
                if data_queue is not None: data_queue.put(f"Error processing file - Could not extract text from the document | failure")
                handle_error_no_return(f"Could not extract text from the PDF document. WARNING: {doc.get('document_name_and_extension', 'Unknown document name')} will not be uploaded to RAG & Records databases. Encountered error: ", e)
                continue

            try:
                doc['txt_filepath'] = txt_filepath
                doc['status'] = 'Text Extraction Completed - Path to TXT File Saved'
                insert_into_staging_db(doc) # update the `txt_filepath` key in the staging DB
            except Exception as e:
                if data_queue is not None: data_queue.put(f"Error processing file - Could not save text extract | failure")
                handle_error_no_return(f"Could not update the `txt_filepath` key in the staging DB. WARNING: {doc.get('document_name_and_extension', 'Unknown document name')} will not be uploaded to RAG & Records databases. Encountered error: ", e)
        
        try:
            if must_enable_kosmos_vram_offloading_after_bulk_upload_completes: 
                enable_kosmos_vram_offloading()
                invoke_offload_kosmos_vram_endpoint()
        except Exception as e:
            handle_error_no_return("Could not enable Kosmos VRAM offloading after bulk upload completes, skipping. Encountered error: ", e)

    finally:    # Cleanup
        global DOCLING_CONVERTER, VLM_PIPELINE_OPTIONS, PDF_PIPELINE_OPTIONS
        DOCLING_CONVERTER, VLM_PIPELINE_OPTIONS, PDF_PIPELINE_OPTIONS = None, None, None
        utils.safe_empty_cuda_cache()

    return True


@app.route('/bulk_upload_files', methods=['POST'])
def bulk_upload_files():
    '''
    Step Two of the bulk upload process!

    Will receive a JSON list of dictionaries, each entry comprising a complete set of information on a file to be uploaded:
        user_id: str
        upload_id: str
        staged_filepath: str
        txt_filepath: str
        document_name_and_extension: str
        embedding_model: str
        knowledge_domain: str
        source: str
        text_extraction_method: str
        upload_initiated_datetime: str
        staged_datetime: str
        status: str
    
    This list is used to first bulk text-extract (eg OCR) all files, and then bulk upload to the RAG & Records databases.

    Files are removed from the staging area as text extraction progresses and their respective entries are removed from the staging database after sucessful RAG upload.
    They can then be found in the docs_loaded DB.
    '''
    try:
        docs_to_upload = request.form['docs_to_upload']
        docs_to_upload = json.loads(docs_to_upload)
    except Exception as e:
        return handle_api_error("Server-side error, could not read bulk upload request, encountered error: ", e)
    
    data_queue = queue.Queue()
    stop_event = threading.Event()

    def sync_task():
        try:
            try:
                bulk_text_extract_from_staging_area(docs_to_upload, data_queue)
            except Exception as e:
                return handle_api_error("Server-side error, could not perform bulk text extraction, encountered error: ", e)
            
            try:
                bulk_upload_files_to_rag_and_records_databases(docs_to_upload, data_queue)
            except Exception as e:
                return handle_api_error("Server-side error, could not bulk upload files to RAG & Records databases, encountered error: ", e)
        finally:
            data_queue.put(None)
            print("\n\nDocument upload stream done, breaking thread\n\n")

    def bulk_process_files():

        try:
            thread = threading.Thread(target=sync_task)
            thread.start()
        except Exception as e:
            data_queue.put(f"Could not start sync process for bulk file upload | failure")
            return handle_api_error("Could not start thread in the bulk_process_files() method, encountered error: ", e)

        while True:
            if stop_event.is_set(): #TODO: Add Cancel-Sync button to UI! Logic here will be simialr to STOP_GENERATION in hf_waitress.py
                print("\n\nStopping bulk file upload stream as requested by stop_event\n\n")
                thread.join()
                break
            output = data_queue.get()
            if output is None:
                print("\n\nNone read, breaking and stopping thread\n\n")
                thread.join()
                break
            yield f"data: {json.dumps(output)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"
    
    print("\n\nBulk file upload begins!\n\n")
    return Response(bulk_process_files(), content_type='text/event-stream')


def save_file_to_staging_dir(input_file):   # input_file is a Request.files object
    try:
        filename = secure_filename(input_file.filename)
        filename = filename.replace("PDF", "pdf") if "PDF" in filename else filename
        filepath = os.path.join(app.config['UPLOAD_STAGING_FOLDER'], filename)
        input_file.save(filepath)
        print(f"\nSaved file {filename} to {filepath} successfully.\n")
        return filename, filepath
    except Exception as e:
        return handle_local_error("Could not save file to staging directory, encountered error: ", e)


@app.route('/file_transfer_to_staging', methods=['POST'])
def file_transfer_to_staging():
    '''
    ### MAIN ENTRY POINT FOR (BULK) FILE UPLOADS ###

    Step One - Transfer to Staging Area and create entry in Staging-Records DB, after checking if the file and record are already staged.

    Step Two - Invoke bulk_upload_files to perform two-step upload:
        - First, extracting text for all staged documents (benefit :when using local OCR models such as Kosmos, they can be loaded once and used to process all files before offloading)
        - Second, uploading to RAG & Records databases

    Receives a JSON object named `file_transfer_info` comprising the following keys detailing the file to be uploaded:
        user_id: str
        upload_id: str
        document_name_and_extension: str
        embedding_model: str
        knowledge_domain: str
        source: str
        text_extraction_method: str
        upload_initiated_datetime: str
    
    And the file to uploaded as a Request.files object.

    Will save the file to the staging area, and add the following keys to the `file_transfer_info` dictionary and return it:
        staged_datetime: str
        staged_filepath: str
        txt_filepath: str   # will be left blank for now and updated after text extraction by the bulk_text_extract method
        status: str

    Additionally, the `document_name_and_extension` key will be updated to the secure_filename version of the original filename.

    So this method receives a single dict, the front-end JS is responsible for collecting them into a list for step two.
    '''

    print("\n\nFile transfer to staging area begins!\n\n")
    
    try:    # First validate file-transfer info and check if file is already staged
        file_transfer_info = request.form['file_transfer_info']
        file_transfer_info = json.loads(file_transfer_info)
    except Exception as e:
        return handle_api_error("Server-side error, could not read file transfer data, encountered error: ", e)

    try:    # validate file transfer data
        validate_file_transfer_info(file_transfer_info) # exception will be raised if invalid
    except Exception as e:
        return handle_api_error("Error validating request: ", e)

    try:
        staged_file_info = check_if_file_already_staged(file_transfer_info)
        if staged_file_info is not None:
            return jsonify(success=True, file_previously_staged=True, staged_file_info=staged_file_info)
    except Exception as e:
        handle_error_no_return("Server-side error, could not check if files are already staged, proceeding with upload. Encountered error: ", e)
    
    try:    # New file, proceed with fresh upload
        file_to_stage = request.files['file']
    except Exception as e:
        return handle_api_error("Server-side error recieving file: ", e)
    
    try:
        staged_filename, staged_filepath = save_file_to_staging_dir(file_to_stage)
    except Exception as e:
        return handle_api_error("Server-side error, could not save file to staging area, encountered error: ", e)

    try:
        file_transfer_info['document_name_and_extension'] = staged_filename # secure_filename version of original filename
        file_transfer_info['staged_datetime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_transfer_info['staged_filepath'] = staged_filepath
        file_transfer_info['txt_filepath'] = ''
        file_transfer_info['status'] = 'Staged - File Saved to Staging Area'
        insert_into_staging_db(file_transfer_info, skip_check=True)  # `file_already_staged` check has already been performed, so this will insert directly into the DB barring some error.
    except Exception as e:
        return handle_api_error("Server-side error, could not insert file into staging DB, encountered error: ", e)
    
    return jsonify(success=True, file_previously_staged=False, staged_file_info=file_transfer_info)
    


### *** FUTURE USE: Check if files are already staged for a user, and cancel pending uploads ***
def check_for_pending_uploads_for_user(user_id: str):
    try:
        conn, cursor = init_and_connect_to_upload_staging_db()
    except Exception as e:
        return handle_local_error("Could not connect to upload staging DB to check for pending uploads, encountered error: ", e)

    try:
        cursor.execute('''
            SELECT * FROM upload_staging
            WHERE user_id = ?
            ''',
            (user_id,)
        )
        result = cursor.fetchall()  # fetchall() returns a list of tuples equal in length to the number of columns in the query. For `SELECT *` this is the number of columns in the table.
        if result:
            files_staged = []
            for file in result:
                file_info = {
                    'upload_id': file[1],
                    'user_id': file[2],
                    'document_name_and_extension': file[3],
                    'txt_filepath': file[4],
                    'staged_filepath': file[5],
                    'embedding_model': file[6],
                    'knowledge_domain': file[7],
                    'source': file[8],
                    'text_extraction_method': file[9],
                    'upload_initiated_datetime': file[10],
                    'staged_datetime': file[11],
                    'status': file[12]
                }
                files_staged.append(file_info)
            return files_staged
        else:
            return []
    except Exception as e:
        return handle_local_error("Could not check for pending uploads, encountered error: ", e)
    finally:
        cursor.close()
        conn.close()


@app.route('/check_for_pending_uploads', methods=['POST'])
def check_for_pending_uploads():
    try:
        user_id = request.form['user_id']
    except Exception as e:
        return handle_api_error("Server-side error, could not read user ID, encountered error: ", e)
    
    try:
        files_staged = check_for_pending_uploads_for_user(user_id)
        return jsonify(success=True, files_already_staged = True, list_of_files_staged=files_staged) if len(files_staged) > 0 else jsonify(success=True, files_already_staged = False, list_of_files_staged=[])
    except Exception as e:
        return handle_api_error("Server-side error, could not check for pending uploads, encountered error: ", e)
    

def cancel_upload_for_staged_files(user_id: str):
    try:
        conn, cursor = init_and_connect_to_upload_staging_db()
    except Exception as e:
        return handle_local_error("Could not connect to upload staging DB to cancel pending uploads, encountered error: ", e)
    
    # 1. Check for staged files and attmept disk deletion
    try:
        cursor.execute('''
            SELECT * FROM upload_staging
            WHERE user_id = ?
            ''',
            (user_id,)
        )
        result = cursor.fetchall()
        if result:
            for file in result:
                safe_remove_file_from_filepath(file['staged_filepath'])
                safe_remove_file_from_filepath(file['txt_filepath'])
    except Exception as e:
        return handle_local_error("Could not check for pending uploads, encountered error: ", e)
    
    # 2. Delete from staging DB
    try:
        cursor.execute('''
            DELETE FROM upload_staging
            WHERE user_id = ?
            ''',
            (user_id,)
        )
        conn.commit()
    except Exception as e:
        return handle_local_error("Could not delete pending uploads from staging DB, encountered error: ", e)
    
    finally:
        cursor.close()
        conn.close()
    
    return jsonify(success=True)


@app.route('/cancel_pending_uploads', methods=['POST'])
def cancel_pending_uploads():
    try:
        user_id = request.form['user_id']
    except Exception as e:
        return handle_api_error("Server-side error, could not read user ID, encountered error: ", e)
    
    try:
        cancel_upload_for_staged_files(user_id)
    except Exception as e:
        return handle_api_error("Server-side error, could not cancel pending uploads, encountered error: ", e)
    
    return jsonify(success=True)

#TODO: Resume interrupted uploads! - Should be triggered by LARS Server (this file) on boot, not the UI on refresh!

### *** END FUTURE USE: Check if files are already staged for a user, and cancel pending uploads ***



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
            handle_error_no_return("Could not read llama_cpp_access_url and llama_cpp_server_port from config.json, using default localhost:8080 instead. Encountered error: ", e)
            return 'http://localhost:8080'
    elif server_to_check == 'hf-waitress':
        try:
            read_return = read_config(['hf_waitress_access_url', 'hf_waitress_server_port'])
            return f'http://{read_return["hf_waitress_access_url"]}:{read_return["hf_waitress_server_port"]}'
        except Exception as e:
            handle_error_no_return("Could not read hf_waitress_access_url and hf_waitress_server_port from config.json, using default localhost:9069 instead. Encountered error: ", e)
            return 'http://localhost:9069'
    else:
        return handle_api_error("Invalid server_to_check in method get_url_for_server, encountered error: ", e)


@app.route('/llama_cpp_server_starter')
def llama_cpp_server_starter():
    print("\n\nStarting llama.cpp Server\n\n")

    global LLM_CHANGE_RELOAD_TRIGGER_SET
    global LLAMA_CPP_PROCESS
    global LLM_LOADED_UP

    other_server_running = False
    hf_waitress_base_url = get_url_for_server('hf-waitress')
    llama_cpp_base_url = get_url_for_server('llama-cpp')

    # Before attempting to start the llama.cpp server, check if HF-Waitress is running and if so, shut it down:
    try:
        if utils.is_local_server_online(hf_waitress_base_url)['server_available']:
            print("\n\nThe HF-Waitress server is running. Attempting to shut it down before starting the llama.cpp server.\n\n")
            try:
                if utils.shutdown_waitress_server(hf_waitress_base_url)['success']:
                    print(f"\nSuccessfully shut down HF-Waitress server at URL {hf_waitress_base_url}\n")
                else:
                    raise Exception(f"\nCould not shut down HF-Waitress server at URL {hf_waitress_base_url}, proceeding regardless...\n")
                LLM_LOADED_UP = False
            except Exception as e:
                LLM_LOADED_UP = True    # We know the HF-Waitress server is running, which means `hf_waitress.py` is available, so we set LLM_LOADED_UP to True
                other_server_running = True # Set to True as we've determined the other server is running and we failed to terminate it
                handle_error_no_return("Could not terminate running HF-Waitress process before launching llama.cpp. Your IP is likely not whitelisted and thus unauthorized for this action, contact the administrator to whitelist your IP. Additional technical details follow: ", e)
    except Exception as e:
        handle_error_no_return("Warning: Could not check if HF-Waitress server is running. Proceeding to launch llama.cpp server. Encountered error: ", e)   

    is_llama_cpp_running = False
    try:
        is_llama_cpp_running = utils.is_local_server_online(llama_cpp_base_url)['server_available']
    except Exception as e:
        handle_error_no_return("Warning: Could not check if llama.cpp server is running. Proceeding to launch llama.cpp server. Encountered error: ", e)

    model_choice = 'undefined'
    try:
        read_return = read_config(['model_choice'])
        model_choice = read_return['model_choice']
    except Exception as e:
        handle_error_no_return("Missing model_choice in config.json in method llama-cpp-server-starter. Printing error and proceeding with model_choice: 'undefined' ", e)

    if is_llama_cpp_running and not LLM_CHANGE_RELOAD_TRIGGER_SET:
        LLM_LOADED_UP = True
        print(f'\n\nThe llama.cpp server is already loaded and the reload trigger is not set. Simply returning with model choice: {model_choice}\n\n')
        return {'success': True, 'llm_model': model_choice, 'other_server_running': other_server_running}
    
    elif is_llama_cpp_running and LLM_CHANGE_RELOAD_TRIGGER_SET:
        print("\n\nllama.cpp server online and LLM_CHANGE_RELOAD_TRIGGER_SET is set. Attempting to terminate and reload from config.json\n\n")
        try:
            utils.terminate_local_llm_server_process(LLAMA_CPP_PROCESS)
            LLAMA_CPP_PROCESS = None
            LLM_CHANGE_RELOAD_TRIGGER_SET = False
        except Exception as e:
            LLM_LOADED_UP = True  # We know the llama.cpp server is running but there was an error terminating it, so we set LLM_LOADED_UP to True while leaving LLM_CHANGE_RELOAD_TRIGGER_SET to True as we know the server needs to be re-loaded.
            handle_error_no_return("Failed to terminate running llama.cpp process, server was likely launched by a previous session. Returning with the currently loaded LLM. To change, shutdown the previously launched server manually and reload this page. Technical error-details follow: ", e)
            return {'success': True, 'llm_model': 'undefined', 'other_server_running': other_server_running}   # We still return success:True as we've at least determined llama.cpp is running and loaded with a model, even if we cannot reload it.
                 
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
            if utils.is_local_server_online(llama_cpp_base_url)['server_available']:
                print("\n\nllama.cpp server launched succesfully! Returning.\n\n")
                LLM_LOADED_UP = True
                return {'success': True, 'llm_model': model_choice, 'other_server_running': other_server_running}
            time.sleep(timeout)
    except Exception as e:
        handle_error_no_return("Could not check server status after launch attempt, printing error and retrying: ", e)

    return handle_api_error("Failed to start llama.cpp local-server")


def get_hf_waitress_serving_host_and_port():
    try:
        read_return = read_config(['hf_waitress_serving_url', 'hf_waitress_server_port'])
        return read_return['hf_waitress_serving_url'], read_return['hf_waitress_server_port']
    except Exception as e:
        handle_error_no_return("Could not read url & port data for HF-Waitress from config.json, using default localhost:9069 instead. Encountered error: ", e)
        return '0.0.0.0', 9069


def hf_waitress_server_starter(hard_reboot_required = False):
    print("\n\nStarting HF-Waitress Server\n\n")

    global LLM_CHANGE_RELOAD_TRIGGER_SET    # This is only set by LARS basis llama.cpp, so we simply handle it here!
    global LLM_LOADED_UP
    global LLAMA_CPP_PROCESS

    other_server_running = False
    hf_waitress_base_url = get_url_for_server('hf-waitress')
    llama_cpp_base_url = get_url_for_server('llama-cpp')

    if hard_reboot_required:
        print("\nHard-Reboot of HF-Waitress server requested.\n")
        if utils.is_local_server_online(hf_waitress_base_url)['server_available']:
            print("\nHF-Waitress server is running, terminating it before hard-reboot.\n")
            try:
                if utils.shutdown_waitress_server(hf_waitress_base_url)['success']:
                    print(f"\nSuccessfully shut down HF-Waitress server at URL {hf_waitress_base_url}\n")
                else:
                    raise Exception(f"\nCould not shut down HF-Waitress server at URL {hf_waitress_base_url}, proceeding regardless...\n")
            except Exception as e:
                other_server_running = True
                handle_error_no_return("Could not terminate running HF-Waitress process before hard-reboot. Your IP is likely not whitelisted and thus unauthorized for this action, contact the administrator to whitelist your IP. Additional technical details follow: ", e)
            finally:
                LLM_LOADED_UP = False
        else:
            print("\nHF-Waitress server is not running, proceeding to hard-reboot.\n")

    # Before attempting to start the HF-Waitress server, check if llama.cpp is running and if so, shut it down:
    try:
        if LLAMA_CPP_PROCESS is not None and utils.is_local_server_online(llama_cpp_base_url)['server_available']:
            print("\n\nThe llama.cpp server is running. Attempting to shut it down before starting the HF-Waitress server.\n\n")
            try:
                utils.terminate_local_llm_server_process(LLAMA_CPP_PROCESS)
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
        return handle_api_error("Could not read hf_config.json, encountered error: ", e)

    if utils.is_local_server_online(hf_waitress_base_url)['server_available']:
        print("\n\nHF-Waitress server already running. Resetting LLM_CHANGE_RELOAD_TRIGGER_SET and simply returning!\n\n")
        LLM_LOADED_UP = True
        LLM_CHANGE_RELOAD_TRIGGER_SET = False   # The only instance where we're in this method and LLM_CHANGE_RELOAD_TRIGGER_SET is set while the HF-Waitress server is running is when we're trying to switch back to it after running llama.cpp. So we simply reset the flag and return.
        return {'success': True, 'llm_model': model_choice, 'other_server_running': other_server_running}
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
            subprocess.Popen(full_command, creationflags=subprocess.CREATE_NEW_CONSOLE)   #Popen is non-blocking, so the server will keep running in the background
        else:
            # Platform & container agnostic - On Linux/Unix, you need to explicitly provide the arguments as a list to avoid shell interpretation issues:
            command_list = [base_command]  # 'python3'
            # Add script name
            command_list.append('hf_waitress.py')
            # Add any additional arguments
            if launch_args.strip():  # Only add if there are actual arguments
                command_list.extend(launch_args.split())
            
            with open('hf_waitress_output_log.txt', 'w') as f:
                subprocess.Popen(command_list)

    except Exception as e:
        return handle_api_error(f"Could not launch HF-Waitress process in directory: {os.getcwd()}, encountered error: ", e)

    timeout = 5   # seconds
    attempts = 120  # 120 * 5 = 600 seconds = 10 minutes

    try:
        for _ in range(attempts):
            if utils.is_local_server_online(hf_waitress_base_url)['server_available']:
                print("\n\nHF-Waitress server launched succesfully! Returning.\n\n")
                LLM_LOADED_UP = True
                return {'success': True, 'llm_model': model_choice, 'other_server_running': other_server_running}
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
        return handle_api_error("Could not read request, encountered error: ", e)

    try:
        result = hf_waitress_server_starter(hard_reboot_required)
        return jsonify(result)
    except Exception as e:
        return handle_api_error("Could not start HF-Waitress server, encountered error: ", e)


@app.route('/check_local_llm_server_status', methods=['POST'])
def check_local_llm_server_status():    
    try:
        server_to_check = request.form['server_to_check']
    except Exception as e:
        return handle_api_error("Server-side error, could not read server_to_check from the POST request, encountered error: ", e)
    
    try:
        server_url = get_url_for_server(server_to_check)
        server_online = utils.is_local_server_online(server_url)['server_online']
        return jsonify({'success': True, 'server_online': server_online})
    except Exception as e:
        return handle_api_error(f"Error checking {server_to_check} server status, encountered error: ", e)


@app.route('/local_llm_server_starter')
def local_llm_server_starter():
    print("\n\nStarting Local LLM Server\n\n")

    try:
        read_return = read_config(['local_llm_server'])
        server_to_start = read_return['local_llm_server']
    except Exception as e:
        return handle_api_error("Server-side error, could not read local_llm_server from config.json, encountered error: ", e)

    try:
        if server_to_start == 'hf-waitress':
            result = hf_waitress_server_starter()
            return jsonify(result)
        elif server_to_start == 'llama-cpp':
            result = llama_cpp_server_starter()
            return jsonify(result)
        else:
            return handle_api_error(f"Invalid local LLM server choice: {server_to_start}")
    except Exception as e:
        return handle_api_error("Server-side error, could not start local LLM server, encountered error: ", e)

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
        if not check_if_container_is_running('falkor-db'):
            bring_graph_db_online()
            time.sleep(5)   # While the bring-graph-db-online() method waits for the container to start, loading the datasets takes a bit longer so we wait 5 seconds before proceeding.
    except Exception as e:
        return handle_local_error(f"Could not bring graph DB online, unable to delete knowledge graph for {knowledge_domain} domain. Encountered error: ", e)
    
    try:
        client = get_graph_db_client()
        graph = client.select_graph(knowledge_domain)
        graph.delete()  # client.delete_graph() is unsupported by FalkorDB. Delete individual nodes with Cypher: `MATCH (n) DETACH DELETE n`
        print(f"Successfully deleted graph for {knowledge_domain} domain in graph DB")
    except Exception as e:
        handle_error_no_return(f"Could not delete graph for {knowledge_domain} domain in graph DB, encountered error: ", e)


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

    try:
        delete_knowledge_domain_graph(knowledge_domain)
    except Exception as e:
        handle_error_no_return(f"Could not delete knowledge graph for {knowledge_domain} domain in graph DB, encountered error: ", e)
    
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
        # selected_embedding_model = 'all-MiniLM-L6-v2'
        selected_embedding_model = 'Qwen/Qwen3-Reranker-0.6B'

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

        '''
        Studies show that LLMs and Transformers in-general tend to perform better when the most relevant context is towards the beginning or end of the input, while important context in between tends to get 'lost in the middle'! 
        This can be a serious problem for a large multi-turn conversation, wherein extensive back-and-forth query-response history exists and grows with each prompt. 
        Therefore, the re-ranker method has been modified below to return a reversed context docs list, placing the most relevant docs at the end, so the list is now in ascending order of relevance. 
        This should be helpful right from query 1 especially when the system prompt is large!
        '''
        return ranked_documents[::-1]   #Slice to reverse the list, as `.reverse()` would return None because it creates an inplace change on the original list without returning anything

    except Exception as e:
        handle_local_error("Could not reorder documents, encountered error: ", e)
        return [doc.page_content for doc in documents]


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


def format_prompt_for_hf_waitress(formatted_prompt:str, user_query:str, current_sequence_id:int, system_prompt:str, skip_system_prompt:bool) -> str:

    print("\n\nFormatting prompt for hf-waitress\n\n")

    try:
        exl2, exl2_prompt_template_format, vision, flux_diffusers = read_config_for_hf_waitress_prompt_formatting()
    except Exception as e:
        handle_error_no_return("Could not read exl2 details from config.json / hf-config.json, encountered error: ", e)

    if exl2:
        return(prompt_formatting_module.manually_format_prompt_with_prompt_template(formatted_prompt, user_query, current_sequence_id, system_prompt, exl2_prompt_template_format, skip_system_prompt))

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
                                    {{"role": "system", "content": {json.dumps(system_prompt)}}},
                                    {{"role": "user", "content": {json.dumps(user_query)}}}
                                ]
                            }}
                        '''                    

                    formatted_prompt = str(first_prompt_json)
    except Exception as e:
        handle_error_no_return("Could not format prompt for hf-waitress in method format_prompt_for_hf_waitress, encountered error: ", e)

    return formatted_prompt


def get_hf_waitress_formatted_user_prompt(formatted_user_prompt: str, llm_response: str) -> str:
    history_prompt_json = json.loads(formatted_user_prompt)
    new_response = {"role":"assistant", "content":llm_response}
    history_prompt_json['messages'].append(new_response)
    updated_history_prompt_json = json.dumps(history_prompt_json, indent=4)
    return str(updated_history_prompt_json)


def combine_and_deduplicate_search_results(whoosh_results, vector_results):
    print("\n\nCombining whoosh and vector results\n\n")

    combined_results = []

    # Convert whoosh results to Document objects
    for result in whoosh_results:
        combined_results.append(Document(
            page_content=result['content'].strip().replace('\n', ' '),
            metadata={
                'source_link': result['source_link'],
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
        handle_error_no_return("Could not filter out duplicate documents in method combine_and_deduplicate_search_results. Returning all results. Encountered error: ", e)
    
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


def read_hf_waitress_multimodal_config() -> tuple[bool, bool]:
    try:
        hf_read_return = read_hf_config(['flux_diffusers', 'vision'])
        return (
            str(hf_read_return['flux_diffusers']).lower() == 'true',
            str(hf_read_return['vision']).lower() == 'true'
        )
    except Exception as e:
        handle_error_no_return("Could not determine if flux_diffusers or vision model in method read_hf_waitress_multimodal_config, encountered error: ", e)
        return False, False


def prepare_special_model_response(formatted_prompt:str, user_query:str, current_sequence_id:int, new_sequence_id:int, stream_session_id:str, local_llm_server:str) -> dict:
    print("\n\nPreparing special model response\n\n")
    try:
        is_diffusers = local_llm_server == 'hfw-diffusers'
        formatted_prompt = format_prompt_for_hf_waitress(
            formatted_prompt="" if is_diffusers else formatted_prompt, 
            user_query=user_query, 
            current_sequence_id=0 if is_diffusers else current_sequence_id, 
            system_prompt="",  
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
        formatted_updated_prompt = prompt_formatting_module.manually_format_prompt_with_prompt_template(formatted_history_prompt, user_query, current_sequence_id, base_template, local_llm_chat_template_format, skip_system_prompt)
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
    print("\nChecking if special handling for multi-modal models is required...\n")
    if local_llm_server != 'hf-waitress':   #if llama.cpp
        return local_llm_server, None
    
    flux_diffusers, vision = read_hf_waitress_multimodal_config()

    if flux_diffusers or file_attached:
        new_sequence_id = prepare_for_quick_response(current_sequence_id, regeneration_request)
        new_local_llm_server='hfw-diffusers' if flux_diffusers else 'hfw-vision'    # 'hfw-vision' since a file is attached for visual analysis!
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
            return handle_local_error("Could not prepare special model response in method setup_for_local_llm_response, encountered error: ", e)
    
    if vision: 
        return 'hfw-vision', None
    
    return local_llm_server, None   # Likely hf-waitress, but not multi-modal (hfw-diffusers or hfw-vision)


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


def extract_content_source_and_page_data_from_summary_text(summary_text: str) -> tuple[str, str, list]:
    '''
    Extracts content data, source document name and page numbers from a text string ending with the pattern:
    {Source Document Name: xxx}\n{Page Number(s): [y,z]}\n\n
    This pattern is established in the process_nodes_and_relationships method of hf_waitress.py
    
    Args:
        summary_text (str): The input text containing the metadata
    
    Returns:
        tuple[str, str, list]: (content_data, source_document_name, page_numbers_list)
    '''
    try:
        source_pattern = r'{Source Document Name: (.*?)}'   # () creates a capturing group and .*? matches any char except newline zero or more times, non-greedily
        source_match = re.search(source_pattern, summary_text)
        source_doc_name = source_match.group(1) if source_match else ""  # group(1) returns the first (and in this case, only) capturing group. 0 would return the entire match.
        
        page_pattern = r'{Page Number\(s\): \[(.*?)\]}'
        page_match = re.search(page_pattern, summary_text)
        if page_match:  # Convert string representation of list to actual list of integers
            pages_str = page_match.group(1)
            pages = [int(p.strip()) for p in pages_str.split(',')]
        else:
            pages = []
        
        content_data = summary_text[:source_match.start()].strip() if source_match else summary_text.strip()

        return content_data, source_doc_name, pages
    except Exception as e:
        handle_error_no_return("Could not extract content data, source document name and page numbers from summary text, returning unchanged summary text. Encountered error: ", e)
        return summary_text, "", []


def get_summary_report(summarized_chunk_entities: dict, graph_rag_context_length_limit_chars: int, user_query: str) -> str:
    print(f"\n\nGetting summary report\n\n")
    
    summary_report = set()
    summary_doc_objects = []
    
    try:
        
        for _, chunk_data in summarized_chunk_entities.items():
            source_doc_name = chunk_data['source_doc_name']
            
            if source_doc_name == 'user_query':
                print("\nSkipping user query chunk\n")
                continue
            
            try:
                for node in chunk_data['entities_and_relationships']['nodes']:

                    if not node.get('summary'):
                        continue    # Skip nodes with no summaries
                    
                    for summary in node.get('summary', []): # There may be multiple summaries for a single node, so we iterate over the list of summaries.
                        try:
                            if summary is not None and summary != '':
                                summary_preface_string = f"Summary for entity '{node['name']}' of type '{node['type']}'"

                            try:
                                content_data, source_doc_name, pages = extract_content_source_and_page_data_from_summary_text(summary)
                                source_link = f"http://llm-citations-database.net/source?doc_name={source_doc_name}&page_number={[pages[0]]}"
                                summary_doc_objects.append(Document(page_content=f"{summary_preface_string} -\nsource_link:{source_link}:\n{summary}\nsource_link:{source_link}\n\n", metadata={'page_number': pages, 'source': source_doc_name}))
                            except Exception as e:
                                handle_error_no_return("Could not convert GraphRAG context to Document object, skipping. Encountered error: ", e)

                            entry = (
                                f"{summary_preface_string} - {summary}" #The summary, as generated in the process_nodes_and_relationships method of hf_waitress.py, contains metadata and newline spacing.
                            )
                            summary_report.add(entry)
                        except Exception as e:
                            handle_error_no_return("Error processing a node's summary when adding to summary report. Skipping this summary. encountered error: ", e)
            
            except Exception as e:
                handle_error_no_return("Error processing node in chunk_data when adding to summary report, likely a corrupt dict. Skipping node summaries for this chunk. encountered error: ", e)
            
            try:
                for relationship in chunk_data['entities_and_relationships']['relationships']:
                
                    if not relationship.get('summary'):
                        continue    # Skip relationships with no summaries

                    for summary in relationship.get('summary', []):
                        try:
                            if summary is not None and summary != '':
                                summary_preface_string = f"Summary for relationship '{relationship['relationship']}' between entities '{relationship['source']}' and '{relationship['target']}'"

                            try:
                                content_data, source_doc_name, pages = extract_content_source_and_page_data_from_summary_text(summary)
                                source_link = f"http://llm-citations-database.net/source?doc_name={source_doc_name}&page_number={[pages[0]]}"
                                summary_doc_objects.append(Document(page_content=f"{summary_preface_string} -\nsource_link:{source_link}:\n{summary}\nsource_link:{source_link}\n\n", metadata={'page_number': pages, 'source': source_doc_name}))
                            except Exception as e:
                                handle_error_no_return("Could not convert GraphRAG context to Document object, skipping. Encountered error: ", e)

                                entry = (
                                    f"{summary_preface_string} - {summary}"
                                )
                                summary_report.add(entry)
                        except Exception as e:
                            handle_error_no_return("Error processing a relationship's summary when adding to summary report. Skipping this summary. encountered error: ", e)
            
            except Exception as e:
                handle_error_no_return("Error processing relationship in chunk_data when adding to summary report, likely a corrupt dict. Skipping relationship summaries for this chunk. encountered error: ", e)
    
    except Exception as e:
        handle_error_no_return("Could not process summary report, skipping remaining items and exiting. Encountered error: ", e)
    
    textual_summary_report = ''.join(summary_report)

    if len(textual_summary_report) > graph_rag_context_length_limit_chars:
        try:
            textual_summary_report = ''
            try:
                reranked_summaries_list_ascending = rerank_results_ml(user_query, summary_doc_objects, top_n=len(summary_doc_objects))
            except Exception as e:
                handle_error_no_return("Could not rerank search results, skipping. Encountered error: ", e)
                reranked_summaries_list_ascending = summary_doc_objects
            reranked_summaries_list_descending = reranked_summaries_list_ascending[::-1]    # The `rerank_results_ml` method returns a list of docs in ascending order of relevance, so we need to reverse it so we may iterate starting with the most relevant docs!
            for doc in reranked_summaries_list_descending:
                if len(textual_summary_report) + len(str(doc.page_content)) > graph_rag_context_length_limit_chars:
                    break
                textual_summary_report += str(doc.page_content)
            
            # print(f"\n\nReturning Textual summary report: {textual_summary_report}\n\n")
            return textual_summary_report, reranked_summaries_list_descending
        except Exception as e:
            return handle_local_error("Could not handle summary report that is too long, encountered error: ", e)
    else:
        return textual_summary_report, summary_doc_objects


def get_summary_and_source_documents_for_node(graph, name, node_type):
    # print(f"\nChecking if summary for node {name} of type {node_type} exists in graph\n")

    try:
        node_name = sanitize_names(name)

        query = f"""
            MATCH (n:{node_name} {{name: '%s', type: '%s'}})
            RETURN n.summary AS summary, n.source_documents AS source_documents
        """ % (name.replace("'", ""), node_type.replace("'", ""))

        result = graph.query(query)
        
        if hasattr(result, 'result_set') and result.result_set:
            # print(f"\nExisting summary for node found: {result.result_set[0][0]}\n")
            summary_list = list(result.result_set[0][0]) if result.result_set[0][0] else []
            source_documents_list = list(result.result_set[0][1]) if result.result_set[0][1] else []
            return summary_list, source_documents_list
        else:
            # print(f"\nNo existing summary for node found...\n")
            return [], []   # If no summary is found, return an empty list:

    except Exception as e:
        handle_error_no_return(f"Could not check if node {name} of type {node_type} exists in graph, returning empty list. Encountered error: ", e)
        return [], []


def get_summaries_for_all_nodes(nodes: list, graph: FalkorDB, print_string: str = "", get_source_documents: bool = False):
    nodes_with_existing_summaries = []
    processed_nodes = {}    # Will de-duplicate nodes!

    for count, node in enumerate(nodes):
        # print(f"Checking for existing summary for node {count+1} of {len(nodes)} {print_string}...")
        try:
            name = str(node['name'])
            node_type = str(node['type'])

            node_key = (name, node_type)

            if node_key in processed_nodes:
                # print(f"Skipping duplicate node {name} of type {node_type} when checking for existing summaries in graph DB")
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


def get_summary_and_source_documents_for_relationship(graph, source, target, relationship_type):
    # print(f"\nChecking if summary for relationship {source} -> {target} ({relationship_type}) exists in graph\n")

    source_label = sanitize_names(source)
    target_label = sanitize_names(target)

    try:
        query = f"""
            MATCH (s:{source_label} {{name: '{source}'}})-[r:{relationship_type}]->(t:{target_label} {{name: '{target}'}})
            RETURN r.summary AS summary, r.source_documents AS source_documents
        """

        result = graph.query(query)

        if hasattr(result, 'result_set') and result.result_set:
            # print(f"\nExisting summary for relationship found: {result.result_set[0][0]}\n")
            summary_list = list(result.result_set[0][0]) if result.result_set[0][0] else []
            source_documents_list = list(result.result_set[0][1]) if result.result_set[0][1] else []
            return summary_list, source_documents_list
        else:
            # print(f"\nNo existing summary for relationship found...\n")
            return [], []   # If no summary is found, return an empty list:

    except Exception as e:
        handle_error_no_return(f"Could not check if summary for relationship {source} -> {target} ({relationship_type}) exists in graph, returning empty list. Encountered error: ", e)
        return [], []


def get_summaries_for_all_relationships(relationships: list, graph: FalkorDB, print_string: str = "", get_source_documents: bool = False):
    relationships_with_existing_summaries = []
    processed_relationships = {}    # Will de-duplicate relationships!

    for count, relationship in enumerate(relationships):
        # print(f"Checking for existing summary for relationship {count+1} of {len(relationships)} {print_string}...")
        try:
            source = str(relationship['source'])
            target = str(relationship['target'])
            relationship_type = sanitize_names(str(relationship['relationship']).upper())   # Added as sanitize_names().upper() hence formatting here too!

            relationship_key = (source, target, relationship_type)

            if relationship_key in processed_relationships:
                # print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type}) when checking for existing summaries in graph DB")
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

    #print(f"\n\nMerging chunk entities for graph RAG. Received chunk_entities: \n {chunk_entities}\n\n")
    print(f"\nMerging chunk entities for graph RAG.\n")

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
            chunk_entities_merged[0]['chunk_text'] += f"{chunk_data['chunk_text']} "
            chunk_entities_merged[0]['entities_and_relationships']['nodes'].extend(chunk_data['entities_and_relationships']['nodes'])   # extend() is used to add multiple elements to the end of the list...
            chunk_entities_merged[0]['entities_and_relationships']['relationships'].extend(chunk_data['entities_and_relationships']['relationships'])   # and we don't care about de-duplicating here as that'll happen anyways in the get_summaries step!
            chunk_entities_merged[0]['source_chunks'].extend(chunk_data['source_chunks'])
            chunk_entities_merged[0]['source_doc_name'] += f"{chunk_data['source_doc_name']} "
        
        print(f"\n\nMerged chunk entities for graph RAG. Resulting chunk_entities_merged: \n {chunk_entities_merged}\n\n")
        return chunk_entities_merged
    except Exception as e:
        handle_error_no_return("Could not merge chunk entities for graph RAG, proceeding with original chunk_entities dict. WARNING: Duplicates may be present and negatively impact response quality! Encountered error: ", e)
        return chunk_entities


def assemble_chunks_for_graph_rag(docs, user_query=None):
    '''
    Transforms docs, which is a list of Document objects:

        docs = [
            Document(
                page_content = '<page_content>',
                metadata = {
                    'source': '<source_filepath>',
                    'page_number': '<page_number>'
                }
            ),
            ...
        ]

    into:

        chunk_entities = {
            '<graph_chunk_number>': {
                'chunk_text': '<text>',
                'source_chunks': '<chunk_numbers>', #eg: [12,13,14]
                'source_doc_name': '<name>'
            },
            ...
        }
    
    For the purposes of GraphRAG's query-response pipeline.
    Since chunk content is not necessarily contigous document content, we need to treat each chunk as a separate entity.
    '''
    try:
        chunk_entities = {}
        graph_chunk_count = 1

        if user_query is not None:  # For GraphRAG response query-pipeline, we need to add the user query as a chunk
            user_query = user_query.replace("'", "").replace("<br>", "").replace("?", "")
            user_query_chunk_text = f"Do not attempt to answer any query that follows, simply proceed to extract nodes and relationships from the following text:\n{user_query}"
            chunk_entities[graph_chunk_count] = {
                'chunk_text': user_query_chunk_text,
                'source_chunks': [],
                'source_doc_name': 'user_query'
            }
            graph_chunk_count += 1

        print("\nGenerating Graphing Chunks Dictionary...\n")
        
        for count, doc in enumerate(docs):

            try:
                chunk_source_filepath = str(doc.metadata.get('source'))
                source_filename = os.path.basename(chunk_source_filepath)

                page_number_list = []
                try:    # page numbers while useful are non-essential which is why I'm wrapping in a dedicated try-except block that does not raise an error!
                    page_number_list.append(int(doc.metadata.get('page_number')))
                    page_number_list = list(set(page_number_list))   # Remove duplicates
                except Exception as e:
                    handle_error_no_return(f"Could not obtain page number from context document number {count} of {len(docs)} documents, encountered error: ", e)

                chunk_entities[graph_chunk_count] = {
                    'chunk_text': str(doc.page_content).strip().replace("'", ""),
                    'source_chunks': [count],
                    'source_doc_name': source_filename,
                    'page_number': page_number_list
                }

                graph_chunk_count += 1
            except Exception as e:
                handle_error_no_return(f"Error processing context document number {count} of {len(docs)} documents in assemble_chunks_for_graph_db(), encountered error: ", e)

    except Exception as e:
        return handle_local_error(f"Could not assemble chunk_entities dictionary for GraphRAG, encountered error: ", e)

    return chunk_entities


def execute_graph_rag(user_query:str, docs: list[Document]) -> str:
    '''
    Assembles document chunks into a dictionary of entities (via the assemble_chunks_for_graph_rag method, check it for detailed documentation on the structure of docs and chunk_entities) for queries on the GraphDB.

    These chunk_entities are then passed to the graphing model which will process each graph_chunk and append the `entities_and_relationships` key to each chunk_entities dict:
        
        '<entities_and_relationships>': {"nodes": [{"type": "organization","name": "Intel"},{"type": "object","name": "Intel Products"},...], "relationships": [{"source": "Intel","target": "Intel Products","relationship": "business unit"},...]}

    The various chunk_entities in the dict are then merged into a singular chunk_entity for querying the GraphDB to obtain summaries. 
    The merge_chunk_entities_for_graph_rag and get_summaries_from_graph_db methods are respectively used for this purpose.

    The obtained summaries are deduplicated and formatted into a summary report via the get_summary_report method, and finally re-ranked and trimmed to obtain the final graphRAG context, which is then returned.
    '''
    
    print(f"\n\nExecuting GraphRAG. Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    try:
        bring_graph_db_online()
        bring_graph_extraction_model_online()
    except Exception as e:
        return handle_local_error("Could not bring graph DB or graphing model online, encountered error: ", e)

    try:
        chunk_entities = assemble_chunks_for_graph_rag(docs, user_query)
    except Exception as e:
        return handle_local_error("Could not assemble chunks for graph DB, encountered error: ", e)
    
    try:
        selected_knowledge_domain = read_config(['selected_knowledge_domain'])['selected_knowledge_domain']    
        client = get_graph_db_client()
        graph = client.select_graph(selected_knowledge_domain)  # Will create the graph if it doesn't exist
    except Exception as e:
        return handle_local_error(f"Could not connect to / initialize graph for '{selected_knowledge_domain}' domain in graph DB, encountered error: ", e)

    try:
        complete_chunk_entities = extract_all_entities_and_relationships(chunk_entities=chunk_entities, rag_response_mode=True)  # RAG response mode = True
    except Exception as e:
        return handle_local_error("Failed to extract entities and relationships from chunk entities, encountered error: ", e)

    try:
        merged_graph_rag_entities_and_relationships_dict = merge_chunk_entities_for_graph_rag(complete_chunk_entities)
    except Exception as e:
        return handle_local_error("Fatal error merging chunk entities for GraphRAG: ", e)

    try:
        summarized_and_deduplicated_chunk_entities = get_summaries_from_graph_db(merged_graph_rag_entities_and_relationships_dict, selected_knowledge_domain, graph)
    except Exception as e:
        return handle_local_error("Could not store entities and relationships in graph DB, encountered error: ", e)

    try:
        graph_rag_context_length_limit_chars = read_config(['graph_rag_context_length_limit_chars'])['graph_rag_context_length_limit_chars']
        summary_report, reranked_summaries_list_descending = get_summary_report(summarized_and_deduplicated_chunk_entities, graph_rag_context_length_limit_chars, user_query)
    except Exception as e:
        return handle_local_error("Could not get summary report, encountered error: ", e)

    return summary_report, reranked_summaries_list_descending


def hf_waitress_streaming_request_response_handler(endpoint_url, headers, payload):
    print(f"\nHF-Waitress Streaming Request Response Handler Invoked\n")
    try:
        response = requests.post(endpoint_url, headers=headers, data=payload, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes so we can catch them in the except block

        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data:"):
                    event_data = line[6:].strip()
                    try:
                        token = str(json.loads(event_data))
                        full_response += token
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
        return handle_local_error(f"Failed request to HF-Waitress {endpoint_url} API, encountered error: ", e)


def get_request_params_for_generic_hf_waitress_request():
    try:
        hf_config_data = read_hf_config(['exl2'])
        config_data = read_config(['hf_waitress_access_url', 'hf_waitress_server_port'])
        exl2 = str(hf_config_data['exl2']).lower() == 'true'
    except Exception as e:
        return handle_local_error("Could not get graphing request params, encountered error: ", e)

    headers = {
        'Content-Type': 'application/json'
    }

    waitress_url = f"http://{config_data['hf_waitress_access_url']}:{config_data['hf_waitress_server_port']}"

    if not exl2:
        headers['X-Return-Full-Text'] = 'False'
        waitress_url += "/completions_stream"
    else:
        headers['Connection'] = 'keep-alive'
        waitress_url += "/exl2_stream"

    return waitress_url, headers, exl2


def make_request_to_hf_waitress(user_query:str):
    try:
        waitress_url, headers, _ = get_request_params_for_generic_hf_waitress_request()
    except Exception as e:
        return handle_local_error("Could not get request params for generic HF-Waitress request, encountered error: ", e)

    try:
        payload = format_prompt_for_hf_waitress(formatted_prompt="", user_query=user_query, current_sequence_id=0, system_prompt="", skip_system_prompt=True)
        json_payload = json.dumps(payload)
    except Exception as e:
        return handle_local_error("Could not format prompt for generic HF-Waitress request, encountered error: ", e)

    try:
        return hf_waitress_streaming_request_response_handler(waitress_url, headers, json_payload)
    except Exception as e:
        return handle_local_error("Error making request to HF-Waitress, encountered error: ", e)


def parse_service_response(response:str):
    """
    Helper function to parse and clean the service response.
    
    Args:
        response: Raw response string from HF-Waitress
        
    Returns:
        dict: Parsed service selection or default RAG service
    """
    try:
        json_parsed_response = json.loads(response)
        return json_parsed_response
    except Exception as e:
        print(f"\nFailed to parse service response, attempting to literal-eval and maybe even trim. Encountered error: {e}\n")

    try:
        return ast.literal_eval(response)
    except (ValueError, SyntaxError):
        # Sometimes additional text may be present so we need to strip it:
        try:
            print(f"\nAdditional text present, trimming response...\n")
            cleaned_response = prompt_formatting_module.trim_response(
                response,
                '"service":', '}',
                include_start_substring=True,
                include_end_substring=False
            )
            cleaned_response = "{" + cleaned_response.strip() + "}"
            print(f"\nTrimmed response to dictionary: {cleaned_response}\n")
            return ast.literal_eval(cleaned_response)
        except (ValueError, SyntaxError):
            print("Failed to identify selected service even after trimming, defaulting to RAG...")
            return {'service': 'RAG'}


VALID_SERVICES = {
    'rag': {'do_rag': True, 'perform_graph_rag': False},
    'graphdb': {'do_rag': True, 'perform_graph_rag': True},
    'direct response': {'do_rag': False, 'perform_graph_rag': False}
}

DEFAULT_CONFIG = {'do_rag': True, 'perform_graph_rag': False}


def determine_response_service(user_query:str):
    """
    Determines which service should handle the user query.
    
    Args:
        user_query: The user's input query
        
    Returns:
        dict: Configuration for the selected service with 'do_rag' and 'perform_graph_rag' settings
    """
    print(f"\nDetermining response service...\n")

    try:
        service_request_prompt = prompt_formatting_module.get_service_request_prompt(user_query)
    except Exception as e:
        return handle_local_error("Could not get service request prompt, encountered error: ", e)

    try:
        full_response = make_request_to_hf_waitress(service_request_prompt)
    except Exception as e:
        return handle_local_error("Could not make streaming request to HF-Waitress, encountered error: ", e)
    
    try:
        selected_service = parse_service_response(full_response)
        print(f"\nLLM selected service: {selected_service}\n")
    except Exception as e:
        handle_error_no_return("Could not parse service response, encountered error: ", e)
        selected_service = {'service': 'RAG'}

    try:
        service_name = selected_service.get('service', 'RAG').lower().strip()
        print(f"\nService name: {service_name}\n")
        config = VALID_SERVICES.get(service_name, DEFAULT_CONFIG)
        write_config(config)
        print(f"\nConfig: {config}\n")
        return config['do_rag']
    except Exception as e:
        handle_error_no_return("Could not determine service selected by the LLM, defaulting to use Naive RAG. Encountered error: ", e)
        return DEFAULT_CONFIG['do_rag']


def search_knowledge_base(user_query:str, embedding_function:str, force_enable_rag:bool, force_disable_rag:bool, filter_top_k_results_by_reranking:int, fetch_top_k_results_from_vectordb:int, ) -> tuple[list[Document], bool]:
    print("Searching knowledge base")

    if force_disable_rag:
        print("\n\nFORCE_DISABLE_RAG True, force disabling RAG and returning\n\n")
        return [], False, None

    try:
        do_rag = determine_response_service(user_query)
    except Exception as e:
        handle_error_no_return("Could not determine response service, defaulting to use Naive RAG. Encountered error: ", e)
        do_rag = DEFAULT_CONFIG['do_rag']
    
    if force_enable_rag:
        print("\n\nFORCE_ENABLE_RAG True, force enabling RAG and returning\n\n")
        do_rag = True   # We don't check this earlier as if force_enable_rag is True, we also want to determine if GraphRAG should be performed! We only set it to True here to ensure force_enable_rag takes precedence over the LLM's selection.

    if do_rag is False:
        print("Do RAG is False, returning...")
        return [], False, None

    print("Do RAG is True, continuing...")
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

    combined_docs = []
    try:
        combined_docs = combine_and_deduplicate_search_results(whoosh_results, filtered_docs)   # Combine the whoosh and vector results
    except Exception as e:
        handle_error_no_return("Could not combine and deduplicate search results, skipping. Encountered error: ", e)
        combined_docs = filtered_docs

    if not combined_docs:   # i.e. if blank
        print("No documents for citations, setting do_rag to False")
        do_rag = False
        return [], do_rag, None

    try:
        docs = rerank_results_ml(user_query, combined_docs, top_n=filter_top_k_results_by_reranking)
    except Exception as e:
        handle_error_no_return("Could not rerank search results, skipping. Encountered error: ", e)
        docs = combined_docs
    
    # do_rag = determine_do_rag(user_query, docs, force_enable_rag, force_disable_rag)
    
    perform_graph_rag = read_config(['perform_graph_rag'])['perform_graph_rag']

    graph_rag_context = None
    if perform_graph_rag and do_rag:
        try:
            graph_rag_context, reranked_summaries_list_descending = execute_graph_rag(user_query, docs)
            if reranked_summaries_list_descending != []:
                return reranked_summaries_list_descending, do_rag, graph_rag_context
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
    
    try:
        local_llm_server, special_response = handle_special_model_case(config['local_llm_server'], current_sequence_id, file_attached, stream_session_id, user_query, formatted_history_prompt, regeneration_request)
    except Exception as e:
        return handle_api_error("Error determining appropriate model type and server for setup_for_local_llm_response: ", e)
    
    if special_response is not None:    # If a special model response is returned, quick-return here
        print(f"Returning special model response: {special_response}")
        return special_response
    
    if config['force_disable_rag'] or regenerate_with_citations_force_disabled:
        return handle_force_disabled_rag(local_llm_server, formatted_history_prompt, user_query, current_sequence_id, stream_session_id, regeneration_request, config['base_template'], config['local_llm_chat_template_format'], config['skip_system_prompt'])
            
    try:    # RAG Routine Begins: Perform semantic search on the vector DB, lexical search on the whoosh index, combine and rerank results and determine if RAG is necessary
        print("\n\nRAG Routine Begins: Performing semantic search on VectorDB, lexical search on Whoosh index, combining and reranking results and determining if RAG is necessary\n\n") 
        selected_embedding_function = read_config(['selected_embedding_model'])['selected_embedding_model']
        docs, do_rag, graph_rag_context = search_knowledge_base(user_query, selected_embedding_function,
        (config['force_enable_rag'] or regenerate_with_citations_force_enabled), 
        (config['force_disable_rag'] or regenerate_with_citations_force_disabled), 
        config['filter_top_k_results_by_reranking'], config['fetch_top_k_results_from_vectordb'])
    except Exception as e:
        return handle_api_error("Could not process vector search in method setup_for_local_llm_response, encountered error: ", e)

    try:    # Write do_rag to config and prepare RAG context if necessary
        print(f'Do RAG? {do_rag}')
        write_config({'do_rag':do_rag})
        if do_rag:    # Add similarity search results for RAG if necessary!
            QUERIES[key_for_vector_results] = docs
            user_query += f"\n\nThe following context might be helpful in answering the user query above. If so, please reference useful documents by name and specific page numbers in your response:\n"
            if graph_rag_context is not None:
                user_query += f"{graph_rag_context}"
            else:
                user_query += f"{docs}"
    except Exception as e:
        reject_rag()
        handle_error_no_return("Could not write do_rag or prepare RAG context during setup_for_streaming_response, encountered error: ", e)

    try:    # Get full prompt for server
        formatted_updated_prompt = get_full_prompt_for_server(local_llm_server, formatted_history_prompt, user_query, current_sequence_id, config['base_template'], config['local_llm_chat_template_format'], config['skip_system_prompt'])
    except Exception as e:
        return handle_api_error("Could not get formatted_updated_prompt in method setup_for_streaming_response, encountered error: ", e)

    if not regeneration_request or current_sequence_id == 0: current_sequence_id = int(current_sequence_id) + 1
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag, "formatted_user_prompt": formatted_updated_prompt, "sequence_id":current_sequence_id, "server_type":local_llm_server})


def final_cleanup_of_llm_response(llm_response: str) -> str:
    pattern = r'\s*<br>\s+<br>\s*<br>\s*'    # Replace multiple <br> tags with a single <br> tag; \r?\n means optional carriage return and/or newline
    llm_response = re.sub(pattern, '<br><br>', llm_response)
    llm_response = llm_response.strip()
    return llm_response


def construct_citation_html(pdf_tab_buttons_set: set[str], pdf_tab_content_set: set[str], stream_session_id: str) -> str:
    try:
        pdf_section_html = f'<div class="pdf-viewer-container" id="stream{stream_session_id}PdfPane">'
        pdf_section_html += '<div class="tab-buttons">'

        for pdf_tab_button in pdf_tab_buttons_set:
            pdf_section_html += pdf_tab_button
        pdf_section_html += '</div>'   # Close the tab-buttons

        for pdf_tab_content in pdf_tab_content_set:
            pdf_section_html += pdf_tab_content   # tab-content elements are complete divs

        pdf_section_html += "</div>"  # Close the pdf-viewer-container

        return pdf_section_html.strip()
    except Exception as e:
        return handle_local_error("Could not construct citation html, encountered error: ", e)


def save_pdf_to_download_dir(doc_name: str, stream_session_id: str):

    try:
        read_return = read_config(['upload_folder', 'highlighted_docs'])
        upload_folder = read_return['upload_folder']
        highlighted_pdfs_path = read_return['highlighted_docs']
    except Exception as e:
        return handle_local_error("Missing upload_folder in config.json for method save_pdf_to_download_dir. Error: ", e)

    try:
        doc_name_without_extension = os.path.splitext(doc_name)[0]
        pdf_path = os.path.join(upload_folder, doc_name_without_extension + ".pdf")
        output_file_name = f"{doc_name_without_extension}_{stream_session_id}.pdf"        
        output_pdf_path = os.path.join(highlighted_pdfs_path, output_file_name)
        
        shutil.copy2(pdf_path, output_pdf_path) # .copy2() preserves the file permissions and metadata, and is similar to `cp -p`, while .copy() is similar to `cp`

        return output_file_name
    except Exception as e:
        return handle_local_error("Could not save PDF to download dir, encountered error: ", e)


def get_doc_name_and_page_number_from_url(url: str) -> tuple[str, str]:
    try:
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        doc_name = params.get('doc_name', [None])[0]
        page_number = params.get('page_number', [None])[0]
        return doc_name, page_number
    except Exception as e:
        return handle_local_error("Could not get doc_name and page_number from url, encountered error: ", e)


def obtain_singular_page_number_from_url(page_number: str) -> str:
    try:
        page_number = str(page_number)
        if page_number.startswith('[') and not page_number.endswith(']'):
            page_number = page_number + ']'
        elif page_number.endswith(']') and not page_number.startswith('['):
            page_number = '[' + page_number
        
        if page_number.startswith('[') and page_number.endswith(']'):
            try:
                page_number = ast.literal_eval(page_number)
                page_number = page_number[0] if len(page_number) > 0 else "1"
            except Exception as e:
                handle_error_no_return("Could not ast.literal_eval page_number, returning 1. Encountered error: ", e)
                return "1"
        
        return str(page_number)
    except Exception as e:
        handle_error_no_return("Could not obtain singular page number from url, returning 1. Encountered error: ", e)
        return "1"


def obtain_citation_urls_from_llm_response(llm_response: str) -> list[str]:
    try:
        extractor = URLExtract()
        urls = extractor.find_urls(llm_response)
        return [url for url in urls if 'llm-citations-database.net/source' in url]  # Filter out our citation urls
    except Exception as e:
        return handle_local_error("Could not obtain citation urls from llm_response, encountered error: ", e)


def remove_embedded_links_from_llm_response(llm_response: str) -> str:
    try:
        embedded_link_start_pattern = r'\(*\[[^\]]+\]\(\s*'  # Matches embedded links in the format: (*[link text](url))
        embedded_link_end_pattern = r'\s*\){1,2}\.?'          # Matches the closing parenthesis and optional period
        llm_response_without_embedded_links = re.sub(embedded_link_start_pattern, '', llm_response)
        llm_response_without_embedded_links = re.sub(embedded_link_end_pattern, '', llm_response_without_embedded_links)
        return llm_response_without_embedded_links
    except Exception as e:
        return handle_local_error("Could not remove embedded links from llm_response, encountered error: ", e)


def add_citations_and_pdf_browser_to_llm_response(llm_response: str, stream_session_id: str) -> str:
    '''
    This function takes a LLM response, identifies all the documents linked within the response, and returns a new LLM response with citations and a PDF browser.
    '''
    print(f"\n\nAdding citations and PDF browser to LLM response\n\n")

    try:
        llm_response_without_embedded_links = remove_embedded_links_from_llm_response(llm_response)
    except Exception as e:
        return handle_local_error("Could not remove embedded links from llm_response, encountered error: ", e)
    
    tab_count = 1
    tabs_created_for_doc = {}
    pdf_tab_buttons_set = set()
    pdf_tab_content_set = set()

    try:
        urls = obtain_citation_urls_from_llm_response(llm_response) # Extract all URLs from the LLM response
    except Exception as e:
        return handle_local_error("Could not obtain citation urls from llm_response, encountered error: ", e)
    
    for url in urls:    # For each citation (URL), create the HTML to be substituted into the LLM response, and populate the tab button and content for the PDF viewer
        try:
            try:
                doc_name, page_number = get_doc_name_and_page_number_from_url(url)
            except Exception as e:
                handle_error_no_return("Could not get doc_name and page_number from url, skipping this citation. Encountered error: ", e)
                continue

            try:
                doc_name_with_stream_id = save_pdf_to_download_dir(doc_name, stream_session_id) # if we didn't create a unique PDF per stream session, clicking a button would scroll all open instances of the same document in the chat!
            except Exception as e:
                handle_error_no_return("Could not save PDF to download dir, skipping this citation. Encountered error: ", e)
                continue
            
            if not page_number: page_number = 1
            page_number = obtain_singular_page_number_from_url(page_number)
            
            tab_previously_created = False
            if doc_name in tabs_created_for_doc:
                count = tabs_created_for_doc[doc_name]
                tab_previously_created = True
            else:
                count = tab_count

            # 1. Create clickable HTML link for the citation mentioned
            pdf_iframe_id = f"stream{stream_session_id}PdfViewer{str(count)}"
            frame_doc_path = f"/pdf/{doc_name_with_stream_id}#page={str(page_number)}"
            tab_name_string = f"stream{stream_session_id}tabName{str(count)}"            

            citation_html = f'''<a href="javascript:void(0)" onclick="goToPageAndSwitchTab(\'{pdf_iframe_id}\', \'{frame_doc_path}\', \'tab{tab_name_string}\', \'{stream_session_id}\')">Reference: {doc_name}</a>'''
            llm_response_without_embedded_links = llm_response_without_embedded_links.replace(url, citation_html.strip())   # Swap the current URL with the citation HTML
            
            if not tab_previously_created:  # 2. Create tab button and content for the cited document, if not already created
                default_open = ' defaultTabs' if count == 1 else '' # First tab of every response will be open by default
                pdf_tab_button = f'<button class="tab-button{default_open}" stream-session-id="{stream_session_id}" onclick="openTab(event, \'tab{tab_name_string}\', \'{stream_session_id}\')">{doc_name}</button>'
                pdf_tab_buttons_set.add(pdf_tab_button)

                download_link_url = url_for('download_file', filename=doc_name_with_stream_id)
                pdf_tab_content = f'''<div id="tab{tab_name_string}" class="tab-content" stream-session-id="{stream_session_id}"><iframe class="citations-pdf-iframe" id="{pdf_iframe_id}" src="{download_link_url}"></iframe></div>'''
                pdf_tab_content_set.add(pdf_tab_content.strip())

                tabs_created_for_doc[doc_name] = count
                tab_count += 1
        except Exception as e:
            handle_error_no_return("Could not process citation, the LLM may have mis-spelled the document name, skipping. Encountered error: ", e)
            continue

    try:
        pdf_section_html = construct_citation_html(pdf_tab_buttons_set, pdf_tab_content_set, stream_session_id) if len(urls) > 0 else None
    except Exception as e:
        handle_error_no_return("Could not construct citation html, returning blank. Encountered error: ", e)
        pdf_section_html = None

    try:
        llm_response_with_citations_cleaned = final_cleanup_of_llm_response(llm_response_without_embedded_links.strip())
    except Exception as e:
        handle_error_no_return("Could not clean up llm_response_without_embedded_links, skipping. Encountered error: ", e)
        llm_response_with_citations_cleaned = llm_response_without_embedded_links.strip()

    return llm_response_with_citations_cleaned, pdf_section_html


def prepare_model_response_for_storage_to_history_db(download_link_html: str, llm_response: str) -> str:
    model_response_for_history_db = str(llm_response)
    model_response_for_history_db += f"\n\npdf_pane_data={download_link_html}"
    model_response_for_history_db = model_response_for_history_db.strip('\n')
    return model_response_for_history_db


def determine_if_flux_diffusers_is_enabled() -> bool:
    try:
        hf_read_return = read_hf_config(['flux_diffusers'])
        flux_diffusers = str(hf_read_return['flux_diffusers']).lower() == 'true'
        return flux_diffusers
    except Exception as e:
        return False


def get_vector_results_for_get_references(stream_session_id: str) -> tuple[list[Document], bool]:
    try:
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        docs = QUERIES.pop(key_for_vector_results, None)
        return docs, docs is not None
    except Exception as e:
        return handle_local_error("Could not get vector results for stream_session_id in method get_vector_results_for_get_references(), encountered error: ", e)


def get_request_parameters_for_get_references(request: Request) -> tuple[str, str, str, str, str, str, bool, bool]:
    try:
        stream_session_id = request.json['stream_session_id']
        user_query = request.json['user_query']
        llm_response = request.json['llm_response']
        formatted_user_prompt = request.json['formatted_user_prompt']
        chat_id = request.json['chat_id']
        sequence_id = request.json['sequence_id']
        regeneration_request = request.json['regeneration_request']
        return stream_session_id, user_query, llm_response, formatted_user_prompt, chat_id, sequence_id, regeneration_request
    except Exception as e:
        return handle_local_error("Could not read request content in method get_request_parameters_for_get_references(), encountered error: ", e)


def read_config_for_get_references() -> tuple[str, str, str, str, bool]:
    try:
        read_return = read_config(['local_llm_server', 'upload_folder', 'local_llm_chat_template_format', 'exl2_prompt_template_format', 'perform_graph_rag'])
        local_llm_server = read_return['local_llm_server']
        upload_folder = read_return['upload_folder']
        local_llm_chat_template_format = read_return['local_llm_chat_template_format']
        exl2_prompt_template_format = read_return['exl2_prompt_template_format']
        perform_graph_rag = str(read_return['perform_graph_rag']).lower() == 'true'
        return local_llm_server, upload_folder, local_llm_chat_template_format, exl2_prompt_template_format, perform_graph_rag
    except Exception as e:
        return handle_local_error("Could not read config.json in method read_config_for_get_references(), encountered error: ", e)


@app.route('/get_references', methods=['POST'])
def get_references():

    print("\n\nStoring History Post-Response -- Determining if Citations are Necessary\n\n")

    try:
        local_llm_server, upload_folder, local_llm_chat_template_format, exl2_prompt_template_format, perform_graph_rag = read_config_for_get_references()
        exl2 = read_hf_config(['exl2'])['exl2']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to get_references. Error: ", e)

    try:
        stream_session_id, user_query, llm_response, formatted_user_prompt, chat_id, sequence_id, regeneration_request = get_request_parameters_for_get_references(request)
    except Exception as e:
        return handle_api_error("Could not read request content in method get_references, encountered error: ", e)

    do_rag = False
    try:
        docs, do_rag = get_vector_results_for_get_references(stream_session_id)
    except Exception as e:
        handle_error_no_return("Error determining if RAG was used in method get_references - Could not check the QUERIES dict. Proceeding without RAG. Encountered error: ", e)

    if local_llm_server == 'llama-cpp':
        formatted_user_prompt += prompt_formatting_module.append_eot_token_to_llm_response(local_llm_chat_template_format, llm_response)
    elif local_llm_server == 'hf-waitress':
        local_llm_chat_template_format = "hf-transformers"
        flux_diffusers = determine_if_flux_diffusers_is_enabled()
        if flux_diffusers:
            do_rag = False
        else:
            if exl2:
                if perform_graph_rag:
                    try:
                        last_context_index = formatted_user_prompt.rindex("The following context might be helpful in answering the user query above.")
                        formatted_user_prompt = formatted_user_prompt[:last_context_index]
                    except Exception as e:
                        handle_error_no_return("Trimming RAG context unnecessary, skipping. Encountered error: ", e)
                formatted_user_prompt += prompt_formatting_module.append_eot_token_to_llm_response(exl2_prompt_template_format, llm_response)
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

    pdf_section_html = None
    llm_response_with_citation_links = llm_response
    try:
        llm_response_with_citation_links, pdf_section_html = add_citations_and_pdf_browser_to_llm_response(llm_response, stream_session_id)
    except Exception as e:
        handle_error_no_return("Could not add citations and pdf browser to llm_response, encountered error: ", e)
    
    try:
        model_response_for_history_db = prepare_model_response_for_storage_to_history_db(pdf_section_html, llm_response_with_citation_links)
    except Exception as e:
        handle_error_no_return("Could not prep data to store to chat history DB in get_references(), encountered error: ", e)

    try:
        if not regeneration_request:
            stored_datetime, chat_id = store_local_llm_chat_history_to_db(chat_id, sequence_id, stream_session_id, user_query, model_response_for_history_db, formatted_user_prompt, local_llm_server, local_llm_chat_template_format)
        else:
            stored_datetime, chat_id = update_llm_response_in_history_db(chat_id, stream_session_id, user_query, model_response_for_history_db)
    except Exception as e:
        handle_error_no_return("Could not store or update chat history DB in get_references(), encountered error: ", e)

    return jsonify({'success': True, 'response': llm_response_with_citation_links, 'pdf_frame':pdf_section_html, 'stored_datetime':stored_datetime, 'local_llm_server':local_llm_server, 'local_llm_chat_template_format':local_llm_chat_template_format, 'chat_id':chat_id})



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