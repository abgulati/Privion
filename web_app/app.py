from flask import Flask, render_template, request, redirect, url_for, Response
from flask import send_from_directory
from flask_cors import CORS
from flask import jsonify

from sentence_transformers import SentenceTransformer
import torch
import numpy as np

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
# accepts `pdf_path` param as either a String or PurePath. 
# Can directly pass pathlib.Path as it's a subclass of PurePath!

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
import pathlib
import sqlite3
import PyPDF2
import shutil   # Shell Utilities is part of Python's stdlib & is used for file ops
import signal
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

from waitress import serve

import fitz # PyMuPDF - fitz.open(filename) accepts a string or pathlib.Path

from llm_apis import hf_waitress_non_streaming_api_handler, make_request_to_llm_server
import rag_support as rag_support_module
import rag_core as rag_core_module
import butler as butler_module

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

    from docling_core.types.doc import ImageRefMode
except Exception as e:
    print(
        'Could not import Docling OCR, skipping. '
        'If not installed, please run `pip install docling`. '
        f'Encountered error: {e}'
    )

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    import azure.ai.vision as sdk
except Exception as e:
    print(
        'Could not import Azure libraries for ComputerVision & Document-Intelligence OCR. '
        'WARNING: These services will not work, use a local OCR service instead. '
        f'Encountered error: {e}'
    )

try:
    from graph_clustering import apply_leiden_clustering
except Exception as e:
    print(
        'Could not import graph_clustering (likely not installed), skipping. '
        f'Encountered error: {e}'
    )

try:
    import prompt_formatting as prompt_formatting_module
except ImportError:
    print(
        'WARNING: Prompt Formatter module `prompt_formatting.py` is not present. '
        'Skipping import. Exl2 and llama.cpp will not work if using legacy mode!'
    )

try:
    import utils
    from utils import get_url_for_server
except ImportError:
    print("WARNING: utils.py is not present. Skipping import.")

def flash_attention_is_installed() -> bool:
    try:
        import flash_attn
        print(f"Flash Attention is installed: {flash_attn.__version__}")
        return True
    except ImportError:
        print("Flash Attention is not installed.")
        return False
    
try:
    from kokoro import KPipeline
    import soundfile as sf
except Exception as e:
    print(
        'Could not import necessary modules for Kokoro 82M TTS pipeline, '
        f'encountered error: {e}'
    )

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
'''
Note: Allow insecure traffic - Needed to bypass HTTPS requirement for Google Drive OAuth. 
This is for DEV USE ONLY! SWITCH TO SELF-SIGNED CERTIFICATES & HTTPS FOR PRODUCTION!
'''

app = Flask(__name__)
CORS(app)

# Route for the home page, rendering the initial model selection form (legacy)
@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

# Route to display the file loading form
@app.route('/load_file')
def load_file():
    return render_template('model_selection.html', show_file_form=True)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(
        app.config['DOWNLOAD_FOLDER'], filename, as_attachment=False, mimetype='application/pdf'
    )

@app.route('/pdf/<filename>')
def pdf_viewer(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)



#########################------------------GLOBALS!----------------------###############################
LLAMA_CPP_PROCESS = None   #  used in llama-cpp_server_starter and hf-waitress_server_starter
# Primary purpose is to manage the termination of the llama.cpp server process

LLM_CHANGE_RELOAD_TRIGGER_SET = False   # set in config.json and used in llama-cpp_server_starter
ASR_CHANGE_RELOAD_TRIGGER_SET = False   # set in config.json and used in asr-server_starter

DOCLING_CONVERTER = None    # used in docling-ocr_page and bulk-text_extract_from_staging_area
TTS_PIPELINE = None    # used in load_tts_pipeline

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



########################---------------------config setup---------------------###############################
def _early_os_default_base_dir():
    try:
        sysname = platform.system()
    except Exception as e:
        print(f"Could not determine OS, defaulting to 'app' dir. Encountered error: {e}")
        sysname = ''
    if sysname == 'Windows':
        return 'C:/privion_lars_storage'
    elif sysname == 'Linux':
        return '/app/privion_lars_storage'
    else:   # For Darwin (Mac) and otherwise
        return 'app'


def _early_resolve_base_and_config():
    # Code-local bootstrap pointer (same dir as app.py)
    bootstrap_path = os.path.join(os.getcwd(), 'storage_config.json')
    base = _early_os_default_base_dir()
    cfg_path = os.path.join(base, 'config.json')
    if os.path.exists(bootstrap_path):
        try:
            with open(bootstrap_path, 'r') as f:
                boot = json.load(f) or {}
            base = boot.get('base_directory', base)
            cfg_path = boot.get('config_path', cfg_path)
        except Exception as e:
            print(
                f'Could not read storage_config.json, defaulting to base dir: {base}. '
                f'Encountered error: {e}'
            )

    try:
        os.makedirs(base, exist_ok=True)
    except Exception as e:
        print(f"Could not create base directory: {base}. Encountered error: {e}")
    
    return base, cfg_path, bootstrap_path

BASE_DIRECTORY, CONFIG_PATH, BOOTSTRAP_PATH = _early_resolve_base_and_config()

# Create real config if missing - No error handling as exceptions should halt execution.
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as file:
        json.dump({}, file, indent=4)

# Update config with base_directory
try:
    with open(CONFIG_PATH, 'r+') as file:   # 
        '''
        Unlike w, r+ allows updates without overwriting the whole file, 
        but requires seek, to move the file-pointer back to the start of 
        the file before writing, and truncation to remove the original data.

        TODO: `if config.get('base_directory') != BASE_DIRECTORY:`, 
        then move contents of old dir to new dir! 
        Invoke: `_move_contents_of_old_dir_to_new_dir(old_dir, new_dir)`
        Verify if referencing system can handle moves first!
        '''
        config = json.load(file)
        config['base_directory'] = BASE_DIRECTORY
        file.seek(0)
        json.dump(config, file, indent=4)
        file.truncate()

except Exception as e:
    print(f"Could not read config.json, encountered error: {e}")

# If the file doesn't exist (first-run), write defaults. 
# On subsequent runs, simply write-back to storage_config.json,
# which serves as a confirmation mechanism of the paths set.
# Users simply edit base_dir in storage_config.json to move.
try:
    with open(BOOTSTRAP_PATH, 'w') as file:
        json.dump({
                'base_directory': str(pathlib.Path(BASE_DIRECTORY).resolve()),
                'config_path': str(pathlib.Path(CONFIG_PATH).resolve())
            },file,indent=4)
except Exception as e:
    print(f"Could not write bootstrap storage_config.json, encountered error: {e}")

#######################################################################################################



#########################------------Setup & Handle Logging-------------###############################
LOGS_DIR = os.path.join(BASE_DIRECTORY, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOGS_DIR, 'privion_lars_server_log.log')

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

    # 4 - Add the handler to the logger for final LOGGER - Usage: LOGGER.error(f"Error: {e}")
    LOGGER.addHandler(handler)
except Exception as e:
    print(f"\n\nCould not establish logger, encountered error: {e}")


def central_error_logging(message:str, exception:Exception=None):
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


def handle_api_error(message:str, exception:Exception=None):
    error_message = central_error_logging(message, exception)
    return jsonify(success=False, error=error_message), 500 #internal server error


def handle_local_error(message:str, exception:Exception=None):
    _ = central_error_logging(message, exception)
    if exception:
        raise  # preserves the original error & traceback
    raise RuntimeError(message)


def handle_error_no_return(message:str, exception:Exception=None):
    _ = central_error_logging(message, exception)

############################----------------------------------------------###############################


from privion_config_concierge import read_config, write_config, read_hf_config, safe_write_config

@app.route('/config_reader_api', methods=['POST'])
def config_reader_api():    
    try:
        keys = request.json['keys']
    except Exception as e:
        return handle_api_error("Request error - could not read keys from config_reader_api request body:", e)

    try:
        values = read_config(keys)  # send list of keys, get dict of key:values
    except Exception as e:
        return handle_api_error("Server-side error - could not read keys from config.json: ", e)
    
    return jsonify(success=True, values=values)


@app.route('/config_writer_api', methods=['POST'])
def config_writer_api():

    try:
        config_updates = request.json['config_updates']
        print(f"config_updates for config_writer_api: {config_updates}")
    except Exception as e:
        return handle_api_error("Request error - could not read config_writer_api request body: ", e)
    
    try:
        write_return = write_config(config_updates)
        if write_return['restart_required'] and not write_return['skip_reload_trigger']:
            global LLM_CHANGE_RELOAD_TRIGGER_SET
            LLM_CHANGE_RELOAD_TRIGGER_SET = True

        if write_return['reload_tts']:
            global TTS_PIPELINE
            TTS_PIPELINE = None

        if write_return['reload_asr']:
            
            global ASR_CHANGE_RELOAD_TRIGGER_SET
            ASR_CHANGE_RELOAD_TRIGGER_SET = True

    except Exception as e:
        return handle_api_error("Server-side error - could not write to config.json: ", e)
    
    return jsonify({
        "success": write_return['success'],
        "restart_required": write_return['restart_required'],
        "reload_asr": write_return['reload_asr']
    })

############################----------------------------------------------###############################


#########################------------Setup Directories-------------###############################

'''
## Notes on the above workflow for setting up the directories:
1. Everytime the app runs, the OS platform is detected and the appropriate 
    OS-specific base directory is requested above
2. If this is the very first run:
   a. read_config does not find the directory data in config.json
   b. The methods `else` clause is triggered and defaults are written to 
   config.json and subsequently returned
3. If this isn't the very first run, read_config simply returns the OS specific 
directory (windows_base_directory, unix_and_docker_base_directory, or mac_base_directory)
4. Basis this, BASE_DIRECTORY is written to config.json
5. This setup ensures that:
   a. Directories are set correctly at each run
   b. The user can set their preferred directory by easily editing config.json!

Having set the values for the directories above, proceed to actually create them on 
disk IF they don't already exist!
'''
       
try:
    read_return = read_config([
        'model_dir', 'highlighted_docs', 'upload_folder', 'ocr_pdfs', 'pdfs_to_txts', 
        'docs_to_knowledge_graph_dir', 'upload_staging_folder', 
        'graph_db_data_directory', 'draft_models_dir'
    ])
except Exception as e:
    handle_local_error("Could not read paths for core app directories from config, encountered error: ", e)

try:
    os.makedirs(read_return['model_dir'], exist_ok=True)
    os.makedirs(read_return['draft_models_dir'], exist_ok=True)
    os.makedirs(read_return['highlighted_docs'], exist_ok=True)
    os.makedirs(read_return['upload_folder'], exist_ok=True)
    os.makedirs(read_return['ocr_pdfs'], exist_ok=True)
    os.makedirs(read_return['pdfs_to_txts'], exist_ok=True)
    os.makedirs(read_return['docs_to_knowledge_graph_dir'], exist_ok=True)
    os.makedirs(read_return['upload_staging_folder'], exist_ok=True)
    os.makedirs(read_return['graph_db_data_directory'], exist_ok=True)
except Exception as e:
    handle_local_error("Failed to create app directories, encountered error: ", e)

app.config['UPLOAD_FOLDER'] = read_return['upload_folder']
app.config['DOWNLOAD_FOLDER'] = read_return['highlighted_docs']
app.config['UPLOAD_STAGING_FOLDER'] = read_return['upload_staging_folder']

############################----------------------------------------------###############################



############################------------File & Folder Management-------------###############################

def load_json_file(file_path:pathlib.Path) -> dict:
    '''
    Load a JSON file into a dictionary

    Args:
        - file_path: pathlib.Path object of the file path

    Returns:
        - dict: The dictionary loaded from the JSON file

    Raises:
        - Exception: If the JSON file cannot be loaded
    '''
    if file_path.exists():
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            handle_local_error("Could not load JSON file, encountered error: ", e)
    else:
        return None   # NOTE: isinstance({}, dict) will return True so better to return None!


def update_and_save_json_file(data: dict, file_path: pathlib.Path) -> bool:
    '''
    - Updates a JSON file with the given data.

    Args:
        - data: dict of the data to be updated and saved
        - file_path: pathlib.Path object of the file path

    - Returns:
        - bool: True if the file was updated and saved successfully, False otherwise.

    Raises:
        - Exception: If the JSON file cannot be updated and saved
    '''
    current_cache = {}
    if file_path.exists():
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


def overwrite_json_file(data: dict, file_path: pathlib.Path) -> bool:
    '''
    Overwrite a JSON file with the given data.

    Args:
        - data: dict of the data to be overwritten
        - file_path: pathlib.Path object of the file path

    Returns:
        - bool: True if the file was overwritten successfully, False otherwise

    Raises:
        - Exception: If the JSON file cannot be overwritten
    '''

    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        handle_local_error("Could not save JSON file, encountered error: ", e)

    return True


def remove_file_from_filepath(filepath: pathlib.Path):
    print(f"\n\nRemoving file from filepath: {filepath}\n\n")
    try:
        os.remove(str(filepath))
        print(f"Successfully deleted file: {filepath}")
    except Exception as e:
        handle_local_error(f"Error removing file from filepath: {filepath}: ", e)


def safe_remove_file_from_filepath(filepath: pathlib.Path):
    try:
        remove_file_from_filepath(filepath)
    except Exception as e:
        handle_error_no_return(f"Error removing file from filepath: {filepath}: ", e)


def remove_folder_from_filepath(folderpath:pathlib.Path):
    '''
    Remove a folder from a filepath.

    Args:
        - folderpath: pathlib.Path object of the folder path
    

    Returns:
        - bool: True if the folder was removed successfully, False otherwise

    Raises:
        - Exception: If the folder cannot be removed
    '''
    print(f"\n\nRemoving folder from filepath: {folderpath}\n\n")
    try:
        shutil.rmtree(str(folderpath))
        print(f"Successfully deleted folder: {folderpath}")
    except Exception as e:
        handle_local_error(f"Error removing folder from filepath: {folderpath}: ", e)
    

def safe_remove_folder_from_filepath(folderpath:pathlib.Path):
    '''
    Wrapper method to remove a folder from a filepath safely.

    Args:
        - folderpath: pathlib.Path object of the folder path

    Returns:
        - bool: True if the folder was removed successfully, False otherwise

    Raises:
        - Exception: If the folder cannot be removed
    '''
    try:
        remove_folder_from_filepath(folderpath)
    except PermissionError:
        print(
            "\nPermissionError: Ensure OneDrive or other file sync services aren't running! "
            "If not, manually delete the folder at the path that follows and try again. "
            f"Error -- Could not remove folder from filepath: {folderpath}\n"
        )
    except FileNotFoundError:   # Path doesn't exist at all
        print(f"\nFileNotFoundError: Could not remove folder from filepath: {folderpath}\n")
    except NotADirectoryError:   # Path exists but is not a dir - shutil.rmtree() is for dirs
        print(f"\nNotADirectoryError: Could not remove folder from filepath: {folderpath}\n")
    except Exception as e:
        handle_error_no_return(f"Error removing folder from filepath: {folderpath}: ", e)

############################----------------------------------------------###############################


def whoosh_indexer(new_chunks:list[dict]):

    print("\n\nWhoosh Indexing Chunks\n\n")
    
    try:
        whoosh_index_folder = rag_support_module.determine_whoosh_index_folder()
    except Exception as e:
        handle_local_error("Error determining Whoosh Index Folder: ", e)
    
    try:
        ix = rag_support_module.get_whoosh_index_object_for_folder(whoosh_index_folder)
    except Exception as e:
        handle_local_error("Error getting Whoosh Index Object: ", e)
        
    # init writer and write to the index:
    try:
        writer = ix.writer()
        for chunk in new_chunks:
            writer.add_document(
                content=chunk['content'], 
                source_link=chunk['source_link'],
                source=chunk['source'], 
                page_number=str(chunk['page_number']),
                entities_and_relationships=str(
                    chunk.get('entities_and_relationships', {})
                )
            )
        
        writer.commit()
        print(f"Added {len(new_chunks)} chunks to Whoosh Index")
        
    except Exception as e:
        handle_local_error("Error writing to Whoosh Index: ", e)


def PDFtoAzureDocAiTXT(input_pdf_filepath:pathlib.Path) -> pathlib.Path:
    '''
    OCR PDFs using Azure Document Intelligence OCR:
        - Iterating through each page
        - Converting to a binary stream
        - Invoking `begin_analyze_document()`

    Args:
        - input_pdf_filepath: pathlib.Path to the target PDF

    Returns:
        - pathlib.Path to the output text file
    
    Raises:
        - Exception: 
            - If the PDF file cannot be opened
            - If the output text file cannot be initialized
            - If the OCR process fails
    '''
    
    try:
        config = read_config([
            'azure_doc_ai_endpoint',
            'azure_doc_ai_subscription_key',
            'ocr_pdfs','force_re_extract'
        ])
    except Exception as e:
        handle_local_error(
            "Missing Azure DocAI URL and/or Subscription Key, "
            "please provide required API config. Error: ", e
        )

    try:
        source_filename = input_pdf_filepath.name
        print(f"\nApplying Azure DocAI OCR to PDF: {source_filename}\n")

        output_text_file_name = input_pdf_filepath.with_suffix(".txt").name
        output_text_file_path = pathlib.Path(
            rf"{config['ocr_pdfs']}"
        ).resolve() / output_text_file_name   # normalize and append filename
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    if output_text_file_path.exists() and not config['force_re_extract']:
        if (
            output_text_file_path.is_file() and 
            output_text_file_path.stat().st_size > 0
        ):
            print("OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("OCR'ed doc already exists but is empty! Proceeding with OCR extraction.")

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize output text file, encountered error: ", e)

    try:
        docai_client = DocumentAnalysisClient(
            config['azure_doc_ai_endpoint'],
            AzureKeyCredential(config['azure_doc_ai_subscription_key'])
        )
    except Exception as e:
        handle_local_error("Could not create DocumentAnalysisClient, encountered error: ", e)

    try:
        with open(input_pdf_filepath, "rb") as pdf_file:
            # 1 - Get page count:
            try:
                pypdf_reader = PyPDF2.PdfReader(pdf_file)
                page_count = len(pypdf_reader.pages)
                page_range = f"1-{page_count}" if page_count > 1 else "1"
                print(f"page_range: {page_range}")
            except Exception as e:
                handle_local_error("Could not get page count for Azure DocAI, Error: ", e)

            # 2 - Reset file pointer to the beginning of the file.
            pdf_file.seek(0)

            # 3 - Call Azure DocAI:
            try:
                poller = docai_client.begin_analyze_document(
                    "prebuilt-layout",
                    pdf_file,
                    pages=page_range
                )
                result = poller.result()
            except Exception as e:
                handle_local_error("Error with Azure begin_analyze_document: ", e)

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
                                cell_polygon_tuple = tuple(
                                    (point.x, point.y) for point in cell_polygon
                                )    # lists aren't hashable to cast to a tuple
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
    print(f"\n\nCompleted Azure-DocAI OCR for PDF file: {input_pdf_filepath}\n\n")
    return output_text_file_path


def PDFtoAzureOCRTXT(input_pdf_filepath:pathlib.Path) -> pathlib.Path:
    '''
    OCR PDFs using Azure Computer Vision OCR by iterating through each page, converting to a binary stream and then invoking `recognize_printed_text_in_stream()`

    Args:
        - input_pdf_filepath: pathlib.Path object of the PDF file to be OCR'ed

    Returns:
        - pathlib.Path object of the output text file

    Raises:
        - Exception: If the PDF file cannot be opened, the output text file cannot be initialized, or the OCR process fails
    '''
    try:
        read_return = read_config(['azure_ocr_endpoint', 'azure_ocr_subscription_key', 'ocr_pdfs', 'azure_cv_free_tier', 'force_re_extract'])
    except Exception as e:
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for Azure OCR, please provide required API config. Error: ", e)

    try:
        source_filename = input_pdf_filepath.name
        print(f"\n\nApplying Azure OCR to PDF file: {source_filename}\n\n")

        output_text_file_name = input_pdf_filepath.with_suffix(".txt").name
        output_text_file_path = pathlib.Path(rf"{read_return['ocr_pdfs']}").resolve() / output_text_file_name   # normalize and append filename
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    if output_text_file_path.exists() and not read_return['force_re_extract']:
        if output_text_file_path.is_file() and output_text_file_path.stat().st_size > 0:
            print("OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_pdf_filepath, 300) # The convert-from_path() function from pdf2image lib intertnally uses Poppler to convert PDF pages to images, and then creates PIP Image objects from them. 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    try:
        computervision_client = ComputerVisionClient(read_return['azure_ocr_endpoint'], CognitiveServicesCredentials(read_return['azure_ocr_subscription_key']))
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
            if read_return['azure_cv_free_tier']:
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
    print(f"\n\nCompleted Azure-ComputerVision OCR for PDF file: {input_pdf_filepath}\n\n")
    return output_text_file_path


def get_vision_llm_request_params() -> tuple[str, dict, dict]:
    try:
        read_return = read_config(['vision_llm_local_url', 'vision_ocr_prompt'])
    except Exception as e:
        handle_local_error("Missing Vision LLM URL or Vision OCR Prompt for get_vision_llm_request_params, please provide required API config. Error: ", e)

    vision_request_payload = {
        'messages': json.dumps([
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": read_return['vision_ocr_prompt']}
                ]
            }
        ])
    }

    headers = {
        'X-DPI': '300',
        'X-Max-New-Tokens': '5000'
    }

    return read_return['vision_llm_local_url'], vision_request_payload, headers


def PDFtoVisionLLMOCRTXT(input_pdf_filepath:pathlib.Path) -> pathlib.Path:
    '''
    OCR PDFs using Vision LLM by iterating through each page, converting to a binary stream and then invoking the Vision LLM

    Args:
        - input_filepath: pathlib.Path object of the PDF file to be OCR'ed

    Returns:
        - pathlib.Path object of the output text file

    Raises:
        - Exception: If the PDF file cannot be opened, the output text file cannot be initialized, or the OCR process fails
    '''

    try:
        print("\n\nChecking if HF-Waitress Server is Online\n\n")
        hf_waitress_base_url = get_url_for_server('hf-waitress')
        if not utils.is_local_server_online(hf_waitress_base_url)['server_available']:
            handle_local_error("HF-Waitress Server is Offline! Please start the HF-Waitress Server and try again.")
        else:
            print("\n\nHF-Waitress Server is Online\n\n")
    except Exception as e:
        handle_error_no_return("Could not check if HF-Waitress Server is Online, presuming online and proceeding. Encountered error: ", e)

    try:
        read_return = read_config(['ocr_pdfs', 'force_re_extract'])
    except Exception as e:
        handle_local_error("Missing OCR PDFs directory for Vision LLM-based OCR, please provide required API config. Error: ", e)

    try:
        source_filename = input_pdf_filepath.name
        print(f"\n\nApplying Vision LLM OCR to PDF file: {source_filename}\n\n")

        output_text_file_name = input_pdf_filepath.with_suffix(".txt").name
        output_text_file_path = pathlib.Path(rf"{read_return['ocr_pdfs']}").resolve() / output_text_file_name   # normalize and append filename
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    if output_text_file_path.exists() and not read_return['force_re_extract']:
        if output_text_file_path.is_file() and output_text_file_path.stat().st_size > 0:
            print("Vision LLM OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("Vision LLM OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")

    # Convert PDF to  a list of images
    pil_image_object_list = []
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pil_image_object_list = convert_from_path(input_pdf_filepath, 300) # The convert-from_path() function from pdf2image lib intertnally uses Poppler to convert PDF pages to images, and then creates PIL Image objects from them. 300dpi - good balance between quality and performance
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
        handle_local_error("Could not get Vision LLM request parameters, encountered error: ", e)

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
    print(f"\n\nCompleted Local Vision-LLM OCR for PDF file: {input_pdf_filepath}\n\n")
    return output_text_file_path


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
        handle_local_error(f"Could not check if {container_name} Docker container is running, encountered error: ", e)
    

def check_if_container_is_running(container_name:str) -> bool:
    try:
        container_id = get_container_id_by_container_name(container_name)
        return container_id is not None and container_id != ""
    except Exception as e:
        handle_error_no_return(f"Could not check if {container_name} Docker container is running, encountered error: ", e)
        return False


def start_kosmos_container() -> bool:
    try:
        read_return = read_config(['kosmos_container_name', 'minimum_free_vram_for_kosmos_ocr'])
        kosmos_container_name = read_return['kosmos_container_name']
        minimum_free_vram_for_kosmos_ocr = read_return['minimum_free_vram_for_kosmos_ocr']
    except Exception as e:
        handle_local_error("Could not read Kosmos container name from config.json, encountered error: ", e)
    
    try:
        hf_waitress_base_url = get_url_for_server('hf-waitress')
        url_config = read_config(['graph_model_access_url', 'graph_model_server_port', 'graph_summarizer_access_url', 'graph_summarizer_server_port'])
        graph_model_base_url = f"http://{url_config['graph_model_access_url']}:{url_config['graph_model_server_port']}"
        graph_summarizer_base_url = f"http://{url_config['graph_summarizer_access_url']}:{url_config['graph_summarizer_server_port']}"
        utils.ensure_minimum_free_vram(minimum_free_vram_for_kosmos_ocr, [graph_summarizer_base_url, graph_model_base_url, hf_waitress_base_url])   
        # Graph-Extraction model is used for GraphRAG responses, so we try shutting down the summarizer model first!
    except Exception as e:
        handle_local_error(f"Could not ensure minimum free GPU memory ({minimum_free_vram_for_kosmos_ocr}MB) required for Kosmos OCR, encountered error: ", e)

    # Check if Docker Engine is running
    try:
        subprocess.run(['docker', 'info'], capture_output=True, check=True)  # check=True will raise an exception if the command returns a non-zero exit code
    except Exception as e:
        handle_local_error("Docker Engine is not running, encountered error: ", e)

    print("\nDocker Engine is running, proceeding with Kosmos Docker container launch...\n")

    if check_if_container_is_running(kosmos_container_name):
        print("\nKosmos Docker container is already running, skipping launch...\n")
        return True

    command = ['docker', 'start', f'{kosmos_container_name}']

    try:
        # subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE) if platform.system() == 'Windows' else subprocess.Popen(command, shell=True)

        if platform.system() == 'Windows':
            kosmos_container_process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        else:
            with open('kosmos_container_output_log.txt', 'w') as f:
                kosmos_container_process = subprocess.Popen(
                    command,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
        
        # Check if the container is running
        timeout = 5
        attempts = 50
        for _ in range(attempts):
            # Detect early crash of Kosmos Docker container:
            if kosmos_container_process is not None:
                exit_code = kosmos_container_process.poll()
                if exit_code is not None:
                    if exit_code != 0:  # `docker start` is expected to exit quickly with code 0 when it succeeds!
                        handle_local_error(
                            "Kosmos Docker container process exited early before becoming available. "
                            f"Return code: {exit_code}. Check the Kosmos Docker container log for details. ",
                            RuntimeError(f"Kosmos Docker container exited with code {exit_code}")
                        )

            if check_if_container_is_running(kosmos_container_name):
                print(f"\nKosmos Docker container launched successfully! Returning in {timeout} seconds...\n")
                time.sleep(timeout) # small delay to ensure the container is fully started
                return True
            
            else:
                print(f"Kosmos Docker container not yet running, waiting {timeout} seconds before retrying...")
                time.sleep(timeout)

    except Exception as e:
        handle_local_error("Could not launch Kosmos Docker container, encountered error: ", e)

    return False


def get_kosmos_request_params() -> tuple[str, dict, dict]:
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


def kosmos_ocr_page(page_as_pil_image_object, retry_count:int=0) -> str:
    print(f"\n\nProcessing Page - Kosmos OCR\n\n")
    try:
        kosmos_local_url, payload, headers = get_kosmos_request_params()
    except Exception as e:
        handle_local_error("Could not get Kosmos request parameters, encountered error: ", e)
    
    try:
        print("\n\nConverting PIL-image object to Byte-Stream\n\n")
        img_stream = io.BytesIO()
        page_as_pil_image_object.save(img_stream, format='PNG')
        img_stream.seek(0)
    except Exception as e:
        handle_local_error("Could not convert PIL-image object to Byte-Stream, encountered error: ", e)
    
    try:
        print("Preparing file payload for Kosmos\n")
        file_payload = [
            ('file', ('page-image.png', img_stream, 'image/png'))
        ]
    except Exception as e:
        handle_local_error("Could not prepare file payload for Kosmos, encountered error: ", e)
    
    try:
        start_kosmos_container()    # Will do nothing if the container is already running!
    except Exception as e:
        handle_local_error("Could not start Kosmos Docker container, encountered error: ", e)
    
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
                            handle_local_error("Could not process event from Kosmos, encountered error: ", e)
    except requests.exceptions.RequestException as e:
        handle_error_no_return(f"Could not send request to Kosmos, checking if Kosmos is running and attempting to start the service if not. Retry attempt {retry_count+1} of 3. Encountered error: ", e)
        if retry_count >= 3:
            handle_local_error("Failed to receive a proper response from the Kosmos OCR service even after 3 retries, stopping execution. Encountered error: ", e)
        else:
            try:
                start_kosmos_container()    # Will do nothing if the container is already running!
                return kosmos_ocr_page(page_as_pil_image_object, retry_count=retry_count+1)
            except Exception as e:
                handle_local_error("Could not start Kosmos Docker container, encountered error: ", e)
    except Exception as e:
        handle_local_error("Could not receive a complete response from the Kosmos OCR service, encountered error: ", e)
    finally:
        if 'img_stream' in locals() and img_stream:
            img_stream.close()  # io.BytesIO objects are in-memory streams, and while the GC will eventually close them, it's best to do it explicitly here.


def PDFtoKosmosOCRTXT(input_pdf_filepath:pathlib.Path) -> pathlib.Path:
    '''
    OCR PDFs using Kosmos by iterating through each page, converting to a binary stream and then invoking `infer_file_stream()`

    Args:
        - input_pdf_filepath: pathlib.Path object of the PDF file to be OCR'ed

    Returns:
        - pathlib.Path object of the output text file

    Raises:
        - Exception: If the PDF file cannot be opened, the output text file cannot be initialized, or the OCR process fails
    '''
    try:
        read_return = read_config(['ocr_pdfs', 'force_re_extract'])
    except Exception as e:
        handle_local_error("Missing OCR PDFs directory for Kosmos OCR, please provide required API config. Error: ", e)

    try:
        source_filename = input_pdf_filepath.name
        print(f"\n\nApplying Kosmos OCR to PDF file: {source_filename}\n\n")

        output_text_file_name = input_pdf_filepath.with_suffix(".txt").name
        output_text_file_path = pathlib.Path(rf"{read_return['ocr_pdfs']}").resolve() / output_text_file_name   # normalize and append filename
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    if output_text_file_path.exists() and not read_return['force_re_extract']:
        if output_text_file_path.is_file() and output_text_file_path.stat().st_size > 0:
            print("Kosmos OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("Kosmos OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")

    # Convert PDF to  a list of images
    pil_image_object_list = []
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pil_image_object_list = convert_from_path(input_pdf_filepath, 300) # The convert-from_path() function from pdf2image lib intertnally uses Poppler to convert PDF pages to images, and then creates PIP Image objects from them. 300dpi - good balance between quality and performance
        pdf_document_length = len(pil_image_object_list)
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)
    
    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    # Initialize page number
    for page_number, image in enumerate(pil_image_object_list, start=1):
        try:
            print(f"\nProcessing Page: {page_number} of {pdf_document_length} from file: {source_filename}\n")
            full_parsed_text = kosmos_ocr_page(image)
            output_text_file.write(f"[PAGE:{page_number}]\n{full_parsed_text}\n")
        except Exception as e:
            handle_error_no_return("Could not process page, encountered error: ", e)
            continue
    
    # Close & return
    output_text_file.close()
    print(f"\n\nCompleted Kosmos OCR for PDF file: {input_pdf_filepath}\n\n")
    return output_text_file_path


def get_docling_ocr_model(model_name_string:str):
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
        handle_local_error("Could not get Docling OCR model, encountered error: ", e)
        

def get_docling_vlm_model(model_name_string:str):
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
        handle_local_error("Could not get Docling VLM model, encountered error: ", e)


def get_docling_config() -> dict:
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
        handle_local_error("Could not read Docling config, encountered error: ", e)


def get_docling_converter(docling_config:dict):
    try:

        if docling_config['docling_pipeline'] == 'vlm':
            
            # a. Set VLM Pipeline Options
            vlm_pipeline_options = None
            vlm_pipeline_options = VlmPipelineOptions()
            vlm_pipeline_options.vlm_options = get_docling_vlm_model(docling_config['docling_vlm_model'])

            # b. VLM Converter
            vlm_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=vlm_pipeline_options,
                    ),
                }
            )

            return vlm_converter

        # Standard Pipeline
        # a. Set PDF Pipeline Options
        pdf_pipeline_options = None
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = str(docling_config['docling_do_ocr']).lower() == 'true'
        pdf_pipeline_options.do_code_enrichment = str(docling_config['docling_do_code_enrichment']).lower() == 'true'
        pdf_pipeline_options.do_formula_enrichment = str(docling_config['docling_do_formula_enrichment']).lower() == 'true'
        pdf_pipeline_options.do_table_structure = str(docling_config['docling_do_table_structure']).lower() == 'true'
        pdf_pipeline_options.do_picture_classification = str(docling_config['docling_do_picture_classification']).lower() == 'true'
        pdf_pipeline_options.do_picture_description = str(docling_config['docling_do_picture_description']).lower() == 'true'
        pdf_pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE if str(docling_config['docling_table_structure_mode']) == 'accurate' else TableFormerMode.FAST
        pdf_pipeline_options.table_structure_options.do_cell_matching = str(docling_config['docling_do_cell_matching']).lower() == 'true'
        pdf_pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads = int(docling_config['docling_num_threads']),
            device = AcceleratorDevice.AUTO,
            # cuda_use_flash_attention_2 = str(docling_config['docling_cuda_use_flash_attention_2']).lower() == 'true'
        )

        # b. Set OCR Options
        ocr_options = get_docling_ocr_model(str(docling_config['docling_ocr_model']))
        ocr_options.force_full_page_ocr = str(docling_config['docling_force_full_page_ocr']).lower() == 'true'
        pdf_pipeline_options.ocr_options = ocr_options

        # c. Initialize converter and process
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options,
                )
            }
        )
        
        return converter

    except Exception as e:
        handle_local_error("Could not get Docling converter, encountered error: ", e)


def docling_ocr_page(page_as_pdf_bytes:bytes, page_number:int, retry_count:int=0) -> str:
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
        return str(result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER))

    except Exception as e:
        if retry_count < 3:
            print(f"Retrying Docling OCR for page {page_number}, retry attempt {retry_count+1} of 3. Encountered error: {e}")
            return docling_ocr_page(page_as_pdf_bytes, page_number, retry_count + 1)
        else:
            handle_local_error("Failed to receive a proper response from the Docling OCR service even after 3 retries, stopping execution. Encountered error: ", e)


def PDFtoDoclingOCRTXT(input_pdf_filepath:pathlib.Path) -> pathlib.Path:
    '''
    OCR PDFs using Docling by iterating through each page, converting to a binary stream and then invoking `dolcing_ocr_page()`

    Args:
        - input_pdf_filepath: pathlib.Path object of the PDF file to be OCR'ed

    Returns:
        - pathlib.Path object of the output text file

    Raises:
        - Exception: If the PDF file cannot be opened, the output text file cannot be initialized, or the OCR process fails
    '''
    try:
        read_return = read_config(['force_re_extract', 'ocr_pdfs'])
    except Exception as e:
        handle_local_error("Could not read required values from config.json when attempting to convert PDF to TXT, encountered error: ", e)
    
    try:
        source_filename = input_pdf_filepath.name
        print(f"\n\nApplying Docling OCR to PDF file: {source_filename}\n\n")

        output_text_file_name = input_pdf_filepath.with_suffix(".txt").name
        output_text_file_path = pathlib.Path(rf"{read_return['ocr_pdfs']}").resolve() / output_text_file_name   # normalize and append filename
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    if output_text_file_path.exists() and not read_return['force_re_extract']:
        if output_text_file_path.is_file() and output_text_file_path.stat().st_size > 0:
            print("Docling OCR'ed doc already exists and is not empty! Returning existing file.")
            return output_text_file_path
        else:
            print("Docling OCR'ed doc already exists but is empty! Overwriting with new OCR'ed file.")

    # Open with PyMuPDF for conversion to binary byte stream
    try:
        pdf_document = fitz.open(input_pdf_filepath)
        pdf_document_length = len(pdf_document)
    except Exception as e:
        handle_local_error("Could not open PDF file, encountered error: ", e)
    
    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)
    
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
    print(f"\n\nCompleted Docling OCR for PDF file: {input_pdf_filepath}\n\n")
    return output_text_file_path


def UseBackupOcrOnPage(input_pdf_filepath:pathlib.Path, current_page_num:int) -> str:
    '''
    OCR a single page using the backup OCR service
    
    Args:
        - input_pdf_filepath: pathlib.Path object of the PDF file to be OCR'ed
        - current_page_num: int of the page number to be OCR'ed

    Returns:
        - str of the OCR'ed text
    
    Raises:
        - Exception: If the PDF file cannot be opened, the output text file cannot be initialized, or the OCR process fails
    '''
    try:
        read_return = read_config(['backup_ocr_service_choice'])
    except Exception as e:
        handle_local_error("Could not read required values from config.json when attempting to use backup OCR on page, encountered error: ", e)
    
    if read_return['backup_ocr_service_choice'] == 'Kosmos':
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
            handle_local_error("Could not OCR page with Kosmos, skipping. Encountered error: ", e)
        
    elif read_return['backup_ocr_service_choice'] == 'Docling':
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
            handle_local_error("Could not OCR page with Docling, skipping. Encountered error: ", e)

    else:
        handle_local_error("Invalid backup OCR service choice, skipping. Encountered error: ", read_return['backup_ocr_service_choice'])
    

def PDFtoTXT(input_pdf_filepath:pathlib.Path) -> pathlib.Path:
    '''
    OCR PDFs using PyPDF2 by iterating through each page, converting to a binary stream and then invoking `Use-BackupOcrOnPage()`

    Args:
        - input_pdf_filepath: pathlib.Path object of the PDF file to be OCR'ed

    Returns:
        - pathlib.Path object of the output text file

    Raises:
        - Exception: If the PDF file cannot be opened, the output text file cannot be initialized, or the OCR process fails
    '''
    try:
        read_return = read_config(['pdfs_to_txts', 'force_re_extract', 'ocr_pdfs', 'min_char_threshold_for_backup_ocr'])
    except Exception as e:
        handle_local_error("Could not read required values from config.json when attempting to convert PDF to TXT, encountered error: ", e)

    try:
        source_filename = input_pdf_filepath.name
        print(f"\n\nApplying PyPDF2 OCR to PDF file: {source_filename}\n\n")

        output_text_file_name = input_pdf_filepath.with_suffix(".txt").name
        output_text_file_path = pathlib.Path(rf"{read_return['pdfs_to_txts']}").resolve() / output_text_file_name   # normalize and append filename
        ocr_pdf_file_path = pathlib.Path(rf"{read_return['ocr_pdfs']}").resolve() / output_text_file_name   # normalize and append filename

        if not read_return['force_re_extract']:
            if output_text_file_path.is_file() and output_text_file_path.stat().st_size > 0:
                print("PyPDF2-extracted .txt already exists and is not empty! Returning existing file.")
                return output_text_file_path
            elif ocr_pdf_file_path.is_file() and ocr_pdf_file_path.stat().st_size > 0:
                print("OCR'ed .txt already exists and is not empty! Returning existing file.")
                return ocr_pdf_file_path
            else:
                print("PyPDF2-extracted .txt already exists but is empty! Overwriting with new .txt file.")
    
    except Exception as e:
        handle_local_error("Could not complete necessary file system operations when attempting to convert PDF to TXT, encountered error: ", e)
    
    try:
        with open(input_pdf_filepath, 'rb') as pdf_file_obj:    # Open PDF file as binary stream. The `with open()` block will automatically close the file after the block is executed.
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            num_pages = len(pdf_reader.pages)

            with open(output_text_file_path, 'w', encoding='utf-8') as output_text_file:

                # Loop through all the pages and extract text
                for page_num in range(num_pages):
                    current_page_num = int(page_num) + 1
                    print(f"\nProcessing page {current_page_num} of {num_pages} from file: {source_filename}\n")

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

                    if use_backup_ocr or pypdf2_text is None or len(pypdf2_text) < read_return['min_char_threshold_for_backup_ocr']:
                        print(f"OCR Necessary - Attempting to OCR page {current_page_num} of {num_pages} using backup OCR method.")
                        try:
                            ocr_text = UseBackupOcrOnPage(input_pdf_filepath, current_page_num)
                        except Exception as e:
                            handle_error_no_return("Could not OCR page with backup OCR method, skipping. Encountered error: ", e)
                            ocr_text = ""
                        
                    try:
                        text_to_write = pypdf2_text if len(pypdf2_text) >= len(ocr_text) else ocr_text
                        output_text_file.write(f"[PAGE:{current_page_num}]\n{text_to_write}\n")
                    except Exception as e:
                        handle_error_no_return("Could not write to output text file, encountered error: ", e)
                        continue
    except Exception as e:
        handle_local_error("Could not process PDF file, encountered error: ", e)

    print(f"\n\nCompleted default-extraction for PDF file: {input_pdf_filepath}\n\n")
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
        handle_local_error("Missing sqlite_docs_loaded_db in config.json for method init_and_get_cursor_fordocs_loaded_db. Error: ", e)

    try:
        conn = sqlite3.connect(sqlite_docs_loaded_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to sqlite_docs_loaded_db, encountered error: ", e)

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
        handle_local_error("Could not create document_records DB, encountered error: ", e)
    
    try:
        add_column_if_not_exists(cursor, 'document_records', 'document_name', 'TEXT')
        add_column_if_not_exists(cursor, 'document_records', 'embedding_model', 'TEXT')
        add_column_if_not_exists(cursor, 'document_records', 'vectordb_used', 'TEXT')
        add_column_if_not_exists(cursor, 'document_records', 'chunk_size', 'INTEGER')
        add_column_if_not_exists(cursor, 'document_records', 'chunk_overlap', 'INTEGER')
        add_column_if_not_exists(cursor, 'document_records', 'knowledge_domain', 'TEXT')
    except Exception as e:
        handle_local_error("Could not add necessary columns to document_records db, encountered error: ", e)

    return conn, cursor


def is_doc_already_loaded_to_db(document_name:str, embedding_model:str, chunk_size:int, chunk_overlap:int, knowledge_domain:str) -> bool:
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


def record_doc_loaded_to_db(document_name:str, chunk_size:int, chunk_overlap:int) -> bool:

    print("\n\nRecording document loading to records DB\n\n")

    try:
        read_return = read_config(['selected_embedding_model', 'selected_knowledge_domain'])
        embedding_model = read_return['selected_embedding_model']
        knowledge_domain = read_return['selected_knowledge_domain']
    except Exception as e:
        handle_local_error("Could not determine embedding model and knowledge domain in config.json. Error: ", e)

    if is_doc_already_loaded_to_db(document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain):
        print("Document already exists in records DB, skipping insertion.")
        return True

    try:
        conn, cursor = init_and_connect_to_docs_loaded_db()
    except Exception as e:
        handle_local_error("Could not connect to docs_loaded_db, encountered error: ", e)
    
    try:
        cursor.execute("INSERT INTO document_records (document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain) VALUES (?, ?, ?, ?, ?)", (document_name, embedding_model, chunk_size, chunk_overlap, knowledge_domain))
        conn.commit()
        conn.close()
        print("\nDocument record added to document_records DB\n")
        return True
    except Exception as e:
        handle_local_error("Could not update document_records DB, encountered error: ", e)
    


# List-splitter function for a large number of embeddings!
def split_embeddings_list(all_splits, max_emmbeddings_list_size):
    for i in range(0, len(all_splits), max_emmbeddings_list_size):  # Step through the large list in steps of max size
        yield all_splits[i:i + max_emmbeddings_list_size]   # Yield a slice of all_splits from index i upto but NOT including i+max_size. 
        '''
        While memory efficient, there is a key limitation: cannot use len() on the yielded object as if it were a list! This is because the yielded object is a generator object created via iteration, not a list!
        Also, issues may arise if attmepting to enumerate() as there's an implicit split_docs[i] indexing, especially if anywhere in the loop you then try a len() on the yielded object!
        This is because you would be using the generator multiple times (once for length, then again for iteration): Generators are "single-use" iterators - once you iterate through them, 
        they're exhausted and can't be used again without recreating them. Plus, they don't support random access or length checking because they generate values on-the-fly!
        Hence this method is not used by core-embedder(), but the code retained here for future reference and debugging/documentation purposes!
        '''


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self): #to provide string-representation of an object
        # return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"    # Does not truncate the actual page_content or even str(doc.page_content), rather it only comes into play for display purposes when we print the entire object as a string!
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

# Consider turning this into a generator function in the future for efficiency when dealing with large files!
def chunk_docs_with_page_numbers(input_filepath:pathlib.Path, chunk_size:int=250) -> list[dict]:
    '''
    Chunk the document into smaller chunks, and then store the chunks in the Documents list

    Args:
        - input_file: pathlib.Path object of the input file
        - chunk_size: int of the chunk size

    Returns:
        - list[dict]: The list of document chunks

    Raises:
        - Exception: If the document cannot be chunked
    '''

    print("\nGenerating Document Chunks\n")
    documents = []
    current_chunk = ""
    current_page = 1

    def add_chunk(chunk, page):
        if chunk.strip():   #if chunk is not empty!
            #print(f"\n\nAdding chunk from page {page}: {chunk.strip()}\n\n")
            input_file_name = input_filepath.name
            source_link = f"http://llm-citations-database.net/source?doc_name={input_file_name}&page_number={page}"
            documents.append({
                'content': chunk.strip(),
                'source_link': source_link,
                'source': str(input_filepath),
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
        with open(input_filepath, 'r', encoding='utf-8') as file:
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


def core_embedder(chunks:list[dict], selected_embedding_model:str, path_to_knowledge_domain:pathlib.Path) -> bool:
    '''
    Embed the chunks and store them in the VectorDB.

    Args:
        - chunks: list[dict] of the chunks
        - selected_embedding_model: str of the selected embedding model
        - path_to_knowledge_domain: pathlib.Path of the path to the knowledge domain

    Returns:
        - bool: True if the chunks were embedded and stored in the VectorDB, False otherwise

    Raises:
        - Exception: If the chunks cannot be embedded or stored in the VectorDB
    '''

    # Convert Chunks to Document objects:
    try:    # To generate a list of Document objects, each containing a 'page_content' string, and a 'metadata' dictionary with 'source_link', 'page_number', and 'source' keys
        numbered_splits = [Document(page_content=chunk['content'], metadata={'source_link':chunk['source_link'], 'page_number': chunk['page_number'], 'source': chunk['source'], 'entities_and_relationships': str(chunk.get('entities_and_relationships', {}))}) for chunk in chunks]
    except Exception as e:
        handle_local_error("Failed to convert chunks to Document objects for storage to VectorDB, encountered error: ", e)

    # Load Embedding Model
    embedding_model = None
    try:
        embedding_model = SentenceTransformer(selected_embedding_model, trust_remote_code=True)
    except Exception as e:
        handle_local_error("Could not load embedding model in core-embedder, encountered error: ", e)

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
    vector_db_path = rag_support_module.create_vector_db_directory(path_to_knowledge_domain, selected_embedding_model)

    # Store Chunks in VectorDB
    print("Storing to VectorDB: ChromaDB")
    try:
        # Initialize Chroma Client and collection
        chroma_client = chromadb.PersistentClient(path=str(vector_db_path), settings=chromadb.Settings(allow_reset=True))
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
            'graph_model_top_p', 'graph_model_min_p', 'graph_chunk_overlap',
            'reuse_graph_extraction_cache_with_validation'
        ])
        
        generation_config = {
            'max_new_tokens': config_data['graph_model_max_new_tokens'],
            'temperature': config_data['graph_model_temperature'],
            'top_k': config_data['graph_model_top_k'],
            'top_p': config_data['graph_model_top_p'],
            'graph_chunk_overlap': '0' if rag_response_mode else config_data['graph_chunk_overlap'],
            'reuse_extraction_cache': config_data['reuse_graph_extraction_cache_with_validation'],
            'exl2_quantize_graph_model': config_data['exl2_quantize_graph_model']
        }

        if not config_data['exl2_quantize_graph_model']:
            generation_config['do_sample'] = config_data['graph_model_do_sample']
            generation_config['min_p'] = config_data['graph_model_min_p']
    
    except Exception as e:
        handle_local_error("Could not get graphing request params, encountered error: ", e)

    headers = {
        'Content-Type': 'application/json'
    }

    grapher_url = f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}"

    if not config_data['exl2_quantize_graph_model']:
        grapher_url += "/completions"
    else:
        headers['Connection'] = 'keep-alive'
        grapher_url += "/exl2_graph_extractor"

    return grapher_url, headers, generation_config


def get_graphing_request_payload(chunk:str, exl2_quantize_graph_model:bool) -> str:
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


# def hf_waitress_non_streaming_api_handler(endpoint_url:str, headers:dict, payload:str) -> str:
#     print(f"\nHF-Waitress Non-Streaming Request Response Handler Invoked\n")
#     try:
#         response = requests.post(endpoint_url, headers=headers, data=payload)
#         print("\nCompleted, returning response\n")
#         return (response.json()['response'])
#     except Exception as e:
#         handle_local_error("Failed /completions request to extract entities and relationships from chunk, encountered error: ", e)


def hf_waitress_exl2_graph_api_stream_handler(endpoint_url:str, headers:dict, payload:str, cache_filepath: pathlib.Path = None) -> dict:
    '''
    Handle the response from the exl2-grapher API

    Args:
        - endpoint_url: str of the endpoint URL
        - headers: dict of the headers
        - payload: str of the payload
        - cache_filepath: pathlib.Path object of the cache filepath

    Returns:
        - dict: The full response from the exl2-grapher API

    Raises:
        - Exception: If the response cannot be handled
    '''

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
                                While the received data is a dict, the value of the `entities_and_relationships` key itself may be returned as a string by exl2-grapher's extraction_task(), 
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

                        if cache_filepath is not None:  # Will be None when `extract-all_entities_and_relationships()` is invoked with rag-response_mode set to True
                            try:
                                update_and_save_json_file({chunk_number: full_response[chunk_number]}, cache_filepath)
                                print(f"\nSaved graph-components for chunk {chunk_number} to cache file at path {cache_filepath}\n")
                            except Exception as e:
                                handle_error_no_return(f"Could not cache graph-components for chunk {chunk_number} to cache file at path {cache_filepath}, skipping. Encountered error: ", e)
                        
                        chunk_number += 1
                    
                    except json.JSONDecodeError as e:
                        handle_error_no_return(f"Failed to parse event data: {event_data}, encountered error: ", e)
                elif line.startswith("event: END"):
                    break
                else:
                    print(f"\nUnexpected Line Format: {line}\n")

        if not full_response:
            print("\nWarning: No response from exl2-stream / exl2-grapher request\n")
            return None

        print("\nCompleted, returning response\n")
        return full_response
        
    except Exception as e:
        handle_local_error("Failed request to exl2-stream or exl2-grapher APIs, encountered error: ", e)


def graphing_request_response_handler(grapher_url:str, headers:dict, payload:str) -> dict:
    print(f"\nHF-Waitress Non-Streaming Graphing-Request Response Handler Invoked\n")
    try:
        response = hf_waitress_non_streaming_api_handler(grapher_url, headers, payload)
        print(f"\nResponse (Entities and Relationships): {response}\n")
        return ast.literal_eval(response)
    except Exception as e:
        handle_local_error("Failed /completions request to extract entities and relationships from chunk, encountered error: ", e)


def extract_all_entities_and_relationships(chunk_entities: dict, rag_response_mode: bool = False, cache_filepath: pathlib.Path = None) -> dict:
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
        grapher_url, headers, config_data = get_request_params_for_graph_entity_extraction_model(rag_response_mode=rag_response_mode)
    except Exception as e:
        handle_local_error("Could not get graphing request params, encountered error: ", e)

    if config_data['exl2_quantize_graph_model']:   # invoke exl2-grapher API
        try:
            payload = json.dumps({
                "chunk_entities": chunk_entities,
                "rag_response_mode": rag_response_mode,
                **config_data
            })
            full_response = hf_waitress_exl2_graph_api_stream_handler(grapher_url, headers, payload, cache_filepath)
            # print(f"\nExl2 Bulk-Graphing Response (Entities and Relationships):\n\n{full_response}\n")
            return full_response
        except Exception as e:
            handle_local_error("Error with request to exl2-grapher API, encountered error: ", e)

    else:   # invoke /completions
        '''
        Block Unused!
        '''
        # TODO: Implement non-exl2 bulk-summary generation with /completions

        try:
            for chunk_number, chunk_data in chunk_entities.items(): # chunk_entities.items() returns a dict_items object which is iterable. chunk_data is a dict for each chunk while chunk_number is a string ranging from 0 to len(chunk_entities) - 1
                try:
                    print(f"\nExtracting entities and relationships for chunk {chunk_number} of total {len(chunk_entities)} chunks...\n")
                    payload = get_graphing_request_payload(
                        chunk=chunk_data['chunk_text'], 
                        exl2_quantize_graph_model=False
                    )
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
            handle_local_error("\nCould not iterate over chunk entities, encountered error: ", e)


def get_request_params_for_graph_summarizer_model():
    try:
        config_data = read_config([
            'exl2_quantize_graph_summarizer_model', 'graph_summarizer_access_url',
            'graph_summarizer_server_port', 'graph_summarizer_max_new_tokens',
            'graph_summarizer_temperature', 'graph_summarizer_do_sample', 'graph_summarizer_top_k',
            'graph_summarizer_top_p', 'graph_summarizer_min_p',
            'reuse_graph_summary_cache_with_validation'
        ])

        generation_config = {
            'max_new_tokens': config_data['graph_summarizer_max_new_tokens'],
            'temperature': config_data['graph_summarizer_temperature'],
            'top_k': config_data['graph_summarizer_top_k'],
            'top_p': config_data['graph_summarizer_top_p'],
            'reuse_summary_cache': config_data['reuse_graph_summary_cache_with_validation'],
            'exl2_quantize_graph_summarizer_model': config_data['exl2_quantize_graph_summarizer_model']
        }

        if not config_data['exl2_quantize_graph_summarizer_model']:
            generation_config['do_sample'] = config_data['graph_summarizer_do_sample']
            generation_config['min_p'] = config_data['graph_summarizer_min_p']
    
    except Exception as e:
        handle_local_error("Could not get graphing request params, encountered error: ", e)

    graph_summarizer_url = f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}"

    headers = {
        'Content-Type': 'application/json',
    }

    if not config_data['exl2_quantize_graph_summarizer_model']:
        graph_summarizer_url += "/completions"
    else:
        headers['Connection'] = 'keep-alive'
        graph_summarizer_url += "/exl2_graph_summarizer"

    return graph_summarizer_url, headers, generation_config


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
        hf_waitress_graph_summarizer_server_path = pathlib.Path.cwd() / str(config_data['graph_models_base_directory_name']) / str(config_data['graph_summary_generator_directory_name'])
        print(f"\nLaunching HF-Waitress instance with graph summarizer model at path: {hf_waitress_graph_summarizer_server_path}\n")
    except Exception as e:
        handle_local_error("Could not read graph model config, encountered error: ", e)

    if utils.is_local_server_online(f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}")['server_online']:
        print("\nGraph summarizer model server is already online, skipping launch...\n")
        return True
    
    # Get free GPU memory and shutdown the main Waitress LLM chat server if necessary:
    try:
        hf_waitress_base_url = get_url_for_server('hf-waitress')    # shutdown general-chat LLM server at this URL in case free VRAM is insufficient
        graph_model_base_url = f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}"
        utils.ensure_minimum_free_vram(int(config_data['minimum_free_vram_for_graph_summarizer_model']), [graph_model_base_url, hf_waitress_base_url])
    except Exception as e:
        handle_local_error(f"Could not reserve minimum GPU memory ({config_data['minimum_free_vram_for_graph_summarizer_model']}MB) required for Graph-Summarizer model, encountered error: ", e)

    base_command = 'python' if platform.system() == 'Windows' else 'python3'

    cmd_list = [
        base_command,
        'hf_waitress.py',
        '--port', str(config_data['graph_summarizer_server_port']),
        '--model_id', str(config_data['graph_summarizer_model']),
        '--max_new_tokens', str(config_data['graph_summarizer_max_new_tokens']),
    ]

    if str(config_data['exl2_quantize_graph_summarizer_model']).lower() == 'true':
        cmd_list.extend([
            '--exl2',
            '--exl2_bpw', str(config_data['exl2_quantize_graph_summarizer_model_bpw']),
            '--exl2_max_seq_len', str(config_data['graph_summarizer_max_seq_len']),
        ])
    else:
        cmd_list.extend([
            '--quantize', str(config_data['quantize_graph_summarizer_model']),
            '--quant_level', str(config_data['quantize_graph_summarizer_model_bits']),
        ])

    try:
        if platform.system() == 'Windows':
            graph_summarizer_process = subprocess.Popen(
                cmd_list,
                cwd=hf_waitress_graph_summarizer_server_path,
                creationflags=subprocess.CREATE_NEW_CONSOLE,  # or CREATE_NEW_PROCESS_GROUP
            )
        else:
            with open('graph_summarizer_model_output_log.txt', 'w') as f:
                graph_summarizer_process = subprocess.Popen(
                    cmd_list,
                    cwd=hf_waitress_graph_summarizer_server_path,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

        timeout = 5
        attempts = 25
        for _ in range(attempts):

            # Detect early crash of KB-Generator HF-Waitress process:
            if graph_summarizer_process is not None:
                exit_code = graph_summarizer_process.poll()
                if exit_code is not None:
                    handle_local_error(
                        "Graph-Summarizer model HF-Waitress server process exited early before becoming available. "
                        f"Return code: {exit_code}. Check the Graph-Summarizer model server log for details. ",
                        RuntimeError(f"Graph-Summarizer model HF-Waitress exited with code {exit_code}")
                    )
                    return False

            if utils.is_local_server_online(f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}")['server_online']:
                print(f"\nGraph-Summarizer model launched successfully!\n")
                return True

            print(f"Graph-Summarizer model not yet running, waiting {timeout} seconds before retrying...")
            time.sleep(timeout)

    except Exception as e:
        handle_local_error("Could not launch HF-Waitress instance with graph-summarizer model, encountered error: ", e)
    
    return False


def summary_generator_for_graph_db(chunk_entities=None, cache_filepath: pathlib.Path = None) -> dict:
    '''
    Generate summaries for the graph DB.

    Args:
        - chunk_entities: dict of the chunk entities
        - cache_filepath: pathlib.Path object of the cache filepath

    Returns:
        - dict: The full response from the exl2-grapher API

    Raises:
        - Exception: If the summaries cannot be generated
    '''
    print("\nGenerating summaries...\n")

    if chunk_entities is None:
        handle_local_error("Chunk entities are required to generate summaries")

    try:
        bring_graph_summarizer_model_online()
    except Exception as e:
        handle_local_error("Could not bring graph summarizer model online, encountered error: ", e)

    exl2 = str(read_config(['exl2_quantize_graph_summarizer_model'])['exl2_quantize_graph_summarizer_model']).lower() == 'true'        
    
    if exl2:

        try:
            summarizer_url, headers, generation_config = get_request_params_for_graph_summarizer_model()
        except Exception as e:
            handle_local_error("Could not get graphing request params, encountered error: ", e)

        try:
            payload = json.dumps({
                "chunk_entities": chunk_entities,
                **generation_config
            })
            full_response = hf_waitress_exl2_graph_api_stream_handler(summarizer_url, headers, payload, cache_filepath)
            # print(f"\nExl2 Bulk-Summary Generation Response:\n\n{full_response}\n")
            return full_response
        except Exception as e:
            handle_local_error("Error with request to exl2-grapher API, encountered error: ", e)

    else:
        # TODO: Implement non-exl2 bulk-summary generation with /completions
        pass


def add_nodes_to_graph(selected_knowledge_domain: str, nodes: list, graph: FalkorDB, source_document: str = '', page_number: list = None, summary: list = []):
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
            if not isinstance(node, dict):
                print(f"Skipping node from document {source_document} - Invalid Type: Expected a dict, got {type(node).__name__}")
                continue
            
            name = str(node.get('name', ''))
            node_type = str(node.get('type', ''))
            #updated_summary = list(node.get('summary', []))   # dict .get() method is safer than `if node['summary']` because it provides a default value if the key doesn't exist and handles NoneType errors gracefully!
            
            if name == '' or node_type == '':
                print(f"Skipping node from document {source_document} because it's missing required fields: name={name}, type={node_type}")
                continue

            node_key = (name, node_type)
            if node_key in processed_nodes:
                # print(f"Skipping duplicate node {name} of type {node_type} in {selected_knowledge_domain} graph DB")
                continue

            node_name = rag_support_module.sanitize_names(name)

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
                'summary': summary,
                'source_document': source_document.replace("'", ""),
                'page_number': page_number
            })
            
            # Mark node as processed:
            processed_nodes[node_key] = True
            
            # print(f"\nCreated node - name: {name}, type: {node_type} - in {selected_knowledge_domain} graph DB\n")
        except Exception as e:
            handle_error_no_return(f"Could NOT create node - name: {name}, type: {node_type} - in {selected_knowledge_domain} graph DB, skipping. Encountered error: ", e)


def add_relationships_to_graph(selected_knowledge_domain: str, relationships: list, graph: FalkorDB, source_document: str = '', page_number: list = None, summary: list = []):
    # print(f"\nStoring relationships to {selected_knowledge_domain} graph DB\n")

    processed_relationships = {}    # Format: {(source, target, relationship): True}
    page_number = [] if page_number is None else page_number

    for relationship in relationships:
        try:
            if not isinstance(relationship, dict):
                print(f"Skipping relationship from document {source_document} - Invalid Type: Expected a dict, got {type(relationship).__name__}")
                continue

            source = str(relationship.get('source', ''))
            target = str(relationship.get('target', ''))
            relationship_type = str(relationship.get('relationship', ''))
            #updated_summary = list(relationship.get('summary', []))   # dict .get() method is safer than `if relationship['summary']` because it provides a default value if the key doesn't exist and handles NoneType errors gracefully!
            
            if source == '' or target == '' or relationship_type == '':
                print(f"Skipping relationship from document {source_document} because it's missing required fields: source={source}, target={target}, relationship={relationship_type}")
                continue

            relationship_key = (source, target, relationship_type)
            if relationship_key in processed_relationships:
                # print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type}) in {selected_knowledge_domain} graph DB")
                continue

            source = rag_support_module.sanitize_names(source)
            target = rag_support_module.sanitize_names(target)
            relationship_type = rag_support_module.sanitize_names(relationship_type).upper()

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
                'summary': summary,
                'source_document': source_document.replace("'", ""),
                'page_number': page_number
            })

            # Mark relationship as processed:
            processed_relationships[relationship_key] = True
            
            # print(f"\nCreated relationship - source: {source}, target: {target}, relationship: {relationship} - in {selected_knowledge_domain} graph DB\n")
        except Exception as e:
            handle_error_no_return(f"Could NOT create relationship from data: {relationship} in {selected_knowledge_domain} graph DB, skipping. Encountered error: ", e)


def store_entities_and_relationships_in_graph_db(chunk_entities: dict, selected_knowledge_domain: str, graph: FalkorDB, skip_summary_generation: bool = False, summaries_filepath: pathlib.Path = None):
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
            - The updated chunk_entities dict is then stored to the `summaries-filepath` cache file for persistence and re-use before being stored in the graph DB.
        2. Stores the nodes and relationships in the graph DB.
    '''

    try:
        # First, Get/Generate summaries for each node and relationship if applicable:
        if not skip_summary_generation:

            try:
                chunks = summary_generator_for_graph_db(chunk_entities=chunk_entities, cache_filepath=summaries_filepath)
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
                add_nodes_to_graph(selected_knowledge_domain, chunk_data['entities_and_relationships']['nodes'], graph, chunk_data['source_doc_name'], chunk_data['page_number'], chunk_data.get('summary', []))
                add_relationships_to_graph(selected_knowledge_domain, chunk_data['entities_and_relationships']['relationships'], graph, chunk_data['source_doc_name'], chunk_data['page_number'], chunk_data.get('summary', []))
            except Exception as e:
                handle_error_no_return(f"Could not store entities and relationships for chunk {chunk_number} in {selected_knowledge_domain} graph DB, encountered error: ", e)
    except Exception as e:
        handle_local_error("Could not iterate over chunk entities for storage in {selected_knowledge_domain} graph DB, encountered error: ", e)

    print(f"\nSuccessfully stored {len(chunk_entities)} chunks in {selected_knowledge_domain} graph DB\n")
    return True


def append_graph_entities_to_chunks(chunks, complete_chunk_entities):
    '''
    Appends `entities_and_relationships` to regular-RAG doc chunks. Receives a complete chunk_entities dict:

    complete_chunk_entities = {
        '<graph_chunk_number>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>'
        }
    }
    (created by convert-doc_chunks_to_graph_entities and updated by extract-all_entities_and_relationships)

    And updates 'chunks' basis the 'source_chunks' key to include the `entities_and_relationships`:

    chunks = [
        {
            'content': '<text>',
            'source_link': '<source_link>',
            'source': '<filename>',
            'page_number': '<page_number>',
            'entities_and_relationships': '<node_relationships_dict>'
        }
    ]
    (created by chunk-docs_with_page_numbers)

    By adding this data to the chunks, we no longer need to launch the graph extraction model when executing GraphRAG,
    saving massive time and significant resources (~8GB VRAM savings)!

    While graph-chunk sizes ('chunk_text' in chunk_entities) are typically larger than plain RAG chunks ('content' in chunks) as
    graph entity extraction benefits from greater context than the 200-300 word chunks used in plain RAG to match the typical search-query length,
    by storing entities and relationships derived from multiple chunks into individual chunks, we benefit from an overlap effect when performing GraphRAG!
    '''
    for _, chunk_data in complete_chunk_entities.items():
        
        try:
            for source_chunk in chunk_data['source_chunks']:
                
                try:    # try to extend existing metadata...

                    if not isinstance(source_chunk, int) or source_chunk < 0 or source_chunk >= len(chunks):
                        continue    # index guard (fallback won't help if out of range!)

                    chunks[source_chunk]['entities_and_relationships']['nodes'].extend(chunk_data['entities_and_relationships']['nodes'])
                    chunks[source_chunk]['entities_and_relationships']['relationships'].extend(chunk_data['entities_and_relationships']['relationships'])
                except (KeyError, TypeError, AttributeError):  # ...or append afresh!
                    chunks[source_chunk]['entities_and_relationships'] = chunk_data['entities_and_relationships']

                '''
                Here, we follow Python's "EAFP" style: "Easier to Ask Forgiveness Than Permission"!
                That is, if we were to perform safety checks in an effort to catch any potential issue before extending, we'd have a ton of 'if' checks:
                if isinstance dict, if is instance list, if is instance int, if is instance str, etc. This would be a nightmare to maintain and read.
                Then there would be the length checking if blank etc, just terrible.
                Instead, we simply try to extend and catch relevant exceptions to handle them with a safe fallback: fresh append!
                '''
        except Exception as e:
            handle_error_no_return(f"Could not append graph entities to RAG chunk {source_chunk} in complete_chunk_entities, encountered error: ", e)

    return chunks


@app.route('/start_falkordb')
def start_falkordb():
    print("\nStarting FalkorDB Docker container...\n")
    try:
        rag_support_module.bring_graph_db_online()
        return jsonify({"message": "FalkorDB Docker container started successfully", "success": True}), 200
    except Exception as e:
        return handle_api_error("Could not start FalkorDB Docker container, encountered error: ", e)


@app.route('/start_perplexica')
def start_perplexica():
    print("\nStarting Perplexica Docker container...\n")
    try:
        rag_support_module.bring_perplexica_online()
        return jsonify({"message": "Perplexica Docker container started successfully", "success": True}), 200
    except Exception as e:
        return handle_api_error("Could not start Perplexica Docker container, encountered error: ", e)


@app.route('/start_searxng')
def start_searxng():
    print("\nStarting SearXNG Docker container...\n")
    try:
        rag_support_module.bring_searxng_online()
        return jsonify({"message": "SearXNG Docker container started successfully", "success": True}), 200
    except Exception as e:
        return handle_api_error("Could not start SearXNG Docker container, encountered error: ", e)


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
        hf_waitress_kb_generator_server_path = pathlib.Path.cwd() / str(config_data['graph_models_base_directory_name']) / str(config_data['graph_extraction_model_directory_name'])
        print(f"\nLaunching HF-Waitress instance with kb-generator model at path: {hf_waitress_kb_generator_server_path}\n")
    except Exception as e:
        handle_local_error("Could not read graph model config, encountered error: ", e)

    if utils.is_local_server_online(f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}")['server_online']:
        print("\nGraphing model server is already online, skipping launch...\n")
        return True
    
    # Get free GPU memory and shutdown the main Waitress LLM chat server if necessary:
    try:
        hf_waitress_base_url = get_url_for_server('hf-waitress')
        graph_summarizer_base_url = f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}"
        utils.ensure_minimum_free_vram(int(config_data['minimum_free_vram_for_graph_extraction_model']), [graph_summarizer_base_url, hf_waitress_base_url])
    except Exception as e:
        handle_local_error(f"Could not reserve minimum GPU memory ({config_data['minimum_free_vram_for_graph_extraction_model']}MB) required for Graph-Extraction model, encountered error: ", e)

    base_command = 'python' if platform.system() == 'Windows' else 'python3'

    cmd_list = [
        base_command,
        'hf_waitress.py',
        '--port', str(config_data['graph_model_server_port']),
        '--model_id', str(config_data['graph_generator_model']),
        '--max_new_tokens', str(config_data['graph_model_max_new_tokens']),
    ]

    if str(config_data['exl2_quantize_graph_model']).lower() == 'true':
        cmd_list.extend([
            '--exl2',
            '--exl2_bpw', str(config_data['exl2_quantize_graph_model_bpw']),
            '--exl2_max_seq_len', str(config_data['graph_model_max_seq_len']),
        ])
    else:
        cmd_list.extend([
            '--quantize', str(config_data['quantize_graph_model']),
            '--quant_level', str(config_data['quantize_graph_model_bits']),
        ])

    try:
        if platform.system() == 'Windows':
            graph_extraction_process = subprocess.Popen(
                cmd_list,
                cwd=hf_waitress_kb_generator_server_path,
                creationflags=subprocess.CREATE_NEW_CONSOLE,  # or CREATE_NEW_PROCESS_GROUP
            )
        else:
            with open('graph_extraction_model_output_log.txt', 'w') as f:
                graph_extraction_process = subprocess.Popen(
                    cmd_list,
                    cwd=hf_waitress_kb_generator_server_path,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

        timeout = 5
        attempts = 25
        for _ in range(attempts):

            # Detect early crash of Graph-Extraction model HF-Waitress process:
            if graph_extraction_process is not None:
                exit_code = graph_extraction_process.poll()
                if exit_code is not None:
                    handle_local_error(
                        "Graph-Extraction model HF-Waitress server process exited early before becoming available. "
                        f"Return code: {exit_code}. Check the Graph-Extraction model server log for details. ",
                        RuntimeError(f"Graph-Extraction model HF-Waitress exited with code {exit_code}")
                    )
                    return False

            if utils.is_local_server_online(f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}")['server_online']:
                print(f"\nGraph-Extraction model launched successfully!\n")
                return True

            print(f"Graph-Extraction model not yet running, waiting {timeout} seconds before retrying...")
            time.sleep(timeout)

    except Exception as e:
        handle_local_error("Could not launch HF-Waitress instance with graph-extraction model, encountered error: ", e)
    
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
        client = rag_support_module.get_graph_db_client()
        clustering_success, status_message = apply_leiden_clustering(client, str(selected_knowledge_domain))
        if not clustering_success:
            raise Exception(status_message)
        return True
    except Exception as e:
        handle_local_error(f"Could not apply Leiden clustering to the graph for {selected_knowledge_domain}, encountered error: ", e)


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


def convert_doc_chunks_to_graph_entities(chunks:list[dict]) -> dict:
    '''
    Assembles document chunks into a dictionary of entities for storage to the GraphDB:
    '''
    try:
        read_return = read_config(['graph_chunk_size', 'graph_chunk_overlap'])
        graph_chunk_size = int(read_return.get('graph_chunk_size', 1500))
        graph_chunk_overlap = int(read_return.get('graph_chunk_overlap', 300))
        
        graphing_chunk = ""     # chunks will be combined upto `graph_chunk_size` and stored here
        chunks_in_storage_queue = []    # V.Important for graph-generator()! Serves as 'source_chunks' which are later mapped to extracted graph entities by append-graph_entities_to_chunks()
        source_filename = ""
        page_number_list = []
        chunk_entities = {}
        graph_chunk_count = 1
        overlap_text = ""

        print("\nGenerating Graphing Chunks Dictionary...\n")
        
        for count, chunk in enumerate(chunks):

            try:
                source_filename = pathlib.Path(rf"{str(chunk['source'])}").resolve().name

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

                    # Add overlap text to the next chunk - The [-<number>:] syntax is used to get the last <number> of characters from the string
                    overlap_text = graphing_chunk[-graph_chunk_overlap:]
                    # Python's negative-indexing handles edge cases gracefully and will return the entire string if somehow the overlap is larger than the chunk size, 
                    # which should never happen anyways, especially after the if-condition check above! 

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
                handle_error_no_return(f"Error processing final batch of chunks in assemble-chunks_for_graph_db(), encountered error: ", e)
    except Exception as e:
        handle_local_error(f"Error assembling chunks for graph DB, encountered error: ", e)

    return chunk_entities


def determine_graph_cache_reuse(entities_and_relationships_filepath:pathlib.Path, summaries_filepath:pathlib.Path) -> tuple[bool, bool, dict]:
    '''
    Determine if the graph cache should be reused, and if so, load the complete chunk entities from the cache.

    Args:
        - entities_and_relationships_filepath: pathlib.Path object of the entities and relationships cache filepath
        - summaries_filepath: pathlib.Path object of the summaries cache filepath
    
    Returns:
        - tuple[bool, bool, dict]: A tuple containing:
            - bool: True if the graph cache should be reused, False otherwise
            - bool: True if the summary generation should be skipped, False otherwise
            - dict: The complete chunk entities if the graph cache should be reused, None otherwise

    Raises:
        - Exception: If the graph cache config cannot be determined
    '''

    try:
        read_return = read_config([
            'reuse_graph_extraction_cache_without_validation',
            'reuse_graph_summary_cache_without_validation',
            'skip_summary_generation'
        ])
        
        reuse_graph_extraction_cache_without_validation = str(read_return['reuse_graph_extraction_cache_without_validation']).lower() == 'true'
        reuse_graph_summary_cache_without_validation = str(read_return['reuse_graph_summary_cache_without_validation']).lower() == 'true'
        skip_summary_generation = str(read_return['skip_summary_generation']).lower() == 'true'
        complete_chunk_entities = None
        reuse_previous_extract = False
        loaded_entities_and_relationships = False
    except Exception as e:
        handle_local_error("Could not determine graph cache config, encountered error: ", e)
    
    if reuse_graph_extraction_cache_without_validation and entities_and_relationships_filepath.exists():
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
    
    if reuse_graph_summary_cache_without_validation and summaries_filepath.exists():
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

    return reuse_previous_extract, skip_summary_generation, complete_chunk_entities


def get_graph_cache_filepaths(input_filepath:pathlib.Path, docs_to_knowledge_graph_dir:pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    try:
        input_filename_no_ext = str(input_filepath.stem)
        
        entities_and_relationships_filename = input_filename_no_ext + '_entities_and_relationships.json'
        entities_and_relationships_filepath = docs_to_knowledge_graph_dir / entities_and_relationships_filename
        
        summaries_filename = input_filename_no_ext + '_summaries.json'
        summaries_filepath = docs_to_knowledge_graph_dir / summaries_filename

        return entities_and_relationships_filepath, summaries_filepath
    
    except Exception as e:
        handle_local_error("Could not set graph generator file names and paths, encountered error: ", e)


def read_graph_generator_config():
    try:
        read_return = read_config(['docs_to_knowledge_graph_dir', 'selected_knowledge_domain', 'apply_clustering_to_graph_db_on_doc_load'])
        return read_return['docs_to_knowledge_graph_dir'], read_return['selected_knowledge_domain'], str(read_return['apply_clustering_to_graph_db_on_doc_load']).lower() == 'true'
    except Exception as e:
        handle_local_error("Could not read graph generator config, encountered error: ", e)


def graph_generator(chunks:list[dict], input_filepath:pathlib.Path) -> bool:
    '''
    Assembles document chunks (via convert-doc_chunks_to_graph_entities()) into a dictionary of entities for storage to the GraphDB:

    chunk_entities = {
        '<graph_chunk_number>': {
            '<chunk_text>': '<text>',
            '<source_chunks>': '<chunk_numbers>', #eg: [12,13,14]
            '<source_doc_name>': '<name>',
            '<page_number>': '<page_number>'
        }
    }

    This is then passed to the graphing model which will process each graph_chunk and append the `entities_and_relationships` key to each chunk_entity in the dict:
        
        '<entities_and_relationships>': {"nodes": [{"type": "organization","name": "Intel"},{"type": "object","name": "Intel Products"},...], "relationships": [{"source": "Intel","target": "Intel Products","relationship": "business unit"},...]}

    The final `chunk_entities` dict is then passed to the `store-entities_and_relationships_in_graph_db()` function which will invoke the summary generator if applicable, 
    and proceed to store all entities and relationships in the GraphDB.

    Future TODO: Split work across multiple instances of the Graphing Model and Summarizer LLM. Requires minimum two GPUs.
    '''
    
    print(f"\n\nGraph Generator Invoked. Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    try:
        rag_support_module.bring_graph_db_online()
    except Exception as e:
        handle_local_error("Could not bring graph DB online, encountered error: ", e)

    try:
        docs_to_knowledge_graph_dir, selected_knowledge_domain, apply_clustering_to_graph_db_on_doc_load = read_graph_generator_config()
        complete_chunk_entities = None
    except Exception as e:
        handle_local_error("Could not read graph generator config, encountered error: ", e)

    try:
        entities_and_relationships_filepath, summaries_filepath = get_graph_cache_filepaths(input_filepath, pathlib.Path(rf"{docs_to_knowledge_graph_dir}").resolve())
    except Exception as e:
        handle_local_error("Could not set graph generator file names and paths, encountered error: ", e)

    try:
        reuse_previous_extract, skip_summary_generation, complete_chunk_entities = determine_graph_cache_reuse(entities_and_relationships_filepath, summaries_filepath)
    except Exception as e:
        handle_local_error("Could not determine graph cache reuse, encountered error: ", e)
    
    if not reuse_previous_extract:
        try:
            bring_graph_extraction_model_online()
        except Exception as e:
            handle_local_error("Could not bring graphing model online, encountered error: ", e)

        try:
            chunk_entities = convert_doc_chunks_to_graph_entities(chunks)
        except Exception as e:
            handle_local_error("Could not assemble chunks for graph DB, encountered error: ", e)

        try:
            complete_chunk_entities = extract_all_entities_and_relationships(chunk_entities=chunk_entities, cache_filepath=entities_and_relationships_filepath)
        except Exception as e:
            handle_local_error("Failed to extract entities and relationships from chunk entities, encountered error: ", e)
    
    try:
        client = rag_support_module.get_graph_db_client()
        graph = client.select_graph(selected_knowledge_domain)  # Will create the graph if it doesn't exist
    except Exception as e:
        handle_local_error(f"Could not connect to / initialize graph for '{selected_knowledge_domain}' domain in graph DB, encountered error: ", e)
    
    try:
        store_entities_and_relationships_in_graph_db(complete_chunk_entities, selected_knowledge_domain, graph, skip_summary_generation, summaries_filepath)
    except Exception as e:
        handle_local_error("Could not store entities and relationships in graph DB, encountered error: ", e)

    if apply_clustering_to_graph_db_on_doc_load:
        try:
            apply_leiden_clustering_to_graph(selected_knowledge_domain)
        except Exception as e:
            handle_error_no_return("Could not apply Leiden clustering to the graph, encountered error: ", e)

    try:
        chunks_with_graph_entities = append_graph_entities_to_chunks(chunks, complete_chunk_entities)
    except Exception as e:
        handle_local_error("Could not append graph entities to chunks, encountered error: ", e)

    print(f"\n\nCompleted Graph Generator at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return chunks_with_graph_entities


def read_embeddings_config() -> tuple[str, pathlib.Path, int, int, bool, bool]:
    print("\n\nReading embeddings config\n\n")
    try:
        read_return = read_config(['selected_embedding_model', 'chunk_size', 'chunk_overlap', 'upload_doc_to_graph_db', 'perform_only_graph_rag'])
    except Exception as e:
        handle_local_error("Missing values in config.json, could not read embeddings config. Error: ", e)

    path_to_knowledge_domain = rag_support_module.get_path_to_knowledge_domain()
    return read_return['selected_embedding_model'], path_to_knowledge_domain, int(read_return['chunk_size']), int(read_return['chunk_overlap']), str(read_return['upload_doc_to_graph_db']).lower() == 'true', str(read_return['perform_only_graph_rag']).lower() == 'true'


# Document vectorization and chunking
def whoosh_embed_and_graph_doc_chunks(input_filepath:pathlib.Path) -> tuple[int, int]:
    '''
    Vectorize and chunk the document, and then store the chunks in the VectorDB and GraphDB

    Args:
        - input_file: pathlib.Path object of the input file

    Returns:
        - tuple[int, int]: The chunk size and chunk overlap

    Raises:
        - Exception: If the document cannot be vectorized and chunked
    '''

    print("\n\nCore Document Vectorization and Chunking Function Invoked\n\n")

    # Read Embeddings Config
    try:
        selected_embedding_model, path_to_knowledge_domain, chunk_sz, chunk_olp, upload_doc_to_graph_db, perform_only_graph_rag = read_embeddings_config()
    except Exception as e:
        handle_local_error("Could not read embeddings config, encountered error: ", e)

    # Chunk Source Data
    print("Chunking Doc")
    try:
        chunks = chunk_docs_with_page_numbers(input_filepath, chunk_sz) # Generates a list of dictionaries, each containing 'content', 'source', and 'page_number' as keys
        if len(chunks) > 0:

            if upload_doc_to_graph_db:
                try:
                    chunks = graph_generator(chunks, input_filepath)
                except Exception as e:
                    handle_error_no_return("Could not graph chunks, skipping. Encountered error: ", e)
            
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
            
            print("Document added to knowledge domain.")
        else:
            print("No chunks generated, skipping indexing and embedding")
    except Exception as e:
        handle_local_error("Failed to chunk document for storage to VectorDB, encountered error: ", e)
    finally:
        print("\nCompleted Document-Upload Processing.\n")
        return chunk_sz, chunk_olp


def determine_sequence_id_for_chat(chat_id: int) -> int:

    print(f"\n\nDetermining sequence ID for chat with chat_id: {chat_id}")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing keys in config.json for method determine-sequence_id_for_chat(). Error: ", e)

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
        current_sequence_id = int(result[0])     # 'result' will be a list, so extract the first value

        print(f"Current max sequence_id for chat_id: {chat_id} is: {current_sequence_id}")
        return current_sequence_id  # returning current max sequence_id, this will be incremented by 1 when a new response is stored to the db
        
    except Exception as e:
        handle_local_error("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)


def store_to_chat_history_db(
        chat_id: str,
        sequence_id: str,
        stream_session_id: str,
        user_query: str,
        user_query_html: str,
        model_response_for_history_db: str,
        messages_object: str,
        local_llm_server: str,
        local_llm_chat_template_format: str
    ) -> tuple[str, str]:

    print(f"\n\nStoring chat history for chat with chat_id: {chat_id} and sequence_id: {sequence_id}")

    try:
        read_return = read_config(['sqlite_history_db', 'local_llm_server', 'model_choice'])
        sqlite_history_db = read_return['sqlite_history_db']
        local_llm_server = read_return['local_llm_server']
        model_choice = read_return['model_choice']
    except Exception as e:
        handle_local_error("Missing keys in config.json for method store-local_llm_chat_history_to_db. Error: ", e)

    if local_llm_server == "hf-waitress":
        try:
            model_choice = read_hf_config(['model_id'])['model_id']
        except Exception as e:
            handle_error_no_return("Could not read model_id from hf_config.json, setting to 'unknown'. Encountered error: ", e)
            model_choice = 'unknown'

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
        prev_max_sequence_id = determine_sequence_id_for_chat(int(chat_id))
        sequence_id = prev_max_sequence_id + 1
    except Exception as e:
        handle_local_error("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)

    try:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime('%d %b %Y - %I:%M %p %Z')
    except Exception as e:
        handle_local_error("Could not obtain timestamp in store-local_llm_chat_history_to_db, encountered error: ", e)

    try:
        # Store conversation history into DB
        cursor.execute('''
            INSERT INTO chat_history (
                chat_id,
                sequence_id,
                stream_session_id,
                user_query,
                user_query_html,
                llm_response,
                llm_model,
                prompt_template,
                local_llm_server, 
                prompt_template_format,
                date_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                int(chat_id),
                int(sequence_id),
                str(stream_session_id),
                str(user_query), 
                str(user_query_html),
                str(model_response_for_history_db),
                str(model_choice),
                str(messages_object),
                str(local_llm_server),
                str(local_llm_chat_template_format),
                str(formatted_datetime)
            )
        )
        conn.commit()
        print(f"\n\nInserted chat history into DB with chat_id: {chat_id}\n\n")
    except Exception as e:
        handle_local_error("Could not insert chat history into DB, encountered error: ", e)

    return formatted_datetime, chat_id


# Route for loading all models from model dir
@app.route('/load_local_models')
def load_local_models():
    try:
        read_return = read_config(['model_dir', 'draft_models_dir', 'llama_cpp_spec_draft_model'])
        model_dir = read_return['model_dir']
        draft_models_dir = read_return['draft_models_dir']
        llama_cpp_spec_draft_model = read_return['llama_cpp_spec_draft_model']
    except Exception as e:
        return handle_api_error("Missing model_dir in config.json for method load-local_models. Error: ", e)
    
    try:
        models = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
        draft_models = [f for f in os.listdir(draft_models_dir) if os.path.isfile(os.path.join(draft_models_dir, f))]
    except Exception as e:
        return handle_api_error("Could not load list of local models, encountered error: ", e)
        
    #print(f"locally available models: {models}")
    return jsonify({
        'success': True,
        'models': models,
        'draftModels': draft_models,
        'draftModelSelected': llama_cpp_spec_draft_model
    })


@app.route('/gguf_reader', methods=['POST'])
def gguf_reader():

    try:
        model_name = request.json.get('model_name')
        model_dir = read_config(['model_dir'])['model_dir']
        model_path = pathlib.Path(rf"{str(model_dir)}").resolve() / str(model_name)
    except Exception as e:
        return handle_api_error("Could not obtain model path, encountered error: ", e)

    try:
        from gguf.gguf_reader import GGUFReader
    except Exception as e:
        return handle_api_error("Could not import GGUFReader, please `pip install gguf`. Encountered error: ", e)

    gguf_details = ""
    try:
        reader   = GGUFReader(model_path)   # loads and memory-maps the GGUF file :contentReference[oaicite:0]{index=0}

        #print(f"=== Tensors in {model_path.name} ===")
        # reader.tensors is now a list of ReaderTensor(NamedTuple) :contentReference[oaicite:1]{index=1}
        for tensor in reader.tensors:
            name        = tensor.name                     # tensor name, e.g. "layers.0.ffn_up_proj_exps"
            dtype       = tensor.tensor_type.name         # quantization / dtype, e.g. "Q4_K", "F32"
            shape       = tuple(int(dim) for dim in tensor.shape)  # e.g. (4096, 11008)
            n_elements  = tensor.n_elements                # total number of elements
            n_bytes     = tensor.n_bytes                   # total byte size on disk

            #print(f"{name}\tshape={shape}\tdtype={dtype}\telements={n_elements}\tbytes={n_bytes}")
            gguf_details += f"{name}\tshape={shape}\tdtype={dtype}\telements={n_elements}\tbytes={n_bytes}\n"
    except Exception as e:
        return handle_api_error("Could not create GGUFReader, encountered error: ", e)

    return jsonify(success=True, gguf_details=gguf_details)


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
        filepath = pathlib.Path(rf"{str(model_dir)}").resolve() / str(filename)

        print("Loading new LLM - filename: ", filename)
        print("Loading new LLM - filepath: ", filepath)

        # Save the uploaded file to the specified path
        input_file.save(str(filepath))
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
        return jsonify(success=False)


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
        handle_local_error(f"Server-side error - could not prepare staging info record for Google Drive file: '{filename_with_extension}', encountered error: ", e)
    
    try:
        staged_file_info = check_if_file_already_staged(staging_info_record)
        if staged_file_info is not None:
            return staged_file_info
    except Exception as e:
        handle_error_no_return(f"Could not check if GDrive file '{filename_with_extension}' is already staged, proceeding afresh. Encountered error: ", e)

    # New file, proceed with fresh upload
    try:    # convert file_content to FileStorage object and pass to save-file_to_staging_dir
        file_to_stage = FileStorage(
            stream=io.BytesIO(file_content),
            filename=filename_with_extension,
            content_type=mime_type
        )
    except Exception as e:
        handle_local_error("Server-side error recieving file: ", e)
    
    try:
        staged_filename, staged_filepath = save_file_to_staging_dir(file_to_stage)
    except Exception as e:
        handle_local_error("Server-side error, could not save file to staging area, encountered error: ", e)
    
    try:
        staging_info_record['document_name_and_extension'] = staged_filename # secure_filename version of original filename
        staging_info_record['staged_datetime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        staging_info_record['staged_filepath'] = str(staged_filepath)
        staging_info_record['txt_filepath'] = ''
        staging_info_record['status'] = 'Staged - File Saved to Staging Area'
        insert_into_staging_db(staging_info_record, skip_check=True)  # `file_already_staged` check has already been performed, so this will insert directly into the DB barring some error.
    except Exception as e:
        handle_local_error("Server-side error, could not insert file into staging DB, encountered error: ", e)
    
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
                
                #base_filename, _ = os.path.splitext(filename) # Remove existing extension if any
                base_filename = pathlib.Path(rf"{str(filename)}").resolve().stem
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
        handle_local_error("Error downloading file from Google Drive: ", e)
    
    try:
        file_content = file.getvalue()
        if file_content is None:
            raise Exception("Server-side error - Could not download file from Google Drive")
    except Exception as e:
        handle_local_error("Server-side error downloading file from Google Drive: ", e)
    
    try:
        return stage_gdrive_file(filename_with_extension, file_content, mime_type, lars_user_id)
    except Exception as e:
        handle_local_error("Server-side error staging file from Google Drive: ", e)


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
        handle_local_error(f"Error downloading Google Drive folder: '{folder_name}' in the GDrive download folder method, encountered error: ", e)

    return staged_file_records


def gdrive_downloader(service, file_or_folder_id, filename, mime_type, mime_type_category, lars_user_id):
    try:
        if mime_type_category == "folder":
            return download_folder(service, file_or_folder_id, filename, lars_user_id)    # filename is the folder name in this case
        else:
            return download_gdrive_file(service, file_or_folder_id, filename, mime_type, lars_user_id)
    except Exception as e:
        handle_local_error("Error downloading file from Google Drive in the gdrive-downloader() method, encountered error: ", e)


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


def get_text_extract_from_pdf(pdf_filepath:pathlib.Path) -> pathlib.Path:
    '''
    Determine which OCR service to use and extract text from the PDF document

    Args:
        - pdf_filepath: pathlib.Path object of the PDF file to be OCR'ed

    Returns:
        - pathlib.Path object of the output text file

    Raises:
        - Exception: If the PDF file cannot be opened, the output text file cannot be initialized, or the OCR process fails
    '''
    print("\nGetting text extract from PDF\n")

    try:
        read_return = read_config(['force_ocr', 'ocr_service_choice'])
        force_ocr = read_return.get('force_ocr', False)
        ocr_service_choice = read_return.get('ocr_service_choice', None)
    except Exception as e:
        handle_local_error("Could not determine ocr configuration from config.json. Disabling OCR and proceeding. Error: ", e)
    
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
                handle_local_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)
    else:
        try:
            txt_filepath = PDFtoTXT(pdf_filepath)
        except Exception as e:
            handle_local_error("Failed to extract text from the PDF document via PyPDF2, encountered error: ", e)
    
    return txt_filepath


def convert_to_pdf_with_unoconv(input_file_path: pathlib.Path, output_file_path: pathlib.Path):
    '''
    Convert a non-PDF document to a PDF file using unoconv

    Args:
        - input_file_path: pathlib.Path object of the input file to be converted
        - output_file_path: pathlib.Path object of the output file to be created
    '''
    print(f"\n\nConverting non-PDF document to PDF format. Input file: {input_file_path}. Output file: {output_file_path}\n\n")
    if platform.system() == 'Windows':
        subprocess.run(['python', 'unoconv.py', '-f', 'pdf', '-o', output_file_path, input_file_path], check=True)
    else:
        subprocess.run(['unoconv', '-f', 'pdf', '-o', output_file_path, input_file_path], check=True)


def prep_and_execute_unoconv_conversion(input_filepath: pathlib.Path, target_dir_path: pathlib.Path) -> pathlib.Path:
    '''
    Prepare and execute the unoconv conversion of a non-PDF document to a PDF file

    Args:
        - input_filepath: pathlib.Path object of the input file to be converted
        - target_dir_path: pathlib.Path object of the directory to save the converted file

    Returns:
        - pathlib.Path object of the output file

    Raises:
        - Exception: If the input file cannot be converted, the output file cannot be created, or the conversion fails
    '''
    print("Converting to PDF file")

    try:
        conv_filename = input_filepath.with_suffix(".pdf").name
        output_filepath = target_dir_path / conv_filename
        convert_to_pdf_with_unoconv(input_filepath, output_filepath)
        return output_filepath
    except subprocess.CalledProcessError as e:
        handle_local_error("Could not convert file to PDF, encountered error: ", e)
    except Exception as e:
        handle_local_error("Unexpected error when converting file to PDF, encountered error: ", e)


def check_if_converted_file_exists(pdf_filename: str) -> tuple[bool, pathlib.Path]:
    '''
    Invoked for non-PDF files to check if a converted file already exists

    Args:
        - pdf_filename: pathlib.Path object of the PDF file to be checked

    Returns:
        - tuple[bool, pathlib.Path]: True if the converted file exists, False otherwise, and the path to the converted file

    Raises:
        - Exception: If the converted file cannot be found, the output file cannot be created, or the conversion fails
    '''
    try:
        pdf_filepath = pathlib.Path(rf"{app.config['UPLOAD_FOLDER']}").resolve() / pdf_filename
        return pdf_filepath.exists(), pdf_filepath
    except Exception as e:
        handle_error_no_return("Could not determine if converted file already exists, proceeding to convert file regardless. Encountered error: ", e)
        return False, None


def get_pdf_filepath_for_upload(filepath: pathlib.Path) -> pathlib.Path:
    '''
    Determine which PDF filepath to use for upload - either from staging or converted directories

    Args:
        - filepath: pathlib.Path object of the file to be uploaded

    Returns:
        - pathlib.Path object of the PDF filepath to be uploaded

    Raises:
        - Exception: If the PDF filepath cannot be determined, the conversion fails, or the file cannot be uploaded
    '''
    try:
        if not filepath.suffix.lower() == '.pdf':
            converted_file_exists, converted_pdf_file_path = check_if_converted_file_exists(filepath.with_suffix(".pdf").name)
            if not converted_file_exists:
                pdf_filepath = prep_and_execute_unoconv_conversion(filepath, converted_pdf_file_path.parent)
            else:
                pdf_filepath = converted_pdf_file_path
            return pdf_filepath
        else:
            return filepath
    except Exception as e:
        handle_local_error("Could not get PDF filepath for upload, encountered error: ", e)


def upload_to_rag_and_records_databases(original_filename:str, txt_filepath:pathlib.Path) -> bool:
    '''
    Upload the document to the RAG and Records databases

    Args:
        - original_filename: str of the original filename
        - txt_filepath: pathlib.Path object of the text filepath

    Returns:
        - bool: True if the document was uploaded to the RAG and Records databases, False otherwise

    Raises:
        - Exception: If the document cannot be uploaded to the RAG and Records databases
    '''
    try:
        if txt_filepath.is_file() and txt_filepath.stat().st_size > 0:
            chunk_size, chunk_overlap = whoosh_embed_and_graph_doc_chunks(txt_filepath)
        else:
            print("Extracted document is empty! Skipping upload to RAG databases.")
    except Exception as e:
        handle_local_error("Failed to embed & index document: ", e)

    try:
        if txt_filepath.is_file() and txt_filepath.stat().st_size > 0:
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
        handle_local_error("Could not read config when preparing basic staging info record, encountered error: ", e)

    try:
        upload_id = str(uuid.uuid4())
    except Exception as e:
        handle_local_error("Could not generate unique ID for staging info record, encountered error: ", e)
    
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
        handle_local_error("Could not prepare basic staging info record, encountered error: ", e)
    
    return staging_info_record
    

def init_and_connect_to_upload_staging_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    try:
        read_return = read_config(['upload_staging_db'])
        upload_staging_db = read_return['upload_staging_db']
    except Exception as e:
        handle_local_error("Missing upload_staging_db in config.json for method init_and_connect_to_upload_staging_db. Error: ", e)
    
    try:
        conn = sqlite3.connect(upload_staging_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to upload_staging_db, encountered error: ", e)
    
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
        handle_local_error("Could not create upload_staging table, encountered error: ", e)
    
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
        handle_local_error("Could not add necessary columns to upload_staging table, encountered error: ", e)
    
    return conn, cursor


def update_existing_entry_in_staging_db(doc_info: dict, id: int):
    try:
        conn, cursor = init_and_connect_to_upload_staging_db()
    except Exception as e:
        handle_local_error("Could not connect to upload staging DB, encountered error: ", e)
    
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
        handle_local_error("Could not update existing entry in upload staging DB, encountered error: ", e)
    finally:
        cursor.close()
        conn.close()
    return True


def check_for_staged_file_info_in_staging_db(file_transfer_info: dict):
    try:
        conn, cursor = init_and_connect_to_upload_staging_db()
    except Exception as e:
        handle_local_error("Could not connect to upload staging DB, encountered error: ", e)
    
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
        handle_local_error("Could not check if files are already staged, encountered error: ", e)
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
        handle_local_error("File transfer info validation failed, encountered error: ", e)


def check_if_file_already_staged(file_transfer_info: dict):
    try:
        staged_file_info = check_for_staged_file_info_in_staging_db(file_transfer_info)
        # print(f"\n\nStaged file info: {staged_file_info}\n\n")
        if staged_file_info is not None:
            print(f"\n\nStaged file info is not None! Validating...\n\n")
            try:    # validate file transfer data
                validate_file_transfer_info(staged_file_info) # exception will be raised if invalid
                print(f"\n\nStaged file info validated successfully!\n\n")
                if (pathlib.Path(rf"{app.config['UPLOAD_STAGING_FOLDER']}").resolve() / staged_file_info['document_name_and_extension']).exists():
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
        handle_local_error("Could not check if file is already staged, encountered error: ", e)


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
                handle_local_error("Could not delete entry from upload staging DB, encountered error: ", e)
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
        handle_local_error("Could not connect to upload staging DB, encountered error: ", e)
    
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
        handle_local_error("Could not add new entry to upload staging DB, encountered error: ", e)
    finally:
        cursor.close()
        conn.close()
    
    return True


def perform_post_bulk_upload_cleanup(data_queue: queue.Queue = None):
    # 1. Shutdown the graph summarizer model
    try:
        config_data = read_config(['graph_summarizer_access_url', 'graph_summarizer_server_port', 'graph_model_access_url', 'graph_model_server_port'])
        summarizer_url = f"http://{config_data['graph_summarizer_access_url']}:{config_data['graph_summarizer_server_port']}"
        if data_queue is not None: data_queue.put(f"Shutting down graph summarizer model at URL {summarizer_url}... | waiting")
        response = utils.shutdown_waitress_server(summarizer_url)
        if isinstance(response, dict) and response.get('success', False):
            print(f"\nSuccessfully shut down graph summarizer model at URL {summarizer_url}\n")
            if data_queue is not None: data_queue.put(f"Successfully shut down graph summarizer model at URL {summarizer_url} | success")
        else:
            handle_error_no_return(f"\nCould not shut down graph summarizer model at URL {summarizer_url}, proceeding regardless...\n")
            if data_queue is not None: data_queue.put(f"Error shutting down graph summarizer model at URL {summarizer_url} | failure")
    except Exception as e:
        handle_error_no_return("Could not shutdown graph summarizer model, encountered error: ", e)

    # 2. Shutdown the graph extraction model
    try:
        graph_extraction_url = f"http://{config_data['graph_model_access_url']}:{config_data['graph_model_server_port']}"
        if data_queue is not None: data_queue.put(f"Shutting down graph extraction model at URL {graph_extraction_url}... | waiting")
        response = utils.shutdown_waitress_server(graph_extraction_url)
        if isinstance(response, dict) and response.get('success'):
            print(f"\nSuccessfully shut down graph extraction model at URL {graph_extraction_url}\n")
            if data_queue is not None: data_queue.put(f"Successfully shut down graph extraction model at URL {graph_extraction_url} | success")
        else:
            handle_error_no_return(f"\nCould not shut down graph extraction model at URL {graph_extraction_url}, proceeding regardless...\n")
            if data_queue is not None: data_queue.put(f"Error shutting down graph extraction model at URL {graph_extraction_url} | failure")
    except Exception as e:
        handle_error_no_return("Could not shutdown graph extraction model, encountered error: ", e)

    # 3. Ensure chat-LLM server is online
    try:
        read_return = read_config(['local_llm_server', 'exclusive_server_mode'])
        server_to_start = read_return['local_llm_server']
        exclusive_server_mode = str(read_return['exclusive_server_mode']).lower() == 'true'
        if data_queue is not None: data_queue.put(f"Checking if {server_to_start} chat-LLM server is online... | waiting")
    except Exception as e:
        handle_error_no_return("Server-side error, could not read local_llm_server from config.json, encountered error: ", e)
    
    try:
        if server_to_start == 'hf-waitress':
            hf_waitress_base_url = get_url_for_server('hf-waitress')
            if not utils.is_local_server_online(hf_waitress_base_url)['server_online']: # This means the main chat server was likely shut due to insufficient VRAM

                server_starter_response = hf_waitress_server_starter(exclusive_server_mode = exclusive_server_mode)
                if server_starter_response is not None and server_starter_response.get('success'):
                    print("\nSuccessfully started HF-Waitress Chat server\n")
                    if data_queue is not None: data_queue.put(f"Successfully started HF-Waitress Chat server | success")
                    return True
                else:
                    handle_error_no_return(f"\nCould not start HF-Waitress server, proceeding regardless...\n")
                    if data_queue is not None: data_queue.put(f"Error starting HF-Waitress Chat server | failure")

            else:
                print("\nHF-Waitress chat server already online\n")
                if data_queue is not None: data_queue.put(f"HF-Waitress chat server is online | success")
                return True
        
        elif server_to_start == 'llama-cpp':
            server_starter_response = llama_cpp_server_starter(exclusive_server_mode)
            if server_starter_response is not None and server_starter_response.get('success'):
                print("\nSuccessfully started llama-cpp Chat server\n")
                if data_queue is not None: data_queue.put(f"Successfully started llama-cpp Chat server | success")
                return True
            else:
                handle_error_no_return(f"\nCould not start llama-cpp server, proceeding regardless...\n")
                if data_queue is not None: data_queue.put(f"Error starting llama-cpp Chat server | failure")
        
        else:
            handle_error_no_return(f"Invalid local LLM server choice: {server_to_start}")
            if data_queue is not None: data_queue.put(f"Invalid local LLM server choice: {server_to_start} | failure")
    except Exception as e:
        handle_error_no_return("Could not start local LLM server post-document upload, encountered error: ", e)
        if data_queue is not None: data_queue.put(f"Error starting local LLM server post-document upload | failure")

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
                upload_to_rag_and_records_databases(doc['document_name_and_extension'], pathlib.Path(rf"{doc['txt_filepath']}").resolve())
                if data_queue is not None: data_queue.put(f"Successfully Uploaded Document: {doc.get('document_name_and_extension', 'Unknown document name')} | success")
                safe_delete_entry_from_staging_db(doc)
            except Exception as e:
                if data_queue is not None: data_queue.put(f"Error uploading file to RAG & Records databases | failure")
                handle_error_no_return(f"Could not upload {doc.get('document_name_and_extension', 'Unknown document name')} to RAG & Records databases, skipping this document. Encountered error: ", e)
            if data_queue is not None: data_queue.put(f"Completed processing file {count + 1} of {len(docs_to_upload)}! | success")
    finally:    # Cleanup!
        perform_post_bulk_upload_cleanup(data_queue)

    return True


def move_file_to_upload_dir(staged_filename: str, staged_filepath: pathlib.Path) -> pathlib.Path:
    
    try:    # copy from staging to upload folder
        final_filepath = pathlib.Path(rf"{app.config['UPLOAD_FOLDER']}").resolve() / staged_filename
        shutil.copy2(str(staged_filepath), str(final_filepath))   # copy2 preserves more metadata than copy()
        safe_remove_file_from_filepath(staged_filepath)
        return final_filepath   # access filename using final_filepath.name
    except FileNotFoundError as f:
        handle_local_error(f"File not found: {staged_filepath}", f)
    except Exception as e:
        handle_local_error("Could not move file to upload folder, encountered error: ", e)


def enable_kosmos_vram_offloading():
    return safe_write_config({'kosmos_offload_vram': True})['success']


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
    
    And iterates through the list, adding documents to LARS/Privion
    '''
    print("\nBulk uploading documents from staging area\n")

    try:

        must_enable_kosmos_vram_offloading_after_bulk_upload_completes = disable_kosmos_vram_offloading()

        for count, doc in enumerate(staged_docs_to_upload):
            if data_queue is not None: data_queue.put(f"Performing Step 1 of 2 for document: {doc.get('document_name_and_extension', 'Unknown document name')} - Data Extraction. Progress: {count + 1} of {len(staged_docs_to_upload)}... | waiting")
            try:    # Move to upload dir
                filepath = move_file_to_upload_dir(doc['document_name_and_extension'], pathlib.Path(rf"{doc['staged_filepath']}").resolve()) # will also delete from staging dir
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
                pdf_filepath = get_pdf_filepath_for_upload(filepath)
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
                doc['txt_filepath'] = str(txt_filepath)
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
        global DOCLING_CONVERTER
        DOCLING_CONVERTER = None

    return True


@app.route('/bulk_upload_files', methods=['POST'])
def bulk_upload_files():
    '''
    Step Two of the bulk upload process!
    For Step One, see below for file-transfer_to_staging()

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
                handle_local_error("Server-side error, could not perform bulk text extraction, encountered error: ", e)
            
            try:
                bulk_upload_files_to_rag_and_records_databases(docs_to_upload, data_queue)
            except Exception as e:
                handle_local_error("Server-side error, could not bulk upload files to RAG & Records databases, encountered error: ", e)
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


def save_file_to_staging_dir(input_file) -> tuple[str, pathlib.Path]:   # input_file is a Request.files object
    try:
        filename = secure_filename(input_file.filename)
        filename = filename.replace("PDF", "pdf") if "PDF" in filename else filename
        filepath = pathlib.Path(rf"{app.config['UPLOAD_STAGING_FOLDER']}").resolve() / filename
        input_file.save(str(filepath))   # save() is a method of the Request.files object, and it expects a string filepath, not a pathlib.Path object
        print(f"\nSaved file {filename} to {filepath} successfully.\n")
        return filename, filepath
    except Exception as e:
        handle_local_error("Could not save file to staging directory, encountered error: ", e)


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
    
    And the file itself as a Request.files object.

    Will save the file to the staging area, and add the following keys to the `file_transfer_info` dictionary and return it:
        staged_datetime: str
        staged_filepath: str
        txt_filepath: str   # will be left blank for now and updated after text extraction by the bulk_text_extract method
        status: str

    Additionally, the `document_name_and_extension` key will be updated to the secure-filename version of the original filename.

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
        file_transfer_info['document_name_and_extension'] = staged_filename # secure-filename version of original filename
        file_transfer_info['staged_datetime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_transfer_info['staged_filepath'] = str(staged_filepath)
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
        handle_local_error("Could not connect to upload staging DB to check for pending uploads, encountered error: ", e)

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
        handle_local_error("Could not check for pending uploads, encountered error: ", e)
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
        handle_local_error("Could not connect to upload staging DB to cancel pending uploads, encountered error: ", e)
    
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
                safe_remove_file_from_filepath(pathlib.Path(rf"{file['staged_filepath']}").resolve())
                safe_remove_file_from_filepath(pathlib.Path(rf"{file['txt_filepath']}").resolve())
    except Exception as e:
        handle_local_error("Could not check for pending uploads, encountered error: ", e)
    
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
        handle_local_error("Could not delete pending uploads from staging DB, encountered error: ", e)
    
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
        return handle_api_error("Missing sqlite_history_db in config.json for method store_user_rating. Error: ", e)
    
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


def run_prechecks_for_asr_server_starter(hard_reboot_required: bool):
    '''
    This method checks if:
        - The HF-Waitress ASR-ASGI server is already running, and if so, checks if a hard-reboot is required (hard_reboot_required set).
            - If set, the running server process is terminated and a new instance can be launched.
            - If not, the server starter method may simply return.

    Args:
        hard_reboot_required (bool): Whether to check if a hard-reboot is required - will be set and specified by HF-Waitress itself basis changes to Core Settings.

    Returns:
        - dict: A dictionary containing the following keys:
            - 'success': True if the prechecks were successful, False otherwise.
            - 'asr_model': The model choice, or 'undefined' if not set.
            - 'hf_waitress_server_running': True if the HF-Waitress server is running, False otherwise.
            - 'skip_fresh_start': True if the prechecks determined that the HF-Waitress server should not be launched, False otherwise.
            - 'reboot_failed': True if the hard-reboot failed, False otherwise.
    '''
    
    global ASR_CHANGE_RELOAD_TRIGGER_SET
    asr_server_running = False
    asr_base_url = get_url_for_server('asr-waitress')
    try:
        asr_model = read_config(['asr_model'])['asr_model']
    except Exception as e:
        handle_local_error("Could not read asr_model from config.json - prechecks failed. Encountered error: ", e)
        asr_model = 'undefined'

    if hard_reboot_required:    # will be set and specified by ASR-Waitress itself
        print("\nHard-Reboot of ASR-Waitress server requested.\n")
        try:
            if utils.shutdown_waitress_server(asr_base_url)['success']: # will handle online check and immediately return if the server is already offline
                print(f"\nSuccessfully shut down HF-Waitress ASR-ASGI server at URL {asr_base_url}, proceeding to restart...\n")
                asr_server_running = False
            else:
                raise Exception(f"\nCould not shut down HF-Waitress ASR-ASGI server at URL {asr_base_url}...\n")
        
        except Exception as e:
            asr_server_running = True
            err_msg = f"Could not terminate running HF-Waitress process before hard-reboot. Your IP is likely not whitelisted and thus unauthorized for this action, contact the administrator for help."
            handle_error_no_return(f"{err_msg} Additional technical details follow: ", e)            
            return {'success': True, 'asr_model': asr_model, 'asr_server_running': asr_server_running, 'skip_fresh_start': True, 'reboot_failed': True}    # success True as the prechecks method has successfully determined the server's status
    else:
        try:
            asr_server_running = utils.is_local_server_online(asr_base_url)['server_available']
        except Exception as e:
            asr_server_running = False
            handle_error_no_return("Warning: Could not check if ASR-Waitress server is running. Proceeding to launch ASR-Waitress server. Encountered error: ", e)

    return {'success': True, 'asr_model': asr_model, 'asr_server_running': asr_server_running, 'skip_fresh_start': asr_server_running, 'reboot_failed': False}     # skip fresh start if ASR-Waitress server is already running



def asr_server_starter(hard_reboot_required: bool = False):
    print("\n\nStarting HF-Waitress Server\n\n")

    try:
        precheck_result = run_prechecks_for_asr_server_starter(hard_reboot_required)
    except Exception as e:
        handle_error_no_return(
            "Could not run prechecks for HF-Waitress server starter, proceeding with safe defaults for first launch. Encountered error: ",
            e
        )
        precheck_result = {}
    
    if precheck_result.get('skip_fresh_start', False):
        return precheck_result
        
    try:
        config_data = read_config([
            'asr_model',
            'asr_waitress_server_port',
            'voice_base_directory_name',
            'asr_subdirectory_name',
            'asr_torch_device',
            'hf_waitress_server_retry_attempts',
            'hf_waitress_server_timeout_seconds'
        ])
        asr_base_url = get_url_for_server('asr-waitress')
        
        hf_waitress_asr_server_path = pathlib.Path.cwd() / str(config_data['voice_base_directory_name']) / str(config_data['asr_subdirectory_name'])
        print(f"\nLaunching HF-Waitress instance for ASR at path: {hf_waitress_asr_server_path}\n")
    except Exception as e:
        handle_local_error("Could not read ASR model config, encountered error: ", e)
    
    print("\n\nProceeding to launch HF-Waitress ASR-ASGI server\n\n")
    
    asr_base_command = 'python' if platform.system() == 'Windows' else 'python3'

    cmd_list = [
        asr_base_command,
        'asr_waitress.py',
        '--port', str(config_data['asr_waitress_server_port']),
        '--model_id', str(config_data['asr_model']),
        '--torch_device_map', str(config_data['asr_torch_device']),
        '--quantize', 'n',
    ]

    print(f"\n\nLaunching ASR-Waitress server with command: {' '.join(cmd_list)}\n\n")

    try:
        if platform.system() == 'Windows':
            asr_waitress_process = subprocess.Popen(
                cmd_list,
                cwd=hf_waitress_asr_server_path,
                creationflags=subprocess.CREATE_NEW_CONSOLE,  # or CREATE_NEW_PROCESS_GROUP
            )
        else:
            with open('asr_waitress_output_log.txt', 'w') as f:
                asr_waitress_process = subprocess.Popen(
                    cmd_list,
                    cwd=hf_waitress_asr_server_path,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
    except Exception as e:
        handle_local_error("Could not launch ASR-Waitress instance, encountered error: ", e)

    try:
        for _ in range(config_data['hf_waitress_server_retry_attempts']):

            # Detect early crash of ASR-Waitress process:
            if asr_waitress_process is not None:
                exit_code = asr_waitress_process.poll()
                if exit_code is not None:
                    # Process died before the HTTP server came up
                    # Optional: on non‑Windows you already log to asr_waitress_output_log.txt
                    # so you can tell the user to check that file.
                    handle_local_error(
                        "ASR-Waitress server process exited early before becoming available. "
                        f"Return code: {exit_code}. Check the ASR-Waitress server log for details. ",
                        RuntimeError(f"ASR-Waitress exited with code {exit_code}")
                    )
            
            if utils.is_local_server_online(asr_base_url)['server_online']:
                print(f"\nASR server launched successfully!\n")
                return {
                    'success': True,
                    'asr_model': config_data['asr_model'],
                    'asr_server_running': True,
                    'skip_fresh_start': False,
                    'reboot_failed': False
                }
            
            time.sleep(config_data['hf_waitress_server_timeout_seconds'])
    except Exception as e:
        handle_local_error("Could not check status of ASR-Waitress server after launch attempt, printing error and returning: ", e)

    return {
        'success': False,
        'asr_model': config_data['asr_model'],
        'asr_server_running': False,
        'skip_fresh_start': False,
        'reboot_failed': precheck_result.get('reboot_failed', False)
    }


@app.route('/asr_server_starter_endpoint', methods=['POST'])
def asr_server_starter_endpoint():
    print("\n\nASR server starter endpoint called\n\n")
    global ASR_CHANGE_RELOAD_TRIGGER_SET
    try:
        data = request.get_json()
        hard_reboot_required = data.get('hard_reboot_required', 'false')
    except Exception as e:
        return handle_api_error("Could not read request, encountered error: ", e)
    
    try:
        result = asr_server_starter(hard_reboot_required = (hard_reboot_required or ASR_CHANGE_RELOAD_TRIGGER_SET)) # If either is True, the ASR server will be restarted.
        ASR_CHANGE_RELOAD_TRIGGER_SET = False
        return jsonify(result)
    except Exception as e:
        return handle_api_error("Could not start ASR server, encountered error: ", e)


def get_tts_pipeline(tts_name: str, **kwargs):
    
    if tts_name == 'kokoro_82m':
        
        try:
            return KPipeline(lang_code=kwargs['kokoro_language_code'])
        except Exception as e:
            raise Exception(f"Could not load Kokoro 82M TTS pipeline, encountered error: {e}")
    else:
        raise Exception(f"Invalid TTS choice: {tts_name}")


def tts_pipeline_core_loader():
    print("\nLoading TTS pipeline...\n")
    global TTS_PIPELINE

    try:
        read_return = read_config(['selected_tts', 'selected_kokoro_voice', 'tts_sample_rate', 'kokoro_language_code'])
    except Exception as e:
        handle_local_error("Server-side error, could not read tts configuration, encountered error: ", e)
    
    try:
        kwargs = {'tts_sample_rate': read_return['tts_sample_rate'], 'kokoro_voice': read_return['selected_kokoro_voice'], 'kokoro_language_code': read_return['kokoro_language_code']}
        TTS_PIPELINE = get_tts_pipeline(tts_name=read_return['selected_tts'], **kwargs)
    except Exception as e:
        handle_local_error("Server-side error, could not load tts pipeline, encountered error: ", e)

    return True


@app.route('/load_tts_pipeline')
def load_tts_pipeline():
    
    try:
        tts_pipeline_core_loader()
        return jsonify(success=True)
    except Exception as e:
        return handle_api_error("Server-side error, could not load tts pipeline, encountered error: ", e) 
    

def synth_to_wav_bytes(text: str) -> bytes:
    global TTS_PIPELINE

    try:
        read_return = read_config(['selected_tts', 'selected_kokoro_voice', 'tts_sample_rate'])
    except Exception as e:
        raise Exception(f"Could not read tts configuration, encountered error: {e}")
    
    try:
        if read_return['selected_tts'] == 'kokoro_82m':
            if TTS_PIPELINE is None:
                tts_pipeline_core_loader()
            
            chunks = []
            generator = TTS_PIPELINE(text, voice=read_return['selected_kokoro_voice'])
            for i, (gs, ps, audio) in enumerate(generator):
                # print(i, gs, ps)
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                chunks.append(audio.astype(np.float32))
            if not chunks:
                return b''
            waveform = np.concatenate(chunks)
            buf = io.BytesIO()
            sf.write(buf, waveform, int(read_return['tts_sample_rate']), format='WAV', subtype='PCM_16')
            return buf.getvalue()
        else:
            raise Exception(f"Invalid TTS choice: {read_return['selected_tts']}")
    except Exception as e:
        raise Exception(f"Could not synthesize TTS audio, encountered error: {e}")


@app.route('/tts_voice', methods=['POST'])
def tts_voice():
    try:
        payload = request.get_json(silent=True) or {}
        text = (payload.get('text') or '').strip()
    except Exception as e:
        return handle_api_error("Server-side error, could not read text or voice, encountered error: ", e)
    
    try:
        wav_bytes = synth_to_wav_bytes(text)
        return Response(
            wav_bytes,
            mimetype='audio/wav',
            headers={
                'Content-Disposition': 'inline; filename="tts.wav"',
                'Cache-Control': 'no-store'
            }
        )
    except Exception as e:
        return handle_api_error("Server-side error, could not synthesize TTS audio, encountered error: ", e)


def check_status_and_shutdown_llm_server(server_to_shutdown: str):
    print(f"\n\nShutdown Requested: Checking Status & Shutting Down Server: {server_to_shutdown} ...\n\n")
    
    try:
        target_server_url = get_url_for_server(server_to_shutdown)  # will throw error if invalid server choice
    except Exception as e:
        err_msg = f"Could not check if the server is running, encountered error: "
        handle_error_no_return(err_msg, e)
        return {'success': False, 'message': f"{err_msg} {e}"}
    
    try:
        if server_to_shutdown == 'hf-waitress' or server_to_shutdown == 'asr-waitress':
            result = utils.shutdown_waitress_server(target_server_url)
        elif server_to_shutdown == 'llama-cpp':
            result = utils.shutdown_local_llm_server_process(LLAMA_CPP_PROCESS)
        else:
            raise Exception(f"\nInvalid server choice: {server_to_shutdown}\n")
        
        if result['success']:
            return result
        else:
            raise Exception(result['message'])
    
    except Exception as e:
        err_msg = f"Could not terminate running server at URL {target_server_url}. The server may have been launched externally, in which case Privion does not have control over it, or your IP is not whitelisted / authorized for this action. Contact the administrator for help with the following technical details: "
        handle_error_no_return(err_msg, e)
        return {'success': False, 'message': f"{err_msg} {e}"}


@app.route('/shutdown_local_llm_server', methods=['POST'])
def shutdown_local_llm_server():
    '''
    API interface to check-status_and_shutdown_llm_server, leaving that method free to be invoked directly in other methods here.
    '''
    try:
        data = request.get_json()
        server_to_shutdown = data.get('server_to_shutdown')
    except Exception as e:
        return handle_api_error("Server-side error, could not read server_to_shutdown from the POST request, encountered error: ", e)

    try:
        return jsonify(check_status_and_shutdown_llm_server(server_to_shutdown = server_to_shutdown))
    except Exception as e:
        return handle_api_error("Server-side error, could not shutdown local LLM server, encountered error: ", e)


def run_prechecks_for_llama_cpp_server_starter(exclusive_server_mode: bool):
    '''
    This method checks if:
        - The llama.cpp server is already running, and if so, checks if a restart is required (reload trigger set) due to setting changes.
            - If set, the running server process is terminated and a new instance can be launched.
            - If not, the server starter method may simply return.
        
    Furthermore, it can be made to check if other LLM servers (eg: HF-Waitress) are running, and shut them down if so, if the related input arg is set to True.

    Args:
        exclusive_server_mode (bool): Whether to check if other LLM servers (eg: HF-Waitress) are running and shut them down if so.

    Returns:
        - dict: A dictionary containing the following keys:
            - 'success': True if the prechecks were successful, False otherwise.
            - 'llm_model': The model choice, or 'undefined' if not set.
            - 'hf_waitress_server_running': True if the HF-Waitress server is running, False otherwise.
            - 'llama_cpp_server_running': True if the llama.cpp server is running, False otherwise.
            - 'skip_fresh_start': True if the prechecks determined that the llama.cpp server should not be launched, False otherwise.
            - 'reboot_failed': True if the hard-reboot failed, False otherwise.
    '''

    global LLAMA_CPP_PROCESS, LLM_CHANGE_RELOAD_TRIGGER_SET
    hf_waitress_server_running, llama_cpp_server_running = False, False
    hf_waitress_base_url = get_url_for_server('hf-waitress')
    llama_cpp_base_url = get_url_for_server('llama-cpp')

    try:
        model_choice = read_config(['model_choice'])['model_choice']
    except Exception as e:
        handle_error_no_return("Missing model_choice in config.json in method llama-cpp-server-starter. Printing error and proceeding with model_choice: 'undefined' ", e)
        model_choice = 'undefined'

    # First, determine status of HF-Waitress server
    if exclusive_server_mode:
        waitress_shutdown_result = utils.shutdown_waitress_server(hf_waitress_base_url)
        if waitress_shutdown_result['success']:
            print(f"\nSuccessfully shut down HF-Waitress server at URL {hf_waitress_base_url}\n")
            hf_waitress_server_running = False
        else:
            hf_waitress_server_running = True # Set to True as we've determined the other server is running and we failed to terminate it
            err_msg = "Could not terminate running HF-Waitress process before launching llama.cpp, proceeding regardless. Your IP may not be whitelisted for this action, contact the administrator for help."
            handle_error_no_return(f"{err_msg} Provide the following technical details: ", waitress_shutdown_result['message'])
    # else:
    #     try:    # to set an accurate value for the boolean regardless of exclusive_server_mode
    #         hf_waitress_server_running = utils.is_local_server_online(hf_waitress_base_url)['server_available']
    #     except Exception as e:
    #         handle_error_no_return("Warning: Could not check if HF-Waitress server is running. Proceeding to launch llama.cpp server. Encountered error: ", e)

    try:
        llama_cpp_server_running = utils.is_local_server_online(llama_cpp_base_url)['server_available']
    except Exception as e:
        handle_error_no_return("Warning: Could not check if llama.cpp server is running. Proceeding to launch llama.cpp server. Encountered error: ", e)

    if llama_cpp_server_running and LLAMA_CPP_PROCESS is not None:

        if LLM_CHANGE_RELOAD_TRIGGER_SET:
            
            print("\n\nllama.cpp server online, but reload requested. Attempting to terminate and restart...\n\n")
            try:
                if utils.shutdown_local_llm_server_process(LLAMA_CPP_PROCESS)['success']:
                    print(f"\nSuccessfully shut down llama.cpp server at URL {llama_cpp_base_url}\n")
                    LLAMA_CPP_PROCESS = None
                    LLM_CHANGE_RELOAD_TRIGGER_SET = False
                    return {'success': True, 'llm_model': model_choice, 'hf_waitress_server_running': hf_waitress_server_running, 'llama_cpp_server_running': False, 'skip_fresh_start': False, 'reboot_failed': False}
                else:
                    raise Exception(f"\nFailed to shutdown llama.cpp server at URL {llama_cpp_base_url}\n")
            except Exception as e:
                LLM_CHANGE_RELOAD_TRIGGER_SET = False
                handle_error_no_return("Failed to terminate running llama.cpp process, server was likely launched by a previous session. To change, shutdown the previously launched server manually and reload this page. Technical error-details follow: ", e)
                return {'success': True, 'llm_model': model_choice, 'hf_waitress_server_running': hf_waitress_server_running, 'llama_cpp_server_running': True, 'skip_fresh_start': True, 'reboot_failed': True}    # success True as the prechecks method has successfully determined the server's status

        else:
            print(f'\n\nThe llama.cpp server is already loaded and the reload trigger is not set. Presently selected model: {model_choice}. Returning...\n\n')
            return {'success': True, 'llm_model': model_choice, 'hf_waitress_server_running': hf_waitress_server_running, 'llama_cpp_server_running': True, 'skip_fresh_start': True, 'reboot_failed': False}
        
    elif llama_cpp_server_running and LLAMA_CPP_PROCESS is None:
        print(f'\n\nThe llama.cpp server is running but Privion does not have control over it, as it was launched by another process/session. Presently selected model: {model_choice}. Returning...\n\n')
        LLM_CHANGE_RELOAD_TRIGGER_SET = False   # LLM-CHANGE_RELOAD_TRIGGER_SET does not matter in this case, as we cannot restart the server anyways!
        return {'success': True, 'llm_model': model_choice, 'hf_waitress_server_running': hf_waitress_server_running, 'llama_cpp_server_running': True, 'skip_fresh_start': True, 'reboot_failed': False}
    
    else:
        print(f'\n\nThe llama.cpp server is not running. Proceeding to launch llama.cpp server.\n\n')
        return {'success': True, 'llm_model': model_choice, 'hf_waitress_server_running': hf_waitress_server_running, 'llama_cpp_server_running': False, 'skip_fresh_start': False, 'reboot_failed': False}


def llama_cpp_server_starter(exclusive_server_mode: bool):
    print("\n\nStarting llama.cpp Server\n\n")

    global LLM_CHANGE_RELOAD_TRIGGER_SET, LLAMA_CPP_PROCESS

    try:
        precheck_result = run_prechecks_for_llama_cpp_server_starter(exclusive_server_mode)
    except Exception as e:
        handle_local_error("Could not run prechecks for llama.cpp server starter, encountered error: ", e)
    
    if precheck_result['skip_fresh_start']:
        return precheck_result
    
    LLM_CHANGE_RELOAD_TRIGGER_SET = False       # Reset the flags to False as we're launching a new server - Do NOT reset before prechecks, and do not force-set LLAMA_CPP_PROCESS to None as it may be running!
    
    try:
        read_return = read_config(
            [
                'model_dir',
                'model_choice',
                'draft_models_dir',
                'llama_cpp_spec_type',
                'llama_cpp_spec_draft_model',
                'llama_cpp_spec_draft_n_max',
                'llama_cpp_warmup',
                'llama_cpp_alias',
                'llama_cpp_misc_args',
                'llama_cpp_check_tensors',
                'llama_cpp_verbose',
                'llama_cpp_special_tokens_output',
                'llama_cpp_reasoning_budget',
                'llama_cpp_enable_thinking',
                'llama_cpp_preserve_thinking',
                'llama_cpp_dio',
                'llama_cpp_main_gpu_index',
                'llama_cpp_fit',
                'llama_cpp_fit_min_ctx',
                'llama_cpp_fit_target',
                'llama_cpp_flash_attn',
                'llama_cpp_context_length',
                'llama_cpp_batch_size',
                'llama_cpp_ubatch_size',
                'llama_cpp_max_new_tokens',
                'llama_cpp_gpu_layers', 
                'llama_cpp_server_timeout_seconds',
                'llama_cpp_server_retry_attempts',
                'llama_cpp_use_gpu',
                'llama_cpp_unified_kv_buffer',
                'llama_cpp_disable_kv_offloading',
                'llama_cpp_key_cache_data_type',
                'llama_cpp_value_cache_data_type',
                'llama_cpp_no_of_seqs_to_par_decode',
                'llama_cpp_mlock',
                'llama_cpp_no_mmap',
                'llama_cpp_offload_to_devices',
                'llama_cpp_cpu_only_moe',
                'llama_cpp_num_cpu_moe',
                'llama_cpp_split_mode',
                'llama_cpp_tensor_split',
                'llama_cpp_override_tensor',
                'llama_cpp_serving_url',
                'llama_cpp_server_port'
            ]
        )
        llama_cpp_base_url = get_url_for_server('llama-cpp')
        cpp_model = str(
            pathlib.Path(rf"{read_return['model_dir']}").resolve() / read_return['model_choice']
        )

        spec_draft_model = read_return.get('llama_cpp_spec_draft_model', '')
        draft_model = None
        if spec_draft_model and spec_draft_model != 'none':
            draft_model = str(
                pathlib.Path(rf"{read_return['draft_models_dir']}").resolve() / spec_draft_model
            )

    except Exception as e:
        handle_local_error("Error with core configuration in config.json for llama.cpp server starter method: ", e)

    if not read_return['llama_cpp_use_gpu']:
        read_return['llama_cpp_gpu_layers'] = 0

    try:
        # ---------------------------------------------------------
        # Branch 0: Init with Base Arguments
        # ---------------------------------------------------------
        llama_cpp_safe_launch_args = [
            'llama-server',
            '--warmup' if read_return['llama_cpp_warmup'] else '--no-warmup',
            '--reasoning-budget', str(read_return['llama_cpp_reasoning_budget']),
            '--model', cpp_model,
            '--ctx-size', str(read_return["llama_cpp_context_length"]),
            '--batch-size', str(read_return["llama_cpp_batch_size"]),
            '--ubatch-size', str(read_return["llama_cpp_ubatch_size"]),
            '--n-predict', str(read_return["llama_cpp_max_new_tokens"]),
            '--parallel', str(read_return["llama_cpp_no_of_seqs_to_par_decode"]),
            '--jinja',
            '--host', read_return['llama_cpp_serving_url'],
            '--port', str(read_return['llama_cpp_server_port']),
            '--chat-template-kwargs', 
            json.dumps({
                'enable_thinking': bool(read_return['llama_cpp_enable_thinking']),
                'preserve_thinking': bool(read_return['llama_cpp_preserve_thinking'])
            }, separators=(',', ':'))   # no spaces or other separators!
        ]

        spec_type = read_return.get('llama_cpp_spec_type', '')

        if spec_type and spec_type != 'none':
            llama_cpp_safe_launch_args.extend([
                '--spec-type', str(spec_type),
                '--spec-draft-n-max', str(read_return['llama_cpp_spec_draft_n_max'])
            ])
            if draft_model:
                llama_cpp_safe_launch_args.extend([
                    '--spec-draft-model', str(draft_model)
                ])

        if read_return['llama_cpp_alias'] and read_return['llama_cpp_alias'] != '':
            llama_cpp_safe_launch_args.extend(['--alias', read_return['llama_cpp_alias']])

        if read_return['llama_cpp_verbose']:
            llama_cpp_safe_launch_args.append('--verbose')

        if read_return['llama_cpp_check_tensors']:
            llama_cpp_safe_launch_args.append('--check-tensors')

        if read_return['llama_cpp_special_tokens_output']:
            llama_cpp_safe_launch_args.append('--special')

        # ---------------------------------------------------------
        # Branch 1: Context / Layers / Fit
        # ---------------------------------------------------------
        if read_return['llama_cpp_fit']:
            llama_cpp_safe_launch_args.extend([
                '--fit', 'on',
                '--fit-target', str(read_return['llama_cpp_fit_target']),
                '--fit-ctx', str(read_return['llama_cpp_fit_min_ctx'])
            ])
        else:
            llama_cpp_safe_launch_args.extend([
                '--fit', 'off',
                '--device', str(read_return["llama_cpp_offload_to_devices"]),
                '--split-mode', str(read_return["llama_cpp_split_mode"])
            ])

            if read_return['llama_cpp_split_mode'] == 'none':
                llama_cpp_safe_launch_args.extend(['--main-gpu', str(read_return['llama_cpp_main_gpu_index'])])

            if read_return['llama_cpp_tensor_split']:
                llama_cpp_safe_launch_args.extend(['--tensor-split', str(read_return['llama_cpp_tensor_split'])])

            # MoE Management:
            if read_return['llama_cpp_override_tensor'] and read_return['llama_cpp_override_tensor'] != '':
                llama_cpp_safe_launch_args.extend([
                    '--override-tensor', str(read_return['llama_cpp_override_tensor']),
                    '--n-gpu-layers', '99'
                ])
            else:
                llama_cpp_safe_launch_args.extend(['--n-gpu-layers', str(read_return["llama_cpp_gpu_layers"])])
                
                # MoE specific logic
                num_cpu_moe = int(read_return.get('llama_cpp_num_cpu_moe', 0))
                if num_cpu_moe == 0 and read_return['llama_cpp_cpu_only_moe']:
                    llama_cpp_safe_launch_args.append('--cpu-moe')
                elif num_cpu_moe > 0:
                    llama_cpp_safe_launch_args.extend(['--n-cpu-moe', str(num_cpu_moe)])
        
        # ---------------------------------------------------------
        # Branch 2: Optimizations & Flags
        # ---------------------------------------------------------
            
        if read_return['llama_cpp_unified_kv_buffer']:
            llama_cpp_safe_launch_args.append('--kv-unified')

        # Flash Attention requires GPU KV Cache - it cannot be enabled if kv-offload is disabled!
        want_flash_attn = (read_return['llama_cpp_flash_attn'] in ['auto', 'on'])
        kv_offload_disabled = read_return.get('llama_cpp_disable_kv_offloading', False)

        if want_flash_attn and flash_attention_is_installed() and not kv_offload_disabled:  
            llama_cpp_safe_launch_args.extend(['--flash-attn', 'on'])
            llama_cpp_safe_launch_args.extend(['--cache-type-k', str(read_return["llama_cpp_key_cache_data_type"])])
            llama_cpp_safe_launch_args.extend(['--cache-type-v', str(read_return["llama_cpp_value_cache_data_type"])])
        
        if kv_offload_disabled:
            llama_cpp_safe_launch_args.append('--no-kv-offload')
        
        if read_return['llama_cpp_mlock']:
            llama_cpp_safe_launch_args.append('--mlock')
        
        if read_return['llama_cpp_no_mmap']:
            llama_cpp_safe_launch_args.append('--no-mmap')

        if read_return['llama_cpp_dio']:
            llama_cpp_safe_launch_args.append('--direct-io')

        if read_return['llama_cpp_misc_args'] and read_return['llama_cpp_misc_args'] != '':
            llama_cpp_safe_launch_args.append(read_return['llama_cpp_misc_args'].strip())
        

        print(f"\n\nLaunching llama.cpp server with command: {' '.join(llama_cpp_safe_launch_args)}\n\n")
        # full_command = ' '.join(llama_cpp_safe_launch_args)
        # print(f"Full command: {full_command}\n\n")

        if platform.system() == 'Windows':
            # windows_command = f'start cmd /k "{full_command}"'  # /c - closes window after command finishes, /k - keeps window open (useful for debugging)
            # LLAMA_CPP_PROCESS = subprocess.Popen(windows_command, shell=True)

            # with open('llama_cpp_server_output_log.txt', 'w') as f:
            #     LLAMA_CPP_PROCESS = subprocess.Popen(
            #         llama_cpp_safe_launch_args,
            #         stdout=f,
            #         stderr=subprocess.STDOUT,
            #         text=True,
            #         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            #     )
            LLAMA_CPP_PROCESS = subprocess.Popen(llama_cpp_safe_launch_args, creationflags=subprocess.CREATE_NEW_CONSOLE)   # only for testing & debugging, see note below!
            '''
            ## Why CREATE_NEW_CONSOLE is less reliable than CREATE_NEW_PROCESS_GROUP:
            
                - Control events don't reach the child: Python's send_signal(signal.CTRL_BREAK_EVENT) uses GenerateConsoleCtrlEvent, which only delivers to processes that share 
                the same console as the caller. With CREATE_NEW_CONSOLE, the child is in a different console, so the event is dropped.
                
                - Process group targeting requires same console: Even with CREATE_NEW_PROCESS_GROUP, Windows won't deliver the control event to a process group in another console. 
                You'd have to attach to that console first (not something subprocess does).

                - Graceful shutdown becomes hit-or-miss: In a separate console, you typically fall back to terminate()/kill(), which is less graceful and can leave resources in a 
                bad state. This can be observed when invoking `utils.send_ctrl_c_to_process()` on the LLAMA-CPP_PROCESS.

                - Logging trade-offs: CREATE_NEW_CONSOLE shows output in a new window but makes file redirection and unified logging harder unless you manually wire stdout/stderr.
            '''
        else:           
            # Platform & container agnostic:
            with open('llama_cpp_server_output_log.txt', 'w') as f:
                LLAMA_CPP_PROCESS = subprocess.Popen(llama_cpp_safe_launch_args, stdout=f, stderr=subprocess.STDOUT, text=True)    #stdout has already been redirected to the file, so simply direct stderr to stdout!

    except Exception as e:
        handle_local_error("Could not launch llama.cpp process, encountered error: ", e)

    try:
        for _ in range(read_return['llama_cpp_server_retry_attempts']):

            # Detect early crash of llama.cpp process:
            if LLAMA_CPP_PROCESS is not None:
                exit_code = LLAMA_CPP_PROCESS.poll()
                if exit_code is not None:
                    # Process died before the HTTP server came up
                    # Optional: on non‑Windows you already log to llama_cpp_server_output_log.txt
                    # so you can tell the user to check that file.
                    handle_local_error(
                        "llama.cpp server process exited early before becoming available. "
                        f"Return code: {exit_code}. Check the llama.cpp server log for details. ",
                        RuntimeError(f"llama.cpp exited with code {exit_code}")
                    )
            
            if utils.is_local_server_online(llama_cpp_base_url)['server_available']:
                print("\n\nllama.cpp server launched succesfully! Returning.\n\n")
                return {
                    'success': True,
                    'llm_model': read_return['model_choice'],
                    'hf_waitress_server_running': precheck_result.get('hf_waitress_server_running', False),
                    'llama_cpp_server_running': True,
                    'skip_fresh_start': False,
                    'reboot_failed': False
                }
            
            time.sleep(read_return['llama_cpp_server_timeout_seconds'])
    except Exception as e:
        handle_error_no_return("Could not check server status after launch attempt, printing error and returning: ", e)

    return {
        'success': False,
        'llm_model': None,
        'hf_waitress_server_running': precheck_result.get('hf_waitress_server_running', False),
        'llama_cpp_server_running': False,
        'skip_fresh_start': False,
        'reboot_failed': precheck_result.get('reboot_failed', False)
    }


def get_hf_waitress_serving_host_and_port():
    try:
        read_return = read_config(['hf_waitress_serving_url', 'hf_waitress_server_port'])
        return read_return['hf_waitress_serving_url'], read_return['hf_waitress_server_port']
    except Exception as e:
        handle_error_no_return("Could not read url & port data for HF-Waitress from config.json, using default localhost:9069 instead. Encountered error: ", e)
        return '0.0.0.0', 9069


def run_prechecks_for_hf_waitress_server_starter(exclusive_server_mode: bool, hard_reboot_required: bool):
    '''
    This method checks if:
        - The HF-Waitress server is already running, and if so, checks if a hard-reboot is required (hard_reboot_required set).
            - If set, the running server process is terminated and a new instance can be launched.
            - If not, the server starter method may simply return.
        
    Furthermore, it can be made to check if other LLM servers (eg: llama.cpp) are running, and shut them down if so, if the related input arg is set to True.

    Args:
        exclusive_server_mode (bool): Whether to check if other LLM servers (eg: llama.cpp) are running, and shut them down if so.
        hard_reboot_required (bool): Whether to check if a hard-reboot is required - will be set and specified by HF-Waitress itself basis changes to Core Settings.

    Returns:
        - dict: A dictionary containing the following keys:
            - 'success': True if the prechecks were successful, False otherwise.
            - 'llm_model': The model choice, or 'undefined' if not set.
            - 'hf_waitress_server_running': True if the HF-Waitress server is running, False otherwise.
            - 'llama_cpp_server_running': True if the llama.cpp server is running, False otherwise.
            - 'skip_fresh_start': True if the prechecks determined that the HF-Waitress server should not be launched, False otherwise.
            - 'reboot_failed': True if the hard-reboot failed, False otherwise.
    '''
    
    global LLAMA_CPP_PROCESS
    hf_waitress_server_running = False
    llama_cpp_server_running = False if LLAMA_CPP_PROCESS is None else True
    hf_waitress_base_url = get_url_for_server('hf-waitress')
    llama_cpp_base_url = get_url_for_server('llama-cpp')
    try:
        model_choice = read_hf_config(['model_id'])['model_id']
    except Exception as e:
        handle_local_error("Could not read model_id from hf_config.json - prechecks failed. Encountered error: ", e)

    # First, determine status of llama.cpp server
    if exclusive_server_mode and LLAMA_CPP_PROCESS is not None:
        llama_cpp_shutdown_result = utils.shutdown_local_llm_server_process(LLAMA_CPP_PROCESS)
        if llama_cpp_shutdown_result['success']:
            print(f"\nSuccessfully shut down llama.cpp server at URL {llama_cpp_base_url}\n")
            llama_cpp_server_running = False
            LLAMA_CPP_PROCESS = None
        else:
            llama_cpp_server_running = True    # We know the llama.cpp server is running, which means `llama-server` is available, so we set LLM-LOADED_UP to True
            err_msg = "Could not terminate running llama.cpp process before launching HF-Waitress, proceeding regardless. Privion cannot control the llama.cpp server as it was likely launched by another user or process, contact the administrator for help."
            handle_error_no_return(f"{err_msg} Provide the following technical details: ", llama_cpp_shutdown_result['message'])
    # else:
    #     try:    # to set an accurate value for the boolean regardless of exclusive_server_mode - LLAMA-CPP_PROCESS might be None but server may be online from elsewhere!
    #         llama_cpp_server_running = utils.is_local_server_online(llama_cpp_base_url)['server_available']
    #     except Exception as e:
    #         llama_cpp_server_running = False
    #         handle_error_no_return("Warning: Could not check if llama.cpp server is running. Proceeding to launch HF-Waitress server. Encountered error: ", e)

    if hard_reboot_required:    # will be specified by HF-Waitress itself and passed to this method
        print("\nHard-Reboot of HF-Waitress server requested.\n")
        try:
            if utils.shutdown_waitress_server(hf_waitress_base_url)['success']:
                print(f"\nSuccessfully shut down HF-Waitress server at URL {hf_waitress_base_url}, proceeding to restart...\n")
                hf_waitress_server_running = False
            else:
                raise Exception(f"\nCould not shut down HF-Waitress server at URL {hf_waitress_base_url}...\n")
        
        except Exception as e:
            hf_waitress_server_running = True
            err_msg = f"Could not terminate running HF-Waitress process before hard-reboot. Your IP is likely not whitelisted and thus unauthorized for this action, contact the administrator for help."
            handle_error_no_return(f"{err_msg} Additional technical details follow: ", e)
            return {'success': True, 'llm_model': model_choice, 'hf_waitress_server_running': hf_waitress_server_running, 'llama_cpp_server_running': llama_cpp_server_running, 'skip_fresh_start': True, 'reboot_failed': True}    # success True as the prechecks method has successfully determined the server's status
    else:   # Just make sure the server is running!
        try:
            hf_waitress_server_running = utils.is_local_server_online(hf_waitress_base_url)['server_available']
        except Exception as e:
            hf_waitress_server_running = False
            handle_error_no_return("Warning: Could not determine if the HF-Waitress server is running, proceeding to launch afresh. Encountered error: ", e)

    return {'success': True, 'llm_model': model_choice, 'hf_waitress_server_running': hf_waitress_server_running, 'llama_cpp_server_running': llama_cpp_server_running, 'skip_fresh_start': hf_waitress_server_running, 'reboot_failed': False}     # skip fresh start if HF-Waitress server is already running


def hf_waitress_server_starter(exclusive_server_mode: bool, hard_reboot_required: bool = False):
    print("\n\nStarting HF-Waitress Server\n\n")

    try:
        precheck_result = run_prechecks_for_hf_waitress_server_starter(exclusive_server_mode, hard_reboot_required)

    except Exception as e:
        handle_error_no_return("Could not run prechecks for HF-Waitress server starter, proceeding with safe defaults for first launch. Encountered error: ", e)
        precheck_result = {}
    
    if precheck_result.get('skip_fresh_start', False):
        return precheck_result
    
    try:
        hf_read_return = read_hf_config(
            [
                'model_id', 'awq', 'use_flash_attention_2', 'flux_diffusers', 
                'flux_low_vram_optimizations', 'load_quantized_flux', 'vision',
                'exl2', 'exl2_no_flash_attn','exl3', 'exl3_resume_quant_job'
            ]
        )   # ONLY include those settings that default to False and are not persisted in hf_config.json!
    except Exception as e:
        handle_error_no_return("Could not read hf_config.json - proceeding with safe defaults for first launch. Encountered error: ", e)
        hf_read_return = {}

    try:
        lars_read_return = read_config(['hf_waitress_server_timeout_seconds', 'hf_waitress_server_retry_attempts'])
        hf_waitress_base_url = get_url_for_server('hf-waitress')
    except Exception as e:
        handle_local_error("Could not read hf_config.json, encountered error: ", e)
    
    print("\n\nProceeding to launch HF-Waitress server\n\n")
    
    # Basic params
    hf_waitress_host, hf_waitress_port = get_hf_waitress_serving_host_and_port()
    launch_args = f'--host={hf_waitress_host} --port={hf_waitress_port} '

    # Transformers params
    if hf_read_return.get('model_id'):
        launch_args += f'--model={hf_read_return.get("model_id")} '
    if hf_read_return.get('awq'):
        launch_args += '--awq '
    if hf_read_return.get('use_flash_attention_2'):
        launch_args += '--use_flash_attention_2 '
    if hf_read_return.get('flux_diffusers'):
        launch_args += '--flux_diffusers '
    if hf_read_return.get('flux_low_vram_optimizations'):
        launch_args += '--flux_low_vram_optimizations '
    if hf_read_return.get('load_quantized_flux'):
        launch_args += '--load_quantized_flux '
    if hf_read_return.get('vision'):
        launch_args += '--vision '
    
    # ExLlama Config
    if hf_read_return.get('exl2'):
        launch_args += '--exl2 '
        if hf_read_return.get('exl2_no_flash_attn'):
            launch_args += '--exl2_no_flash_attn '
    
    elif hf_read_return.get('exl3'):
        launch_args += '--exl3 '
        if hf_read_return.get('exl3_resume_quant_job'):
            launch_args += '--exl3_resume_quant_job '

    launch_args = launch_args.strip()
    base_command = 'python' if platform.system() == 'Windows' else 'python3'
    command_list = [base_command, 'hf_waitress.py']
    if launch_args:  # Add any additional arguments - Only if there are actual arguments
        command_list.extend(launch_args.split())

    print(f"\n\nLaunching HF-Waitress server with command: {' '.join(command_list)}\n\n")

    try:
        if platform.system() == 'Windows':
            hf_waitress_process = subprocess.Popen(command_list, creationflags=subprocess.CREATE_NEW_CONSOLE)
            
            # full_command = f"{base_command} hf_waitress.py {launch_args}"
            # print(f"Full command: {full_command}")
            # windows_command = f'start cmd /c "{full_command}"'  # /c - closes window after command finishes, /k - keeps window open (useful for debugging)
            # subprocess.Popen(windows_command, shell=True)

        else:
            # Platform & container agnostic - On Linux/Unix, you need to explicitly provide the arguments as a list to avoid shell interpretation issues:
            with open('hf_waitress_output_log.txt', 'w') as f:
                # hf_waitress_process = subprocess.Popen(command_list, shell=True)
                hf_waitress_process = subprocess.Popen(command_list, stdout=f, stderr=subprocess.STDOUT, text=True)

    except Exception as e:
        handle_local_error(f"Could not launch HF-Waitress process in directory: {os.getcwd()}, encountered error: ", e)

    try:
        for _ in range(lars_read_return['hf_waitress_server_retry_attempts']):

            # Detect early crash of HF-Waitress process:
            if hf_waitress_process is not None:
                exit_code = hf_waitress_process.poll()
                if exit_code is not None:
                    # Process died before the HTTP server came up
                    # Optional: on non‑Windows you already log to hf_waitress_output_log.txt
                    # so you can tell the user to check that file.
                    handle_local_error(
                        "HF-Waitress server process exited early before becoming available. "
                        f"Return code: {exit_code}. Check the HF-Waitress server log for details. ",
                        RuntimeError(f"HF-Waitress exited with code {exit_code}")
                    )

            if utils.is_local_server_online(hf_waitress_base_url)['server_available']:
                print(f"\n\nHF-Waitress server launched succesfully with model: {hf_read_return.get('model_id')}! Returning.\n\n")
                return {
                    'success': True,
                    'llm_model': hf_read_return.get('model_id'),
                    'hf_waitress_server_running': True,
                    'llama_cpp_server_running': precheck_result.get('llama_cpp_server_running', False),
                    'skip_fresh_start': False,
                    'reboot_failed': False
                }
            
            time.sleep(lars_read_return['hf_waitress_server_timeout_seconds'])
    except Exception as e:
        handle_error_no_return("Could not check server status after launch attempt, printing error and returning: ", e)

    return {
        'success': False,
        'llm_model': None,
        'hf_waitress_server_running': False,
        'llama_cpp_server_running': precheck_result.get('llama_cpp_server_running', False),
        'skip_fresh_start': False,
        'reboot_failed': precheck_result.get('reboot_failed', False)
    }


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
        exclusive_server_mode = str(read_config(['exclusive_server_mode'])['exclusive_server_mode']).lower() == 'true'
        result = hf_waitress_server_starter(exclusive_server_mode = exclusive_server_mode, hard_reboot_required = hard_reboot_required)
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
        read_return = read_config(['local_llm_server', 'exclusive_server_mode'])
        server_to_start = read_return['local_llm_server']
        exclusive_server_mode = str(read_return['exclusive_server_mode']).lower() == 'true'
    except Exception as e:
        return handle_api_error("Server-side error, could not read local_llm_server from config.json, encountered error: ", e)

    try:
        if server_to_start == 'hf-waitress':
            result = hf_waitress_server_starter(exclusive_server_mode = exclusive_server_mode)
            return jsonify(result)
        elif server_to_start == 'llama-cpp':
            result = llama_cpp_server_starter(exclusive_server_mode)
            return jsonify(result)
        else:
            return handle_api_error(f"Invalid local LLM server choice: {server_to_start}")
    except Exception as e:
        return handle_api_error("Server-side error, could not start local LLM server, encountered error: ", e)



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


def shell_delete_folder(target_path:pathlib.Path, delete_vector_db:bool = False) -> bool:
    print(f"Deleting folder / resetting vector_db at path: {target_path}")
    try:
        if target_path.exists():

            if delete_vector_db:
                try:
                    print(f"Attempting to close connection to VectorDB at path: {target_path} before deleting it...")
                    client = chromadb.PersistentClient(path=str(target_path), settings=chromadb.Settings(allow_reset=True))
                    client.reset()  # Specifically mentioned in the chromadb docs as the way to cleanup and remove vector_dbs
                    print(f"Successfully reset vectorDB at path: {target_path}")
                    return True
                except Exception as e:
                    handle_error_no_return("Could not reset vector_db, encountered error: ", e)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(str(target_path))
                    print(f"Removed existing folder: {target_path}")
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"Permission denied. Retrying in 1 second... (Attempt {attempt + 1} of {max_retries})")
                        time.sleep(1)
                    else:
                        print(f"Failed to remove folder after {max_retries} attempts.")
                        break
        else:
            print(f"No existing folder found at: {target_path}")
    except Exception as e:
        handle_error_no_return("Could not remove existing folder, encountered error: ", e)
        return False


def clean_up_docs_loaded_db(selected_embedding_model_choice:str, knowledge_domain:str) -> bool:
    print(f"Cleaning up docs_loaded_db for embedding model: {selected_embedding_model_choice} and knowledge domain: {knowledge_domain}")
    try:
        conn, cursor = init_and_connect_to_docs_loaded_db()
    except Exception as e:
        handle_local_error("Could not initialize and connect to docs_loaded_db to clean up, encountered error: ", e)

    try:
        with conn:  # Handles commit/rollback automatically
            cursor.execute("DELETE FROM document_records WHERE embedding_model LIKE ? AND knowledge_domain LIKE ?", (selected_embedding_model_choice, knowledge_domain))
    except Exception as e:
        handle_local_error("Could not delete document records from docs_loaded_db, encountered error: ", e)
    finally:
        cursor.close()


def delete_knowledge_domain_graph(knowledge_domain:str) -> bool:
    print(f"Deleting knowledge domain graph for: {knowledge_domain}")

    try:
        if not check_if_container_is_running('falkor-db'):
            rag_support_module.bring_graph_db_online()
            time.sleep(5)   # While the bring-graph-db-online() method waits for the container to start, loading the datasets takes a bit longer so we wait 5 seconds before proceeding.
    except Exception as e:
        handle_error_no_return(f"Could not bring graph DB online, unable to delete knowledge graph for {knowledge_domain} domain. Encountered error: ", e)
        return False
    
    try:
        client = rag_support_module.get_graph_db_client()
        graph = client.select_graph(knowledge_domain)
        graph.delete()  # client.delete_graph() is unsupported by FalkorDB. Delete individual nodes with Cypher: `MATCH (n) DETACH DELETE n`
        print(f"Successfully deleted graph for {knowledge_domain} domain in graph DB")
        return True
    except Exception as e:
        handle_error_no_return(f"Could not delete graph for {knowledge_domain} domain in graph DB, encountered error: ", e)
        return False


@app.route('/reset_vector_db_on_disk', methods=['POST'])
def reset_vector_db_on_disk():

    print("Resetting selected VectorDB")

    try:
        selected_embedding_model_choice = str(request.form['selected_embedding_model'])
        knowledge_domain = str(request.form['selected_knowledge_domain'])
    except Exception as e:
        return handle_api_error("Server-side error, could not read selected_embedding_model or selected_knowledge_domain from the POST request in method reset_vector_db_on_disk, encountered error: ", e)

    try:
        knowledge_domain_base_directory = str(read_config(['knowledge_domain_base_directory'])['knowledge_domain_base_directory'])
    except Exception as e:
        handle_local_error("Missing values in config.json, could not determine knowledge_domain_base_directory. Error: ", e)

    try:
        path_to_knowledge_domain = pathlib.Path(rf"{knowledge_domain_base_directory}").resolve() / knowledge_domain
        if not path_to_knowledge_domain.exists():
            path_to_knowledge_domain.mkdir(parents=True, exist_ok=True)
            print(f"\n\nCreated knowledge domain directory: {path_to_knowledge_domain}\n\n")
    except Exception as e:
        return handle_api_error("Could not determine path to knowledge domain, encountered error: ", e)

    try:
        vector_db_path = path_to_knowledge_domain / "vector_db_and_whoosh_index" / selected_embedding_model_choice
        whoosh_index_path = path_to_knowledge_domain / "vector_db_and_whoosh_index" / selected_embedding_model_choice / "whoosh_index"
    except Exception as e:
        return handle_api_error("Could not determine path to vector_db or whoosh_index, encountered error: ", e)
    
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


def generate_user_message_html(user_message: str, user_message_html: str, stream_session_id: str, chat_id: int, sequence_id: int) -> str:
    try:
        if not user_message_html or user_message_html == "" or user_message_html == "None" or user_message_html == "null":
            user_message = user_message.strip('\n')
            user_message = f'''
            <div class="user-message glassmorphism" data-stream-session-id="{stream_session_id}" data-chat-id="{chat_id}" data-sequence-id="{sequence_id}">
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
            return user_message
        else:
            return user_message_html
    except Exception as e:
        handle_local_error("Could not generate user message HTML, encountered error: ", e)


def generate_llm_response_html(llm_response: str, stream_session_id: str, user_rating: str, chat_id: int, sequence_id: int) -> str:
    try:
        result_parts = llm_response.split("pdf_pane_data=",1)
        llm_response = f'<div class="response-and-viewer-container" data-stream-session-id="{stream_session_id}"><div class="llm-wrapper" id="ResponseWrapper{stream_session_id}"><div class="llm-response" id="ResponseContent{stream_session_id}">' + result_parts[0]
        llm_response = llm_response.strip('\n')

        response_rated = False
        if user_rating and (isinstance(user_rating, str) or isinstance(user_rating, int)):
            user_rating = int(user_rating)
            if user_rating > 0:
                response_rated = True

        # Using string-join to avoid newline characters \n's appearing in the HTML output as breakline tags !:
        llm_rating_html_parts = [
            '<br>',
            f'<div class="star-rating" data-rated="{response_rated}" data-rating-chat-id="{chat_id}" data-rating-sequence-id="{sequence_id}">',
            '<i class="far fa-star" data-rate="1"></i>',
            '<i class="far fa-star" data-rate="2"></i>',
            '<i class="far fa-star" data-rate="3"></i>',
            '<i class="far fa-star" data-rate="4"></i>',
            '<i class="far fa-star" data-rate="5"></i>',
            '</div>',
            '</div>',
            '</div>'
        ]
        llm_rating = ''.join(llm_rating_html_parts)

        if response_rated:
            rating_parts = llm_rating.split("far", user_rating)
            if len(rating_parts) <= user_rating:
                llm_rating = "fas".join(rating_parts)
            else:
                llm_rating = "fas".join(rating_parts[:-1]) + "fas" + "far".join(rating_parts[-1:])

        llm_response += llm_rating

        if len(result_parts) > 1:
            llm_response += result_parts[1]
            llm_response += "</div>"
            llm_response = llm_response.strip('\n')

        return llm_response
    except Exception as e:
        handle_local_error("Could not generate LLM response HTML, encountered error: ", e)


def load_messages_list_from_history_db(cursor: sqlite3.Cursor, chat_id: int, sequence_id: int = None) -> list:
    messages_list = None
    try:
        if sequence_id is None:
            sequence_id = determine_sequence_id_for_chat(chat_id)
        
        if sequence_id > 0:
            cursor.execute("SELECT prompt_template FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (chat_id, sequence_id))
            result = cursor.fetchone()

            if result:
                messages_object = result[0]

                # Parse if it's a string (JSON)
                if isinstance(messages_object, str):
                    try:
                        messages_object = json.loads(messages_object)   # since this is JSON.stringified by the cleint when invoking get-references(), we deserialize it with json.loads()
                    except Exception as e:
                        messages_object = {}

                if isinstance(messages_object, dict):
                    messages_list = messages_object.get('messages', [])
                elif isinstance(messages_object, list):
                    messages_list = messages_object
    
        return messages_list
                
    except Exception as e:
        handle_local_error("Could not load messages_list from chat history DB, encountered error: ", e)


@app.route('/load_chat_history', methods=['POST'])
def load_chat_history():

    print("loading chat history")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_api_error("Missing sqlite_history_db in config.json in method load-chat_history. Error: ", e)

    try:
        chat_id = int(request.form['chat_id'])
    except Exception as e:
        return handle_api_error("Could not retrieve Chat ID from request form, encountered error: ", e)

    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to chat history database, encountered error: ", e)

    sequence_id_for_history_search = 1
    chat_history = []
    old_chat_model = "" # Will be set to the last model used for the given chat_id

    while True:

        try:
            c.execute('''
                SELECT user_query, user_query_html, stream_session_id, llm_model, llm_response, user_rating FROM chat_history 
                WHERE chat_id = ? AND sequence_id = ?
                ''',
                (chat_id, sequence_id_for_history_search)
            )
            result = c.fetchone()
            if result is None:  # No more history to retrieve, so break out of the while loop
                break
            
            try:
                user_message, user_message_html, stream_session_id, old_chat_model, llm_response, user_rating = result # list unpacking as result is a tuple
                chat_history.append(generate_user_message_html(user_message, user_message_html, stream_session_id, chat_id, sequence_id_for_history_search))
                chat_history.append(generate_llm_response_html(llm_response, stream_session_id, user_rating, chat_id, sequence_id_for_history_search))
            except Exception as e:
                handle_error_no_return(f"Could not retrieve chat history for chat with chat_id: {chat_id} and sequence_id: {sequence_id_for_history_search}, skipping. Encountered error: ", e)

            sequence_id_for_history_search += 1 # Increment sequence ID for next iteration
        
        except Exception as e:
            return handle_api_error("Could not retrieve chat history, encountered error: ", e)
    
    print(f'\n\nChat history loaded for chat with model: {old_chat_model}\n\n')

    try:
        sequence_id = determine_sequence_id_for_chat(chat_id)
    except Exception as e:
        return handle_api_error("Could not determine sequence_id, encountered error: ", e)

    try:
        messages_list = load_messages_list_from_history_db(c, chat_id, sequence_id)
    except Exception as e:
        handle_error_no_return("Could not load messages_list from chat history DB, encountered error: ", e)
        messages_list = None
    
    c.close()
    conn.close()
    return jsonify({'success': True, 'chat_history': chat_history, 'old_chat_model': old_chat_model, 'sequence_id': sequence_id, 'messages_list': messages_list})


def determine_latest_chat_id(c: sqlite3.Cursor) -> int:
    print("Determining chat ID")
    c.execute("SELECT COALESCE(MAX(chat_id), 0) FROM chat_history") # "The COALESCE function accepts two or more arguments and returns the first non-null argument."
    result = c.fetchone()
    max_chat_id = int(result[0])
    new_max_chat_id = max_chat_id + 1
    print(f"Chat ID determined: {new_max_chat_id}")
    return new_max_chat_id


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
                        user_query_html TEXT,
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
        add_column_if_not_exists(c, 'chat_history', 'sequence_id', 'INTEGER')
        add_column_if_not_exists(c, 'chat_history', 'stream_session_id', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'user_query', 'TEXT')
        add_column_if_not_exists(c, 'chat_history', 'user_query_html', 'TEXT')
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
        handle_local_error(f"Could not delete chat history for chat with chat_id: {chat_id} and sequence_id greater than: {sequence_id}, encountered error: ", e)


@app.route('/delete_messages', methods=['POST'])
def delete_messages():
    print("delete_messages route triggered")

    try:
        chat_id = request.form['chat_id']
        sequence_id = request.form['sequence_id']
    except Exception as e:
        handle_local_error("Could not read chat_id or sequence_id from request, encountered error: ", e)
    
    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json in method delete_messages(). Error: ", e)

    # Connect to chat_history.db to determine appropriate chat_id
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        handle_local_error("Could not connect to chat history database, encountered error: ", e)

    try:
        c.execute("DELETE FROM chat_history WHERE chat_id = ? AND sequence_id >= ?", (chat_id, sequence_id))
        deleted_count = c.rowcount
        conn.commit()
        print(f"Deleted {deleted_count} rows of chat history for chat with chat_id: {chat_id} beginning with sequence_id: {sequence_id}")
    except Exception as e:
        handle_local_error(f"Could not delete chat history for chat with chat_id: {chat_id} beginning with sequence_id: {sequence_id}, encountered error: ", e)
    
    try:
        messages_list = load_messages_list_from_history_db(c, int(chat_id), (int(sequence_id) - 1))
    except Exception as e:
        handle_error_no_return("Could not load messages_list from chat history DB, encountered error: ", e)
        messages_list = None

    conn.close()
    return jsonify({'success': True, 'messages_list': messages_list})


def update_record_in_history_db(
        chat_id: str,
        stream_session_id: str,
        user_query: str,
        user_query_html: str,
        llm_response: str
    ) -> tuple[datetime.datetime, str]:
    '''
    Updates a record in the chat history DB for the given stream_session_id.
    Also, determines the sequence ID of the updated record and deletes any subsequent chat history for the given chat_id.
    This is because the chat template is updated with each new message, and the subsequent chat history is no longer relevant.

    Returns:
        formatted_datetime: str
        chat_id: int
    '''

    print(f"Updating LLM response in chat history DB for chat_id: {chat_id} and stream_session_id: {stream_session_id}")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json in method update-llm_response_in_history_db. Error: ", e)

    # Connect to chat_history.db to determine appropriate chat_id
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        handle_local_error("Could not connect to chat history database, encountered error: ", e)

    try:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime('%d %b %Y - %I:%M %p %Z')
    except Exception as e:
        handle_local_error("Could not obtain timestamp in update-llm_response_in_history_db, encountered error: ", e)
    
    # Update the LLM response in the chat history DB for the given stream_session_id:
    try:
        c.execute("UPDATE chat_history SET user_query = ?, user_query_html = ?, llm_response = ?, date_time = ? WHERE stream_session_id = ?", (user_query, user_query_html, llm_response, formatted_datetime, stream_session_id))
        conn.commit()
    except Exception as e:
        handle_local_error("Could not update LLM response in chat history DB, encountered error: ", e)
    
    try:
        c.execute("SELECT sequence_id FROM chat_history WHERE chat_id = ? AND stream_session_id = ?", (chat_id, stream_session_id))
        result = c.fetchone()
        sequence_id = result[0]
        delete_chat_history_for_chat_id_from_sequence_id(c, conn, chat_id, sequence_id)
    except Exception as e:
        handle_local_error("Could not determine sequence_id / delete chat history in update-llm_response_in_history_db, encountered error: ", e)
    
    conn.close()

    return formatted_datetime, chat_id


def get_formatted_prompt_from_history_db(chat_id, sequence_id):

    print(f"\n\nFormatting prompt from history for chat with chat_id: {chat_id} and sequence_id: {sequence_id}\n\n")

    formatted_prompt = ""

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_error_no_return("Missing keys in config.json for method get-formatted_prompt_from_history_db(). Error: ", e)

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


def get_hf_waitress_formatted_user_prompt(messages_object: str, llm_response: str) -> str:
    messages_dict = json.loads(messages_object)
    new_response = {"role":"assistant", "content":llm_response}
    messages_dict['messages'].append(new_response)
    updated_messages_object = json.dumps(messages_dict, indent=4)
    return str(updated_messages_object)


def get_session_id_and_vector_key(reusable_ssid:str = None) -> tuple[str, str]:
    '''
    # Generates a unique session ID using universally Unique Identifier via the uuid4() method, wherein the randomness of the result is dependent on the randomness of the underlying operating system's random number generator
    # UUI is a standard used for creating unique strings that have a very high likelihood of being unique across all time and space, for ex: f47ac10b-58cc-4372-a567-0e02b2c3d479
    '''
    try:
        stream_session_id = str(uuid.uuid4()) if reusable_ssid is None else reusable_ssid
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        return stream_session_id, key_for_vector_results
    except Exception as e:
        handle_local_error("Could not generate stream_session_id with UUID4, encountered error: ", e)


def read_config_for_llm_response_setup() -> dict:
    try:
        return read_config([
            'local_llm_server',
            'selected_embedding_model',
            'force_enable_rag', 
            'force_disable_rag', 
            'base_template',
            'fetch_top_k_results_from_vectordb', 
            'filter_top_k_results_by_reranking', 
            'skip_system_prompt',
            'enable_butler_mode_selection'
        ])
    except Exception as e:
        handle_local_error("Could not read config for setup-for_local_llm_response, encountered error: ", e)


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
        handle_error_no_return("Could not determine if flux_diffusers or vision model in method read-hf_waitress_multimodal_config, encountered error: ", e)
        return False, False


def prepare_quick_response_for_special_model(full_prompt:str, user_query:str, current_sequence_id:int, new_sequence_id:int, stream_session_id:str, local_llm_server:str) -> dict:
    print("\n\nPreparing special model response\n\n")
    try:
        is_diffusers = local_llm_server == 'hfw-diffusers'
        messages_dict = prompt_formatting_module.prepare_prompt_for_auto_templating(
            formatted_prompt="" if is_diffusers else full_prompt, 
            user_query=user_query, 
            current_sequence_id=0 if is_diffusers else current_sequence_id, 
            system_prompt="",  
            skip_system_prompt=True
        )
        return {
            "success": True, 
            "stream_session_id": stream_session_id,
            "llm_set_rag_config": DISABLED_CONFIG,
            "formatted_user_prompt": json.dumps(messages_dict), 
            "sequence_id":new_sequence_id, 
            "server_type":local_llm_server
        }
    except Exception as e:
        handle_local_error("Could not prepare special model response in method prepare-special_model_response, encountered error: ", e)


def reject_rag() -> dict:
    try:
        write_config({'do_rag':False, 'perform_graph_rag':False})
        return {"success": True}
    except Exception as e:
        handle_error_no_return("Could not default do_rag to False in method reject-rag, encountered error: ", e)
        return {"success": False}


def prepare_for_quick_response(current_sequence_id:int, regeneration_request:bool) -> int:
    print("Invoking quick-return route for hfw-diffusers or hfw-vision(file_attached)")
    if not regeneration_request or current_sequence_id == 0:
        # Increment the seqID only for regular requests (as seqID stays the same when regenerating a response), or if it's currently 0.
        current_sequence_id = int(current_sequence_id) + 1
    reject_rag()
    return current_sequence_id


def get_full_llm_prompt_with_history(chat_id:int, current_sequence_id:int) -> str:
    try:
        print("\n\nGetting formatted prompt for setup-for_local_llm_response\n\n")
        formatted_prompt = ""
        if current_sequence_id > 0:    # get the last prompt so we can continue the completions
            formatted_prompt = get_formatted_prompt_from_history_db(chat_id, current_sequence_id)
        return formatted_prompt
    except Exception as e:
        handle_local_error("Could not get formatted prompt in method get-full_llm_prompt_with_history, encountered error: ", e)


def read_request_data_for_setup_response(request: Request) -> tuple[str, str, int, int, bool, bool, bool, bool]:
    try:
        stream_session_id = request.json.get('stream_session_id')   # stream_session_id = None if not regeneration_request
        user_query = request.json.get('user_query')
        chat_id = int(request.json.get('chat_id'))
        raw_sequence_id = request.json.get('sequence_id')   # since seq_id is only sent by the client at this stage for regen requests, and will be null for regular requests, we need to handle both cases.
        sequence_id = int(raw_sequence_id) if raw_sequence_id is not None else 0
        file_attached = request.json.get('file_attached')
        regeneration_request = request.json.get('regeneration_request')
        regenerate_with_citations_force_enabled = request.json.get('regenerate_with_citations_force_enabled')
        regenerate_with_citations_force_disabled = request.json.get('regenerate_with_citations_force_disabled')
        return stream_session_id, user_query, chat_id, sequence_id, file_attached, regeneration_request, regenerate_with_citations_force_enabled, regenerate_with_citations_force_disabled
    except Exception as e:
        handle_local_error("Could not read request data for query-setup API, encountered error: ", e)


def read_request_data_for_tools_response(request: Request) -> tuple[str, str, int, dict, bool, int]:
    try:
        stream_session_id = request.json.get('stream_session_id')   # stream_session_id = same as generated by determine service_and_ids_for_query in call 1
        user_query = request.json.get('user_query')
        chat_id = int(request.json.get('chat_id'))
        llm_set_rag_config = request.json.get('llm_set_rag_config')
        regeneration_request = request.json.get('regeneration_request')
        sequence_id = int(request.json.get('sequence_id'))  # at this stage, the client sends the seqs_id set in the previous request so no need to handle null case.
        return stream_session_id, user_query, chat_id, llm_set_rag_config, regeneration_request, sequence_id
    except Exception as e:
        handle_local_error("Could not read request data for tool-invocation API, encountered error: ", e)


def get_full_prompt_for_server(local_llm_server: str, full_prompt: str, user_query: str, current_sequence_id: int, base_template: str, skip_system_prompt: bool) -> str:
    if local_llm_server == 'llama-cpp':
        return json.dumps(prompt_formatting_module.prepare_prompt_for_auto_templating(full_prompt, user_query, current_sequence_id, base_template, skip_system_prompt))
    elif local_llm_server == 'hf-waitress':
        return json.dumps(prompt_formatting_module.prepare_prompt_for_auto_templating(full_prompt, user_query, current_sequence_id, base_template, skip_system_prompt))
    elif local_llm_server == 'hfw-vision':
        return json.dumps(prompt_formatting_module.prepare_prompt_for_auto_templating(full_prompt, user_query, current_sequence_id, "", True))  # No base_template for hfw-vision


def get_ids_for_llm_response_setup(stream_session_id:str, chat_id:int, sequence_id:int = 0, regeneration_request:bool = False, reuse_ssid:bool = False) -> tuple[str, str, int]:
    '''
    Determines and assigns the stream_session_id, key_for_vector_results and current_sequence_id keys for any query received from the client. 
    Every query has a unique stream_session_id.\n
    Handles two cases:
        - Regenration requests:
            - The client sends the stream_session_id and sequence_id of the message to regenerate.
            - The key_for_vector_results is set afresh basis the stream_session_id, while the stream_session_id and sequence_id themselves are returned unchanged.
        - Regular requests:
            - The client sends the chat_id.
            - The stream_session_id is generated afresh, and the key_for_vector_results is set subsequently.
            - The sequence_id is determined by obtaining the present max sequence_id for the given chat_id, and will be incremented by 1 at a later stage 
            (after the chat history is fetched and the prompt template is applied) before returning to the client.\n
    '''
    if regeneration_request:
        if stream_session_id is None or stream_session_id == "" or sequence_id == 0:
            return handle_local_error("Could not process regeneration-request in get-base_values_for_setup_for_local_llm_response, encountered error: ", ValueError("stream_session_id is blank or sequence_id is 0."))
        
        print(f"\nSetting defaults for regeneration for request ID: {stream_session_id}\n")
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        current_sequence_id = sequence_id  # return unchanged sequence_id for regeneration_request. 
        return stream_session_id, key_for_vector_results, current_sequence_id
    
    try:
        current_sequence_id = determine_sequence_id_for_chat(chat_id)  # sequence_id received in the request is ignored. NOTE: this will obtain the present max seqID, not the new max which is present_max + 1! This is so that the present chat history is fetched correctly in the next step.
        new_stream_session_id, key_for_vector_results = get_session_id_and_vector_key(stream_session_id if reuse_ssid else None)
        print(f"Current Chat ID: {chat_id} & Sequence ID: {current_sequence_id}")
        return new_stream_session_id, key_for_vector_results, current_sequence_id
    except Exception as e:
        handle_local_error("Error determining sequence_id and/or getting session_id and vector_key in get-base_values_for_setup_for_local_llm_response, encountered error: ", e)


def handle_special_model_case(local_llm_server:str, current_sequence_id:int, file_attached:bool, stream_session_id:str, user_query:str, full_prompt:str, regeneration_request:bool) -> tuple[str, Response]:
    print("\nChecking if special handling for multi-modal models is required...\n")
    if local_llm_server != 'hf-waitress':   #if llama.cpp
        return local_llm_server, None
    
    flux_diffusers, vision = read_hf_waitress_multimodal_config()

    if flux_diffusers or file_attached:
        new_sequence_id = prepare_for_quick_response(current_sequence_id, regeneration_request)
        new_local_llm_server='hfw-diffusers' if flux_diffusers else 'hfw-vision'    # 'hfw-vision' since a file is attached for visual analysis!
        try:
            response = prepare_quick_response_for_special_model(
                local_llm_server=new_local_llm_server,
                stream_session_id=stream_session_id,
                user_query=user_query,
                current_sequence_id=current_sequence_id,
                new_sequence_id=new_sequence_id,
                full_prompt=full_prompt
            )
            return new_local_llm_server, jsonify(response)
        except Exception as e:
            handle_local_error("Could not prepare special model response in method setup-for_local_llm_response, encountered error: ", e)
    
    if vision: 
        return 'hfw-vision', None
    
    return local_llm_server, None   # Likely hf-waitress, but not multi-modal (hfw-diffusers or hfw-vision)


def handle_force_disabled_rag(local_llm_server:str, full_prompt:str, user_query:str, current_sequence_id:int, stream_session_id:str, regeneration_request:bool, base_template:str, skip_system_prompt:bool) -> Response:
    print(f"\nForce disabling RAG for request ID {stream_session_id}\n")
    reject_rag()
    try:
        formatted_updated_prompt = get_full_prompt_for_server(local_llm_server, full_prompt, user_query, current_sequence_id, base_template, skip_system_prompt)
    except Exception as e:
        handle_local_error("Could not get formatted_updated_prompt in method setup_for_streaming_response, encountered error: ", e)
    if not regeneration_request or current_sequence_id == 0: current_sequence_id = int(current_sequence_id) + 1 # Increment the seqID only for regular requests (as seqID stays the same when regenerating a response), or if it's currently 0.
    return jsonify({"success": True, "stream_session_id": stream_session_id, "llm_set_rag_config": DISABLED_CONFIG, "formatted_user_prompt": formatted_updated_prompt, "sequence_id":current_sequence_id, "server_type":local_llm_server})


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
    'rag': {'do_rag': True, 'perform_graph_rag': False, 'butler_mode': False},
    'graphdb': {'do_rag': True, 'perform_graph_rag': True, 'butler_mode': False},
    'direct response': {'do_rag': False, 'perform_graph_rag': False, 'butler_mode': False},
    'home assistant': {'do_rag': False, 'perform_graph_rag': False, 'butler_mode': True}
}

DEFAULT_CONFIG = {'do_rag': True, 'perform_graph_rag': False, 'butler_mode': False}
DISABLED_CONFIG = {'do_rag': False, 'perform_graph_rag': False, 'butler_mode': False}


def determine_response_service(user_query:str, force_enable_rag:bool = False, enable_butler_mode_selection:bool = False) -> dict:
    """
    Determines which service should handle the user query. Accepts args to override the LLM's selection.
    
    Args:
        user_query: The user's input query
        force_enable_rag: If True, will force enable RAG and return the RAG config
        
    Returns:
        dict: Configuration for the selected service with 'do_rag' and 'perform_graph_rag' settings
    """
    print(f"\nDetermining response service...\n")

    try:
        service_request_prompt = prompt_formatting_module.get_service_request_prompt(user_query, enable_butler_mode_selection)
    except Exception as e:
        handle_local_error("Could not get service request prompt, encountered error: ", e)

    try:
        full_response = make_request_to_llm_server(service_request_prompt)
    except Exception as e:
        handle_local_error("Could not make service-determination request to LLM server, encountered error: ", e)
    
    try:
        selected_service = parse_service_response(full_response)
        print(f"\nLLM selected service: {selected_service}\n")  # Will be a dict with a single key: 'service' and a value as per those in get-service_request_prompt() above
    except Exception as e:
        handle_error_no_return("Could not parse service response, encountered error: ", e)
        selected_service = {'service': 'RAG'}

    try:
        service_name = selected_service.get('service', 'RAG').lower().strip()   # safe fallback to RAG, since this is a RAG app!
        print(f"\nService name: {service_name}\n")
        llm_set_rag_config = VALID_SERVICES.get(service_name, DEFAULT_CONFIG)   # return default-config if service-name is not in valid-services (both defined above)
        print(f"\nLLM Set Config: {llm_set_rag_config}\n")
    except Exception as e:
        handle_error_no_return("Could not determine service selected by the LLM, defaulting to use Naive RAG. Encountered error: ", e)
        llm_set_rag_config = DEFAULT_CONFIG
    
    if force_enable_rag:
        print("\n\nFORCE_ENABLE_RAG True, force enabling RAG and returning\n\n")
        llm_set_rag_config['do_rag'] = True   # We don't check this earlier as we also want to determine if GraphRAG should be performed! We only set it to True here to ensure force_enable_rag's do_rag setting takes precedence over the LLM's selection.
    
    safe_write_config(llm_set_rag_config)
    return llm_set_rag_config


@app.route('/get_tools_schema')
def get_tools_schema():
    '''
    This API is used to get the tools schema for the butler mode.
    '''
    try:
        _, full_tools_config = butler_module._full_read_butler_tools_config()
        tools_schema = butler_module.generate_tools_schema(full_tools_config.get('services', {}))
    except Exception as e:
        return handle_api_error("Error getting tools schema in method get-tools_schema, encountered error: ", e)
    return jsonify({"success": True, "tools_schema": tools_schema})


@app.route('/execute_tools', methods=['POST'])
def execute_tools():
    '''
    This API is used to execute the tools for the butler mode.
    '''
    try:
        tools_json = request.json
        stream_session_id = tools_json.get('stream_session_id', None)
        tool_results = butler_module.execute_tools(tools_json, stream_session_id)
        print(f"\n\nTool results: {tool_results}\n\n")
        return jsonify({"success": tool_results['success'], "tool_result_list": tool_results['tool_result_list'], "message": tool_results['message']})
    except Exception as e:
        return handle_api_error(f"Error executing tools in method execute-tools - {tool_results['message']}. Encountered error: ", e)


@app.route('/determine_service_and_ids_for_query', methods=['POST'])
def determine_service_and_ids_for_query():
    '''
    This is the first API invoked when a user submits a query via the Privion/LARS frontend.\n
    Main goal is to to determine the sequence_id and stream_session_id to be assigned to the query, handle any special cases and lastly, query the LLM 
    to determine the appropriate search-service/tools to best respond to the user.\n
    One of two downstream outcomes:
        - No further search-tools are needed, either because force_disable_rag is True, or there's a special case (vision/diffusers image-gen model) or of course, 
        if the LLM opts for a direct response. Or,
        - Further processing (RAG) is required, and the client must make another request to actually execute the search.

    We want to ensure that in the case of the former, the client has all the required information to proceed to the next step thus we have some redundancy.\n
    For instance in case another request is necessary, the sequence_id determination and chat-history loading / prompt template application must be redone by 
    invoke-tools_for_query().
    '''
    print("\n\nDetermining service and ids for query\n\n")

    try:    # Read config and request data, determine base values while handling regeneration case
        config = read_config_for_llm_response_setup()
        stream_session_id, user_query, chat_id, sequence_id, file_attached, regeneration_request, regenerate_with_citations_force_enabled, regenerate_with_citations_force_disabled = read_request_data_for_setup_response(request)
        stream_session_id, _, current_sequence_id = get_ids_for_llm_response_setup(stream_session_id, chat_id, sequence_id, regeneration_request)
    except Exception as e:
        return handle_api_error("Error getting base values for determine-service_and_ids_for_query, encountered error: ", e)

    try:    # Get full prompt including history from history-db
        full_prompt = get_full_llm_prompt_with_history(chat_id, current_sequence_id if not regeneration_request else current_sequence_id - 1) # if regeneration_request, then go back one sequence id!
        if full_prompt == "": current_sequence_id = 0   # An empty prompt implies regeneration of the first message in a chat, so we set the sequence_id to 0 for applying the prompt template from scratch. It'll be incremented before final return.
    except Exception as e:
        return handle_api_error("Could not get full prompt from history db in method determine-service_and_ids_for_query, encountered error: ", e)
    
    try:
        local_llm_server, special_response = handle_special_model_case(config['local_llm_server'], current_sequence_id, file_attached, stream_session_id, user_query, full_prompt, regeneration_request)
        if special_response is not None:    # If a special model response is returned, quick-return here
            return special_response
    except Exception as e:
        return handle_api_error("Error determining appropriate model type and server for determine-service_and_ids_for_query: ", e)
    
    if (config['force_disable_rag'] or regenerate_with_citations_force_disabled):
        return handle_force_disabled_rag(local_llm_server, full_prompt, user_query, current_sequence_id, stream_session_id, regeneration_request, config['base_template'], config['skip_system_prompt'])
    
    print(f"\n\nSpecial cases addressed, determining service and returning\n\n")

    try:
        llm_set_rag_config = determine_response_service(user_query, (config['force_enable_rag'] or regenerate_with_citations_force_enabled), config['enable_butler_mode_selection'])
    except Exception as e:
        handle_error_no_return("Could not determine response service, defaulting to use Naive RAG. Encountered error: ", e)
        safe_write_config(DEFAULT_CONFIG)
        llm_set_rag_config = DEFAULT_CONFIG

    try:    
        formatted_updated_prompt = None # In case RAG or other additional preparation is needed, the client must make another request to obtain the final query, so it won't be set here.
        if llm_set_rag_config == DISABLED_CONFIG:   # On the other hand, if no further preparation is needed, the client need not make another request to obtain the final query, so set final here.
            formatted_updated_prompt = get_full_prompt_for_server(local_llm_server, full_prompt, user_query, current_sequence_id, config['base_template'], config['skip_system_prompt'])  # Get full prompt for server
            # NOTE: seq_id should be incremented later (done below) as the prompt must be formatted accordingly if this is the first message in a chat! 
    except Exception as e:
        return handle_api_error("Could not get formatted_updated_prompt in method determine-service_and_ids_for_query, encountered error: ", e)

    if not regeneration_request or current_sequence_id == 0: current_sequence_id = current_sequence_id + 1 # Increment the seqID only for regular requests (as seqID stays the same when regenerating a response), or if it's currently 0.
    return jsonify({"success": True, "stream_session_id": stream_session_id, "llm_set_rag_config": llm_set_rag_config, "formatted_user_prompt":formatted_updated_prompt, "sequence_id": current_sequence_id, "server_type": local_llm_server})
    

@app.route('/invoke_tools_for_query', methods=['POST'])
def invoke_tools_for_query():
    '''
    This API is only to be invoked if in the previous step, the LLM determined that RAG/other tools are necessary.\n
    The client is basically saying, "Hey here's the user query and the RAG config that was deemed necessary, along with the stream_session_id and sequence_id.
    Please execute the search and return the full & final prompt."\n
    The sequence ID sent by the client is used in case this is a regeneration request, as the chat history need only be fetched upto that sequence ID, 
    and the prompt template applied accordingly.\n
    In case of a regular request, the sequence ID is redetermined anyways due to the way the get-ids_for_llm_response_setup() works, so the received value is ignored,
    which is not a problem. The client-sent value is only used for regeneration requests.\n
    The logic behind decrementing and incrementing the sequence-ID stays the same as in determine-service_and_ids_for_query():
        - Increment: By 1 for regular requests, or if it's currently 0 before returning to the client.
        - Decrement: Only in case of regeneration requests, to fetch the chat history upto the provided sequence ID non-inclusively.
    '''
    print("\n\nInvoking tools for query\n\n")

    try:
        config = read_config_for_llm_response_setup()
        stream_session_id, user_query, chat_id, llm_set_rag_config, regeneration_request, sequence_id = read_request_data_for_tools_response(request)
        stream_session_id, key_for_vector_results, current_sequence_id = get_ids_for_llm_response_setup(stream_session_id, chat_id, sequence_id, regeneration_request, reuse_ssid = True)
    except Exception as e:
        return handle_api_error("Error getting base values for invoke-tools_for_query, encountered error: ", e)
    
    try:    # Get full prompt including history from history-db
        full_prompt = get_full_llm_prompt_with_history(chat_id, current_sequence_id if not regeneration_request else current_sequence_id - 1)
        if full_prompt == "": current_sequence_id = 0   # An empty prompt implies regeneration of the first message in a chat, so we set the sequence_id to 0 for applying the prompt template from scratch. It'll be incremented before final return.
    except Exception as e:
        return handle_api_error("Could not get full prompt from history db in method invoke-tools_for_query, encountered error: ", e)

    try:
        if llm_set_rag_config.get('butler_mode', False):
            butler_response = butler_module.execute_butler_tasks(user_query)
            # print(f"Butler response: {butler_response}")
            user_query = butler_response['action_analysis_prompt']
    except Exception as e:
        return handle_api_error("Error handling butler mode in method invoke-tools_for_query, encountered error: ", e)
    
    # print("\n\nRAG Routine Begins: Performing semantic search on VectorDB, lexical search on Whoosh index, combining and reranking results and determining if RAG is necessary\n\n")
            
    try:
        docs, do_rag, graph_rag_context = rag_core_module.legacy_execute_search_tools_on_query(
            user_query, config['selected_embedding_model'], llm_set_rag_config, int(config['filter_top_k_results_by_reranking']), int(config['fetch_top_k_results_from_vectordb'])
        )
    except Exception as e:
        return handle_api_error("Could not execute RAG tools, encountered error: ", e)

    try:
        if do_rag and docs:    # Add RAG results to user-query if necessary! Check docstring in get-vector_results_for_get_references for more details.
            global QUERIES
            QUERIES[key_for_vector_results] = docs
            user_query += f"\n\nThe following context might be helpful in answering the user query above. If so, please reference useful documents by name and specific page numbers in your response:\n"
            user_query += f"{graph_rag_context}" if (graph_rag_context and len(graph_rag_context) > 0) else f"{docs}"
    except Exception as e:
        handle_error_no_return("Could not process RAG results, encountered error: ", e)   # No need to reject-rag as get-vector_results_for_get_references() handles any issues.
    
    try:    # Get full prompt for server - NOTE: seq_id should be incremented later as the prompt must be formatted accordingly if this is the first message in a chat!
        formatted_updated_prompt = get_full_prompt_for_server(config['local_llm_server'], full_prompt, user_query, current_sequence_id, config['base_template'], config['skip_system_prompt'])
    except Exception as e:
        return handle_api_error("Could not get formatted_updated_prompt in method invoke-tools_for_query, encountered error: ", e)

    if not regeneration_request or current_sequence_id == 0: current_sequence_id = current_sequence_id + 1
    return jsonify({"success": True, "reused_stream_session_id": stream_session_id, "tool_formatted_user_prompt": formatted_updated_prompt, "sequence_id":current_sequence_id, "reconfirmed_server_type":config['local_llm_server']})
    

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
        handle_local_error("Could not construct citation html, encountered error: ", e)


def save_pdf_to_download_dir(doc_name: str, stream_session_id: str) -> str:
    '''
    Save a PDF to the download directory.

    Args:
        - doc_name: str of the document name
        - stream_session_id: str of the stream session ID

    Returns:
        - str: The name of the saved PDF

    Raises:
        - Exception: If the PDF cannot be saved
    '''

    try:
        read_return = read_config(['upload_folder', 'highlighted_docs'])
    except Exception as e:
        handle_local_error("Missing upload_folder in config.json for method save-pdf_to_download_dir. Error: ", e)

    try:
        doc_name_without_extension = pathlib.Path(rf"{doc_name}").resolve().stem    # .stem() returns a string
        pdf_path = pathlib.Path(rf"{str(read_return['upload_folder'])}").resolve() / pathlib.Path(doc_name_without_extension).with_suffix(".pdf").name

        output_file_name = f"{doc_name_without_extension}_{stream_session_id}.pdf"
        output_pdf_path = pathlib.Path(rf"{str(read_return['highlighted_docs'])}").resolve() / output_file_name
        
        shutil.copy2(str(pdf_path), str(output_pdf_path)) # .copy2() preserves the file permissions and metadata, and is similar to `cp -p`, while .copy() is similar to `cp`

        return output_file_name
    except Exception as e:
        handle_local_error("Could not save PDF to download dir, encountered error: ", e)


def get_doc_name_and_page_number_from_url(url: str) -> tuple[str, str]:
    try:
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        doc_name = params.get('doc_name', [None])[0]
        page_number = params.get('page_number', [None])[0]
        return doc_name, page_number
    except Exception as e:
        handle_local_error("Could not get doc_name and page_number from url, encountered error: ", e)


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
                handle_error_no_return(f"Could not ast.literal_eval page_number string {page_number}, returning 1. Encountered error: ", e)
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
        handle_local_error("Could not obtain citation urls from llm_response, encountered error: ", e)


def remove_embedded_links_from_llm_response(llm_response: str) -> str:
    try:
        embedded_link_complete_pattern = r'\(*\[[^\]]+\]\(\s*\){1,2}\.?'  # Matches embedded links in the format: ([link text](url)) - For detailed explainer use: https://regex101.com/
        llm_response_without_embedded_links = re.sub(embedded_link_complete_pattern, '', llm_response)
        return llm_response_without_embedded_links
    except Exception as e:
        handle_local_error("Could not remove embedded links from llm_response, encountered error: ", e)


def add_citations_and_pdf_browser_to_llm_response(llm_response: str, stream_session_id: str) -> str:
    '''
    This function takes a LLM response, identifies all the documents linked within the response, and returns a new LLM response with citations and a PDF browser.
    '''
    print(f"\n\nAdding citations and PDF browser to LLM response\n\n")

    try:
        llm_response_without_embedded_links = remove_embedded_links_from_llm_response(llm_response)
    except Exception as e:
        handle_local_error("Could not remove embedded links from llm_response, encountered error: ", e)
    
    tab_count = 1
    tabs_created_for_doc = {}
    pdf_tab_buttons_set = set()
    pdf_tab_content_set = set()

    try:
        urls = obtain_citation_urls_from_llm_response(llm_response) # Extract all URLs from the LLM response
    except Exception as e:
        handle_local_error("Could not obtain citation urls from llm_response, encountered error: ", e)
    
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

            citation_html = f'''<a href="javascript:void(0)" class="citation-link" onclick="goToPageAndSwitchTab(\'{pdf_iframe_id}\', \'{frame_doc_path}\', \'tab{tab_name_string}\', \'{stream_session_id}\')">Reference: {doc_name}</a>'''
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
    final_llm_response = llm_response_without_embedded_links.strip()

    try:
        pdf_section_html = construct_citation_html(pdf_tab_buttons_set, pdf_tab_content_set, stream_session_id) if len(urls) > 0 else None
    except Exception as e:
        handle_error_no_return("Could not construct citation html, returning blank. Encountered error: ", e)
        pdf_section_html = None

    return final_llm_response, pdf_section_html


def prepare_model_response_for_storage_to_history_db(download_link_html: str, llm_response: str) -> str:
    model_response_for_history_db = str(llm_response)
    model_response_for_history_db += f"\n\npdf_pane_data={download_link_html}"
    model_response_for_history_db = model_response_for_history_db.strip('\n')
    return model_response_for_history_db


def flux_diffusers_is_enabled() -> bool:
    try:
        hf_read_return = read_hf_config(['flux_diffusers'])
        flux_diffusers = str(hf_read_return['flux_diffusers']).lower() == 'true'
        return flux_diffusers
    except Exception as e:
        return False


def get_vector_results_for_get_references(stream_session_id: str) -> tuple[list[Document], bool]:
    '''
    The do_rag key in config.json may have been reset by other requests, so by tracking a specific requests RAG results via the queries dict,
    we can be certain if RAG was perform for this query or not. While it's easier to simply store a boolean indicating this in the queries dict for a given 
    stream-session-id, we store the actual RAG results for a query in case it may be needed anywhere downstream, such as for storing seperately to the chat history DB, etc.
    '''
    try:
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        docs = QUERIES.pop(key_for_vector_results, None)
        return docs, docs is not None
    except Exception as e:
        handle_local_error("Could not get vector results for stream_session_id in method get-vector_results_for_get_references(), encountered error: ", e)


def get_request_parameters_for_get_references(request: Request) -> tuple[str, str, str, str, str, str, str, bool]:
    try:
        stream_session_id = request.json['stream_session_id']
        user_query = request.json['user_query']
        user_query_html = request.json['user_query_html']
        llm_response = request.json['llm_response']
        messages_object = request.json['formatted_user_prompt']
        chat_id = request.json['chat_id']
        sequence_id = request.json['sequence_id']
        regeneration_request = request.json['regeneration_request']
        return stream_session_id, user_query, user_query_html, llm_response, messages_object, chat_id, sequence_id, regeneration_request
    except Exception as e:
        handle_local_error("Could not read request content in method get-request_parameters_for_get_references(), encountered error: ", e)


def read_config_for_get_references() -> tuple[str, bool, bool]:
    try:
        read_return = read_config(['local_llm_server', 'perform_graph_rag', 'legacy_mode'])
        return read_return['local_llm_server'], read_return['perform_graph_rag'], read_return['legacy_mode']
    except Exception as e:
        handle_local_error("Could not read config.json in method read-config_for_get_references(), encountered error: ", e)


@app.route('/get_references', methods=['POST'])
def get_references():

    print("\n\nStoring History Post-Response -- Determining if Citations are Necessary\n\n")

    try:
        local_llm_server, perform_graph_rag, legacy_mode = read_config_for_get_references()
        do_rag = False
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to get-references. Error: ", e)

    try:
        stream_session_id, user_query, user_query_html, llm_response, messages_object, chat_id, sequence_id, regeneration_request = get_request_parameters_for_get_references(request)
    except Exception as e:
        return handle_api_error("Could not read request content in method get-references, encountered error: ", e)
    
    if local_llm_server == 'llama-cpp':
        local_llm_chat_template_format = "llama-cpp-jinja"
    elif local_llm_server == 'hf-waitress':
        local_llm_chat_template_format = "Transformers-AutoTokenizer"

    if legacy_mode:
        try:
            _, do_rag = get_vector_results_for_get_references(stream_session_id)    # check docstring in function call below for details
        except Exception as e:
            handle_error_no_return("Error determining if RAG was used in method get-references - Could not check the queries dict. Proceeding without RAG. Encountered error: ", e)
    
        if perform_graph_rag:
            try:
                last_context_index = messages_object.rindex("The following context might be helpful in answering the user query above.")
                graph_context_trimmed_prompt = messages_object[:last_context_index]
                messages_object = graph_context_trimmed_prompt + '"}]}'   # proper closing of the JSON array
            except Exception as e:
                handle_error_no_return("Trimming RAG context unnecessary, skipping. Encountered error: ", e)
        
        messages_object = get_hf_waitress_formatted_user_prompt(messages_object, llm_response)
    else:
        _, do_rag = rag_core_module.fetch_rag_context(stream_session_id, persist_in_db=False)
        messages_object = json.dumps(messages_object, indent=4)

    if not do_rag and not 'http://llm-citations-database.net' in llm_response:
        print("\n\nRAG Citations unnecessary, storing chat history and returning\n\n")
        try:
            if not regeneration_request:
                stored_datetime, chat_id = store_to_chat_history_db(chat_id, sequence_id, stream_session_id, user_query, user_query_html, llm_response, messages_object, local_llm_server, local_llm_chat_template_format)
            else:
                stored_datetime, chat_id = update_record_in_history_db(chat_id, stream_session_id, user_query, user_query_html, llm_response)
        except Exception as e:
            handle_error_no_return("Could not store or update chat history DB in get-references(), encountered error: ", e)
        
        return jsonify({'success': True, 'response': llm_response, 'pdf_frame': None, 'stored_datetime':stored_datetime, 'local_llm_server':local_llm_server, 'local_llm_chat_template_format':local_llm_chat_template_format, 'chat_id':chat_id})

    print("\n\nFetching Citations\n\n")

    try:
        llm_response_with_citation_links, pdf_section_html = add_citations_and_pdf_browser_to_llm_response(llm_response, stream_session_id)
    except Exception as e:
        pdf_section_html = None
        llm_response_with_citation_links = llm_response
        handle_error_no_return("Could not add citations and pdf browser to llm_response, encountered error: ", e)
    
    try:
        if pdf_section_html:
            model_response_for_history_db = prepare_model_response_for_storage_to_history_db(pdf_section_html, llm_response_with_citation_links)
        else:
            model_response_for_history_db = llm_response_with_citation_links
    except Exception as e:
        handle_error_no_return("Could not prep data to store to chat history DB in get-references(), encountered error: ", e)

    try:
        if not regeneration_request:
            stored_datetime, chat_id = store_to_chat_history_db(chat_id, sequence_id, stream_session_id, user_query, user_query_html, model_response_for_history_db, messages_object, local_llm_server, local_llm_chat_template_format)
        else:
            stored_datetime, chat_id = update_record_in_history_db(chat_id, stream_session_id, user_query, user_query_html, model_response_for_history_db)
    except Exception as e:
        handle_error_no_return("Could not store or update chat history DB in get-references(), encountered error: ", e)

    return jsonify({'success': True, 'response': llm_response_with_citation_links, 'pdf_frame':pdf_section_html, 'stored_datetime':stored_datetime, 'local_llm_server':local_llm_server, 'local_llm_chat_template_format':local_llm_chat_template_format, 'chat_id':chat_id})


def parse_arguments() -> argparse.ArgumentParser:

    try:
        parser = argparse.ArgumentParser(description="Server for HuggingFace Transformers models")

        # Even if a parser object could not be created, a read_request will write & return defaults
        read_return = read_config(
            [
                'lars_host',
                'lars_port',
                'hf_waitress_access_url',
                'hf_waitress_serving_url',
                'hf_waitress_server_port',
                'llama_cpp_access_url',
                'llama_cpp_serving_url', 
                'llama_cpp_server_port'
            ]
        )

        if parser:
            parser.add_argument(
                "--reset_to_defaults", action="store_true", default=False,
                help="Erase all custom settings and reset to defaults."
            )
            
            parser.add_argument(
                "--lars_host", type=str, default=read_return['lars_host'],
                help="Specify the host for Privion. Remembers previously set value. Default: 0.0.0.0"
            )
            
            parser.add_argument(
                "--lars_port", type=int, default=read_return['lars_port'],
                help="Specify the port for Privion. Remembers previously set value. Default: 5000"
            )

            parser.add_argument(
                "--hf_waitress_access_url", type=str, default=read_return['hf_waitress_access_url'], 
                help="Specify the access URL for HF-Waitress. Remembers previously set value. Default: localhost"
            )

            parser.add_argument(
                "--hf_waitress_serving_url", type=str, default=read_return['hf_waitress_serving_url'], 
                help="Specify the serving URL for HF-Waitress. Remembers previously set value. Default: 0.0.0.0"
            )

            parser.add_argument(
                "--hf_waitress_server_port", type=int, default=read_return['hf_waitress_server_port'], 
                help="Specify the port for HF-Waitress. Remembers previously set value. Default: 9069"
            )

            parser.add_argument(
                "--llama_cpp_access_url", type=str, default=read_return['llama_cpp_access_url'],
                help="Specify the access URL for Llama-CPP. Remembers previously set value. Default: localhost"
            )

            parser.add_argument(
                "--llama_cpp_serving_url", type=str, default=read_return['llama_cpp_serving_url'],
                help="Specify the serving URL for Llama-CPP. Remembers previously set value. Default: 0.0.0.0"
            )

            parser.add_argument(
                "--llama_cpp_server_port", type=int, default=read_return['llama_cpp_server_port'],
                help="Specify the port for Llama-CPP. Remembers previously set value. Default: 8080"
            )

            args = parser.parse_args()
            # print(f"\n\nparser.parse_args():\n\n{args}\n\n")

            if args.reset_to_defaults:
                print("\n\nLoading Server with Safe Defaults\n\n")
                
                with open(CONFIG_PATH, 'w') as file:
                    json.dump({}, file, indent=4)   # Empty config.json
                
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

            else:
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

            return args

        return None

    except Exception as e:
        handle_local_error("Could not parse arguments, encountered error: ", e)


def get_host_and_port() -> tuple[str, int]:
    try:
        read_return = read_config(['lars_host', 'lars_port'])
        return read_return['lars_host'], read_return['lars_port']
    except Exception as e:
        handle_error_no_return("Could not get host and port from config.json, encountered error: ", e)


def signal_handler(sig, frame):
    '''
    Signal handler for the main process.
    Shuts down the server gracefully by intercepting the interrupt "SIGINT" signal (Ctrl+C).
    No waiting on semaphores here: In case app is hung, waiting here will cause the shutdown to hang!

    Hard exit:
    os._exit(0) is used instead of sys.exit(0) because it skips Python's cleanup handlers (except 
    finally blocks) and immediately terminates the process.
    
    Args:
    - sig: Integer representing the specific signal that triggered the handler. Eg: CTRL+C -> 2
    - frame: Represents the "current execution point" of the code at the exact millisecond of the signal. 
        - Primarily used for debugging.
        - Eg: line 452 inside function `generate_tokens`
    '''
    
    print("\n\n⚠️  CTRL+C Detected! Force stopping workers...\n")
    print("👋 Exiting immediately.")
    os._exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    _ = parse_arguments()
    lars_host, lars_port = get_host_and_port()
    print(f"\n\nServing LARS-Enterprise on {lars_host} port {lars_port}\n\n")
    # app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000)
    max_upload_size = 100 * 1024 * 1024 * 1024  # 100GB upload limit
    serve(app, host=lars_host, port=lars_port, max_request_body_size=max_upload_size)