############################------------configuration manager-------------###############################

import json
import os

def write_config(config_updates:dict, filename=None) -> dict:
    '''
    Method to write app configuration to config.json.\n
    
    Args:
        - config_updates: dict of key:values to be written to config.json
        - filename: name of the file to write to, defaults to None which sets to CONFIG_PATH

    Returns:
        - Confirmation of success: {success: True}

    Raises:
        - Exception: If the file cannot be written to
    '''

    if filename is None:    # look in storage_config.json for the config path
        bootstrap_path = os.path.join(os.getcwd(), 'storage_config.json')
        with open(bootstrap_path, 'r') as f:
            bootstrap_config = json.load(f)
        filename = bootstrap_config['config_path']
    
    # First, open existing config file (if present) to read-in current settings, fallback to an empty dict if file does not exist:
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
    except Exception as e:
        config = {}     #init emply config dict
        print("Could not read config.json when attempting to write, encountered error: ", e)
        
    restart_required = False
    skip_reload_trigger = False
    llm_trigger_keys_for_app_restart = [
        'local_llm_server',
        'use_local_llm',
        'use_azure_open_ai',
        'model_choice',
        'llama_cpp_context_length',
        'llama_cpp_batch_size',
        'llama_cpp_ubatch_size',
        'llama_cpp_max_new_tokens',
        'llama_cpp_use_gpu',
        'llama_cpp_gpu_layers',
        'llama_cpp_unified_kv_buffer',
        'llama_cpp_disable_kv_offloading',
        'llama_cpp_key_cache_data_type',
        'llama_cpp_value_cache_data_type',
        'llama_cpp_no_of_seqs_to_par_decode',
        'llama_cpp_offload_to_devices',
        'llama_cpp_cpu_only_moe',
        'llama_cpp_num_cpu_moe',
        'llama_cpp_mlock',
        'llama_cpp_no_nmap',
        'base_template',
        'skip_system_prompt',
        'hf_waitress_serving_url',
        'hf_waitress_access_url',
        'hf_waitress_server_port'
    ]
            
    for key in llm_trigger_keys_for_app_restart:
        if key in config_updates and config_updates[key] != config.get(key):
            if key == 'local_llm_server' or key == 'base_template':
                restart_required = True # we want the page to refresh but don't set the llama.cpp reload trigger just yet as a server restart is unnecessary if no other settings have changed.
                skip_reload_trigger = True
            else:
                restart_required = True
                skip_reload_trigger = False
                break
    
    reload_tts = False
    triggers_for_tts_reload = [
        'kokoro_language_code'
    ]

    for key in triggers_for_tts_reload:
        if key in config_updates and config_updates[key] != config.get(key):
            reload_tts = True
            break

    reload_asr = False
    triggers_for_asr_reload = [
        'asr_model',
        'asr_torch_device'
    ]

    for key in triggers_for_asr_reload:
        if key in config_updates and config_updates[key] != config.get(key):
            reload_asr = True
            break

    # Handle any special cases here:
    if not reload_asr:  # 1. If the relaod_asr trigger hasn't already been set...
        if config_updates.get('enable_asr') and not config.get('enable_asr'):   #... and only if ASR is being switched on from an off state...
            reload_asr = True   # ...then set the reload trigger as asr-server_starter_endpoint will handle the rest!

    config.update(config_updates)

    # Write updated config.json:
    try:
        with open(filename, 'w') as file:
            json.dump(config, file, indent=4)
    except Exception as e:
        raise Exception(f"Could not update config.json, encountered error: {e}")
     
    return {'success': True, 'restart_required':restart_required, 'skip_reload_trigger':skip_reload_trigger, 'reload_tts':reload_tts, 'reload_asr':reload_asr}


def safe_write_config(config_updates:dict, filename=None) -> dict:
    '''
    Wrapper for write-config() that handles errors silently.
    Directly invoke write-config() instead of this method anytime a write-specific error must be raised!
    '''
    try:
        return write_config(config_updates, filename)
    except Exception as e:
        print("Could not write to config.json, encountered error: ", e)
        return {'success': False, 'restart_required': False}


def read_config(keys:list, default_value=None, filename=None) -> dict:
    '''
    Method to read app configuration from config.json. Central method to configure safe application defaults.
    
    Args:
        - keys: list of keys to read from config.json
        - default_value: default value to return if a key is not found in config.json, defaults to None
        - filename: name of the file to read from, defaults to None which sets to CONFIG_PATH

    Returns:
        - dict of key:values read from config.json

    Raises:
        - KeyError: If a key is not found in config.json and no default value has been defined
    '''

    bootstrap_path = os.path.join(os.getcwd(), 'storage_config.json')   # force-reading as BASE_DIR key is required as a fallback below!
    with open(bootstrap_path, 'r') as f:
        bootstrap_config = json.load(f)
    BASE_DIRECTORY = bootstrap_config['base_directory']
    CONFIG_PATH = bootstrap_config['config_path']

    filename = filename or CONFIG_PATH
    
    # Open config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
    except Exception as e:
        print("Could not read config.json, encountered error: ", e)
        return {key: default_value for key in keys}     #because a read scenario wherein config.json does not exist shouldn't occur!
    
    return_dict = {}
    update_config_dict = {}
    base_directory = config.get('base_directory', BASE_DIRECTORY)   # base_directory is written to config after platform detection and the correct value will be present by the time other app directories are requested 

    for key in keys:
        if key in config:
            return_dict[key] = config[key]
        else:
            default_value = {
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
                'exclusive_server_mode':True,  # If True, only one main LLM server instance will be allowed to run at a time. For example, when launching llama.cpp, HF-Waitress will be shut down.
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
                'butler_mode':False,
                'enable_butler_mode_selection':False,
                'force_enable_rag':False,
                'force_disable_rag':False,
                'use_local_llm':True,
                'use_gpu_for_embeddings':False,
                'azure_cv_free_tier':True,
                'use_azure_open_ai':False,
                'azure_openai_api_type':'azure',
                'azure_openai_api_version':'2023-05-15',
                'azure_openai_max_tokens':4096,
                'azure_openai_temperature':0.7,
                'force_re_extract':False,
                'llm_filter_citations':True,
                'local_llm_model_type':'llama',
                'local_llm_chat_template_format':'Tranformers-AutoTokenizer',
                'llama_cpp_use_gpu':False,
                'llama_cpp_batch_size': 2048,
                'llama_cpp_ubatch_size': 512,
                'llama_cpp_context_length':4096,
                'llama_cpp_max_new_tokens':-1,
                'llama_cpp_gpu_layers':25,
                'llama_cpp_unified_kv_buffer':False,
                'llama_cpp_disable_kv_offloading':False,
                'llama_cpp_key_cache_data_type':'f16',
                'llama_cpp_value_cache_data_type':'f16',
                'llama_cpp_no_of_seqs_to_par_decode':1,
                'llama_cpp_offload_to_devices':'none',
                'llama_cpp_cpu_only_moe':False,
                'llama_cpp_num_cpu_moe': 0,
                'llama_cpp_mlock':False,
                'llama_cpp_no_nmap':False,
                'llama_cpp_temperature':0.8,
                'llama_cpp_top_k':40,
                'llama_cpp_top_p':0.9,
                'llama_cpp_min_p':0.1,
                'llama_cpp_n_keep':0,
                'llama_cpp_server_timeout_seconds':3,
                'llama_cpp_server_retry_attempts':200,
                'hf_waitress_server_timeout_seconds':3,
                'hf_waitress_server_retry_attempts':200,
                'enable_asr':False,
                'asr_wake_word':'simon',
                'asr_model':'openai/whisper-large-v3',
                'asr_torch_device':'cpu',
                'asr_temperature':0.0,
                'asr_max_new_tokens':1500,
                'asr_samplerate':16000,
                'asr_volume_threshold':0.04,
                'asr_silence_duration_s':1.5,
                'asr_min_chunk_duration_s':0.25,
                'asr_min_context_s':11,
                'asr_stale_buffer_timeout_s':20.0,
                'asr_min_meaningful_samples_factor':1.5,
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
                'asr_waitress_serving_url':'0.0.0.0',
                'asr_waitress_access_url':'localhost',
                'asr_waitress_server_port':10087,
                'voice_base_directory_name': 'voice-model-servers',
                'asr_subdirectory_name': 'asr-model-server',
                'enable_tts': False,
                'selected_tts': 'kokoro_82m',
                'selected_kokoro_voice': 'af_heart',
                'kokoro_language_code': 'a',
                'tts_sample_rate': 24000,
                'whoosh_search_weighting':'BM25F',
                'fetch_top_k_results_from_whoosh':50,
                'fetch_top_k_results_from_vectordb':50,
                'filter_top_k_results_by_reranking':11,
                'min_semantic_similarity_threshold':0.5,
                'min_lexical_similarity_threshold':3.0,
                'chunk_size':250,
                'chunk_overlap':0,
                'enable_graph_rag':True,
                'perform_graph_rag':True,   # Determined & managed by the LLM
                'perform_only_graph_rag':False, # dev flag only for testing
                'upload_doc_to_graph_db':True,
                'graph_chunk_size':1500,    # Larger chunks are better because they provide more context for the model to identify meaningful entities and relationships
                'graph_chunk_overlap':300,  # 20% overlap for a 1500 char chunk: provides very reasonable overlap while not being too redundant.
                'graph_generator_model_list':[
                    'Metin/Gemma-2-2B-TR-Knowledge-Graph',
                    'google/gemma-2-2b-it',
                    'google/gemma-2-9b-it'
                ],
                'graph_generator_model':'Metin/Gemma-2-2B-TR-Knowledge-Graph',
                'graph_model_server_port':9070,
                'graph_model_access_url':'localhost',
                'quantize_graph_model':'n',
                'quantize_graph_model_bits':'int8',
                'exl2_quantize_graph_model':True,
                'exl2_quantize_graph_model_bpw':8.0,
                'graph_summarizer_model_list':[
                    "google/gemma-3-4b-it",
                    "microsoft/Phi-4-mini-instruct",
                    "google/gemma-3-1b-it",
                    "google/gemma-3-27b-it",
                    "Qwen/Qwen3-14B",
                    "Qwen/Qwen3-30B-A3B",
                    "Qwen/Qwen3-14B",
                    "Qwen/Qwen3-0.6B",
                    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
                    "Qwen/Qwen3-32B",
                    "Qwen/QwQ-32B",
                    "mistralai/Mistral-Small-24B-Instruct-2501",
                    "microsoft/phi-4",
                    "meta-llama/Llama-3.2-11B-Vision-Instruct",
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "black-forest-labs/FLUX.1-schnell",
                    "black-forest-labs/FLUX.1-dev",
                    "mistralai/Mistral-Nemo-Instruct-2407",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-70B-Instruct",
                    "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
                    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                    "microsoft/Phi-3.5-mini-instruct",
                    "microsoft/Phi-3.5-MoE-instruct",
                    "microsoft/Phi-3-mini-4k-instruct",
                    "microsoft/Phi-3-mini-128k-instruct",
                    "microsoft/Phi-3-small-8k-instruct",
                    "microsoft/Phi-3-small-128k-instruct",
                    "microsoft/Phi-3-medium-4k-instruct",
                    "microsoft/Phi-3-medium-128k-instruct",
                    "CohereForAI/c4ai-command-r-plus",
                    "CohereForAI/c4ai-command-r-v01",
                    "google/gemma-2-2b-it",
                    "google/gemma-2-9b-it",
                    "google/gemma-2-27b-it",
                    "Qwen/Qwen2-7B-Instruct",
                    "Qwen/Qwen2-72B-Instruct",
                    "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/QwQ-32B-Preview",
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen2.5-3B-Instruct",
                    "Qwen/Qwen2.5-14B-Instruct",
                    "deepseek-ai/DeepSeek-R1",
                    "open-thoughts/OpenThinker-32B"
                ],
                'graph_summarizer_model':'google/gemma-2-2b-it',
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
                'skip_summary_generation':False,    # dev flag only for testing
                'reuse_graph_extraction_cache_without_validation':False,
                'reuse_graph_summary_cache_without_validation':False,
                'reuse_graph_extraction_cache_with_validation':True,
                'reuse_graph_summary_cache_with_validation':True,
                'graph_rag_context_length_limit_chars':25000,
                'base_template': (
                            "You are a helpful assistant deployed in a Retrieval Augmented Generation (RAG) system.\n"
                            "Please link to source citations after every significant point and wherever else applicable in your response, by providing the complete 'source_link' link.\n"
                            "Thank you!\n"                            
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
                    'Qwen/Qwen3-Embedding-0.6B',
                    'Qwen/Qwen3-Embedding-4B',
                    'Qwen/Qwen3-Embedding-8B',
                    'BAAI/bge-small-en-v1.5',
                    'BAAI/bge-base-en-v1.5',
                    'BAAI/bge-large-en-v1.5',
                    'nvidia/NV-Embed-v2'
                ],
                'selected_embedding_model':'sentence-transformers/all-mpnet-base-v2',
                'reranker_models_list':[
                    'all-MiniLM-L6-v2',
                    'Qwen/Qwen3-Reranker-0.6B',
                    'Qwen/Qwen3-Reranker-4B',
                    'Qwen/Qwen3-Reranker-8B',
                    'BAAI/bge-small-en-v1.5',
                    'BAAI/bge-base-en-v1.5',
                    'BAAI/bge-large-en-v1.5'
                ],
                'selected_reranker_model':'all-MiniLM-L6-v2',
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

    if update_config_dict: safe_write_config(update_config_dict)   # write defaults to config.json
    
    return return_dict


def read_hf_config(keys:list, default_value=None, filename=None) -> dict:
    
    if filename is None:    # look in waitress_storage_config.json for the config path
        hf_bootstrap_path = os.path.join(os.getcwd(), 'waitress_storage_config.json')
        with open(hf_bootstrap_path, 'r') as f:
            hf_storage_config = json.load(f)
        filename = hf_storage_config.get('config_path')
    
    # Open hf_config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            hf_config = json.load(file)
    except Exception as e:
        print("Could not read hf_config.json, encountered error: ", e)
        return {key: default_value for key in keys}     #because a read scenario wherein hf_config.json does not exist shouldn't occur!

    return_dict = {}

    for key in keys:
        if key in hf_config:
            return_dict[key] = hf_config[key]
        else:
            return_dict[key] = default_value

    return return_dict

############################----------------------------------------------###############################