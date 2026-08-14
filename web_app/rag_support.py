import privion_config_concierge as config_manager
import utils as utils_module

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from falkordb import FalkorDB

import subprocess
import platform
import requests
import textwrap
import pathlib
import json
import time
import re
import os



def sanitize_names(name:str) -> str:
    name_str = str(name).lower()
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name_str)
    '''
    Matches all non-alphanumeric characters disallowed by 
    the OpenCypher spec and replaces them with an underscore.
    '''
    if sanitized[0].isdigit():
        sanitized = 'n_' + sanitized
        '''
        OpenCypher spec disallows digits at the beginning of a node name, 
        even if they're strings eg "2025"!
        '''
    return sanitized


def get_path_to_knowledge_domain() -> pathlib.Path:
    '''Returns the Pathlib.Path to the knowledge domain.'''
    
    try:
        read_return = config_manager.read_config([
            'selected_knowledge_domain',
            'knowledge_domain_base_directory'
        ])
    except Exception as e:
        print(f"Missing config for knowledge domain path. Error: {e}")

    try:
        path_to_knowledge_domain = pathlib.Path(
            (
                rf"{str(read_return['knowledge_domain_base_directory'])}"
            ).resolve() / str(
                read_return['selected_knowledge_domain']
            )
        )

        if not path_to_knowledge_domain.exists():
            path_to_knowledge_domain.mkdir(parents=True, exist_ok=True)
            print(
                "\n\nCreated knowledge domain directory: "
                f"{path_to_knowledge_domain}\n\n"
            )
        
        return path_to_knowledge_domain
    except Exception as e:
        print(f"Could not create knowledge domain folder, error: {e}")


def determine_whoosh_index_folder() -> pathlib.Path:
    '''Returns the Pathlib.Path to the Whoosh Index folder.'''

    try:
        path_to_knowledge_domain = get_path_to_knowledge_domain()
        config_data = config_manager.read_config(['selected_embedding_model'])
        selected_embedding_model = str(config_data['selected_embedding_model'])
    except Exception as e:
        raise Exception(f"Error determining selected embedding model: {e}")

    try:
        whoosh_index_folder = (
            path_to_knowledge_domain
            / "vector_db_and_whoosh_index"
            / selected_embedding_model
            / "whoosh_index"
        )
    except Exception as e:
        raise Exception(f"Error determining whoosh index folder: {e}")

    return whoosh_index_folder


def create_whoosh_index_in_folder(whoosh_index_folder:pathlib.Path):

    print(f"Creating Whoosh Index in folder: {whoosh_index_folder}")
    
     # Define the Index schema: what fields it contains
    schema = Schema(
        content=TEXT(stored=True),
        source_link=ID(stored=True),
        source=ID(stored=True),
        page_number=ID(stored=True),
        entities_and_relationships=ID(stored=True)
    )

    # Create a directory for persistent storage of the index to disk
    try:
        whoosh_index_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise Exception(f"Error creating directory for the Whoosh Index: {e}")
    # Create the index based on the schema definted above
    try:
        ix = create_in(str(whoosh_index_folder), schema)
    except Exception as e:
        raise Exception(f"Error creating Whoosh Index: {e}")

    return ix


def get_whoosh_index_object_for_folder(whoosh_index_folder:pathlib.Path):

    print(f"Getting Whoosh Index Object for folder: {whoosh_index_folder}")

    if not whoosh_index_folder.exists():
        try:
            ix = create_whoosh_index_in_folder(whoosh_index_folder)
        except Exception as e:
            raise Exception(f"Error creating Whoosh Index: {e}")
    else:
        try:
            ix = open_dir(str(whoosh_index_folder))
        except Exception as e:
            raise Exception(f"Error opening Whoosh Index: {e}")

    return ix


def create_vector_db_directory(
    path_to_knowledge_domain:pathlib.Path, 
    embedding_function:str
) -> pathlib.Path:
    '''
    Create the vector_db directory.

    Args:
        - path_to_knowledge_domain: pathlib.Path of the knowledge domain
        - embedding_function: str of the embedding function

    Returns:
        - pathlib.Path: To the vector_db directory

    Raises:
        - Exception: If the vector_db directory cannot be created
    '''

    try:
        vector_db_path = (
            path_to_knowledge_domain
            / "vector_db_and_whoosh_index"
            / embedding_function
        )
        if not vector_db_path.exists():
            vector_db_path.mkdir(parents=True, exist_ok=True)
            print(f"\n\nCreated vector_db directory: {vector_db_path}\n\n")
        else:
            print(
                "\n\nVector_db directory already exists, "
                f"returning path: {vector_db_path}\n\n"
            )
        return vector_db_path
    except Exception as e:
        print(f"Error creating vector_db directory: {e}")


def bring_graph_db_online():
    '''Launches the FalkorDB Docker container.'''
    
    print(f"\nLaunching FalkorDB Docker container...\n")
    try:
        config = config_manager.read_config([
            'launch_graph_db_with_ui',
            'assign_host_port_to_graph_db_server', 
            'assign_host_port_to_graph_db_ui',
            'graph_db_data_directory'
        ])
    except Exception as e:
        raise Exception(f"Config error for FalkorDB launch: {e}")

    try:    # Check if Docker Engine is online
        subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            check=True
        )  # check=True raises if the command returns a non-zero exit code
    except Exception as e:
        raise Exception(f"Docker Engine is not running: {e}")

    print("\nDocker Engine online, proceeding with FalkorDB container launch...\n")

    if utils_module.check_if_container_is_running('falkor-db'):
        print("\nFalkorDB container is already running, skipping...\n")
        return True

    command = [
        'docker', 'run',
        '-p', f"{config['assign_host_port_to_graph_db_server']}:6379",
        *(
            ['-p', f"{config['assign_host_port_to_graph_db_ui']}:3000"]
            if config['launch_graph_db_with_ui'] else []
        ),
        '--name', 'falkor-db',
        '-it', '--rm',
        '-v', f"{config['graph_db_data_directory']}:/var/lib/falkordb/data",
        'falkordb/falkordb:edge'
    ]   # Using conditional list-unpacking with * to handle optional args!

    try:
        if platform.system() == 'Windows':
            subprocess.Popen(
                command, 
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                command, 
                shell=True
            )
   
        # Check if the container is running
        container_name = 'falkor-db'
        timeout = 2
        attempts = 50
        for _ in range(attempts):
            if utils_module.check_if_container_is_running(container_name):
                print(f"\nFalkorDB container launched successfully!\n")
                return True
            else:
                print(
                    "\n\nFalkorDB container not yet running, "
                    f"waiting {timeout} seconds before retrying...\n\n"
                )
                time.sleep(timeout)

    except Exception as e:
        raise Exception(f"Could not launch FalkorDB container: {e}")

    return True


def get_graph_db_client():
    print("\nObtaining Graph DB Client\n")

    try:
        config = config_manager.read_config([
            'graph_db_server_host',
            'assign_host_port_to_graph_db_server'
        ])

        client = FalkorDB(
            host=config['graph_db_server_host'],
            port=config['assign_host_port_to_graph_db_server']
        )
        print(f"\nGraphDB Client obtained successfully!\n")
        return client
    except Exception as e:
        raise Exception(f"Could not get GraphDB Client: {e}")


def bring_perplexica_online():    # launch Perplexica Docker container
    print(f"\nLaunching Perplexica Docker container...\n")

    try:
        config = config_manager.read_config([
            'assign_host_port_to_perplexica_server',
            'perplexica_version'
        ])
        container_name = f"perplexica-{config['perplexica_version']}"
    except Exception as e:
        raise Exception(f"Config error for Perplexica launch: {e}")

    try:    # Check if Docker Engine is online
        subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            check=True
        )  # check=True raises if the command returns a non-zero exit code
    except Exception as e:
        raise Exception(f"Docker Engine is not online: {e}")

    # Check if container is running
    if utils_module.check_if_container_is_running(container_name):
        print("\nPerplexica container is already running, skipping...\n")
        return True

    # Check if container exists, but is stopped
    if utils_module.check_if_container_exists(container_name):
        print("\nPerplexica container exists but is stopped, restarting...\n")
        try:
            return utils_module.start_container(container_name)
        except Exception as e:
            raise Exception(f"Could not start Perplexica Docker container: {e}")

    # Container doesn't exist, create it
    print("\nDocker Engine online, proceeding with Perplexica container launch...\n")

    command = [
        'docker', 'run', '-d',
        '-p', f"{config['assign_host_port_to_perplexica_server']}:3000",
        '-v', 'perplexica-data:/home/perplexica/data',
        '--name', container_name,
        f"itzcrazykns1337/perplexica:{config['perplexica_version']}"
    ]

    try:
        if platform.system() == 'Windows':
            subprocess.Popen(
                command,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                command,
                shell=True
            )
        
        print(f"\nPerplexica container launched successfully!\n")
        return True
        
    except Exception as e:
        raise Exception(f"Could not launch Perplexica container: {e}")


def get_perplexica_providers():
    print("\nGetting Perplexica Providers\n")
    try:
        config = config_manager.read_config([
            'assign_host_port_to_perplexica_server'
        ])
        port = config['assign_host_port_to_perplexica_server']

        url = f"http://localhost:{port}/api/providers"
        payload = {}
        headers = {}

        response = requests.request("GET",url,headers=headers,data=payload)
        response.raise_for_status()
        full_response = response.json()

        provider_ids = {}
        for provider in full_response['providers']:
            if provider['name'] == 'Transformers':
                provider_ids['embeddingModel'] = provider['id']
            if provider['name'] == 'HF-Waitress':
                provider_ids['chatModel'] = provider['id']

        return provider_ids

    except Exception as e:
        raise Exception(f"Could not get Perplexica Providers: {e}")


def perplexica_search(query:str, history:list) -> str:
    print("\nPerforming Perplexica Search\n")

    try:
        config = config_manager.read_config([
            'assign_host_port_to_perplexica_server', 
            'perplexica_embedding_model', 'perplexica_optimization_mode', 
            'perplexica_sources', 'perplexica_include_history', 
            'perplexica_system_instructions'
        ])
        port = config['assign_host_port_to_perplexica_server']

        provider_ids = get_perplexica_providers()
        
        url = f'http://localhost:{port}/api/search'

        payload = json.dumps({
            "chatModel": {
                "providerId": provider_ids['chatModel'],
                "key": "exl3"
            },
            "embeddingModel": {
                "providerId": provider_ids['embeddingModel'],
                "key": "Xenova/all-MiniLM-L6-v2"
            },
            "optimizationMode": config['perplexica_optimization_mode'],
            "sources": config['perplexica_sources'],
            "query": query,
            "history": (
                history 
                if config['perplexica_include_history'] and history 
                else []
            ),
       
            "systemInstructions": config['perplexica_system_instructions'],
            "stream": False
        })
        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST",url,headers=headers,data=payload)

        print(response.text)

    except Exception as e:
        raise Exception(f"Could not perform Perplexica Search: {e}")


def setup_searxng_settings_yaml():
    try:
        config = config_manager.read_config(['searxng_dir'])

        # Ensure path is absolute for Docker volume mounting
        searxng_path = os.path.abspath(config['searxng_dir'])
        os.makedirs(searxng_path, exist_ok=True)

        file_path = os.path.join(searxng_path, 'settings.yml')

        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:

                settings_yaml = textwrap.dedent(f"""
                    use_default_settings: true

                    server:
                        # Internal container port (do not change this)
                        port: 8080
                        bind_address: "0.0.0.0"
                        secret_key: "a_very_long_random_string"

                        # CRITICAL: Disable limiter for RAG/API usage
                        limiter: false

                    search:
                        formats:
                            - html
                            - json  # <--- This is the critical line to enable the API
                    
                    # Optional: Disable throttling if you are the only user
                    # protection_max_requests: 100 
                    # protection_limiter_window: 1
                """)

                f.write(settings_yaml)
            print(f"\nSearXNG `settings.yml` created successfully at {file_path}!\n")

        return True
    except Exception as e:
        raise Exception(f"Could not setup SearXNG `settings.yml`: {e}")


def bring_searxng_online():
    try:

        config = config_manager.read_config([
            'assign_host_port_to_searxng_server',
            'searxng_dir'
        ])
        container_name = 'searxng'

        # Ensure absolute path for volume mount
        abs_searxng_dir = os.path.abspath(config['searxng_dir'])
        host_port = str(config['assign_host_port_to_searxng_server'])

        setup_searxng_settings_yaml()

        # Check if Docker Engine is running
        subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            check=True
        )  # check=True raises if the command returns a non-zero exit code

        # Check if container is running
        if utils_module.check_if_container_is_running(container_name):
            print("\nSearXNG container is already running, skipping...\n")
            return True

        # Check if container exists, but is stopped
        if utils_module.check_if_container_exists(container_name):
            print("\nSearXNG container exists but is stopped, restarting...\n")
            return utils_module.start_container(container_name)

        # Container doesn't exist, create it
        print("\nDocker Engine online, proceeding with SearXNG container launch...\n")

        command = [
            'docker', 'run', '-d',
            '--name', container_name,
            '-p', f"{host_port}:8080",
            '-v', f"{abs_searxng_dir}:/etc/searxng",
            'searxng/searxng:latest'
        ]

        if platform.system() == 'Windows':
            subprocess.Popen(
                command,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            '''
            shell=True is dangerous/buggy with list arguments on Linux. 
            Use shell=False (default) when passing a list.
            '''
            subprocess.Popen(
                command,
                shell=False
            )
        
        print(f"\nSearXNG container launched successfully!\n")
        return True

    except Exception as e:
        raise Exception(f"Could not launch SearXNG container: {e}")