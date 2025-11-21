import requests
import json

from prompt_formatting import prepare_prompt_for_auto_templating
from privion_config_concierge import read_config, read_hf_config


def hf_waitress_non_streaming_api_handler(endpoint_url:str, headers:dict, payload:str) -> str:
    print(f"\nHF-Waitress Non-Streaming Request Response Handler Invoked\n")
    try:
        response = requests.post(endpoint_url, headers=headers, data=payload)
        print("\nCompleted, returning response\n")
        return (response.json()['response'])
    except Exception as e:
        raise Exception(f"Failed /completions request to extract entities and relationships from chunk, encountered error: {e}")


def llama_cpp_non_streaming_api_handler(endpoint_url:str, headers:dict, payload:str) -> str:
    print(f"\nLLama.cpp Non-Streaming Request Response Handler Invoked\n")
    try:
        response = requests.post(endpoint_url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for bad status codes so we can catch them in the except block
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        raise Exception(f"Failed request to LLama.cpp {endpoint_url} API, encountered error: {e}")


def hf_waitress_streaming_api_handler(endpoint_url:str, headers:dict, payload:str) -> str:
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
                        print(f"Failed to parse event data: {event_data}, encountered error: ", e)
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
        raise Exception(f"Failed request to HF-Waitress {endpoint_url} API, encountered error: {e}")


def get_request_params_for_llm_api(messages_dict:dict, stream:bool=False) -> tuple[str, dict, str, str, bool]:
    try:
        read_return = read_config([
            'local_llm_server', 'hf_waitress_access_url', 'hf_waitress_server_port', 
            'llama_cpp_access_url', 'llama_cpp_server_port', 'llama_cpp_temperature', 
            'llama_cpp_top_k', 'llama_cpp_top_p', 'llama_cpp_min_p'
        ])
        local_llm_server = read_return['local_llm_server'].lower().strip()
    except Exception as e:
        raise Exception(f"Could not read request params from config.json, encountered error: {e}")

    headers = {'Content-Type': 'application/json'}

    if local_llm_server == 'hf-waitress':

        json_payload = json.dumps(messages_dict)
        base_url = f"http://{read_return['hf_waitress_access_url']}:{read_return['hf_waitress_server_port']}"

        try:
            read_hf_return = read_hf_config(['exl2', 'exl3'])
            exl2 = str(read_hf_return['exl2']).lower() == 'true'
            exl3 = str(read_hf_return['exl3']).lower() == 'true'
        except Exception as e:
            raise Exception(f"Could not read hf-waitress config, encountered error: {e}")
        
        if not exl2 and not exl3:
            headers['X-Return-Full-Text'] = 'False'
            endpoint_url = f"{base_url}/completions_stream" if stream else f"{base_url}/completions"
        elif (exl2 or exl3) and stream:
            headers['Connection'] = 'keep-alive'
            endpoint_url = f"{base_url}/exl2_stream" if exl2 else f"{base_url}/exl3_stream"
        else:
            raise Exception(f"Invalid local LLM server, expected 'hf-waitress' or 'llama-cpp', received: {local_llm_server}")

        return endpoint_url, headers, json_payload, local_llm_server, exl2
    
    elif local_llm_server == 'llama-cpp':   # llama.cpp LLMs
        messages_dict['stream'] = stream
        messages_dict['temperature'] = read_return['llama_cpp_temperature']
        messages_dict['top_k'] = read_return['llama_cpp_top_k']
        messages_dict['top_p'] = read_return['llama_cpp_top_p']
        messages_dict['min_p'] = read_return['llama_cpp_min_p']
        json_payload = json.dumps(messages_dict)

        endpoint_url = f"http://{read_return['llama_cpp_access_url']}:{read_return['llama_cpp_server_port']}/v1/chat/completions"
        
        return endpoint_url, headers, json_payload, local_llm_server, False

    else:
        raise Exception(f"Invalid local LLM server, expected 'hf-waitress' or 'llama-cpp', received: {local_llm_server}")


def make_request_to_llm_server(user_query:str) -> str:
    try:
        local_llm_server = read_config(['local_llm_server'])['local_llm_server'].lower().strip()
    except Exception as e:
        raise Exception(f"Could not determine local LLM server, encountered error: {e}")

    try:
        messages_dict = prepare_prompt_for_auto_templating(formatted_prompt="", user_query=user_query, current_sequence_id=0, system_prompt="", skip_system_prompt=True)
    except Exception as e:
        raise Exception(f"Could not format prompt for generic HF-Waitress request, encountered error: {e}")

    try:
        if local_llm_server == 'hf-waitress':
            waitress_url, headers, json_payload, _, _ = get_request_params_for_llm_api(messages_dict=messages_dict, stream=True)
            return hf_waitress_streaming_api_handler(waitress_url, headers, json_payload)
        
        elif local_llm_server == 'llama-cpp':
            llama_cpp_url, headers, json_payload, _, _ = get_request_params_for_llm_api(messages_dict=messages_dict, stream=False)
            return llama_cpp_non_streaming_api_handler(llama_cpp_url, headers, json_payload)

        else:
            raise Exception(f"Invalid local LLM server, expected 'hf-waitress' or 'llama-cpp', received: {local_llm_server}")
    
    except Exception as e:
        raise Exception(f"Could not make request to LLM server - {local_llm_server}, encountered error: {e}")