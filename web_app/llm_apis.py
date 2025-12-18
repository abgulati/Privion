import requests
import json
import re

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
    

def llama_cpp_non_streaming_api_handler_with_tools(endpoint_url:str, headers:dict, payload:str) -> dict:
    print(f"\nLLama.cpp Non-Streaming Request Response Handler with Tools Invoked\n")
    try:
        response = requests.post(endpoint_url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for bad status codes so we can catch them in the except block

        # CRITICAL CHANGE: Return the whole message object!
        # This contains {'role': 'assistant', 'content': '...', 'tool_calls': [...]}
        return response.json()['choices'][0]['message']
    
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


def extract_tool_calls_from_response(full_response_text:str) -> list[dict]:
    try:
        print("\n--- Starting tool parsing ---")

        tool_calls = []
        content = full_response_text

        # 1. Regex to find ALL occurrences of <tool_call>...</tool_call>
        # re.DOTALL ensures '.' matches newlines
        tool_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

        for match in tool_regex.finditer(full_response_text):
            raw_tool_content = match.group(1).strip()

            tool_name = None
            tool_args = {}
            parse_success = False

            # --- STRATEGY A: Standard JSON ---
            if raw_tool_content.startswith("{") or raw_tool_content.startswith("["):
                try:
                    json_data = json.loads(raw_tool_content)

                    # Handle case where the content is a LIST of tools inside one tag
                    if isinstance(json_data, list):
                        for item in json_data:
                            tool_calls.append({
                                'id': 'call_' + str(hash(json.dumps(item)))[:8],
                                'type': 'function',
                                'function': {
                                    'name': item.get("name"),
                                    'arguments': json.dumps(item.get("arguments", {}))
                                }
                            })
                        parse_success = True # We handled it, skip to next match
                        continue 

                    # Handle single JSON object
                    else:
                        tool_name = json_data.get('name')
                        tool_args = json_data.get('arguments', {})

                        tool_calls.append({
                            'id': 'call_' + str(hash(tool_name + str(tool_args)))[:8],
                            'type': 'function',
                            'function': {
                                'name': tool_name,
                                'arguments': json.dumps(tool_args)
                            }
                        })
                        parse_success = True
                except json.JSONDecodeError:
                    print("Tool content looked like JSON but failed to parse.")
                    pass # Fall through to Strategy B
            
            # --- STRATEGY B: Custom XML (<function=name>...) ---
            if not parse_success:
                try:
                    # We use finditer here too, in case there are multiple <function> tags inside one <tool_call>
                    function_regex = re.compile(r"<function=(.*?)>(.*?)</function>", re.DOTALL)
                    
                    found_functions = list(function_regex.finditer(raw_tool_content))

                    if found_functions:
                        for func_match in found_functions:
                            t_name = func_match.group(1).strip()
                            t_body = func_match.group(2).strip()
                            t_args = {}
                            
                            # Extract parameters
                            param_matches = re.findall(r"<parameter=(.*?)>(.*?)</parameter>", t_body, re.DOTALL)
                            for p_key, p_val in param_matches:
                                t_args[p_key.strip()] = p_val.strip()

                            # Add to list immediately
                            tool_calls.append({
                                'id': 'call_' + str(hash(t_name + str(t_args)))[:8],
                                'type': 'function',
                                'function': {
                                    'name': t_name,
                                    'arguments': json.dumps(t_args)
                                }
                            })
                        parse_success = True
                except Exception as e:
                    print(f"Failed to parse custom XML format: {e}")

        # Remove ALL <tool_call> blocks from the text shown to the user
        content = tool_regex.sub('', full_response_text).strip()

        # Return standardized dict
        result = {'role': 'assistant', 'content': content}
        if tool_calls:
            result['tool_calls'] = tool_calls

        return result
    except Exception as e:
        raise Exception(f"Failed to extract tool calls from response, encountered error: {e}")


def hf_waitress_streaming_api_handler_with_tools(endpoint_url:str, headers:dict, payload:str) -> dict:
    print(f"\nHF-Waitress Streaming Request Response Handler with Tools Invoked\n")
    
    try:
        response = requests.post(endpoint_url, headers=headers, data=payload, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes so we can catch them in the except block

        full_response_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data:"):
                    event_data = line[6:].strip()
                    try:
                        token = str(json.loads(event_data))
                        full_response_text += token
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse event data: {event_data}, encountered error: ", e)
                elif line.startswith("event: END"):
                    break
                else:
                    print(f"\nUnexpected Line Format: {line}\n")
        
        print(f"Full response text: {full_response_text}")
        
        if not full_response_text:
            print("\nWarning: No response from exl2-stream / exl2-grapher request\n")
            return {'role': 'assistant', 'content': None}
        
        # --- PARSE NATIVE RESPONSE INTO OBJECT ---
        # Since the HF-Waitress custom backend returns raw text (with XML tags), 
        # we parse it here so the Controller treats it exactly like Llama.cpp
        return extract_tool_calls_from_response(full_response_text)
        
    except Exception as e:
        raise Exception(f"Failed request to HF-Waitress {endpoint_url} API, encountered error: {e}")


def get_request_params_for_llm_api(json_request_body:dict, stream:bool=False) -> tuple[str, dict, str, str, bool]:
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

        json_payload = json.dumps(json_request_body)
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
        json_request_body['stream'] = stream
        json_request_body['temperature'] = read_return['llama_cpp_temperature']
        json_request_body['top_k'] = read_return['llama_cpp_top_k']
        json_request_body['top_p'] = read_return['llama_cpp_top_p']
        json_request_body['min_p'] = read_return['llama_cpp_min_p']
        json_payload = json.dumps(json_request_body)

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
        json_request_body = prepare_prompt_for_auto_templating(formatted_prompt="", user_query=user_query, current_sequence_id=0, system_prompt="", skip_system_prompt=True)
    except Exception as e:
        raise Exception(f"Could not format prompt for generic HF-Waitress request, encountered error: {e}")

    try:
        if local_llm_server == 'hf-waitress':
            waitress_url, headers, json_payload, _, _ = get_request_params_for_llm_api(json_request_body=json_request_body, stream=True)
            return hf_waitress_streaming_api_handler(waitress_url, headers, json_payload)
        
        elif local_llm_server == 'llama-cpp':
            llama_cpp_url, headers, json_payload, _, _ = get_request_params_for_llm_api(json_request_body=json_request_body, stream=False)
            return llama_cpp_non_streaming_api_handler(llama_cpp_url, headers, json_payload)

        else:
            raise Exception(f"Invalid local LLM server, expected 'hf-waitress' or 'llama-cpp', received: {local_llm_server}")
    
    except Exception as e:
        raise Exception(f"Could not make request to LLM server - {local_llm_server}, encountered error: {e}")
    

def make_tool_request_to_llm_server(user_query:str, tools_schema: list[dict] = None) -> str:
    try:
        local_llm_server = read_config(['local_llm_server'])['local_llm_server'].lower().strip()
    except Exception as e:
        raise Exception(f"Could not determine local LLM server, encountered error: {e}")

    try:
        json_request_body = prepare_prompt_for_auto_templating(formatted_prompt="", user_query=user_query, current_sequence_id=0, system_prompt="", skip_system_prompt=True)
        if tools_schema: json_request_body['tools'] = tools_schema  # request body contains messages and optionally, tools
    except Exception as e:
        raise Exception(f"Could not format prompt for generic HF-Waitress request, encountered error: {e}")

    try:
        if local_llm_server == 'hf-waitress':
            waitress_url, headers, json_payload, _, _ = get_request_params_for_llm_api(json_request_body=json_request_body, stream=True)
            return hf_waitress_streaming_api_handler_with_tools(waitress_url, headers, json_payload)
        
        elif local_llm_server == 'llama-cpp':
            llama_cpp_url, headers, json_payload, _, _ = get_request_params_for_llm_api(json_request_body=json_request_body, stream=False)
            return llama_cpp_non_streaming_api_handler_with_tools(llama_cpp_url, headers, json_payload)

        else:
            raise Exception(f"Invalid local LLM server, expected 'hf-waitress' or 'llama-cpp', received: {local_llm_server}")
    
    except Exception as e:
        raise Exception(f"Could not make request to LLM server - {local_llm_server}, encountered error: {e}")