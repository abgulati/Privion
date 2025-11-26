import prompt_formatting as prompt_formatting_module
import llm_apis as llm_apis_module

from wakeonlan import send_magic_packet
import ha_device_modules.lg_webos_tv as lg_webos_tv_module
import ha_device_modules.govee_lights as govee_lights_module

from typing import Optional
from pathlib import Path

import asyncio
import json
import ast


########################---------------------config setup---------------------###############################
BUTLER_TOOLS_CONFIG_PATH, FULL_BUTLER_TOOLS_CONFIG = None, None # set in core controller method - execute-butler_tasks()

def _full_read_butler_tools_config() -> dict:
    '''
    Reads the butler-tools.json file and returns the config.
    Returns an empty config if the file does not exist.

    Args:
        None
    
    Returns:
        - butler_tools_config_path: The path to the butler-tools.json file.
        - butler_tools_config: A dictionary with the config.
    '''
    butler_tools_config_path = Path.cwd() / 'butler-tools.json'

    if not butler_tools_config_path.exists():
        with open(butler_tools_config_path, 'w') as file:
            json.dump({}, file, indent=4)

    butler_tools_config = {}
    try:
        with open(butler_tools_config_path, 'r') as file:
            butler_tools_config = json.load(file)
            print(f"Butler tools config: {butler_tools_config}")
    except Exception as e:
        print(f"Error reading butler-tools.json: {str(e)}")

    return butler_tools_config_path, butler_tools_config


def write_butler_tools_config(butler_tools_config_updates:dict, filename:str=None) -> dict:
    '''
    Writes the butler-tools.json file with the config.

    Args:
        - butler_tools_config: A dictionary with the config.
        - filename: The name of the file to write to.
    
    Returns:
        - success: True if the config was written successfully, False otherwise.
    '''

    filename = filename or BUTLER_TOOLS_CONFIG_PATH

    try:
        with open(filename, 'r') as file:
            butler_tools_config = json.load(file)
    except Exception as e:
        butler_tools_config = {}
        print(f"Error reading butler-tools.json: {str(e)}")

    butler_tools_config.update(butler_tools_config_updates)
    
    try:
        with open(filename, 'w') as file:
            json.dump(butler_tools_config, file, indent=4)
    except Exception as e:
        print(f"Error writing butler-tools.json: {str(e)}")
        return False
    
    return True

#######################################################################################################


### TOOLS.JSON PARSERS:   TODO - Filtering out un-needed tools for the prompt!
def get_all_service_names_from_butler_tools_config() -> list:
    '''
    Returns all the services from the butler-tools.json file.
    '''
    return list(FULL_BUTLER_TOOLS_CONFIG['services'].keys())


def get_service_details_from_butler_tools_config(service_name:str) -> dict:
    '''
    For a given service name, returns the service details from the butler-tools.json file.
    '''
    return FULL_BUTLER_TOOLS_CONFIG['services'][service_name]


def get_all_services_with_descriptions_from_butler_tools_config() -> dict:
    '''
    Returns all the services with descriptions from the butler-tools.json file.
    '''
    services_with_descriptions = {}
    for service_name, service_details in FULL_BUTLER_TOOLS_CONFIG['services'].items():
        services_with_descriptions[service_name] = service_details['description']
    
    # print(f"Total number of services: {len(services_with_descriptions)}")
    # print(services_with_descriptions)
    return services_with_descriptions


### TOOL DEFS:
def wol_turn_on_tv(mac_address:str, target_ip: Optional[str] = None, port:int = 9):
    '''
    Uses Wake-On-LAN (WOL) to turn on a TV.
    Args:
        mac_address: The MAC address of the TV to turn on.
        target_ip: The IP address to broadcast the WOL packet to.
        port: The port to send the WOL packet to.
    Returns:
        A dictionary with a success flag and a message.
        success: True if the magic packet was sent, False otherwise.
        message: A message describing the result of the operation.
    '''
    if mac_address is None or mac_address == '':
        return {"success": False, "message": "MAC address is required."}
    
    try:
        send_magic_packet(mac_address, ip_address=target_ip, port=port)
        print("Magic packet sent.")
        return {"success": True, "message": "Device turn-on request successfully transmitted."}
    except Exception as e:
        print(f"Error sending magic packet: {str(e)}")
        return {"success": False, "message": f"Error sending Wake on LAN magic packet: {str(e)}"}


def lg_webos_tv_turn_off(ip_address: Optional[str] = None):
    '''
    Turns on an LG WebOS TV.
    '''
    if not ip_address:
        webos_ip = lg_webos_tv_module.discover_webos_ip()
        if not webos_ip:
            return {"success": False, "message": "No LG WebOS TVs found on the network."}
        ip_address = webos_ip
        print(f"Found LG WebOS TV at IP address: {ip_address}")
    return asyncio.run(lg_webos_tv_module.webos_pair_connect_and_power_off_async(ip_address))


def turn_on_lamp():
    '''
    Turns on a lamp.
    '''
    return asyncio.run(govee_lights_module.light_turn_on_handler())


def turn_off_lamp():
    '''
    Turns off a lamp.
    '''
    return asyncio.run(govee_lights_module.light_turn_off_handler())


def execute_butler_tool(service_name:str) -> dict:
    '''
    Executes a butler tool.
    '''
    # butler_tool_details = get_service_details_from_butler_tools_config(service_name)
    # butler_tool_execution_result = butler_tool_details['function'](**butler_tool_details['fields'])
    try:
        if service_name == 'wol_turn_on_tv' or service_name == 'lg_webos_tv_turn_on':
            print("Executing WOL turn-on TV tool...")
            return wol_turn_on_tv('a4:36:c7:58:3f:18', '192.168.1.142', 9)
        elif service_name == 'lg_webos_tv_turn_off':
            print("Executing LG WebOS TV turn-off tool...")
            return lg_webos_tv_turn_off('192.168.1.142')
        elif service_name == 'lamp_turn_on':
            print("Executing lamp turn-on tool...")
            return turn_on_lamp()
        elif service_name == 'lamp_turn_off':
            print("Executing lamp turn-off tool...")
            return turn_off_lamp()
        else:
            print(f"Service not found: {service_name}")
            return {'success': False, 'message': f"Service not found: {service_name}"}
    except Exception as e:
        print(f"Error executing butler tool: {str(e)}")
        return {'success': False, 'message': f"Error executing butler tool for {service_name} service: {str(e)}"}


### LLM RESPONSE PARSER:
def parse_butler_tool_response(llm_response:str) -> dict:
    """
    Helper function to parse and clean the service response.
    
    Args:
        response: Raw response string from HF-Waitress
        
    Returns:
        dict: Parsed service selection or default RAG service
    """
    try:
        json_parsed_response = json.loads(llm_response)
        return json_parsed_response
    except Exception as e:
        print(f"\nFailed to parse service response, attempting to literal-eval and maybe even trim. Encountered error: {e}\n")

    try:
        return ast.literal_eval(llm_response)
    except (ValueError, SyntaxError):
        # Sometimes additional text may be present so we need to strip it:
        try:
            print(f"\nAdditional text present, trimming response...\n")
            cleaned_response = prompt_formatting_module.trim_response(
                llm_response,
                '"service_list":', '}',
                include_start_substring=True,
                include_end_substring=False
            )
            cleaned_response = "{" + cleaned_response.strip() + "}"
            print(f"\nTrimmed response to dictionary: {cleaned_response}\n")
            return ast.literal_eval(cleaned_response)
        except (ValueError, SyntaxError) as e:
            print(f"Failed to identify selected service even after trimming, encountered error: {e}")
            return {'service': None}


### LLM REQUEST/RESPONSE HANDLERS - FIGURE RE-USE FROM CORE APP.PY
def make_butler_request(llm_prompt:str) -> str:
    '''
    Makes a request to the LLM.
    '''
    return llm_apis_module.make_request_to_llm_server(llm_prompt)


### CORE CONTROLLER METHOD:
def execute_butler_tasks(user_query:str) -> dict:
    '''
    Receives user-query, then invokes in-order:
    1. Services Menu Creation: TOOLS.JSON PARSERS
    2. Prompt Crafting: CORE PROMPT
    3. LLM REQUEST/RESPONSE HANDLERS
    4. Selected Service Determination: LLM RESPONSE PARSER
    5. Service Execution: TOOL DEFS
    6. Return Prep: Format User-Query With Selected Tool, Success/Failure Status
    7. Return: Final Response to Front-End, wherein the selected tool will be printed out to the user, along with a success/failure message.
    '''

    global BUTLER_TOOLS_CONFIG_PATH, FULL_BUTLER_TOOLS_CONFIG
    BUTLER_TOOLS_CONFIG_PATH, FULL_BUTLER_TOOLS_CONFIG = _full_read_butler_tools_config()

    try:
        llm_prompt_for_butler_request = prompt_formatting_module.get_core_prompt_for_butler_tools_config(user_query, get_all_service_names_from_butler_tools_config())
        print(f"LLM prompt for butler request: {llm_prompt_for_butler_request}")
        
        llm_response_for_butler_request = make_butler_request(llm_prompt_for_butler_request)
        print(f"LLM response for butler request: {llm_response_for_butler_request}")
        
        butler_tool_selection = parse_butler_tool_response(llm_response_for_butler_request)
        print(f"Butler tool selection: {butler_tool_selection}")

        action_result = {}
        if butler_tool_selection['service_list'] is None:
            action_result = {'success': False, 'message': "AI-service determination failed."}
            print(f"Action result: {action_result}")
        
        else:

            if isinstance(butler_tool_selection['service_list'], list):

                butler_tool_selection['service_list'] = list(set(butler_tool_selection['service_list']))
                
                for service_name in butler_tool_selection['service_list']:
                    action_result[service_name] = execute_butler_tool(service_name)
                    print(f"Action result after tool execution {service_name}: {action_result[service_name]}")

                print(f"Action results list: {action_result}")
            
            elif isinstance(butler_tool_selection['service_list'], str):
                action_result = execute_butler_tool(butler_tool_selection['service_list'])
                print(f"Action result after tool execution: {action_result}")
            
            else:
                action_result = {'success': False, 'message': f"Invalid service list format: {butler_tool_selection['service_list']}"}
                print(f"Action result after tool execution: {action_result}")

        action_analysis_prompt = prompt_formatting_module.request_action_analysis_prompt(user_query, action_result)
        print(f"Action analysis prompt: {action_analysis_prompt}")
        
        return {'action_result': action_result, 'action_analysis_prompt': action_analysis_prompt}

    except Exception as e:
        print(f"Error executing butler tasks: {str(e)}")
        action_result = {'success': False, 'message': f"Error executing butler tasks: {str(e)}"}
        try:
            action_analysis_prompt = prompt_formatting_module.request_action_analysis_prompt(user_query, action_result)
            print(f"Action analysis prompt in exception handler: {action_analysis_prompt}")
            return {'action_result': action_result, 'action_analysis_prompt': action_analysis_prompt}
        except Exception as e:
            print(f"Both Butler Tool Execution and Action Analysis Prompt Creation Failed: {str(e)}")
            return {'success': False, 'message': f"Both Butler Tool Execution and Action Analysis Prompt Creation Failed: {str(e)}", 'action_analysis_prompt': None}
