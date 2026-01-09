import prompt_formatting as prompt_formatting_module
import rag_core as rag_core_module
import llm_apis as llm_apis_module

from wakeonlan import send_magic_packet
import ha_device_modules.lg_webos_tv as lg_webos_tv_module
import ha_device_modules.govee_lights as govee_lights_module
import ha_device_modules.kasa_plug as kasa_plug_module
import ha_device_modules.kasa_bulb as kasa_bulb_module

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
            # print(f"Butler tools config: {butler_tools_config}")
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


def generate_tools_schema(services_config: dict) -> list[dict]:
    '''
    Generates a tools schema from the services config.
    '''
    tools_schema = []
    
    for service_name, service_details in services_config.items():
        tool = {
            "type": "function",
            "function": {
                "name": service_name,
                "description": service_details.get('description', ''),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False   # Tells the LLM: "Do not hallucinate extra arguments that I didn't define"!
                }
            }
        }

        fields = service_details.get('fields', {})
        for field_name, field_details in fields.items():

            prop_type = field_details.get('type', 'string')
            prop_desc = field_details.get('description', '')

            has_default = 'default' in field_details
            default_val = field_details.get('default')

            if has_default: # CRITICAL: Tell the LLM about the default behavior in plain English
                prop_desc += f" Optional. Defaults to '{default_val}' if not specified. Do not guess."
            
            property_def = {
                "type": prop_type,
                "description": prop_desc.strip()
            }

            if 'enum' in field_details:
                property_def['enum'] = field_details.get('enum', [])

            if 'minimum' in field_details:
                property_def['minimum'] = field_details.get('minimum')

            if 'maximum' in field_details:
                property_def['maximum'] = field_details.get('maximum')

            tool['function']['parameters']['properties'][field_name] = property_def
            
            is_explicitly_required = field_details.get('required', False)
            if is_explicitly_required and not has_default:
                tool['function']['parameters']['required'].append(field_name)

        tools_schema.append(tool)
    
    return tools_schema


### TOOL DEFS:
def search(query:str, **kwargs) -> dict:
    '''
    Searches the local & web knowledge base.
    '''
    stream_session_id = kwargs.get('stream_session_id', None)
    return rag_core_module.execute_full_search(query, stream_session_id)


def wol_turn_on_tv(mac_address:str, target_ip: Optional[str] = None, port: str | int = '9', **kwargs):
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
        send_magic_packet(mac_address, ip_address=target_ip, port=int(port))
        print("Magic packet sent.")
        return {"success": True, "message": "Device turn-on request successfully transmitted."}
    except Exception as e:
        print(f"Error sending magic packet: {str(e)}")
        return {"success": False, "message": f"Error sending Wake on LAN magic packet: {str(e)}"}


def lg_webos_tv_turn_on(mac_address:str, target_ip: Optional[str] = None, port: str | int = '9', **kwargs):
    return wol_turn_on_tv(mac_address, target_ip, port)


def lg_webos_tv_turn_off(ip_address: Optional[str] = None, **kwargs):
    '''
    Turns on an LG WebOS TV.
    '''
    if not ip_address:
        print("No IP address provided, attempting to discover LG WebOS TV on the network.")
        webos_ip = lg_webos_tv_module.discover_webos_ip()
        if not webos_ip:
            return {"success": False, "message": "No LG WebOS TVs found on the network."}
        ip_address = webos_ip
        print(f"Found LG WebOS TV at IP address: {ip_address}")
    return asyncio.run(lg_webos_tv_module.webos_pair_connect_and_power_off_async(ip_address))


def lamp_turn_on(**kwargs):
    '''
    Turns on a lamp.
    '''
    return asyncio.run(govee_lights_module.light_turn_on_handler())


def lamp_turn_off(**kwargs):
    '''
    Turns off a lamp.
    '''
    return asyncio.run(govee_lights_module.light_turn_off_handler())


def set_lamp_brightness(brightness: str | int, **kwargs):
    '''
    Sets the brightness of a lamp.
    '''
    return asyncio.run(govee_lights_module.light_set_brightness_handler(int(brightness)))


def set_lamp_color(red: str | int, green: str | int, blue: str | int, **kwargs):
    '''
    Sets the color of a lamp.
    '''
    return asyncio.run(govee_lights_module.light_set_color_handler(int(red), int(green), int(blue)))


def set_lamp_temperature(temperature: str | int, **kwargs):
    '''
    Sets the temperature of a lamp.
    '''
    return asyncio.run(govee_lights_module.light_set_temperature_handler(int(temperature)))


def set_lamp_scene(scene: str, **kwargs):
    '''
    Sets the scene of a lamp.
    '''
    return asyncio.run(govee_lights_module.light_set_scene_handler(scene))

def plug_turn_on(**kwargs):
    '''
    Turns on a smart plug.
    '''
    return asyncio.run(kasa_plug_module.plug_turn_on_handler())

def plug_turn_off(**kwargs):
    '''
    Turns off a smart plug.
    '''
    return asyncio.run(kasa_plug_module.plug_turn_off_handler())

def bulb_turn_on(**kwargs):
    '''
    Turns on a smart bulb.
    '''
    return asyncio.run(kasa_bulb_module.bulb_turn_on_handler())

def bulb_turn_off(**kwargs):
    '''
    Turns off a smart bulb.
    '''
    return asyncio.run(kasa_bulb_module.bulb_turn_off_handler())

def set_bulb_brightness(brightness: str | int, **kwargs):
    '''
    Sets the brightness of a smart bulb.
    '''
    return asyncio.run(kasa_bulb_module.bulb_set_brightness_handler(int(brightness)))


def set_bulb_color(hue: str | int, saturation: str | int, value: str | int, **kwargs):
    '''
    Sets the color of a smart bulb.
    '''
    return asyncio.run(kasa_bulb_module.bulb_set_color_handler(int(hue), int(saturation), int(value)))


def execute_tool_call(tool_name, llm_args, services_config: dict, stream_session_id: str = None) -> dict:
    '''
    1. Looks up the function in Python.
    2. Looks up the default values in your JSON config.
    3. Merges them and runs the code.
    '''

    # Look up the function in globals
    func_to_call = globals().get(tool_name)

    if not func_to_call:
        return {'success': False, 'message': f"Service not found: {tool_name}"}
    
    try:
        # Get config definitions for defaults (e.g., target_ip, mac_address)
        service_def = services_config.get(tool_name, {})
        fields_def = service_def.get('fields', {})

        final_args = {}
        
        # 1. Load defaults from config (handles hardcoded IPs)
        for field_name, field_info in fields_def.items():
            if "default" in field_info:
                final_args[field_name] = field_info["default"]

        # MERGE: Start with defaults, overwrite with LLM-provided args
        final_args['stream_session_id'] = stream_session_id

        # 2. Update with args the LLM actually sent
        # (e.g. LLM sends 'brightness': 50, but ignores 'mac_address')
        if llm_args:
            final_args.update(llm_args)

        print(f"Executing {tool_name} with args: {final_args}")
        return func_to_call(**final_args)   # **final_args unpacks the dictionary into keyword arguments
    
    except Exception as e:
        print(f"Error executing {tool_name} with args: {final_args}, encountered error: {e}")
        return {'success': False, 'message': f"Error executing {tool_name} with args: {final_args}, encountered error: {e}"}


### LLM REQUEST/RESPONSE HANDLERS - FIGURE RE-USE FROM CORE APP.PY
def make_butler_request(llm_prompt:str, tools_schema:list = None) -> str:
    '''
    Makes a request to the LLM.
    '''
    return llm_apis_module.make_tool_request_to_llm_server(llm_prompt, tools_schema)


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
    

    try:
        BUTLER_TOOLS_CONFIG_PATH, FULL_BUTLER_TOOLS_CONFIG = _full_read_butler_tools_config()

        # 1. Get Tools
        tools_schema = generate_tools_schema(FULL_BUTLER_TOOLS_CONFIG.get('services', {}))
        print(f"\n\nTools schema: {tools_schema}\n\n")

        # 2. Get Prompt
        llm_prompt_for_butler_request = prompt_formatting_module.get_butler_tool_call_prompt(user_query)
        print(f"\n\nLLM prompt for butler request: {llm_prompt_for_butler_request}\n\n")
        
        # 3. Make Request (Returns a DICT!)
        llm_response_obj = make_butler_request(llm_prompt_for_butler_request, tools_schema)
        
        print(f"Butler raw response: {llm_response_obj}")
        action_result = {}

        # 4. Check for Native Tool Calls (Standard Dictionary Access)
        if llm_response_obj.get('tool_calls', None):
            print(f"LLM decided to call {len(llm_response_obj['tool_calls'])} tool(s).")

            for tool_call in llm_response_obj['tool_calls']:
                 # OpenAI format: 'function' is a dict, 'arguments' is a JSON STRING
                fn_name = tool_call['function']['name']
                raw_args = tool_call['function']['arguments']
                
                # Safe Parsing
                if isinstance(raw_args, str):
                    fn_args = json.loads(raw_args)
                else:
                    fn_args = raw_args # In case the custom backend already made it a dict!

                # Execute
                result = execute_tool_call(fn_name, fn_args, FULL_BUTLER_TOOLS_CONFIG.get('services', {}))
                action_result[fn_name] = result
                print(f"Action result after tool execution {fn_name}: {result}")

            # 5. Analysis
            action_analysis_prompt = prompt_formatting_module.request_action_analysis_prompt(user_query, action_result)
            # print(f"Action analysis prompt: {action_analysis_prompt}")
            return {'action_result': action_result, 'action_analysis_prompt': action_analysis_prompt}
        
        else:
            raise Exception("No tool calls found in LLM response.")

    except Exception as e:
        print(f"Error executing butler tasks: {str(e)}")
        action_result = {'success': False, 'message': f"Error executing butler tasks: {str(e)}"}
        try:
            action_analysis_prompt = prompt_formatting_module.request_action_analysis_prompt(user_query, action_result)
            # print(f"Action analysis prompt in exception handler: {action_analysis_prompt}")
            return {'action_result': action_result, 'action_analysis_prompt': action_analysis_prompt}
        except Exception as e:
            print(f"Both Butler Tool Execution and Action Analysis Prompt Creation Failed: {str(e)}")
            return {'success': False, 'message': f"Both Butler Tool Execution and Action Analysis Prompt Creation Failed: {str(e)}", 'action_analysis_prompt': None}
        

def execute_tools(tools_json: dict, stream_session_id: str = None) -> dict:
    print(f"Executing tools: {tools_json}")
    tool_result_list = []
    _, full_tools_config = _full_read_butler_tools_config()

    try:
        if tools_json.get('tool_calls', None):
            print(f"LLM decided to call {len(tools_json['tool_calls'])} tool(s).")

            for tool_call in tools_json['tool_calls']:
                # OpenAI format: 'function' is a dict, 'arguments' is a JSON STRING
                fn_name = tool_call['function']['name']
                raw_args = tool_call['function']['arguments']
                
                # Safe Parsing
                if isinstance(raw_args, str):
                    fn_args = json.loads(raw_args)
                else:
                    fn_args = raw_args # In case the custom backend already made it a dict!

                # Execute
                result = execute_tool_call(fn_name, fn_args, full_tools_config.get('services', {}), stream_session_id)
                tool_result_list.append({
                    'role': 'tool',
                    'tool_call_id': tool_call['id'],
                    'name': fn_name,
                    'content': result['message']
                })
                print(f"Tool execution result for {fn_name}: {result}")

        return {'success': True, 'tool_result_list': tool_result_list, 'message': 'Tools executed successfully.'}
    except Exception as e:
        print(f"Error executing tools: {str(e)}")
        return {'success': False, 'tool_result_list': [], 'message': f"Error executing tools: {str(e)}"}