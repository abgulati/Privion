from privion_config_concierge import read_hf_config

import json

def get_user_query_for_relationship_summary(source, target, relationship, summary, chunk):
    if summary == "":
        return f"""You are helping to populate a knowledge graph database by creating metadata summaries for relationships.
        
        Task: Generate a concise, informative summary (150-200 words) for the following relationship based on the provided text chunk. The summary should capture the core information about this relationship as represented in the text and be written in a factual tone.
        
        Relationship: {{"source": "{source}", "target": "{target}", "relationship": "{relationship}"}}
        
        <text_chunk>
        {chunk}
        </text_chunk>

        Output format:
        {{
            "summary": "Your concise summary here"
        }}
        """
    
    else:
        return f"""You are helping to maintain a knowledge graph database by updating relationship summaries when new information becomes available.
        
        Task: Review the existing summary for this relationship and update it based on the new text chunk provided. Incorporate any new relevant information while maintaining a concise length (250 - 300 words). Keep the factual tone of the original summary.

        Relationship: {{"source": "{source}", "target": "{target}", "relationship": "{relationship}"}}

        Existing Summary: {{"summary": "{summary}"}}

        <text_chunk>
        {chunk}
        </text_chunk>

        Output format:
        {{
            "summary": "Your updated summary here"
        }}

        """


def get_user_query_for_comprehensive_summary(nodes_and_relationships, chunk):
    return f"""For the purpose of creating a Graph Database, nodes and relations were extracted from a chunk of text. Both are provided below, can you provide a concise (under 3000 words) summary, in the style of a report detailing crucial information and insights, for the text_chunk expounding on the nodes and relationships? Thank you!

    <text_chunk>
    {chunk}
    </text_chunk>

    <nodes_and_relationships>
    {json.dumps(nodes_and_relationships)}
    </nodes_and_relationships>


    Output format:
    {{
        "summary": "Your concise summary report (under 3000 words) here"
    }}
    """


def get_minimal_query_for_summary(chunk):
    return f"""For the purpose of creating a Graph Database, nodes and relations were extracted from the below chunk of text. Keeping this in mind, can you provide a concise (under 3000 words) summary, in the style of a report detailing crucial information and insights, for the text_chunk expounding on any nodes and relationships that may be present? Thank you!
    
    <text_chunk>
    {chunk}
    </text_chunk>


    Output format:
    {{
        "summary": "Your concise summary report (under 3000 words) here"
    }}
    """


def get_user_query_for_node_summary(name, node_type, summary, chunk):

    if summary == "":
        return f"""You are helping to populate a knowledge graph database by creating metadata summaries for entities(nodes).
        
        Task: Generate a concise, informative summary (150-200 words) for the following graph node based on the provided text chunk. The summary should capture the core information about this node as represented in the text and be written in a factual tone.

        Node: {{"type": "{node_type}", "name": "{name}"}}
        
        <text_chunk>
        {chunk}
        </text_chunk>

        Output format:
        {{
            "summary": "Your concise summary here"
        }}
        """

    else:
        return f"""You are helping to maintain a knowledge graph database by updating entity(node) summaries when new information becomes available.
        
        Task: Review the existing summary for this node and update it based on the new text chunk provided. Incorporate any new relevant information while maintaining a concise length (250 - 300 words). Keep the factual tone of the original summary.

        Node: {{"type": "{node_type}", "name": "{name}"}}

        Existing Summary: {{"summary": "{summary}"}}

        <text_chunk>
        {chunk}
        </text_chunk>

        Output format:
        {{
            "summary": "Your updated summary here"
        }}

        """



#############################----------Start Service-Request-Prompt-Building Methods----------#############################

def _append_query_and_output_specs_to_service_request_prompt(user_query:str) -> str:
    return f"""
    ### User Query:
    <user_query>
    {user_query}
    </user_query>

    ---

    ### Output Format (Strict JSON):
    ```json
    {{
        "service": "service name here"
    }}
    ```
    """


def _append_butler_guidelines_to_service_request_prompt() -> str:
    return f"""
    4. **Home Assistant**
    *   **Function:** Carry out real-world tasks, such as turning on/off lights, setting alarms, controlling thermostats, etc.
    *   **Use ONLY for:**
        *   Requests related to home automation or smart home devices (e.g., "Turn on the living room light", "Set the alarm for 7 AM", "Switch on or off the TV", etc.).
        *   Basic requests about the home environment (e.g., "What's the temperature in the living room?", "Is the front door locked?", "Is the oven on?", etc.).
        *   The user says something like "I'm bored" or "entertain me" and implies a real-world action by asking if you can do something about it. 
        *   The user asks you to do something for a third person, such as a family member or friend, etc.
        *   The user may use common slang terms when referring to appliances, such as "idiot box" for a TV or "jbl" or "flip" for bluetooth speakers, etc.
        *   Any other requests related to home automation or smart home devices.
        *   **It is NOT for:**
            *   Questions about the AI itself (e.g., "What is your name?", "What can you do?").
            *   Common knowledge facts or data facts lookups better suited for RAG or GraphDB.

    ---
    ### Guiding Principle:
    Focus on the **core intent**: Is the user asking for specific **data points** (even multiple for comparison) -> **RAG**, or are they asking for **analysis, synthesis, interpretation, or understanding relationships** -> **GraphDB**?
    Also, consider if the user is asking for (or indirectly implies) a **real-world action** to be performed -> **Home Assistant**.

    ---

    ### Decision Flow (Follow these steps):

    1.  **Assess Core Need:** Does the query ask for specific facts/values (RAG) or analysis/synthesis/interpretation (GraphDB)? A real-world action (Home Assistant)? Check if it's a simple conversation (Direct Response).
    2.  **Consider Keywords (as supporting evidence):**
        *   **Strong RAG indicators:** "What is/was [specific value]...", "When was...", "Who is...", "List...", "Specific figure for...", "Value of X vs Y...", "How much/many...". *Note: "How was [metric]" often implies asking for the value, favouring RAG unless context clearly demands analysis.*
        *   **Strong GraphDB indicators:** "Analyze...", "Summarize...", "Compare [performance/trends/strategies]...", "Why...", "How did [X relate to Y]...", "Trends...", "Overall performance...", "Correlate...", "Relationship between...", "Risks/Opportunities...".
        *   **Strong Home Assistant indicators:** "Turn on/off [appliance such as TV, speaker, etc.]", "Set the alarm for [time]", "Can you do [implies real-world action]", "I [or friend / family-member / third-person] am/are bored, anything you can do?" etc.
    3.  **Select the Service:** Based on the core need (Step 1) and supported by keywords (Step 2), choose the single most appropriate service. Prioritize the core need over ambiguous keywords.

    ---

    """


def _append_only_rag_guidelines_to_service_request_prompt() -> str:
    return f"""
    ---
    ### Guiding Principle:
    Focus on the **core intent**: Is the user asking for specific **data points** (even multiple for comparison) -> **RAG**, or are they asking for **analysis, synthesis, interpretation, or understanding relationships** -> **GraphDB**?

    ---

    ### Decision Flow (Follow these steps):

    1.  **Assess Core Need:** Does the query ask for specific facts/values (RAG) or analysis/synthesis/interpretation (GraphDB)? A real-world action (Home Assistant)? Check if it's a simple conversation (Direct Response).
    2.  **Consider Keywords (as supporting evidence):**
        *   **Strong RAG indicators:** "What is/was [specific value]...", "When was...", "Who is...", "List...", "Specific figure for...", "Value of X vs Y...", "How much/many...". *Note: "How was [metric]" often implies asking for the value, favouring RAG unless context clearly demands analysis.*
        *   **Strong GraphDB indicators:** "Analyze...", "Summarize...", "Compare [performance/trends/strategies]...", "Why...", "How did [X relate to Y]...", "Trends...", "Overall performance...", "Correlate...", "Relationship between...", "Risks/Opportunities...".
    3.  **Select the Service:** Based on the core need (Step 1) and supported by keywords (Step 2), choose the single most appropriate service. Prioritize the core need over ambiguous keywords.

    ---

    """


def _initialize_base_service_request_prompt() -> str:
    return f"""You're orchestrating a RAG system, and your task is to select the **single best service** from the list below to address the user's query, following the rules precisely.

    ### Services & Usage Rules:

    1.  **RAG**
        *   **Function:** Uses semantic/lexical search to find specific documents/passages containing factual answers.
        *   **Use ONLY if the query primarily asks for:**
            *   **One or more specific, distinct facts:** Numbers, dates, names, definitions, IDs (e.g., "What was <company_or_individual>'s Q2 revenue?", "What is the EIN for <company>?", "List the board members of <company>.").
            *   **Direct comparisons of specific data points:** Retrieving two or more specific values and comparing them directly (e.g., "What was <company_or_individual>'s gross margin in Q4 2024 vs Q4 2023?", "Compare <company_or_individual>'s revenue in 2023 and 2022.").
            *   Questions requiring a precise or definitive answer that can be found by searching documents or databases.

    2.  **GraphDB**
        *   **Function:** Uses a graph database to analyze patterns and relationships between entities to generate comprehensive summary reports.
        *   **Use ONLY if the query requires:**
            *   **Synthesis or complex analysis:** Summarizing performance, explaining trends, identifying correlations, understanding relationships (e.g., "Summarize <company_or_individual>'s 2024 performance", "How did <company_or_individual>'s sales correlate with market trends?", "Analyze the competitive landscape for <product>.", etc.).
            *   **Qualitative assessments:** Questions about strength, weakness, position, or outlook that require interpretation beyond simple facts (e.g., "Was <company_or_individual>'s 2024 performance strong?", "What are the key risks facing <company_or_individual>?").
            *   **Open-ended questions:** Broad questions needing insights derived from connecting multiple pieces of information, not just retrieving facts (e.g., "Why did <company>'s stock price change?", "Tell me about <company_or_individual>'s strategy.").
            *   **It is NOT for** simple factual lookups or direct A-vs-B value comparisons, even if keywords like "compare" or "how" are present in that simple context.

    3.  **Direct Response**
        *   **Function:** Answers directly without needing external data retrieval or complex analysis.
        *   **Use ONLY for:**
            *   Greetings, simple conversational phrases (e.g., "Hi", "Thank you", "How are you?").
            *   Basic requests about the AI itself (e.g., "What is your name?", "What can you do?").
            *   Common knowledge facts not requiring lookup in specific databases (e.g., "What is the capital of France?").

    """


def get_service_request_prompt(user_query:str, enable_butler_mode_selection:bool = False) -> str:
    service_request_prompt = _initialize_base_service_request_prompt()
    
    if enable_butler_mode_selection:
        service_request_prompt += _append_butler_guidelines_to_service_request_prompt()
    else:
        service_request_prompt += _append_only_rag_guidelines_to_service_request_prompt()
    
    service_request_prompt += _append_query_and_output_specs_to_service_request_prompt(user_query)
    
    return service_request_prompt


#############################----------End Service-Request-Prompt-Building Methods----------#############################


def clean_think_tags_from_prompt(formatted_prompt:str) -> str:
    """
    Creates a new list of messages with <think></think> tags cleaned.
    This is because llama.cpp's Jinja parser cannot handle those tags!
    This function does not modify the original list.
    """
    cleaned_prompt = {"messages": []}
    for message in formatted_prompt['messages']:
        new_message = message.copy()    # Create a copy to avoid modifying the original
        
        if "</think>" in new_message['content']:
            clean_content = new_message['content'].split("</think>", 1)[-1].strip()
            new_message['content'] = clean_content
        
        cleaned_prompt['messages'].append(new_message)
    
    return cleaned_prompt


def read_config_for_hf_waitress_prompt_formatting() -> tuple[bool, bool]:
    try:
        vision = read_hf_config(['vision'])['vision']
        flux_diffusers = read_hf_config(['flux_diffusers'])['flux_diffusers']
        return vision, flux_diffusers
    except Exception as e:
        print("Could not read exl2 details from config.json / hf-config.json, encountered error: ", e)


def prepare_prompt_for_auto_templating(formatted_prompt:str, user_query:str, current_sequence_id:int, system_prompt:str, skip_system_prompt:bool) -> dict:

    print("\n\nFormatting prompt for Transformers-AutoTokenizer / Jinja2-based Auto-Templating\n\n")

    try:
        vision, flux_diffusers = read_config_for_hf_waitress_prompt_formatting()
    except Exception as e:
        print("Could not read exl2 details from config.json / hf-config.json, encountered error: ", e)

    try:
        if flux_diffusers:
            return {"messages": [{"prompt": json.dumps(user_query)}]}
            
        else:
            if current_sequence_id > 0:
                # load & clean chat history object
                messages_dict_with_history = json.loads(formatted_prompt)
                messages_without_think_tags = clean_think_tags_from_prompt(messages_dict_with_history)

                # create and append new message
                new_message = {"role":"user", "content":user_query}
                messages_without_think_tags['messages'].append(new_message)
                
                return messages_without_think_tags
            
            else:   # first message in chat
                if vision:
                    return {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_query}]}]}
                else:
                    if skip_system_prompt:
                        return {"messages": [{"role": "user", "content": json.dumps(user_query)}]}
                        
                    else:
                        return {"messages": [{"role": "system", "content": json.dumps(system_prompt)}, {"role": "user", "content": json.dumps(user_query)}]}

    except Exception as e:
        print("Could not format prompt for hf-waitress in method format-prompt_for_hf_waitress, encountered error: ", e)


def manually_format_prompt_with_prompt_template(formatted_prompt:str, user_query:str, current_sequence_id:int, base_template:str, local_llm_chat_template_format:str, skip_system_prompt=False) -> str:

    print(f"\nFormatting prompt for LMM with template format: {local_llm_chat_template_format}\n")

    if skip_system_prompt:
        base_template = ""

    if local_llm_chat_template_format == 'llama3':

        if current_sequence_id > 0:
            formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            if skip_system_prompt:
                formatted_prompt += f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                formatted_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{base_template}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif local_llm_chat_template_format == 'llama2':

        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"<s>[INST] {user_query} [/INST] "
        else:
            formatted_prompt += f"<s>[INST] <<SYS>>\n {base_template} \n<</SYS>>\n\n {user_query}  [/INST] "

    elif local_llm_chat_template_format == 'chatml':
        
        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt += f"<|im_start|>system\n{base_template}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    
    elif local_llm_chat_template_format == 'qwen-chatml':
        
        if current_sequence_id > 0:
            formatted_prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt += f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n{base_template}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

    elif local_llm_chat_template_format == 'phi3':

        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"<|user|>\n{user_query}<|end|>\n<|assistant|>\n"
        else:
            formatted_prompt += f"<|system|>\n{base_template}<|end|>\n<|user|>\n{user_query}<|end|>\n<|assistant|>\n"

    elif local_llm_chat_template_format == 'phi4':
        
        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"<|im_start|>user<|im_sep|>\n{user_query}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n"
        else:
            formatted_prompt += f"<|im_start|>system<|im_sep|>\n{base_template}<|im_end|>\n<|im_start|>user<|im_sep|>\n{user_query}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n"

    elif local_llm_chat_template_format == 'command-r':

        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_query}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        else:
            formatted_prompt += f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{base_template}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_query}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"

    elif local_llm_chat_template_format == 'deepseek':
        
        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"### Instruction:\n{user_query}\n### Response:\n"
        else:
            formatted_prompt += f"{base_template}### Instruction:\n{user_query}\n### Response:\n"

    elif local_llm_chat_template_format == 'deepseek-coder-v2':
        
        if current_sequence_id > 0:
            formatted_prompt += f"User: {user_query}\nAssistant: "
        else:
            formatted_prompt += f"<|begin_of_sentence|>{base_template}\nUser: {user_query}\nAssistant: "

    elif local_llm_chat_template_format == 'vicuna':

        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"USER: {user_query}\nASSISTANT: "
        else:
            formatted_prompt += f"{base_template}\n\nUSER: {user_query}\nASSISTANT: "

    elif local_llm_chat_template_format == 'openchat':

        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"GPT4 Correct User: {user_query}<|end_of_turn|>GPT4 Correct Assistant: "
        else:
            formatted_prompt += f"<s>GPT4 Correct System: {base_template}<|end_of_turn|>GPT4 Correct User: {user_query}<|end_of_turn|>GPT4 Correct Assistant: "

    elif local_llm_chat_template_format == 'gemma2':

        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted_prompt += f"<start_of_turn>user\n{base_template}\n{user_query}<end_of_turn>\n<start_of_turn>model\n"

    elif local_llm_chat_template_format == 'mistral-small-v7':

        if current_sequence_id > 0:
            formatted_prompt += f"<s>[INST]{user_query}[/INST]"
        else:
            mistral_small_system_prompt = """You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
            Your knowledge base was last updated on 2023-10-01. The current date is 2025-01-30.
            When you're not sure about some information, you say that you don't have the information and don't make up anything.
            If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request 
            (e.g. \"What are some good restaurants around me?\" => \"Where are you?\" or \"When is the next flight to Tokyo\" => \"Where do you travel from?\")
            """
            formatted_prompt  += f"<s>[SYSTEM_PROMPT]{mistral_small_system_prompt}\n{base_template}[/SYSTEM_PROMPT][INST]{user_query}[/INST]"

    elif local_llm_chat_template_format == 'mistral-large-v7':

        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"<s>[INST] {user_query}[/INST] "
        else:
            formatted_prompt  += f"<s>[SYSTEM_PROMPT] {base_template}[/SYSTEM_PROMPT][INST] {user_query}[/INST] "

    elif local_llm_chat_template_format == 'raw':

        if current_sequence_id > 0 or skip_system_prompt:
            formatted_prompt += f"User: {user_query}\nAssistant: "
        else:
            formatted_prompt += f"{base_template}\nUser: {user_query}\nAssistant: "

    return formatted_prompt


def append_eot_token_to_llm_response(local_llm_chat_template_format: str, llm_response: str) -> str:
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


def trim_response(response, start_substring, end_substring, include_start_substring=False, include_end_substring=False):
    print("\nAttempting to trim response...\n")
    try:
        if start_substring in response and end_substring in response:
            start_index = response.rindex(start_substring)  # Sometimes the model re-gurgitates multiple copies of the same dict in it's response
            end_index = response.rindex(end_substring) # rindex() returns the index of the last occurrence of the substring
            
            if not include_start_substring:
                start_index += len(start_substring)
            if include_end_substring:
                end_index += len(end_substring)
            
            return response[start_index:end_index]
        else:
            print(f"\nResponse does not contain start_substring: {start_substring} or end_substring: {end_substring}, returning unchanged response...\n")
            return response
    except Exception as e:
        print(f"Failed to trim response, encountered error: {e}")
        return response
    

#############################----------------Home Assistant Stuff!----------------###############################

def get_core_prompt_for_butler_tools_config(user_query:str, all_services_with_descriptions:dict) -> str:
    return f"""You are a home automation assistant tasked with assisting the user in performing real-world actions, such as turning on/off lights or other appliances such as TVs, setting alarms, controlling thermostats, etc.

    Do keep in mind that the user may use common slang terms when referring to appliances, such as "idiot box" for a TV or "jbl" or "flip" for bluetooth speakers, etc.

    You will be given a user query, and you will need to determine the best service to perform the task basis the tool's description and the user's query.

    Pay attention to negation in the user query, for example, if told "not the idiot box", elect to turn off the TV!

    IMPORTANT: There may be multiple services associated with a single device, such as dedciated "on" and "off" services for a TV, etc.
    
    ### Service List:
    {all_services_with_descriptions}

    ### User Query:
    <user_query>
    {user_query}
    </user_query>

    ### Output Format (Strict JSON):
    ```json
    {{
        "service": "service name here"
    }}
    ```
    NOTE: STATE SERVICE NAME EXACTLY AS IT IS IN THE SERVICE LIST.
    """

def request_action_analysis_prompt(user_query:str, action_result:dict) -> str:
    return f"""A service execution action was attempted in accordance with a user request. Below are both, the user request and the outcome of the attempted service execution action:

    ### User Request:
    <user_request>
    {user_query}
    </user_request>

    ### Action Result:
    <action_result>
    {action_result}
    </action_result>

    Please provide a status update on the service execution action. Was the action successful? If not, any reasons provided as to why not? Thank you!

    Note: Your response will be piped to a text-to-speech (TTS) model, but only once you've finished responding. Accordingly, keep in mind the following: 
    - Respond in a manner conducive to a verbal conversation
    - Avoid technical jargon and verbose language.
    - Do NOT use emojis or symbols, or markdown formatting, as it risks the TTS model reading out the raw markdown or emoji code instead of the intended text!
    - Keep your response concise and to the point.
    - Respond in the past tense (v. imp!) as if the action has already been performed, as it is likely to have been by the time the TTS model verbally synthesizes and replies!
    - If the action was not successful, provide a reason for why not.
    - If the action was successful, provide a brief summary of the action performed.
    """

#############################----------------------------------------------#################################