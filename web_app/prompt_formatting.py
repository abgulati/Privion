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


def get_service_request_prompt(user_query:str):
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

    ---

    ### Guiding Principle:
    Focus on the **core intent**: Is the user asking for specific **data points** (even multiple for comparison) -> **RAG**, or are they asking for **analysis, synthesis, interpretation, or understanding relationships** -> **GraphDB**?

    ---

    ### Decision Flow (Follow these steps):

    1.  **Assess Core Need:** Does the query ask for specific facts/values (RAG) or analysis/synthesis/interpretation (GraphDB)? Check if it's a simple conversation (Direct Response).
    2.  **Consider Keywords (as supporting evidence):**
        *   **Strong RAG indicators:** "What is/was [specific value]...", "When was...", "Who is...", "List...", "Specific figure for...", "Value of X vs Y...", "How much/many...". *Note: "How was [metric]" often implies asking for the value, favouring RAG unless context clearly demands analysis.*
        *   **Strong GraphDB indicators:** "Analyze...", "Summarize...", "Compare [performance/trends/strategies]...", "Why...", "How did [X relate to Y]...", "Trends...", "Overall performance...", "Correlate...", "Relationship between...", "Risks/Opportunities...".
    3.  **Select the Service:** Based on the core need (Step 1) and supported by keywords (Step 2), choose the single most appropriate service. Prioritize the core need over ambiguous keywords.

    ---

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


def manually_format_prompt_with_prompt_template(formatted_prompt:str, user_query:str, current_sequence_id:int, base_template:str, local_llm_chat_template_format:str, skip_system_prompt=False) -> str:

    print(f"\n\nFormatting prompt for LMM with template format: {local_llm_chat_template_format}\n\n")

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
    print("\n\nAttempting to trim response...\n\n")
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
            print(f"\nResponse does not contain start_substring: {start_substring} or end_substring: {end_substring}, returning unchanged response: {response}\n")
            return response
    except Exception as e:
        print(f"Failed to trim response, encountered error: {e}")
        return response