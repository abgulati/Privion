

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


def manually_format_prompt_with_prompt_template(formatted_prompt:str, user_query:str, current_sequence_id:int, base_template:str, local_llm_chat_template_format:str, skip_system_prompt=False) -> str:

    print("\n\nFormatting prompt for llama-cpp\n\n")

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