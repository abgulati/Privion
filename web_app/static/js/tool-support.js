

function getToolsSchema() {
    return fetch('/get_tools_schema')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (!data.success) {
                throw new Error('Failed to fetch tools schema: ' + data.error);
            }
            console.log("tools schema successfully fetched: ", data.tools_schema);
            return data.tools_schema;
        })
        .catch(error => {
            errorHandlerNoAlert("getting tools schema", "get-ToolsSchema()", "Error getting tools schema: " + error);
            return null;
        });
}


function executeTools(tool_calls, stream_session_id=null) {
    return fetch('/execute_tools', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ 'tool_calls': tool_calls, 'stream_session_id': stream_session_id })
    });
}



/**
 * -----------------------------------------------------------------------------------------------------------------
 * NOTE: EVERYTHING BELOW THIS POINT HAS BEEN MOVED TO THE hf_waitress.py FILE AS PART OF THE v1-chat-completion API
 * AND IS NO LONGER USED IN THE FRONTEND. IT IS KEPT HERE FOR REFERENCE ONLY, AND MAY BE REMOVED IN THE FUTURE.
 * -----------------------------------------------------------------------------------------------------------------
*/

function extractAssistantAnswerOnly(assistantMessageElement) {
    const llmResponseEl = assistantMessageElement.querySelector('.llm-response');
    if (!llmResponseEl) return '';

    const clean = llmResponseEl.cloneNode(true);

    // remove reasoning + UI widgets before reading text
    clean.querySelectorAll('.llm-think').forEach(n => n.remove());
    clean.querySelectorAll('.star-rating').forEach(n => n.remove());

    return (clean.innerText || clean.textContent || '').trim();
}

function getMessagesObject(regeneration_request, regen_sequence_id) { // LEGACY FUNCTION - 
// USED PRIOR TO COMPLETION OF THE v1-chat-completion API (hf_waitress.py).
    
    const sequenceId = regeneration_request ? regen_sequence_id : getSequenceId();
    if (sequenceId == 0) {s
        // raise an error
        errorHandlerNoAlert("getting messages object", "get-MessagesObject()", "No messages present in the chat-area to get the messages object from.");
        return;
    }

    // Get current Date:
    const currentDate = new Date().toISOString();
    console.log("currentDate: ", currentDate);
    const fullSystemPrompt = "The Current Date is: " + currentDate + "\n\n" + document.getElementById('templateContent').value;

    let messages = [];
    if (!document.getElementById('skip_system_prompt_checkbox').checked) {
        messages.push({
            role: 'system',
            content: fullSystemPrompt
        });
    }

    for (let i = 1; i <= sequenceId; i++) {
        const userMessageElement = document.querySelector(`.user-message[data-sequence-id="${i}"]`);
        const userQueryContent = userMessageElement?.querySelector('.user-query-content').textContent?.trim();
        if (userQueryContent) messages.push({ role: 'user', content: userQueryContent });
        
        const assistantMessageElement = document.querySelector(`.response-and-viewer-container[data-sequence-id="${i}"]`);
        if (assistantMessageElement) {
            const answerOnly = extractAssistantAnswerOnly(assistantMessageElement);
            if (answerOnly) messages.push({ role: 'assistant', content: answerOnly });
        }
    }

    const messages_object = {
        "messages": messages
    }
    console.log("Freshly generated messages object: ", messages_object);
    return messages_object;
}


const llmToExtractorMapping = new Map();

llmToExtractorMapping.set('fallback', fallbackToolsExtractor);

llmToExtractorMapping.set('nvidia/Nemotron-Orchestrator-8B', Qwen3ToolsExtractor);
llmToExtractorMapping.set('Qwen/Qwen3', Qwen3ToolsExtractor);

llmToExtractorMapping.set('Qwen/Qwen3-Coder-Next', Qwen3CoderNextToolsExtractor);
llmToExtractorMapping.set('Qwen/Qwen3.5', Qwen3CoderNextToolsExtractor);

llmToExtractorMapping.set('google/gemma-4', Gemma4ToolsExtractor);


function Qwen3ToolsExtractor(fullResponseText) {
    const toolCalls = [];
        
    // 1. Regex to find ALL occurrences of <tool_call>...</tool_call>
    // [\s\S] is the JS equivalent of Python's re.DOTALL (matches newlines)
    const toolRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
    
    // We use matchAll or a loop with exec to iterate over all <tool_call> tags in the response
    let match;
    while ((match = toolRegex.exec(fullResponseText)) !== null) {

        const rawToolContent = match[1].trim(); // Extract the content inside the <tool_call> tag
        let toolName = null;
        let toolArgs = {};

        try {
            const jsonData = JSON.parse(rawToolContent);

            // Handle case where the content is a LIST of tools inside one tag
            if (Array.isArray(jsonData)) {
                for (const item of jsonData) {
                    toolCalls.push({
                        'id': 'call_' + getUniqueId().substring(0, 8),
                        'type': 'function',
                        'function': {
                            'name': item.name,
                            'arguments': JSON.stringify(item.arguments || {})
                        }
                    });
                }
                continue;
            } 
            // Handle single JSON object
            else {
                toolName = jsonData.name;
                toolArgs = jsonData.arguments || {};

                toolCalls.push({
                    'id': 'call_' + getUniqueId().substring(0, 8),
                    'type': 'function',
                    'function': {
                        'name': toolName,
                        'arguments': JSON.stringify(toolArgs)
                    }
                });
            }
        } catch (e) {
            errorHandlerNoAlert(
                "parsing tool calls",
                "Qwen3-ToolsExtractor()",
                "Tool content looked like JSON but failed to parse."
            );
        }
    }

    // Remove ALL <tool_call> blocks from the text shown to the user
    const plainText = fullResponseText.replace(toolRegex, '').trim();
    
    return [plainText, toolCalls];
}


function Qwen3CoderNextToolsExtractor(fullResponseText) {
    const toolCalls = [];
        
    // 1. Regex to find ALL occurrences of <tool_call>...</tool_call>
    // [\s\S] is the JS equivalent of Python's re.DOTALL (matches newlines)
    const toolRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
    
    // We use matchAll or a loop with exec to iterate over all <tool_call> tags in the response
    let match;
    while ((match = toolRegex.exec(fullResponseText)) !== null) {
        
        const rawToolContent = match[1].trim(); // Extract the content inside the <tool_call> tag

        try {
            // Regex for <function=name>body</function>
            const functionRegex = /<function=(.*?)>([\s\S]*?)<\/function>/g;
            
            // Find all function matches within this tool_call block
            const foundFunctions = [...rawToolContent.matchAll(functionRegex)];

            if (foundFunctions.length > 0) {
                for (const funcMatch of foundFunctions) {
                    const tName = funcMatch[1].trim();
                    const tBody = funcMatch[2].trim();
                    const tArgs = {};

                    // Extract parameters: <parameter=key>value</parameter>
                    const paramRegex = /<parameter=(.*?)>([\s\S]*?)<\/parameter>/g;
                    const paramMatches = [...tBody.matchAll(paramRegex)];

                    for (const pMatch of paramMatches) {
                        const pKey = pMatch[1].trim();
                        const pVal = pMatch[2].trim();
                        tArgs[pKey] = pVal;
                    }

                    // Add to list immediately
                    toolCalls.push({
                        'id': 'call_' + getUniqueId().substring(0, 8),
                        'type': 'function',
                        'function': {
                            'name': tName,
                            'arguments': JSON.stringify(tArgs)
                        }
                    });
                }
            }
        } catch (e) {
            console.log(`Failed to parse custom XML format: ${e}`);
        }
    }

    // Remove ALL <tool_call> blocks from the text shown to the user
    const plainText = fullResponseText.replace(toolRegex, '').trim();

    return [plainText, toolCalls];
}


function Gemma4ToolsExtractor(fullResponseText) {
    const toolCalls = [];
        
    // 1. Regex to find ALL occurrences of <tool_call>...</tool_call>
    // [\s\S] is the JS equivalent of Python's re.DOTALL (matches newlines)
    // Eg: "<|tool_call>call:search{query:<|\"|>weather in Vancouver today<|\"|>}<tool_call|><|tool_call>call:lamp_turn_on{}<tool_call|><|tool_response>"
    const toolRegex = /<\|tool_call>call:([a-zA-Z0-9_-]+)\{([\s\S]*?)\}<tool_call\|>/g;
    const argRegex = /(\w+):(?:<\|"\|>([\s\S]*?)<\|"\|>|([^,]+))/g;
    
    // We use matchAll or a loop with exec to iterate over all <tool_call> tags in the response
    let match;
    while ((match = toolRegex.exec(fullResponseText)) !== null) {
        
        const functionName = match[1];
        const rawArgs = match[2]; // Parse "key:value,key2:value2" into a JS object
        
        const args = {};
        let am;
        while ((am = argRegex.exec(rawArgs)) !== null) {
            const key = am[1].trim();
            const value = am[2].trim();
            args[key] = value;
        }
        
        toolCalls.push({
            'id': 'call_' + getUniqueId().substring(0, 8),
            'type': 'function',
            'function': {
                'name': functionName,
                'arguments': JSON.stringify(args)
            }
        });
    }

    // Remove ALL <tool_call> blocks from the text shown to the user
    const plainText = fullResponseText.replace(toolRegex, '').trim();
    
    return [plainText, toolCalls];
}


function fallbackToolsExtractor(fullResponseText) {
    const toolCalls = [];
    
    // 1. Regex to find ALL occurrences of <tool_call>...</tool_call>
    // [\s\S] is the JS equivalent of Python's re.DOTALL (matches newlines)
    const toolRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
    
    // We use matchAll or a loop with exec to iterate over all <tool_call> tags in the response
    let match;
    while ((match = toolRegex.exec(fullResponseText)) !== null) {
        const rawToolContent = match[1].trim(); // Extract the content inside the <tool_call> tag

        let toolName = null;
        let toolArgs = {};
        let parseSuccess = false;

        // --- STRATEGY A: Standard JSON Eg: [Qwen-3 | nvidia/Nemotron-Orchestrator-8B]---
        if (rawToolContent.startsWith("{") || rawToolContent.startsWith("[")) {
            try {
                const jsonData = JSON.parse(rawToolContent);

                // Handle case where the content is a LIST of tools inside one tag
                if (Array.isArray(jsonData)) {
                    for (const item of jsonData) {
                        toolCalls.push({
                            'id': 'call_' + getUniqueId().substring(0, 8),
                            'type': 'function',
                            'function': {
                                'name': item.name,
                                'arguments': JSON.stringify(item.arguments || {})
                            }
                        });
                    }
                    parseSuccess = true; // We handled it, skip to next match
                    continue;
                } 
                // Handle single JSON object
                else {
                    toolName = jsonData.name;
                    toolArgs = jsonData.arguments || {};

                    toolCalls.push({
                        'id': 'call_' + getUniqueId().substring(0, 8),
                        'type': 'function',
                        'function': {
                            'name': toolName,
                            'arguments': JSON.stringify(toolArgs)
                        }
                    });
                    parseSuccess = true;
                }
            } catch (e) {
                console.log("Tool content looked like JSON but failed to parse.");
                // Fall through to Strategy B
            }
        }

        // --- STRATEGY B: Custom XML (<function=name>...) [Eg: Qwen/Qwen3-Coder-Next]---
        if (!parseSuccess) {
            try {
                // Regex for <function=name>body</function>
                const functionRegex = /<function=(.*?)>([\s\S]*?)<\/function>/g;
                
                // Find all function matches within this tool_call block
                const foundFunctions = [...rawToolContent.matchAll(functionRegex)];

                if (foundFunctions.length > 0) {
                    for (const funcMatch of foundFunctions) {
                        const tName = funcMatch[1].trim();
                        const tBody = funcMatch[2].trim();
                        const tArgs = {};

                        // Extract parameters: <parameter=key>value</parameter>
                        const paramRegex = /<parameter=(.*?)>([\s\S]*?)<\/parameter>/g;
                        const paramMatches = [...tBody.matchAll(paramRegex)];

                        for (const pMatch of paramMatches) {
                            const pKey = pMatch[1].trim();
                            const pVal = pMatch[2].trim();
                            tArgs[pKey] = pVal;
                        }

                        // Add to list immediately
                        toolCalls.push({
                            'id': 'call_' + getUniqueId().substring(0, 8),
                            'type': 'function',
                            'function': {
                                'name': tName,
                                'arguments': JSON.stringify(tArgs)
                            }
                        });
                    }
                    parseSuccess = true;
                }
            } catch (e) {
                console.log(`Failed to parse custom XML format: ${e}`);
            }
        }
    }

    // Remove ALL <tool_call> blocks from the text shown to the user
    const plainText = fullResponseText.replace(toolRegex, '').trim();

    return [plainText, toolCalls];
}


function extractToolCallsFromResponse(fullResponseText) {   // LEGACY FUNCTION - 
// USED PRIOR TO COMPLETION OF THE v1-chat-completion API (hf_waitress.py).

    /**
     * Extracts tool calls from a text response containing <tool_call> tags.
     * Supports standard JSON or custom <function> XML formats.
     * 
     * @param {string} fullResponseText 
     * @returns {{ role: string, content: string, tool_calls?: Array<any> }}
     */
    
    try {
        console.log("\n--- Starting tool parsing ---");

        const llm = getLlmModel();
        let extractor = llmToExtractorMapping.get(llm);
        
        if (!extractor) {
            const llmNameComponents = llm.split('-');

            for (let i = 1; i < llmNameComponents.length; i++) {
                const llmFamily = llmNameComponents.slice(0, i).join('-');
                extractor = llmToExtractorMapping.get(llmFamily);
                if (extractor) break;
            }
        }

        if (!extractor) {
            extractor = llmToExtractorMapping.get('fallback');
        }

        console.log("extractor: ", extractor);
        
        const [plainText, toolCalls] = extractor(fullResponseText);
        const result = { 'plain_text': plainText, 'invoke_tools': false, 'tool_calls': [] };
        if (toolCalls.length > 0) {
            result['invoke_tools'] = true;
            result['tool_calls'] = toolCalls;
        }
        console.log("result: ", result);
        return result;

    } catch (e) {
        errorHandlerNoAlert(
            "extracting tool calls from response", "extract-ToolCallsFromResponse()",
            "Error extracting tool calls from response: " + e.message
        );
        
        return { 'invoke_tools': false, 'tool_calls': null };
    }
}




