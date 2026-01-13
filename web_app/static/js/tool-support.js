

function extractAssistantAnswerOnly(assistantMessageElement) {
    const llmResponseEl = assistantMessageElement.querySelector('.llm-response');
    if (!llmResponseEl) return '';

    const clean = llmResponseEl.cloneNode(true);

    // remove reasoning + UI widgets before reading text
    clean.querySelectorAll('.llm-think').forEach(n => n.remove());
    clean.querySelectorAll('.star-rating').forEach(n => n.remove());

    return (clean.innerText || clean.textContent || '').trim();
}

function getMessagesObject(regeneration_request, regen_sequence_id) {
    // TODO: Parse the chat-area to get the user and assistant messages, and return them as a single object in OpenAI API format:
    const sequenceId = regeneration_request ? regen_sequence_id : getSequenceId();
    if (sequenceId == 0) {
        // raise an error
        errorHandlerNoAlert("getting messages object", "getMessagesObject()", "No messages present in the chat-area to get the messages object from.");
        return;
    }

    let messages = [];
    if (!document.getElementById('skip_system_prompt_checkbox').checked) {
        messages.push({
            role: 'system',
            content: document.getElementById('templateContent').value
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
            errorHandlerNoAlert("getting tools schema", "getToolsSchema()", "Error getting tools schema: " + error);
            return null;
        });
}



function getDbSchema() {
    return fetch('/get_db_schema')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (!data.success) {
                throw new Error('Failed to fetch db schema: ' + data.error);
            }
            console.log("db schema successfully fetched: ", data.db_schema);
            return data.db_schema;
        })
        .catch(error => {
            errorHandlerNoAlert("getting db schema", "getDbSchema()", "Error getting db schema: " + error);
            return null;
        });
}


function generateId(str) {
    /**
     * Simple helper to generate a short hash string from content.
     * Gemini3Pro Generated Function: Mimics Python's hash() usage for generating IDs.
     */
    let hash = 0;
    if (str.length === 0) return hash.toString();
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i); // Convert the character to its Unicode code point
        hash = ((hash << 5) - hash) + char; // Bitwise left shift by 5 bits and add the character
        hash = hash & hash; // Bitwise AND with the hash to get the result - creates a 32-bit integer
    }
    // Return absolute hex string cropped to 8 chars to mimic the python slice
    return Math.abs(hash).toString(16).substring(0, 8);
}
function extractToolCallsFromResponse(fullResponseText) {
    /**
     * Extracts tool calls from a text response containing <tool_call> tags.
     * Supports standard JSON or custom <function> XML formats.
     * 
     * @param {string} fullResponseText 
     * @returns {{ role: string, content: string, tool_calls?: Array<any> }}
     */

    try {
        console.log("\n--- Starting tool parsing ---");
        console.log("Full response text: ", fullResponseText);

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

            // --- STRATEGY A: Standard JSON ---
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

            // --- STRATEGY B: Custom XML (<function=name>...) ---
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

        // Return standardized dict
        const result = { 'plain_text': plainText, 'invoke_tools': false, 'tool_calls': [] };
        
        if (toolCalls.length > 0) {
            result['invoke_tools'] = true;
            result['tool_calls'] = toolCalls;
        }
        console.log("result: ", result);
        return result;

    } catch (e) {
        errorHandlerNoAlert("extracting tool calls from response", "extractToolCallsFromResponse()", "Error extracting tool calls from response: " + e.message);
        return { 'invoke_tools': false, 'tool_calls': null };
    }
}


function executeTools(tool_calls, stream_session_id=null) {
    return fetch('/execute_tools', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ 'tool_calls': tool_calls, 'stream_session_id': stream_session_id })
    });
}


function updateMessagesObjectWithToolResult(
    messages_object,
    tool_calls,
    tool_result_list,
    plain_text,
    toolResponseMode = "role_tool"  // "role_tool" | user_tool_response_tag" | "both"
) {
    messages_object.messages.push({
        role: 'assistant',
        content: (tool_calls && tool_calls.length > 0) ? null : plain_text,
        tool_calls: tool_calls
    });

    for (const tool_result of tool_result_list) {
        // OpenAI-style tool role
        if (toolResponseMode == "role_tool" || toolResponseMode == "both") {
            messages_object.messages.push({
                role: 'tool',
                tool_call_id: tool_result.tool_call_id,
                name: tool_result.name,
                content: tool_result.content
            });
        }

        // Model-template-style tool response (for models like Orchestrator-8B, etc. that require <tool_result>...</tool_result> tags in user roles)
        if (toolResponseMode == "user_tool_response_tag" || toolResponseMode == "both") {
            const payload = {
                tool_call_id: tool_result.tool_call_id,
                name: tool_result.name,
                content: tool_result.content
            };

            messages_object.messages.push({
                role: 'user',
                content: `<tool_response>${JSON.stringify(tool_result.content)}</tool_response>`
            });
        }
        
    }
    return messages_object;
}