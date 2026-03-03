// ES-Module to protect Messages Object from being modified directly from outside the module!

class ConversationState {
    constructor() {
        this.messages = { 'messages': [] };
        this.initialized = false;
    }

    /**
     * Initialize FRESH conversation (adds system prompt if configured)
     */
    init( { sysPrompt, skipSystemPrompt }) {

        
        if (this.initialized || this.messages['messages'].length > 0) {
            console.warn("ConversationState already initialized or messages present, skipping initialization.");
            return;
        }

        if (!skipSystemPrompt && sysPrompt?.trim()) {
            const currentDate = new Date().toISOString();
            this.messages['messages'].push({
                role: 'system',
                content: `The Current Date is: ${currentDate}\n\n${sysPrompt}`
            });
        }

        this.initialized = true;
    }

    /**
     * Hydrate from server/database (loading existing conversation)
     * Replaces current state entirely.
     */
    hydrate(messagesArray) {
        if (!Array.isArray(messagesArray)) {
            console.warn("Invalid messages array provided to hydrate(). Expected an array, got:", messagesArray);
            return;
        }

        this.messages['messages'] = structuredClone(messagesArray);
        this.initialized = true;

        console.log(`Hydrated conversation state with ${messagesArray.length} messages.`);
    }


    clear() {
        this.messages['messages'] = [];
        this.initialized = false;
    }


    addUserMessage(content) {
        if (!content?.trim()) throw new Error("Empty user message rejected");
        this.messages['messages'].push({
            role: 'user',
            content: content
        });
    }


    addAssistantMessage(content) {
        if (!content?.trim()) throw new Error("Empty assistant message rejected");
        this.messages['messages'].push({
            role: 'assistant',
            content: content
        });
    }


    addToolExchange(plainText, toolCalls, toolResultsList, toolResponseMode = "role_tool") {
        
        // 1. HANDLE THE ASSISTANT MESSAGE
        if (toolResponseMode == "role_tool") {
            // Standard OpenAI: Null content
            this.messages['messages'].push({
                role: 'assistant',
                content: null,
                tool_calls: toolCalls
            });
        } else if (toolResponseMode == "user_tool_response_tag") {
            // XML Mode: Custom/Nvidia/Qwen Style Tool Role
            this.messages['messages'].push({
                role: 'assistant',
                content: plainText || "",
                tool_calls: toolCalls
            });
        } else if (toolResponseMode == "qwen_3_5") {
            let tool_calls_list = [];
            for (const toolCall of toolCalls) {
                tool_calls_list.push(toolCall.function);
            }
            this.messages['messages'].push({
                role: 'assistant',
                content: plainText + "\n<|tool_call|>" + JSON.stringify(tool_calls_list)
            });
        }

        // 2. HANDLE THE TOOL EXCHANGES
        toolResultsList.forEach(toolResult => {
            const normalizedContent = typeof toolResult.content === 'object' 
                ? JSON.stringify(toolResult.content) : String(toolResult.content);

            if (toolResponseMode == "role_tool") {
                // Standard OpenAI Style Tool Role
                this.messages['messages'].push({
                    role: 'tool',
                    tool_call_id: toolResult.tool_call_id,
                    name: toolResult.name,
                    content: normalizedContent // OpenAI expects a string here
                });
            } 
            else if (toolResponseMode == "user_tool_response_tag") {
                // Custom/Nvidia/Qwen Style Tool Role (for models like Orchestrator-8B, etc. that require <tool_result>...</tool_result> tags in user roles)
                this.messages['messages'].push({
                    role: 'user',
                    content: `<tool_response>${normalizedContent}</tool_response>`
                });
            } else if (toolResponseMode == "qwen_3_5") {
                this.messages['messages'].push({
                    role: 'tool',
                    content: `{ "name": "${toolResult.name}", "result": "${normalizedContent}" }`
                });
            }

        });
    }


    getForAPI() {
        return structuredClone(this.messages);
    }

    inspect() {
        return structuredClone(this.messages);
    }

}

export const chatState = new ConversationState();