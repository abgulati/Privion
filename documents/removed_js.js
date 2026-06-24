// Old single-turn tool execution cycle in executePrompt():

// let { full_content, plain_text, invoke_tools, tool_calls } = await fetchEventStream(getServerType(), chatState.getForAPI(), responseIDs.responseContentID, chatContainer, file, tools_schema);
// totalContent += full_content;
// if (invoke_tools == true) {
//     traceManager.startStep("Executing Tools...");
//     console.log("invoke_tools is true");
//     const tool_execution_response = await executeTools(tool_calls, stream_session_id);
//     const tool_execution_data = await tool_execution_response.json();
//     const tool_result = tool_execution_data.tool_result_list;
//     const toolResponseMode = document.getElementById('tool_response_mode').value;
//     chatState.addToolExchange(plain_text, tool_calls, tool_result, toolResponseMode);
//     traceManager.startStep("Fetching Follow-Up Response...");
//     let { full_content: followUpFullContent, plain_text: followUpPlainText, invoke_tools: followUpInvokeTools, tool_calls: followUpToolCalls } = await fetchEventStream(getServerType(), chatState.getForAPI(), responseIDs.responseContentID, chatContainer, file);
//     chatState.addAssistantMessage(followUpPlainText);
//     totalContent += followUpFullContent;
// } else {
//     chatState.addAssistantMessage(plain_text);
// }

function sendMessageAndProcessResponseStream() {

    document.getElementById('processingQnS').innerHTML = 'Reading documents...';
    document.getElementById('processingQnS').style.display  = 'block';

    var userInput = document.getElementById('user-input').value;
    //let userInputForHtml = userInput.replace(/\n/g, '<br>');
    let userInputForHtml = formatTabsAndSpaces(userInput);
    
    // Clear the input field
    document.getElementById('user-input').value = '';

    // Append user input to the chat area
    document.getElementById('chat-area').innerHTML += '<div class="user-message">' + userInputForHtml + '</div>';

    // AJAX call to send user query and receive the response
    fetch('/setup_for_streaming_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({'message': userInput})
    }).then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // At this stage, we've passed the user's query to the server and have received an stream_session ID, now we handle the stream
            let stream_session_id = data.stream_session_id;
            let do_rag = data.do_rag;
            //console.log(do_rag);
            
            const responseWrapperID = "ResponseWrapper" + String(stream_session_id);
            const responseContentID = "ResponseContent" + String(stream_session_id)
            const masterWrapperID = "MasterWrapper" + String(stream_session_id);

            document.getElementById('chat-area').innerHTML += `
            <div class="response-and-viewer-container" id="${masterWrapperID}">
                <div class="llm-wrapper" style="display:none;" id="${responseWrapperID}">
                
                    <div class="llm-response" id="${responseContentID}">
                    </div>
                </div>
            </div>
            `
            let chat_container = document.getElementById('chat-area');
            let isResponseDisplayed = false;
            const responseContentElement = document.getElementById(responseContentID);
            const encodedUserInput = encodeURIComponent(userInput);
            const evtSource = new EventSource("/stream/" + stream_session_id + "?input=" + encodedUserInput);
            evtSource.onmessage = function(event) {

                if (!isResponseDisplayed) {
                    document.getElementById('processingQnS').innerHTML = 'Generating...';
                    document.getElementById(responseWrapperID).style.display  = 'block';
                    isResponseDisplayed = true;
                }
                
                // document.getElementById(responseContentID).innerHTML += event.data;
                
                // 'innerHTML' is very inefficent to do repeatedly, as it doesn't simply append but rather re-parses & rebuilds the entire inner content every time! 
                // Instead, using the DOM API as below to create & append elements is much more efficient as it manipulates the DOM by adding a new node, leaving existing nodes untouched. 
                // This is also better from a security perspective as recreating DOM elements via innerHTML can be exploited for XSS!

                let tempDiv = document.createElement('div');
                tempDiv.innerHTML = event.data;

                // streaming response from LLM starts, begin appending to chat-area
                while (tempDiv.firstChild) {
                    //console.log("appending")
                    responseContentElement.appendChild(tempDiv.firstChild); // When 'appendChild' is used on an element already part of the DOM, a copy isn't created, rather it's moved to the new position thus removing that first child from tempDiv here!
                }

                // chatContainer.scrollHeight is the total height of the content within the chat container.
                // chatContainer.clientHeight is the visible height of the chat container.
                // chatContainer.scrollTop is the distance from the top of the chat container to the top of the visible content.
                
                const scrollThreshold = 100; //100px towards the bottom
                const isNearBottom =  chat_container.scrollHeight - chat_container.clientHeight - chat_container.scrollTop < scrollThreshold;   //by calculating this way, we're finding the difference between the total height of the chat area including the invisble part that's overflown (scrollHeight), the visible height of the chat area (clientHeight), and how far down the chat area has been scrolled (scrollTop). If less than the threshold, auto-scroll engages!
                // For example: if the total height is 100px(scrollHeight), 70px is visible (clientHeight), scrollTop increases as we scroll down, so it's 0 at the top and at the very bottom, scrollTop will be equal to scrollHeight - clientHeight = 30px, so the math would equal to 0px and thus within the threshold!
                if (isNearBottom) {
                    console.log("is scrolled to bottom")
                    chat_container.scrollTop = chat_container.scrollHeight;
                }
            }
            evtSource.onerror = function(error) {
                console.error("EventSource failed: ", error);
                evtSource.close();
            }
            evtSource.addEventListener("END", function(event) {
                evtSource.close();

                // Having received the LLM response stream, fetch relevant pages, documents, and images, if any
                if (do_rag) {
                    document.getElementById('processingQnS').innerHTML = 'Fetching any references...';
                }
                fetch('/lc_get_references', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({'stream_session_id': stream_session_id, 'message': userInput})
                }).then(response => {
                    if (!response.ok) {
                        return response.json().then(err => { throw new Error(err.error)});
                    }
                    return response
                })
                .then(response => response.json())
                .then(data => {
                    
                    if (data.success) {
                        //CHAT_ID = data.chat_id;
                        SEQUENCE_ID = data.sequence_id;

                        if (do_rag) {
                            document.getElementById(responseContentID).innerHTML += `
                            </br> 
                            ${data.response}
                            ` 
                        }
                        
                        document.getElementById(responseContentID).innerHTML += `
                        <div class="star-rating" data-rated="False" rating-chat-id=${data.chat_id} rating-sequence-id=${data.sequence_id}>
                            <i class="far fa-star" data-rate="1"></i>
                            <i class="far fa-star" data-rate="2"></i>
                            <i class="far fa-star" data-rate="3"></i>
                            <i class="far fa-star" data-rate="4"></i>
                            <i class="far fa-star" data-rate="5"></i>
                        </div>
                        `
                        if (do_rag) {
                            document.getElementById(masterWrapperID).innerHTML += data.pdf_frame;
                        }

                        document.getElementById('processingQnS').style.display  = 'none';
                        document.getElementById('processingQnS').innerHTML = '';
                    } else {
                        throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
                    }
                })
                .catch(error => {
                    errorHandler("fetching relevant reference material", "/lc_get_references", String(error.message))
                });
            });
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("setting up for the streaming response", "/setup_for_streaming_response", String(error.message))
    });
}




function updateMessagesObjectWithToolResult(
    tool_calls,
    tool_result_list,
    plain_text,
    toolResponseMode = "role_tool"  // "role_tool" | "user_tool_response_tag"
) {
    // 1. HANDLE THE ASSISTANT MESSAGE
    if (toolResponseMode == "role_tool") {
        // Standard OpenAI: Structured tool_calls, content is null (usually)
        MESSAGES_OBJECT['messages'].push({
            role: 'assistant',
            content: null,
            tool_calls: tool_calls
        });
    } else {
        // XML Mode: Custom/Nvidia/Qwen Style Tool Role
        MESSAGES_OBJECT['messages'].push({
            role: 'assistant',
            content: plain_text || "", // Safety fallback
            tool_calls: tool_calls
        });
    }

    for (const tool_result of tool_result_list) {

        // Normalize tool_result.content - ensure objects are stringified, and strings are left alone
        const normalizedContent = typeof tool_result.content === 'object' ? JSON.stringify(tool_result.content) : String(tool_result.content);
        
        if (toolResponseMode == "role_tool") {
            // Standard OpenAI Style Tool Role
            MESSAGES_OBJECT['messages'].push({
                role: 'tool',
                tool_call_id: tool_result.tool_call_id,
                name: tool_result.name,
                content: normalizedContent // OpenAI expects a string here
            });
        }
        else if (toolResponseMode == "user_tool_response_tag") {
            // Custom/Nvidia/Qwen Style Tool Role (for models like Orchestrator-8B, etc. that require <tool_result>...</tool_result> tags in 'user' role)
            MESSAGES_OBJECT['messages'].push({
                role: 'user',
                content: `<tool_response>${normalizedContent}</tool_response>`
            });
        }
        
    }
    // console.log("messages_object: ", messages_object);
    // return messages_object;
}