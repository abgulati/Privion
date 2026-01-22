

function formatTabsAndSpaces(text, tabSize = 4) {
    // Important to handle user-inputted ampersands so they're not confused with HTML entities such as &lt, &copy etc!
    text = text.replace(/&/g, '&amp;');

    // Replace each tab with tabSize number of &nbsp; nbsp - non-breaking space
    text = text.replace(/\t/g, '&nbsp;'.repeat(tabSize));

    // Replace multiple spaces (2 or more) with equivalent number of &nbsp;
    text = text.replace(/ {2,}/g, (match) => '&nbsp;'.repeat(match.length));

    // Replace newlines with <br>
    text = text.replace(/\n/g, '<br>');

    // Handle user-inputted < and > so they're not confused for HTML elements
    text = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // Ensure you're not loosing newline!
    text = text.replace(/&lt;br&gt;/g, '<br>');

    return text;
}


// ###################---------------LLM Markdown Rendering Pipeline---------------###################

// Markdown + sanitize + syntax highlight pipeline
const md = window.markdownit({
    html: true,       // False will block raw HTML from models
    linkify: true,
    breaks: true,      // respect single newlines as <br> (nice for chat)
    highlight: (str, lang) => {
    try {
        if (lang && window.hljs.getLanguage(lang)) {
            const out = window.hljs.highlight(str, { language: lang, ignoreIllegals: true }).value;
            return `<pre><code class="hljs language-${lang}">${out}</code></pre>`;
        }
    } catch(e) {}
    const esc = md.utils.escapeHtml(str);
    return `<pre><code class="hljs">${esc}</code></pre>`;
    }
});

// Split source into normal Markdown vs <think> blocks, render each safely
function renderMarkdownWithThink(markdown) {
    const src = String(markdown || "");
    const re = /<think>([\s\S]*?)<\/think>/gi;
    let m, last = 0, html = "";
    while ((m = re.exec(src)) !== null) {
      if (m.index > last) html += md.render(src.slice(last, m.index));
      const inner = md.render(m[1] || "");
      html += `<details class="llm-think"><summary>Reasoning</summary><div class="llm-think-body">${inner}</div></details>`;
      last = re.lastIndex;
    }
    if (last < src.length) html += md.render(src.slice(last));
    return html;
  }


function renderMarkdownInto(el, markdownText) {
    const unsafe = renderMarkdownWithThink(markdownText);

    // NOTE: Bypassing for now as our citation-link formatting gets mucked up
    // const safe = DOMPurify.sanitize(unsafe, {
    //   ADD_TAGS: ['details', 'summary'],
    //   ADD_ATTR: ['open'] // optional if you ever set it
    // });
    // el.innerHTML = safe;
    el.innerHTML = unsafe;

    // Open external links in new tabs; keep in-window citations intact
    el.querySelectorAll('a[href]:not(.citation-link)').forEach(a => {
      a.setAttribute('target', '_blank');
      a.setAttribute('rel', 'noopener');
    });
  }

// Streaming state: accumulate raw text and re-render at a low frequency
const streamState = new Map(); // sessionId -> { buffer: string, scheduled: boolean }

function getSessionIdFromResponseContentID(responseContentID) {
    // ResponseContent<stream_session_id>
    return String(responseContentID).replace(/^ResponseContent/, '');
}

function appendStreamChunkAndRender(responseContentID, chunk) {
    const sessionId = getSessionIdFromResponseContentID(responseContentID);
    const state = streamState.get(sessionId) || { buffer: '', scheduled: false };
    state.buffer += (chunk || '');
    streamState.set(sessionId, state);

    /*
    Race-avoidance: re-read the latest state from streamState inside the timeout
    to prevent a stale render. A stale render can occur if finalizeStreamRender
    or any post-stream UI update (e.g., appending star-rating) updates the buffer
    after this timeout was queued., i.e., while the scheduled renderer is still pending!

    Eg: The above is most likely to occur when a response stream from the LLM has just concluded, and so the scheduled renderer is still running,
    but the handle-FetchedReferences method tries to append the star-rating element to the response content element and proceeds to call finalize-StreamRender()
    In this case, while the state obj will be updated, as will the innerHTML of the response content element, the scheduled renderer will still render the stale buffer!

    Contract:
    - Never replace the state object in the Map; mutate its fields (buffer, scheduled).
    - scheduled === true ⇒ exactly one pending timer. As in: while the flag is true, there is already one setTimeout queued; don’t queue another.

    Throttle: 80ms (~12fps). Consider requestAnimationFrame if you want frame-sync.
    */

    if (!state.scheduled) {
        state.scheduled = true; // schedule only a single render timer per session.
        const sid = sessionId;
        setTimeout(() => {
            const cur = streamState.get(sid);   // ensure we always render the newest buffer...
            const el = document.getElementById(responseContentID);
            if (cur && el) renderMarkdownInto(el, cur.buffer);
            if (cur) cur.scheduled = false; // ... and then flip scheduled on the same, current state obj "to maintain the one-timer invariant": at any moment, there is at most one pending render timer per session.
            handleAutoScroll(document.getElementById('chat-area'));
        }, 80); // ~12 FPS; tune as needed
    }

    /*
    ### Renderer explanation
    - `scheduled` is a per-session flag that prevents spawning multiple timers at once.
    - It's a coalescing/throttling flag that ensures only one pending timer per session; this prevents overlapping renders and stale overwrites.
    - “scheduled === true ⇒ exactly one pending timer” means: while the flag is true, there is already one setTimeout queued; don’t queue another.
    - “one-timer invariant” is the rule we keep: at any moment, there is at most one pending render timer per session.

    ### How it works (step-by-step)
    1) A chunk arrives:
    - We append it to `state.buffer`.
    - If `state.scheduled` is false, we set it to true and queue a single `setTimeout(..., 80)`.

    2) More chunks arrive before the timeout fires:
    - We just keep appending to `state.buffer`.
    - We do not queue more timers because `state.scheduled` is true.

    3) The timer fires:
    - It re-reads the current state from the Map, renders the latest `buffer`, then sets `state.scheduled = false`.

    4) Next chunk after that:
    - Sees `scheduled` is false, so it queues the next single timer, and the cycle repeats.

    Effect: many fast chunks are “coalesced” into at most one render per 80ms. This avoids:
    - Excessive renders and jank.
    - Out-of-order/stale renders from multiple overlapping timers.

    If we didn’t have `scheduled`:
    - Every chunk would start its own timer.
    - Timers could fire in a tight cluster, re-rendering multiple times with intermediate buffers, and potentially clobbering UI.

    Invariants in one sentence:
    - Exactly one render timer pending per session at a time; when it fires, it clears the flag so the next burst can schedule the next single render.
    */
}

function finalizeStreamRender(responseContentID) {
    const sessionId = getSessionIdFromResponseContentID(responseContentID);
    const state = streamState.get(sessionId);
    if (!state) return;
    const el = document.getElementById(responseContentID);
    if (el) renderMarkdownInto(el, state.buffer);
}

// ###################---------------END OF LLM Markdown Rendering Pipeline---------------###################

function hideWelcomeScreen() {
    const welcomeScreen = document.getElementById('welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
}

function disableSendButton() {
    document.getElementById('sendButton').disabled = true;
}

function enableSendButton() {
    document.getElementById('sendButton').disabled = false;
}

function displayProcessingStatus(status) {
    const processingQnS = document.getElementById('processingQnS');
    processingQnS.innerHTML = status;
    processingQnS.style.display = status ? 'block' : 'none';
}

function initializePromptRequest() {
    hideWelcomeScreen();
    disableSendButton();
    displayProcessingStatus('Reading documents...');
}

function handleServiceSelectionMessage(llm_set_rag_config, traceManager) {
    if (llm_set_rag_config['do_rag'] && !llm_set_rag_config['perform_graph_rag']) {
        traceManager.startStep("The LLM has Elected to Use Search-Tools. Executing Tech Stack: RAG - Semantic & Lexical Search with Re-Ranking & Filtering...");
    } else if (llm_set_rag_config['do_rag'] && llm_set_rag_config['perform_graph_rag']) {
        traceManager.startStep("The LLM has Elected to Use Deep Research Mode. Executing Tech Stack: RAG [Semantic & Lexical Search] + GraphRAG [In-Depth Analysis] with Re-Ranking & Filtering...");
    } else if (llm_set_rag_config['butler_mode']) {
        traceManager.startStep("Butler Mode Engaged! Navigating Real-World to Perform Requested Action...");
    } else {
        traceManager.startStep("The LLM has Elected to Respond Directly - No Tools Required...");
    }
}

function handleAutoScroll(chatContainer) {
    const scrollThreshold = 100; //100px towards the bottom
    const isNearBottom =  chatContainer.scrollHeight - chatContainer.clientHeight - chatContainer.scrollTop < scrollThreshold;   //by calculating this way, we're finding the difference between the total height of the chat area including the invisble part that's overflown (scrollHeight), the visible height of the chat area (clientHeight), and how far down the chat area has been scrolled (scrollTop). If less than the threshold, auto-scroll engages!
    // For example: if the total height is 100px(scrollHeight), 70px is visible (clientHeight), scrollTop increases as we scroll down, so it's 0 at the top and at the very bottom, scrollTop will be equal to scrollHeight - clientHeight = 30px, so the math would equal to 0px and thus within the threshold!
    if (isNearBottom) {
        //console.log("is scrolled to bottom")
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

function createUserMessageHTML(userInputForHtml){
    const uniqueId = getUniqueId();
    const current_chat_id = getChatId();
    const chatArea = document.getElementById('chat-area');

    // Create a temporary container to build the new element
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = `
        <div class="user-message glassmorphism" data-unique-id="${uniqueId}" data-chat-id="${current_chat_id}">
            <i class="fas fa-chevron-right tool-chain-toggle"></i>
            <div class="user-query-content">${userInputForHtml}</div>
            <div class="tool-chain-trace">
                <!-- Trace items will be added dynamically -->
            </div>
            
            <div class="regenerate-menu">
                <i class="fas fa-ellipsis-v"></i>
                <div class="regenerate-menu-options">
                    <span class="regenerate-menu-option regenerate-option">Regenerate Response</span>
                    <span class="regenerate-menu-option regenerate-with-citations-enabled-option">Regenerate Response with Citations Force Enabled</span>
                    <span class="regenerate-menu-option regenerate-with-citations-disabled-option">Regenerate Response with Citations Force Disabled</span>
                    <span class="regenerate-menu-option delete-option">Delete</span>
                </div>
            </div>
        </div>
    `;

    const userMessageElement = tempDiv.firstElementChild;
    
    // Initialize the traceManager and attach it to the element
    const traceManager = new ToolChainTraceManager(userMessageElement);
    userMessageElement._traceManager = traceManager;

    // Append the fully constructed element to the chat area without destroying anything
    chatArea.appendChild(userMessageElement);

    /*
    It's critical to create the HTML for both, the user query and the response container in append-ResponseContainerToChatArea() via appendChild() and not via innerHTML += because
    in the latter case, the entire chat area is destroyed and recreated, as the browser browser doesn't just add the new element. It performs these steps:
        1. Reads the entire existing HTML of the chat-area into a string.
        2. Appends your new HTML string to it.
        3. Destroys all the old elements inside chat-area and replaces them by parsing the new, combined string from scratch.
    
    This means that the original user-message element, which our traceManager is being attached to below, will be destroyed and recreated every time you add
    the LLM's response container. The traceManager instance in memory will then be trying to add new trace items to a DOM element that no longer exists on the page, 
    rendering it uselessly non-functional.
    
    And as noted in appendContentToResponse(), it's also highly inefficient!
    */

    return uniqueId;
}

function updateChatAreaWithUserInput(userInputForHtml) {
    /*
    The uniqueId is used primarily to set the data-chat-id, data-sequence-id and data-stream-session-id attributes on 
    the user-message element, by the methods immediately below.
    Those methods leverage querySelector to find the user-message element by the uniqueId, and then set attributes on it.
    For regeneration requests, these attributes are already set to the correct chat-id & stream-session-id, so they needen't be set again.
    This is why appendChatIdToUserMessage() & appendStreamSessionIdToUserMessage() are only called if (!regeneration_request).
    In other words, for regen requests, the uniqueID is irrelevant and not used, and instead the previously set stream-session-id is reused!
    The old stream-session-id is obtained by a call to the prepareAttributeForUserMessage() method, which
    is called by a click event listener set at DOM load by initializeRegenerateResponseButton().
    */
    const uniqueId = createUserMessageHTML(userInputForHtml);
    appendLoadingAnimation();
    return uniqueId;
}

function appendChatIdToUserMessage(uniqueId, chat_id) {
    const userMessageElement = document.querySelector(`.user-message[data-unique-id="${uniqueId}"]`);
    userMessageElement.setAttribute('data-chat-id', chat_id);
}

function appendStreamSessionIdToUserMessage(uniqueId, stream_session_id) {
    const userMessageElement = document.querySelector(`.user-message[data-unique-id="${uniqueId}"]`);
    userMessageElement.setAttribute('data-stream-session-id', stream_session_id);
}

function appendSequenceIdToUserMessage(uniqueId, sequence_id) {
    const userMessageElement = document.querySelector(`.user-message[data-unique-id="${uniqueId}"]`);
    userMessageElement.setAttribute('data-sequence-id', sequence_id);
}

function getUserInput() {
    const userInput = document.getElementById('user-input').value;
    document.getElementById('user-input').value = '';
    const file = document.getElementById('textAttachmentInput').files[0];
    if (file) { removeTextAttachment(); }
    return {userInput, file};
}

function shouldAppendContent(streamedContent) {
    const excludeList = ['', '<|eot_id|>', '</s>', '<|im_end|>', '<|end_of_turn|>', '<|EOT|>', '|END_OF_TURN_TOKEN|>', '<|end|>'];
    return !excludeList.includes(streamedContent)
}

function appendContentToResponse(responseContentID, content) {
    const responseContentElement = document.getElementById(responseContentID);

    removeLoadingAnimation();

    // document.getElementById(responseContentID).innerHTML += event.data;
                
    // 'innerHTML' is very inefficent to do repeatedly, as it doesn't simply append but rather re-parses & rebuilds the entire inner content every time! 
    // Instead, using the DOM API as below to create & append elements is much more efficient as it manipulates the DOM by adding a new node, leaving existing nodes untouched. 
    // This is also better from a security perspective as recreating DOM elements via innerHTML can be exploited for XSS!
    let tempDiv = document.createElement('div');
    tempDiv.innerHTML = content;

    // streaming response from LLM starts, begin appending to chat-area
    // The while loop below is used to append all child nodes of tempDiv to responseContentElement. This is necessary because tempDiv.innerHTML = streamed_content creates a new div for each chunk of streamed content, so we need to append each child node individually.
    while (tempDiv.firstChild) {
        responseContentElement.appendChild(tempDiv.firstChild); // If we used appendChild() directly, it would only append the first child node, and the rest would be lost. And we'd end up with a new div for each chunk of streamed content, rather than appending the streamed content directly to responseContentElement.
    }
}


function setupLLMResponse(userInput, file_attached, current_chat_id, regeneration_request, regen_stream_session_id, regen_sequence_id, regenerate_with_citations_force_enabled, regenerate_with_citations_force_disabled) { //no need to async-await here as fetch() inherently returns a promise, so by returning fetch() directly we're providing the same Promise that the async function with await would return.
    return fetch('/determine_service_and_ids_for_query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({'user_query': userInput, 'chat_id': current_chat_id, 'file_attached': file_attached, 'regeneration_request': regeneration_request, 'stream_session_id': regen_stream_session_id, 'sequence_id': regen_sequence_id, 'regenerate_with_citations_force_enabled': regenerate_with_citations_force_enabled, 'regenerate_with_citations_force_disabled': regenerate_with_citations_force_disabled})
    });
}

function invokeTools(stream_session_id, userInput, current_chat_id, llm_set_rag_config, regeneration_request, sequence_id) {
    return fetch('/invoke_tools_for_query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({'stream_session_id': stream_session_id, 'user_query': userInput, 'chat_id': current_chat_id, 'llm_set_rag_config': llm_set_rag_config, 'regeneration_request': regeneration_request, 'sequence_id': sequence_id})
    });
}

async function handleToolUse(stream_session_id, userInput, current_chat_id, llm_set_rag_config, regeneration_request, uniqueId, sequence_id_from_previous_request) {
    const invoke_tools_response = await invokeTools(stream_session_id, userInput, current_chat_id, llm_set_rag_config, regeneration_request, sequence_id_from_previous_request);
    const invoke_tools_data = await invoke_tools_response.json();
    const {reused_stream_session_id, tool_formatted_user_prompt, sequence_id, reconfirmed_server_type} = invoke_tools_data;
    console.log("tool_formatted_user_prompt: ", JSON.parse(tool_formatted_user_prompt));
    setSequenceId(sequence_id);
    if (!regeneration_request) { appendSequenceIdToUserMessage(uniqueId, sequence_id); }
    return {reused_stream_session_id, tool_formatted_user_prompt, reconfirmed_server_type};
}

function appendResponseContainerToChatArea(masterWrapperID, responseWrapperID, responseContentID, stream_session_id, sequence_id) {
    // Not called again for regen requests, as the response container is already present in the chat area.
    const chatArea = document.getElementById('chat-area');
    
    // Create the new element without using innerHTML on the main chat area - SEE COMMENT IN create-UserMessageHTML() TO UNDERSTAND WHY!
    const responseContainer = document.createElement('div');
    responseContainer.className = 'response-and-viewer-container';
    responseContainer.id = masterWrapperID;
    responseContainer.setAttribute('data-stream-session-id', stream_session_id);
    responseContainer.setAttribute('data-sequence-id', sequence_id);
    responseContainer.innerHTML = `
        <div class="llm-wrapper" style="display:none;" id="${responseWrapperID}">
            <div class="llm-response" id="${responseContentID}"></div>
        </div>
    `;

    // Append the new container. This does NOT destroy the existing user-message element and it's associated traceManager.
    chatArea.appendChild(responseContainer);

    document.getElementById(responseWrapperID).style.display  = 'block';
    scrollChatAreaToBottom();
}

function handleSetupResponse(data) {
    const {llm_set_rag_config, stream_session_id, formatted_user_prompt, sequence_id, server_type} = data;   // using object-destructuring (ES6-2015) to extract and set specific values from the data object in a single step!
    const {do_rag, perform_graph_rag, butler_mode} = llm_set_rag_config;
    const tool_use = (do_rag || perform_graph_rag || butler_mode) ? true : false;
    console.log("formatted_user_prompt: ", JSON.parse(formatted_user_prompt));
    setSequenceId(sequence_id);

    const responseIDs = {
        responseWrapperID: `ResponseWrapper${stream_session_id}`,
        responseContentID: `ResponseContent${stream_session_id}`,
        masterWrapperID: `MasterWrapper${stream_session_id}`
    }

    if (!data.regeneration_request) { 
        appendResponseContainerToChatArea(responseIDs.masterWrapperID, responseIDs.responseWrapperID, responseIDs.responseContentID, stream_session_id, sequence_id); 
        appendSequenceIdToUserMessage(data.user_message_html_unique_id, sequence_id);
    }
    displayProcessingStatus('Generating...');

    return { tool_use, llm_set_rag_config, stream_session_id, formatted_user_prompt, responseIDs, sequence_id, server_type };
}


function printErrorToChatArea(responseContentID, error_message) {
    // Escape HTML characters to prevent the browser from interpreting parts of the error as tags
    const safeErrorMessage = String(error_message)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    let userFriendlyErrorMessage = `An error occured when attempting to generate the response. Details follow, contact support if the issue persists. Error: <br><br>${safeErrorMessage}`;
    console.error(userFriendlyErrorMessage);
    // Use div instead of span. span is an inline element and markdown-it breaks it when it encounters newlines/blocks in the error message.
    let spannedErrorMessage = `<div class='chat-area-error-message'>${userFriendlyErrorMessage}</div>`;
    appendStreamChunkAndRender(responseContentID, spannedErrorMessage);
    finalizeStreamRender(responseContentID);
    return spannedErrorMessage;
}


async function fetchLlamacppEventStream(formattedPrompt, responseContentID, chatContainer, tools_schema=null) {
    const host = window.location.hostname;
    const url = `http://${host}:8080/v1/chat/completions`;
    let formattedPromptCopy = structuredClone(formattedPrompt); // objects passed by reference so appending tools to the passed object will make it persistent even when tools_schema is null in a future call!

    const requestData = coerceToObject(formattedPromptCopy, "llama.cpp request");
    if (tools_schema) { requestData.tools = tools_schema; }
    else delete requestData.tools;  // if the incoming formattedPrompt already has a .tools property from any earlier mutation (or from elsewhere), so best to explicitly remove it when tools_schema is null
    requestData.stream = true;
    requestData.temperature = parseFloat(document.getElementById('tempSlider').value);
    requestData.top_k = parseInt(document.getElementById('topkSlider').value);
    requestData.top_p = parseFloat(document.getElementById('toppSlider').value);
    requestData.min_p = parseFloat(document.getElementById('minpSlider').value);
    requestData.n_keep = parseInt(document.getElementById('nkeepSlider').value);
    requestData.repeat_penalty = parseFloat(document.getElementById('repetitionPenaltySlider').value);
    requestData.presence_penalty = parseFloat(document.getElementById('presencePenaltySlider').value);
    requestData.frequency_penalty = parseFloat(document.getElementById('frequencyPenaltySlider').value);

    try {

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        scrollChatAreaToBottom();

        const reader = response.body.getReader();   // To handle the Fetch API's 'Response' object when involving a ReadableStream.  By calling getReader(), a 'ReadableStreamDefaultReader' object is obtained
        let totalContent = '';  //String to accumulate content
        let receivedComplete = false;
        let loaderHidden = false;
        let reasoningContentStream = false;
        let reasoningContentFirstToken = true;

        // Function to process each text chunk
        async function processChunk() {
            let partialData = '';   // Holds partially received JSON strings

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    console.log("Stream complete");
                    break;
                }

                const textChunk = new TextDecoder("utf-8").decode(value);   //When reading a stream with 'ReadableStreamDefaultReader', Uint8Array binary-data objects are received. TextDecoder decodes these byte streams into human readable text strings. UTF-8 encodes all possible chars in Unicode, and is the std text encoding in most network comms, thus is used here.
                const messages = textChunk.split('\n'); //one streamed data-message at a time, newlines are standard for seperating SSE-messages which may arrive bunched-up in chunks or partially
                
                messages.forEach(message => {
                    if (message.startsWith('data: ')) {
                        
                        if (!loaderHidden) {
                            removeLoadingAnimation();
                            loaderHidden = true;
                        }
                        
                        const jsonStr = message.slice(6);   // remove 6 chars to get rid of the 'data: ' prefix!
                        try {
                            if (jsonStr == '[DONE]') {
                                receivedComplete = true;
                            } else {
                                const dataObj = JSON.parse(jsonStr);
                                let streamedText = "";
                                if (dataObj.choices[0].delta.reasoning_content) {
                                    if (reasoningContentFirstToken) {
                                        reasoningContentStream = true;
                                        streamedText = '<think>' + dataObj.choices[0].delta.reasoning_content;
                                        reasoningContentFirstToken = false;
                                    } else {
                                        streamedText = dataObj.choices[0].delta.reasoning_content;
                                    }
                                } else {
                                    if (reasoningContentStream) {
                                        streamedText = '</think>' + dataObj.choices[0].delta.content || "";
                                        reasoningContentStream = false;
                                    } else {
                                        streamedText = dataObj.choices[0].delta.content || "";
                                    }
                                }
                                appendStreamChunkAndRender(responseContentID, streamedText);
                                totalContent += streamedText;
                                handleAutoScroll(chatContainer);
                            }
                        } catch (error) {
                            console.error('Error parsing JSON: ', error);
                        }
                    }
                });

                if (receivedComplete) {
                    finalizeStreamRender(responseContentID);
                    break;
                }
            }
        }

        await processChunk();   // processChunk() is an async-Fn and thus returns a promise. Here, via await, we pause execution until the promise is resolved or rejected.
        return totalContent;

    } catch (error) {
        errorHandlerNoAlert("fetching llama.cpp event-streaming response", "localhost:8080/completions", String(error));
        return printErrorToChatArea(responseContentID, String(error));
    }
}

// modify to open the file in a new tab instead of downloading it
function createDownloadLinkForFile(filename) {
    const downloadLink = document.createElement('a');
    const hfWaitress_URL = getHfwUrl();
    downloadLink.href = `${hfWaitress_URL}/serve_uploaded_file/${filename}`;
    downloadLink.target = '_blank';
    downloadLink.download = filename;   // HTML5 attribute that specifies that the linked resource should be downloaded when a user clicks on the hyperlink
    downloadLink.innerHTML = `<i class="fa-solid fa-file-arrow-down"></i> ${filename} Appended to Conversation`;
    downloadLink.classList.add('plain-text-download-link-for-vision-appended-file');
    return downloadLink;
}


function createDownloadContainerForFile(filename) {
    filename = secure_filename(filename);       // get secure filename similar to how it's done in Python - to prevent directory traversal attacks. secure_filename() is implemented in helper-functions.js
    const downloadContainer = document.createElement('div');
    downloadContainer.classList.add('download-container-for-vision-appended-file');

    const downloadLink = document.createElement('a');
    const hfWaitress_URL = getHfwUrl();
    downloadLink.href = `${hfWaitress_URL}/serve_uploaded_file/${filename}`;
    downloadLink.target = '_blank';
    downloadLink.download = filename;   // HTML5 attribute that specifies that the linked resource should be downloaded when a user clicks on the hyperlink
    downloadLink.classList.add('download-link-for-vision-appended-file');

    const iconContainer = document.createElement('div');
    iconContainer.classList.add('download-icon-for-vision-appended-file');
    iconContainer.innerHTML = '<i class="fa-solid fa-download"></i>';

    const textContainer = document.createElement('div');
    textContainer.classList.add('download-text-for-vision-appended-file');

    const fileNameElement = document.createElement('div');
    fileNameElement.classList.add('download-filename-for-vision-appended-file');
    fileNameElement.innerHTML = filename;

    const downnloadText = document.createElement('div');
    downnloadText.classList.add('download-label-for-vision-appended-file');
    downnloadText.innerHTML = 'Download';

    textContainer.appendChild(fileNameElement);
    textContainer.appendChild(downnloadText);

    downloadLink.appendChild(iconContainer);
    downloadLink.appendChild(textContainer);

    downloadContainer.appendChild(downloadLink);

    return downloadContainer;
}


async function fetchHfWaitressEventStream(formattedPrompt, responseContentID, chatContainer, file=null, tools_schema=null) {
    let url;
    let hfwHeaders = new Headers();
    let formdata = null;
    let rawBodyJSONStringified = null;
    let formattedPromptCopy = structuredClone(formattedPrompt); // objects passed by reference so appending tools to the passed object will make it persistent even when tools_schema is null in a future call!

    const vision = getVision();
    const exl2 = getExl2();
    const exl3 = getExl3();
    if (vision === "true") {
        console.log("Invoking vision_stream");
        const hfWaitress_URL = getHfwUrl();
        url = `${hfWaitress_URL}/vision_stream`;

        hfwHeaders = new Headers();
        hfwHeaders.append("X-DPI", "300");
        hfwHeaders.append("X-Max-New-Tokens", document.getElementById('HfwMaxNewToks').value);

        formdata = new FormData();

        const parsedPrompt = coerceToObject(formattedPromptCopy, "vision prompt");
        formdata.append("messages", JSON.stringify(parsedPrompt.messages));
        if (file) { formdata.append("file", file); }
    } else if (exl2 === "true") {
        console.log("Invoking exl2_stream");
        const hfWaitress_URL = getHfwUrl();
        url = `${hfWaitress_URL}/exl2_stream`;

        hfwHeaders = new Headers();
        hfwHeaders.append("Content-Type", "application/json");
        hfwHeaders.append("X-Max-New-Tokens", document.getElementById('HfwMaxNewToks').value);
        hfwHeaders.append("X-Temperature", document.getElementById('HfwTempSlider').value);
        hfwHeaders.append("X-Top-K", document.getElementById('HfwTopkSlider').value);
        hfwHeaders.append("X-Top-P", document.getElementById('HfwToppSlider').value);

        const requestObj = coerceToObject(formattedPromptCopy, "exl2 request");
        if (tools_schema) requestObj.tools = tools_schema;
        else delete requestObj.tools;
        rawBodyJSONStringified = JSON.stringify(requestObj);

    } else if (exl3 === "true") {
        console.log("Invoking exl3_stream");
        const hfWaitress_URL = getHfwUrl();
        url = `${hfWaitress_URL}/exl3_stream`;

        hfwHeaders = new Headers();
        hfwHeaders.append("Content-Type", "application/json");
        hfwHeaders.append("X-Max-New-Tokens", document.getElementById('HfwMaxNewToks').value);
        hfwHeaders.append("X-Temperature", document.getElementById('HfwTempSlider').value);
        hfwHeaders.append("X-Top-K", document.getElementById('HfwTopkSlider').value);
        hfwHeaders.append("X-Top-P", document.getElementById('HfwToppSlider').value);
        hfwHeaders.append("X-Min-P", document.getElementById('HfwMinpSlider').value);
        hfwHeaders.append("X-Repetition-Penalty", document.getElementById('HfwRepetitionPenaltySlider').value);
        hfwHeaders.append("X-Presence-Penalty", document.getElementById('HfwPresencePenaltySlider').value);
        hfwHeaders.append("X-Frequency-Penalty", document.getElementById('HfwFrequencyPenaltySlider').value);

        const requestObj = coerceToObject(formattedPromptCopy, "exl3 request");
        console.log("tools_schema: ", tools_schema);
        if (tools_schema) requestObj.tools = tools_schema;
        else delete requestObj.tools;
        rawBodyJSONStringified = JSON.stringify(requestObj);

    } else {
        console.log("Invoking completions_stream");
        const hfWaitress_URL = getHfwUrl();
        url = `${hfWaitress_URL}/completions_stream`;

        hfwHeaders = new Headers();
        hfwHeaders.append("Content-Type", "application/json");
        hfwHeaders.append("X-Max-New-Tokens", document.getElementById('HfwMaxNewToks').value);
        hfwHeaders.append("X-Temperature", document.getElementById('HfwTempSlider').value);
        hfwHeaders.append("X-Top-K", document.getElementById('HfwTopkSlider').value);
        hfwHeaders.append("X-Top-P", document.getElementById('HfwToppSlider').value);
        hfwHeaders.append("X-Min-P", document.getElementById('HfwMinpSlider').value);
        hfwHeaders.append("X-Do-Sample", document.getElementById('HfwTempSlider').value > 0 ? "True" : "False");
        
        const requestObj = coerceToObject(formattedPromptCopy, "completions request");
        if (tools_schema) requestObj.tools = tools_schema;
        else delete requestObj.tools;
        rawBodyJSONStringified = JSON.stringify(requestObj);
    }
    
    try {

        const request_body = vision === "true" ? formdata : rawBodyJSONStringified;
        console.log("request_body: ", request_body);

        const hfwResponse = await fetch(url, {
            method: 'POST',
            headers: hfwHeaders,
            body: request_body,
            redirect: 'follow'
        }); // due to the async-await syntax, the fetch() call returns a promise, and we await its resolution here.

        if (!hfwResponse.ok) {
            const err = await hfwResponse.json();
            throw new Error(err.error);
        }

        if (file) {
            const downloadContainer = createDownloadContainerForFile(file.name);
            document.getElementById(responseContentID).appendChild(downloadContainer);
        }

        document.getElementById('chat-area').scrollTop = document.getElementById('chat-area').scrollHeight;     //Scroll to the bottom of the page

        const hfwReader = hfwResponse.body.getReader();
        let hfwTotalContent = '';
        if (file) { hfwTotalContent += document.getElementById(responseContentID).innerHTML + '<br>'; } // Ensure the file download link generated above is appended to hfwTotalContent!
        let hfwReceivedComplete = false;
        let loaderHidden = false;

        async function hfwProcessChunk() {
            while (true) {
                const { done, value } = await hfwReader.read();
                if (done) {
                    console.log("HF-Waitress Stream complete");
                    break;
                }
                
                const textChunk = new TextDecoder("utf-8").decode(value);
                const messages = textChunk.split('\n');
                
                messages.forEach(message => {
                    
                    if (message.startsWith('data: ')) {

                        if (!loaderHidden) {
                            removeLoadingAnimation();
                            loaderHidden = true;
                        }

                        const payload = message.slice(6).trim();    // remove the 'data: ' prefix but keep the quoted string for JSON.parse to decode
                        let chunkText = "";
                        try{
                            chunkText = JSON.parse(payload);    // payload is usually a quoted JSON string with escapes; JSON.parse decodes \n, \t, \uXXXX safely
                        } catch {   // Fallback: strip outer quotes if present
                            chunkText = payload.replace(/^"/, '').replace(/"$/, '');
                        }

                        if (chunkText == "null") {
                            console.log("Stream Complete - Received null payload");
                            hfwReceivedComplete = true;
                        } else {
                            appendStreamChunkAndRender(responseContentID, chunkText);
                            hfwTotalContent += chunkText;
                            handleAutoScroll(chatContainer);
                        }

                    } else if (message.startsWith('event: END') || message.startsWith('data: null')) {
                        hfwReceivedComplete = true;
                    }
                });

                if (hfwReceivedComplete) {
                    finalizeStreamRender(responseContentID);
                    break;
                }
            }
        }

        await hfwProcessChunk();
        return hfwTotalContent;

    } catch (error) {
        errorHandlerNoAlert("fetching event-streaming response", "fetchHfWaitressEventStream", String(error));
        return printErrorToChatArea(responseContentID, String(error));
    }

}

async function fetchHfwDiffusersEventStream(formattedPrompt, responseContentID, chatContainer) {
    const hfWaitress_URL = getHfwUrl();
    const url = `${hfWaitress_URL}/completions`;

    const hfwHeaders = new Headers();
    hfwHeaders.append("Content-Type", "application/json");
    hfwHeaders.append("X-Guidance-Scale", document.getElementById('HfDGuidanceScaleSlider').value);
    hfwHeaders.append("X-Height", document.getElementById('HfDHeight').value);
    hfwHeaders.append("X-Width", document.getElementById('HfDWidth').value);
    hfwHeaders.append("X-Num-Inference-Steps", document.getElementById('HfDNumInfSteps').value);
    hfwHeaders.append("X-Max-Sequence-Length", document.getElementById('HfDSeqLen').value);
    hfwHeaders.append("X-Num-Images-Per-Prompt", document.getElementById('HfDNumImgPerPrompt').value);

    const rawBodyJSONObj = JSON.parse(formattedPrompt);                                
    const rawBodyJSONStringified = JSON.stringify(rawBodyJSONObj);

    console.log("rawBodyJSONObj: ", rawBodyJSONObj);
    console.log("rawBodyJSONStringified: ", rawBodyJSONStringified);

    try {
        const hfwResponse = await fetch(url, {
            method: 'POST',
            headers: hfwHeaders,
            body: rawBodyJSONStringified,
            redirect: 'follow'
        });

        if (!hfwResponse.ok) {
            const err = await hfwResponse.json();
            throw new Error(err.error);
        }

        const data = await hfwResponse.json();

        if (data.success) {
            console.log("Diffusers Image Generation Successful");
            console.log("data: ", data);
            const hfWaitress_URL = getHfwUrl();

            // create image div with source = data.image_name (local file path) and append to responseContentID.innerHTML
            const imageDiv = document.createElement('div');
            const img = document.createElement('img');
            img.alt = "Generated Image";
            img.src = `${hfWaitress_URL}/serve_generated_image/${data.image_name}`;
            imageDiv.appendChild(img);
            document.getElementById(responseContentID).appendChild(imageDiv);
            removeLoadingAnimation();
            handleAutoScroll(chatContainer);
            // console.log("returning imageDiv.outerHTML: ", imageDiv.outerHTML);
            return imageDiv.outerHTML;
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    } catch (error) {
        errorHandlerNoAlert("fetching diffusers response", "HF-Waitress/completions", String(error));
        return printErrorToChatArea(responseContentID, String(error));
    }
    
}

async function fetchEventStream(serverType, formattedPrompt, responseContentID, chatContainer, file=null, tools_schema=null) {
    if (serverType == 'llama-cpp') {
        return fetchLlamacppEventStream(formattedPrompt, responseContentID, chatContainer, tools_schema);
    } else if (serverType == 'hf-waitress') {
        return fetchHfWaitressEventStream(formattedPrompt, responseContentID, chatContainer, file=null, tools_schema=tools_schema);
    } else if (serverType == 'hfw-diffusers') {
        return fetchHfwDiffusersEventStream(formattedPrompt, responseContentID, chatContainer);
    } else if (serverType == 'hfw-vision') {
        return fetchHfWaitressEventStream(formattedPrompt, responseContentID, chatContainer, file, tools_schema);
    } else {
        throw new Error(`Invalid server type: ${serverType}`);
    }
}


function handleFetchedReferences(do_rag, data, responseContentID, masterWrapperID, stream_session_id, user_message_html_unique_id, regeneration_request) {
    const latest_sequence_id = getSequenceId();
    const current_chat_id = data.chat_id;

    const chatState_chat_id = document.getElementById('chatState').getAttribute('data-ongoing-chat-id');
    if (chatState_chat_id != current_chat_id) {
        console.log("Updating chatID in the InfoBox");
        setChatId(current_chat_id);
        setModelHeaderInfoBox(current_chat_id, getLlmModel());
        if (!regeneration_request) { appendChatIdToUserMessage(user_message_html_unique_id, current_chat_id); }
    }

    if (parseInt(latest_sequence_id) == 1 && document.querySelector(`.nav-item[data-chat-id="${getChatId()}"]`) == null) {
        const sidenav = document.getElementById('sidenav-content');

        new_chat = {
            'chat_id': current_chat_id,
            'local_llm_server': data.local_llm_server,
            'chat_name': data.stored_datetime,
            'date_time': data.stored_datetime,
            'prompt_template_format': data.local_llm_chat_template_format
        }

        const newDiv = createChatHistoryMenuItem(new_chat);

        const newChatLink = document.getElementById('dynamicChatLink');
        if (newChatLink.nextSibling) {
            sidenav.insertBefore(newDiv, newChatLink.nextSibling);
        } else {
            sidenav.appendChild(newDiv);
        }
    }

    const llm_star_rating_html_parts = [
        '<br>',
        `<div class="star-rating" data-rated="False" data-rating-chat-id="${current_chat_id}" data-rating-sequence-id="${latest_sequence_id}">`,
        '<i class="far fa-star" data-rate="1"></i>',
        '<i class="far fa-star" data-rate="2"></i>',
        '<i class="far fa-star" data-rate="3"></i>',
        '<i class="far fa-star" data-rate="4"></i>',
        '<i class="far fa-star" data-rate="5"></i>',
        '</div>'
    ];  // Using array and join to avoid newline characters \n's appearing in the HTML output as breakline tags
    const llm_star_rating_full_div = llm_star_rating_html_parts.join('');   // Join the array elements into a single string

    const el = document.getElementById(responseContentID);
    const finalResponse = (data?.response && data.response.length > 0) ? data.response : el.innerHTML;
    /* Notes:
        - data.get(...) won't work on JSON objects, and Map.get won't take a default, thus the above
        - data?.response elegantly handles both “data might not exist” and “response might not exist,” yielding undefined instead of throwing.
        - `?.` is called the "optional chaining operator" and is used to safely access properties of an object that might not exist.
        - We could have also used "nullish coalescing operator" `??` E.g. `data?.response ?? el.innerHTML`
        - `??` falls back to the right-hand side if the left-hand side is null or undefined, but not if it's false, 0, or an empty string
    */
    
    // Never replace the state object, mutate it in place to prevent race conditions with any already-scheduled renders - just ensure...
    let stateObj = streamState.get(stream_session_id);   // ...the scheduled renderer in append-StreamChunkAndRender() reads the latest state from the map!
    if (!stateObj) {
        stateObj = { buffer: '', scheduled: false };
        streamState.set(stream_session_id, stateObj);
    }
    stateObj.buffer = '';
    document.getElementById(responseContentID).innerHTML = `${finalResponse}${llm_star_rating_full_div}`;
    stateObj.buffer = `${finalResponse}${llm_star_rating_full_div}`;
    stateObj.scheduled = false;
    finalizeStreamRender(responseContentID);

    if (do_rag && data.pdf_frame != "" && data.pdf_frame != null) {
        document.getElementById(masterWrapperID).innerHTML += data.pdf_frame;
        var defaultTabs = document.getElementsByClassName("defaultTabs");
        for (let i = 0; i < defaultTabs.length; i++) {
            if (defaultTabs[i].getAttribute('stream-session-id') === stream_session_id) {
                defaultTabs[i].click(); // Open the first tab by default
            }
        }
    }
}

async function getReferences(do_rag, params, responseContentID, masterWrapperID, user_message_html_unique_id) {
    try {
        const response = await fetch('/get_references', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(`Internal Server Error: Check server-log and server command-line for more details. Error: ${data.error}`);
        }

        handleFetchedReferences(do_rag, data, responseContentID, masterWrapperID, params.stream_session_id, user_message_html_unique_id, params.regeneration_request);
    } catch (error) {
        errorHandlerNoAlert("fetching relevant reference material", "get-References()", String(error));
        // return printErrorToChatArea(responseContentID, String(error));
    }
}

let currentTtsSourceNode = null;

function stopTTSPlayback() {
    if (currentTtsSourceNode) {
        try {
            currentTtsSourceNode.stop();
            console.log("TTS playback stopped");
        } catch (error) {
            console.error('Error stopping TTS playback:', error);
        } finally {
            currentTtsSourceNode = null;
        }
    }
}
window.stopTTSPlayback = stopTTSPlayback;   // make it available globally for the ASGI server to call

async function fetchTTSVoice(text) {
    // Ensure we have an active AudioContext. If not, we can't play audio this way.
    if (!audioContext || audioContext.state !== 'running') {
        console.warn('AudioContext is not active. TTS audio playback will not work.');
        return;
    }

    // Stop any current TTS playback before starting a new one
    stopTTSPlayback();

    let ttsText = text.includes("</think>") ? text.split("</think>")[1].trim() : text;
    
    // remove all special characters from ttsText, except for spaces, newlines, tabs, commas, question and exclamation marks, and periods
    ttsText = ttsText.replace(/[^a-zA-Z0-9\s\n\t\.,!?"']/g, '');
    console.log("ttsText for AudioContext to play TTS voice: ", ttsText);

    try {
        const response = await fetch('/tts_voice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({'text': ttsText})
        });
        if (!response.ok) {
            const err = await response.json();
            console.error(err.error);
            return;
        }

        // 1. Get the audio data as an ArrayBuffer
        const audioData = await response.arrayBuffer();

        // 2. Decode the audio data into a usable format for the WebAudio API
        const audioBuffer = await audioContext.decodeAudioData(audioData);

        // 3. Create a source node to play the buffer
        const ttsSourceNode = audioContext.createBufferSource();
        ttsSourceNode.buffer = audioBuffer;

        // 4. Connect the TTS source directly to the audio context's destination (the speakers)
        ttsSourceNode.connect(audioContext.destination);

        // Store reference to allow interruptions
        currentTtsSourceNode = ttsSourceNode;

        // 5. Start playback
        ttsSourceNode.start();

        // Clean up when playback completes
        ttsSourceNode.onended = () => {
            if (currentTtsSourceNode === ttsSourceNode) {
                currentTtsSourceNode = null;
            }
        };

    } catch (error) {
        console.error('Error fetching or playing TTS voice:', error);
    }
}


async function requestFormattedPrompt(
    regeneration_request=false, 
    regenerate_with_citations_force_enabled=false, 
    regenerate_with_citations_force_disabled=false, 
    regen_stream_session_id=null, 
    regen_sequence_id=null
) {
    initializePromptRequest();

    const current_chat_id = getChatId();
    const {userInput, file} = getUserInput();
    const userInputForHtml = regeneration_request ? userInput : formatTabsAndSpaces(userInput); // Old input need-not be re-formatted!
    const uniqueId = regeneration_request ? getUniqueId() : updateChatAreaWithUserInput(userInputForHtml);  // The use of the uniqueId is explained in the docstring of update-ChatAreaWithUserInput().
    
    const userMessageElement = regeneration_request ? document.querySelector(`.user-message[data-stream-session-id="${regen_stream_session_id}"]`) : document.querySelector(`.user-message[data-unique-id="${uniqueId}"]`);
    const traceManager = userMessageElement._traceManager;  // created and stored in the userMessageElement at time of create-UserMessageHTML()!
    
    if (regeneration_request) { resetResponseAndViewerContainerWithStreamSessionId(regen_stream_session_id); }

    let file_attached = false;
    if (file) { file_attached = true; }

    try {
        // 1- setup-for_local_llm_response
        traceManager.startStep("Analyzing Query & Determining Required Tools and Settings...");
        const setup_response = await setupLLMResponse(userInput, file_attached, current_chat_id, regeneration_request, regen_stream_session_id, regen_sequence_id, regenerate_with_citations_force_enabled, regenerate_with_citations_force_disabled);
        const setup_data = await setup_response.json();
        setup_data.user_message_html_unique_id = uniqueId;
        setup_data.regeneration_request = regeneration_request;
        const {tool_use, llm_set_rag_config, stream_session_id, formatted_user_prompt, responseIDs, sequence_id, server_type} = handleSetupResponse(setup_data);
        
        let final_stream_session_id = stream_session_id, final_formatted_user_prompt = formatted_user_prompt, final_server_type = server_type;
        handleServiceSelectionMessage(llm_set_rag_config, traceManager);

        // 1b- invoke-tools_if_necessitated_by_llm
        tool_use && ({reused_stream_session_id: final_stream_session_id, tool_formatted_user_prompt: final_formatted_user_prompt, reconfirmed_server_type: final_server_type} = await handleToolUse(stream_session_id, userInput, current_chat_id, llm_set_rag_config, regeneration_request, uniqueId, sequence_id));

        if (!regeneration_request) { 
            appendStreamSessionIdToUserMessage(uniqueId, final_stream_session_id); 
        }

        // 2- fetch-EventStream()
        traceManager.startStep('Generating Response...');
        const chatContainer = document.getElementById('chat-area');
        const totalContent = await fetchEventStream(final_server_type, final_formatted_user_prompt, responseIDs.responseContentID, chatContainer, file);

        // 3- get-References()
        if (tool_use && !llm_set_rag_config.butler_mode) {
            traceManager.startStep('Fetching Document References...');
            displayProcessingStatus('Fetching any references...');
        }

        const getReferencesParams = {
            'chat_id':current_chat_id,
            'sequence_id': getSequenceId(),
            'stream_session_id': final_stream_session_id,
            'user_query': userInputForHtml,
            'user_query_html': userMessageElement.outerHTML,
            'llm_response': totalContent,
            'formatted_user_prompt': final_formatted_user_prompt,
            'regeneration_request': regeneration_request,
            'regenerate_with_citations_force_enabled': regenerate_with_citations_force_enabled
        };

        await getReferences(tool_use, getReferencesParams, responseIDs.responseContentID, responseIDs.masterWrapperID, uniqueId);

        // Complete final step
        traceManager.completeCurrentStep();

        if (document.getElementById('enable_tts').checked) { await fetchTTSVoice(totalContent); }   // Not using getTts() here so simple checkbox change is enough, rather than a config save!

    } catch (error) {
        // Stop timer on error
        if (traceManager) {
            traceManager.completeCurrentStep();
        }
        errorHandlerNoAlert("chatting with the LLM", "request-FormattedPrompt()", String(error.message));
    } finally {
        enableSendButton();
        displayProcessingStatus(false);
    }
}


async function executePrompt(
    regeneration_request=false, 
    regenerate_with_citations_force_enabled=false, 
    regenerate_with_citations_force_disabled=false, 
    regen_stream_session_id=null, 
    regen_sequence_id=null
) {
    
    if (getLegacyMode() === 'true') {
        return await requestFormattedPrompt(
            regeneration_request,
            regenerate_with_citations_force_enabled,
            regenerate_with_citations_force_disabled,
            regen_stream_session_id,
            regen_sequence_id
        );  // must await so we don't simply return a promise immediately, rather wait for the request-FormattedPrompt() to complete!
    }

    initializePromptRequest();
        
    const current_sequence_id = regeneration_request ? regen_sequence_id : incrementSequenceId();
    const current_chat_id = getChatId();
    const {userInput, file} = getUserInput();
    const userInputForHtml = regeneration_request ? userInput : formatTabsAndSpaces(userInput); // Old input need-not be re-formatted!
    const uniqueId = regeneration_request ? getUniqueId() : updateChatAreaWithUserInput(userInputForHtml);  // The use of the uniqueId is explained in the docstring of updateChatAreaWithUserInput().
    
    const userMessageElement = regeneration_request ? document.querySelector(`.user-message[data-stream-session-id="${regen_stream_session_id}"]`) : document.querySelector(`.user-message[data-unique-id="${uniqueId}"]`);
    const traceManager = userMessageElement._traceManager;  // created and stored in the userMessageElement at time of create-UserMessageHTML()!
    
    if (regeneration_request) { resetResponseAndViewerContainerWithStreamSessionId(regen_stream_session_id); }

    let file_attached = false;
    if (file) { file_attached = true; }

    try {
        traceManager.startStep("Processing Prompt...");

        const stream_session_id = regeneration_request? regen_stream_session_id : getUniqueId();
        const responseIDs = {
            responseWrapperID: `ResponseWrapper${stream_session_id}`,
            responseContentID: `ResponseContent${stream_session_id}`,
            masterWrapperID: `MasterWrapper${stream_session_id}`
        }
        
        if (!regeneration_request) {
            appendSequenceIdToUserMessage(uniqueId, current_sequence_id);
            appendStreamSessionIdToUserMessage(uniqueId, stream_session_id);
            appendResponseContainerToChatArea(responseIDs.masterWrapperID, responseIDs.responseWrapperID, responseIDs.responseContentID, stream_session_id, current_sequence_id);
        }
        
        let apiMessages = getMessagesObject(regeneration_request, regen_sequence_id);
        const tools_schema = await getToolsSchema();
        const db_schema = await getDbSchema();

        if (db_schema) {
            const dbContext = `\n\nHere is a list of all the devices, their types, ip addresses, current states, room location, and zone location available in the smart home:\n${JSON.stringify(db_schema, null, 2)}`;
            if (apiMessages.messages.length > 0 && apiMessages.messages[0].role === 'system') {
                apiMessages.messages[0].content += dbContext;
            } else {
                apiMessages.messages.unshift({ role: 'system', content: dbContext });
            }
        }

        const chatContainer = document.getElementById('chat-area');
        let totalContent = await fetchEventStream(getServerType(), apiMessages, responseIDs.responseContentID, chatContainer, file, tools_schema);
        const { plain_text, invoke_tools, tool_calls } = extractToolCallsFromResponse(totalContent);
        if (invoke_tools == true) {
            console.log("invoke_tools is true");
            // TODO: handle tool-call div creation
            const tool_execution_response = await executeTools(tool_calls, stream_session_id);
            const tool_execution_data = await tool_execution_response.json();
            const tool_result = tool_execution_data.tool_result_list;
            const toolResponseMode = document.getElementById('tool_response_mode').value;
            apiMessages = updateMessagesObjectWithToolResult(apiMessages, tool_calls, tool_result, plain_text, toolResponseMode);
            totalContent += await fetchEventStream(getServerType(), apiMessages, responseIDs.responseContentID, chatContainer, file);
        }
        
        const getReferencesParams = {
            'chat_id':current_chat_id,
            'sequence_id': current_sequence_id,
            'stream_session_id': stream_session_id,
            'user_query': userInputForHtml,
            'user_query_html': userMessageElement.outerHTML,
            'llm_response': totalContent,
            'formatted_user_prompt': JSON.stringify(apiMessages),
            'regeneration_request': regeneration_request,
            'regenerate_with_citations_force_enabled': regenerate_with_citations_force_enabled
        };

        await getReferences(invoke_tools, getReferencesParams, responseIDs.responseContentID, responseIDs.masterWrapperID, uniqueId);

        // Complete final step
        traceManager.completeCurrentStep();

        if (document.getElementById('enable_tts').checked) { await fetchTTSVoice(totalContent); }   // Not using getTts() here so simple checkbox change is enough, rather than a config save!

    } catch (error) {
        decrementSequenceId();
        // Stop timer on error
        if (traceManager) {
            traceManager.completeCurrentStep();
        }
        errorHandlerNoAlert("chatting with the LLM", "execute-Prompt()", String(error.message));
    } finally {
        enableSendButton();
        displayProcessingStatus(false);
    }
}


// Call sendMessage() if the user presses the 'Enter' key
const maxRows = 11; // Replace this value with the maximum allowed number of rows you want
const inputTextAreaElement = document.getElementById("user-input");
const lineHeight = parseInt(window.getComputedStyle(inputTextAreaElement).lineHeight);
const maxHeight = lineHeight * maxRows;
let currentHeight = lineHeight; // Start with 1 line height

// Function to automatically adjust textarea height:
function autoAdjustHeight() {
    // Store the current scroll position
    const scrollPos = inputTextAreaElement.scrollTop;
    
    // Temporarily set height to 'auto' - to measure the true height needed for the content in the textarea, we need to temporarily remove any height constraints!
    inputTextAreaElement.style.height = 'auto';
    
    // Calculate new height within bounds
    const newHeight = Math.min(inputTextAreaElement.scrollHeight, maxHeight);
    
    // Set the new height directly
    inputTextAreaElement.style.height = `${newHeight}px`;
    currentHeight = newHeight;
    
    // Restore scroll position - If the user has manually scrolled around the textarea, setting the height to auto would cause this position to be lost, so it's important to restore it!
    inputTextAreaElement.scrollTop = scrollPos;
}


// Add input event listener for dynamic resizing:
const debouncedAdjustHeight = debounce(autoAdjustHeight, 100);  // necessary to declare a const as this ensures only one instance of the debounced function is created, rather than creating a new instance on each call via `inputTextAreaElement.addEventListener('input', debounce(autoAdjustHeight, 70));`
inputTextAreaElement.addEventListener('input', debouncedAdjustHeight);

// Also handle paste events explicitly:
inputTextAreaElement.addEventListener('paste', function() {
    setTimeout(autoAdjustHeight, 0); // Using timeout to let the paste complete before resizing
});

// Handle focus (clciking back into the textarea):
inputTextAreaElement.addEventListener('focus', function() {
    if (this.value.trim()) {
        inputTextAreaElement.style.height = `${currentHeight}px`;
    }
});

// Handle blur events (when the textarea is no longer focused):
inputTextAreaElement.addEventListener('blur', function() {
    inputTextAreaElement.style.height = `${lineHeight}px`;
});

// Handle SHIFT + ENTER for new lines and ENTER for sending messages:
inputTextAreaElement.addEventListener("keydown", function(event) {
    const sendButton = document.getElementById('sendButton');

    if (!event.shiftKey && event.key == "Enter" && this.value.trim() !== "") {
        event.preventDefault();
        if(!sendButton.disabled) {  //Only trigger a send event if the button is not disabled, i.e. another stream is in progress!
            inputTextAreaElement.style.height = `${lineHeight}px`;
            currentHeight = lineHeight;
            executePrompt();
        }
    } 
    else if (event.shiftKey && event.key == "Enter") {
        const potentialNewHeight = currentHeight + lineHeight;
        if (potentialNewHeight <= maxHeight) {
            // The actual height adjustment will be handled by the input event
            // Just update our tracking variable
            currentHeight = potentialNewHeight;
        }
    } 
    // else if (event.key === "Backspace" || event.key === "Delete") { //KeyDown, which used 8 for Backspace and 46 for Delete, is obsolete and deprecated!
    //     // Let the deletion happen and let autoAdjustHeight handle the height adjustment
    //     // We'll update currentHeight in the autoAdjustHeight function
    // }
});


const textAttachmentInput = document.getElementById("textAttachmentInput");
const textAttachmentPreview = document.getElementById("textAttachmentPreview");
const textAttachmentRemoveBtn = document.getElementById("textAttachmentRemoveBtn");
const textAttachmentFileName = document.getElementById("textAttachmentFileName");

textAttachmentInput.addEventListener('change', function(event) {
    console.log("textAttachmentInput: ", event);
    if (event.target.files.length > 0) {
        console.log("file detected: ", event.target.files[0]);
        const file = event.target.files[0];
        textAttachmentFileName.textContent = `Attached: ${file.name}`;
        textAttachmentPreview.style.display = 'block';
    }
});

function removeTextAttachment() {
    textAttachmentInput.value = "";
    textAttachmentFileName.textContent = "";
    textAttachmentPreview.style.display = 'none';
}

textAttachmentRemoveBtn.addEventListener('click', removeTextAttachment);


// Store user rating & update UI - confirm the associated route does not contain print() statements as the stdout is redirected during response generation!
document.getElementById('chat-area').addEventListener('click', function(e) {
    if(e.target.classList.contains('fa-star')) {
        let starContainer = e.target.parentElement; // should be div class="star-rating"
        // if(starContainer.getAttribute('data-rated') === "False") {
            
            let rate = e.target.getAttribute('data-rate');
            let chat_id = starContainer.getAttribute('data-rating-chat-id');
            let sequence_id = starContainer.getAttribute('data-rating-sequence-id');

            for(let i = 0; i < 5; i++) {
                // reset star
                starContainer.children[i].classList.remove('fas');
                starContainer.children[i].classList.add('far');

                // fill star if rated
                if(i < rate) {
                    console.log("filling star-rating")
                    starContainer.children[i].classList.remove('far');
                    starContainer.children[i].classList.add('fas'); // This fills the star
                }
            }
            starContainer.setAttribute('data-rated', "True");
            
            let formData = new FormData();
            formData.append('rating', rate);
            formData.append('chat_id', chat_id);
            formData.append('sequence_id', sequence_id);
            // AJAX time:
            fetch('/store_user_rating', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error)});
                }
            })
            .catch(error => {
                errorHandler("storing the user's rating", "/store_user_rating", String(error.message))
            });
        // }
    }

    if(e.target.classList.contains('tool-chain-toggle')) {
        const userMessage = e.target.closest('.user-message');
        if (userMessage) {
            const trace = userMessage.querySelector('.tool-chain-trace');
            if (trace) {
                trace.classList.toggle('expanded');
                e.target.classList.toggle('expanded');
            }
        }
    }
});


