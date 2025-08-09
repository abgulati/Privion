

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

    if (!state.scheduled) {
        state.scheduled = true;
        setTimeout(() => {
        const el = document.getElementById(responseContentID);
        if (el) renderMarkdownInto(el, state.buffer);
        state.scheduled = false;
        handleAutoScroll(document.getElementById('chat-area'));
        }, 80); // ~12 FPS; tune as needed
    }
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
    The uniqueId is used primarily to set the data-chat-id, data-sequence-id and data-stream-session-id attributes on the user-message element by the methods immediately below.
    These methods leverage querySelector to find the user-message element by the uniqueId, and then set the attributes on it.
    For regeneration requests, these attributes are already set to the correct chat-id & stream-session-id, so they needen't be set again.
    This is why appendChatIdToUserMessage() & appendStreamSessionIdToUserMessage() are only called if (!regeneration_request).
    However, the sequence-id does need to be reset even for regen requests, because regenerating a response deletes any prior messages for the sake of the chat template, 
    thus the sequence ID will need to be reset in any case where the regenration request is for any message other than the latest one.
    In this case, the stream-session-id is for the regen request is obtained by a call to the prepareAttributeForUserMessage() method, which
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

function appendSequenceIdToUserMessage(uniqueId, stream_session_id, regeneration_request, sequence_id) {
    const userMessageElement = regeneration_request ? document.querySelector(`.user-message[data-stream-session-id="${stream_session_id}"]`) : document.querySelector(`.user-message[data-unique-id="${uniqueId}"]`);
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
    return fetch('/setup_for_local_llm_response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({'user_query': userInput, 'chat_id': current_chat_id, 'file_attached': file_attached, 'regeneration_request': regeneration_request, 'stream_session_id': regen_stream_session_id, 'sequence_id': regen_sequence_id, 'regenerate_with_citations_force_enabled': regenerate_with_citations_force_enabled, 'regenerate_with_citations_force_disabled': regenerate_with_citations_force_disabled})
    });
}

function appendResponseContainerToChatArea(masterWrapperID, responseWrapperID, responseContentID, stream_session_id) {
    // Not called again for regen requests, as the response container is already present in the chat area.
    const chatArea = document.getElementById('chat-area');
    
    // Create the new element without using innerHTML on the main chat area - SEE COMMENT IN create-UserMessageHTML() TO UNDERSTAND WHY!
    const responseContainer = document.createElement('div');
    responseContainer.className = 'response-and-viewer-container';
    responseContainer.id = masterWrapperID;
    responseContainer.setAttribute('data-stream-session-id', stream_session_id);
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
    const {do_rag, stream_session_id, formatted_user_prompt, sequence_id, server_type} = data;   // using object-destructuring (ES6-2015) to extract and set specific values from the data object in a single step!

    console.log("do_rag: ", do_rag);
    console.log("stream_session_id: ", stream_session_id);
    console.log("formatted_user_prompt: ", formatted_user_prompt);
    console.log("server_type: ", server_type);
    setSequenceId(sequence_id);
    appendSequenceIdToUserMessage(data.user_message_html_unique_id, stream_session_id, data.regeneration_request, sequence_id);

    const responseIDs = {
        responseWrapperID: `ResponseWrapper${stream_session_id}`,
        responseContentID: `ResponseContent${stream_session_id}`,
        masterWrapperID: `MasterWrapper${stream_session_id}`
    }

    if (!data.regeneration_request) { appendResponseContainerToChatArea(responseIDs.masterWrapperID, responseIDs.responseWrapperID, responseIDs.responseContentID, stream_session_id); }
    displayProcessingStatus('Generating...');

    return { do_rag, stream_session_id, formatted_user_prompt, responseIDs, server_type };
}


async function fetchLlamacppEventStream(formattedPrompt, responseContentID, chatContainer) {
    const url = "http://localhost:8080/completion";
    const requestData = {
        prompt: formattedPrompt,
        stream: true,
        temperature: parseFloat(document.getElementById('tempSlider').value),
        top_k: parseInt(document.getElementById('topkSlider').value),
        top_p: parseFloat(document.getElementById('toppSlider').value),
        min_p: parseFloat(document.getElementById('minpSlider').value),
        n_keep: parseInt(document.getElementById('nkeepSlider').value)
    };

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
                            const dataObj = JSON.parse(jsonStr);
                            const streamedText = dataObj.content || "";
                            appendStreamChunkAndRender(responseContentID, streamedText);
                            handleAutoScroll(chatContainer);

                            if (dataObj.stop) {
                                receivedComplete = true;
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
        errorHandler("fetching llama.cpp event-streaming response", "localhost:8080/completions", String(error))
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


async function fetchHfWaitressEventStream(formattedPrompt, responseContentID, chatContainer, file=null) {
    let url;
    let hfwHeaders = new Headers();
    let formdata = null;
    let rawBodyJSONStringified = null;

    const vision = getVision();
    const exl2 = getExl2();
    console.log("vision: ", vision, "typeof:", typeof vision);
    console.log("exl2: ", exl2, "typeof:", typeof exl2);
    if (vision === "true") {
        console.log("Invoking vision_stream");
        const hfWaitress_URL = getHfwUrl();
        url = `${hfWaitress_URL}/vision_stream`;

        hfwHeaders = new Headers();
        hfwHeaders.append("X-DPI", "300");
        hfwHeaders.append("X-Max-New-Tokens", document.getElementById('HfwMaxNewToks').value);

        formdata = new FormData();

        const parsedPrompt = JSON.parse(formattedPrompt);
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

        rawBodyJSONObj = JSON.parse(formattedPrompt);                                
        rawBodyJSONStringified = JSON.stringify(rawBodyJSONObj);
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
        
        rawBodyJSONObj = JSON.parse(formattedPrompt);                                
        rawBodyJSONStringified = JSON.stringify(rawBodyJSONObj);
    }
    
    try {

        const request_body = vision === "true" ? formdata : rawBodyJSONStringified;
        // console.log("request_body: ", request_body);

        const hfwResponse = await fetch(url, {
            method: 'POST',
            headers: hfwHeaders,
            body: request_body,
            redirect: 'follow'
        }); // due to the async-await syntax, the fetch() call returns a promise, and we await its resolution here.

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
                            console.log("Received null payload");
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
        errorHandler("fetching event-streaming response", "HF-Waitress/completions_stream", String(error));
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
        errorHandler("fetching diffusers response", "HF-Waitress/completions", String(error));
    }
    
}

async function fetchEventStream(serverType, formattedPrompt, responseContentID, chatContainer, file=null) {
    if (serverType == 'llama-cpp') {
        return fetchLlamacppEventStream(formattedPrompt, responseContentID, chatContainer);
    } else if (serverType == 'hf-waitress') {
        return fetchHfWaitressEventStream(formattedPrompt, responseContentID, chatContainer);
    } else if (serverType == 'hfw-diffusers') {
        return fetchHfwDiffusersEventStream(formattedPrompt, responseContentID, chatContainer);
    } else if (serverType == 'hfw-vision') {
        return fetchHfWaitressEventStream(formattedPrompt, responseContentID, chatContainer, file);
    } else {
        throw new Error(`Invalid server type: ${serverType}`);
    }
}


function handleFetchedReferencess(do_rag, data, responseContentID, masterWrapperID, stream_session_id, user_message_html_unique_id, regeneration_request) {
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

    console.log("get_references data: ", data);

    if (do_rag && data.response != "" && data.response != null) {
        document.getElementById(responseContentID).innerHTML = `${data.response}`;
        const state = `${data.response}`;
        streamState.set(stream_session_id, { buffer: state, scheduled: false });
        finalizeStreamRender(responseContentID);
    }
    
    document.getElementById(responseContentID).innerHTML += `
    <br>
    <div class="star-rating" data-rated="False" data-rating-chat-id=${current_chat_id} data-rating-sequence-id=${latest_sequence_id}>
        <i class="far fa-star" data-rate="1"></i>
        <i class="far fa-star" data-rate="2"></i>
        <i class="far fa-star" data-rate="3"></i>
        <i class="far fa-star" data-rate="4"></i>
        <i class="far fa-star" data-rate="5"></i>
    </div>
    `
    if (do_rag && data.pdf_frame != "" && data.pdf_frame != null) {
        document.getElementById(masterWrapperID).innerHTML += data.pdf_frame;
        // Open the first tab by default
        var defaultTabs = document.getElementsByClassName("defaultTabs");
        for (let i = 0; i < defaultTabs.length; i++) {
            if (defaultTabs[i].getAttribute('stream-session-id') === stream_session_id) {
                defaultTabs[i].click();
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

        handleFetchedReferencess(do_rag, data, responseContentID, masterWrapperID, params.stream_session_id, user_message_html_unique_id, params.regeneration_request);
    } catch (error) {
        errorHandler("fetching relevant reference material", "get-References()", String(error));
    }
}


async function requestFormattedPrompt(regeneration_request=false, regenerate_with_citations_force_enabled=false, regenerate_with_citations_force_disabled=false, regen_stream_session_id=null, regen_sequence_id=null) {
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
        traceManager.startStep("Analyzing Query & Executing Search Tools as Required...");
        const response = await setupLLMResponse(userInput, file_attached, current_chat_id, regeneration_request, regen_stream_session_id, regen_sequence_id, regenerate_with_citations_force_enabled, regenerate_with_citations_force_disabled);
        
        traceManager.startStep("Data Prepared - Reaching out to the LLM for a Response...");
        const data = await response.json();
        data.user_message_html_unique_id = uniqueId;
        data.regeneration_request = regeneration_request;
        const {do_rag, stream_session_id, formatted_user_prompt, responseIDs, server_type} = handleSetupResponse(data);
        if (!regeneration_request) { appendStreamSessionIdToUserMessage(uniqueId, stream_session_id); }

        // 2- fetch-EventStream()
        traceManager.startStep('Generating Response...');
        const chatContainer = document.getElementById('chat-area');
        const totalContent = await fetchEventStream(server_type, formatted_user_prompt, responseIDs.responseContentID, chatContainer, file);

        console.log("llm_response post stream: ", totalContent);

        // 3- get-References()
        if (do_rag) {
            traceManager.startStep('Fetching Document References...');
            displayProcessingStatus('Fetching any references...');
        }

        const getReferencesParams = {
            'chat_id':current_chat_id,
            'sequence_id': getSequenceId(),
            'stream_session_id': stream_session_id,
            'user_query': userInputForHtml,
            'user_query_html': userMessageElement.outerHTML,
            'llm_response': totalContent,
            'formatted_user_prompt': formatted_user_prompt,
            'regeneration_request': regeneration_request,
            'regenerate_with_citations_force_enabled': regenerate_with_citations_force_enabled
        };

        await getReferences(do_rag, getReferencesParams, responseIDs.responseContentID, responseIDs.masterWrapperID, uniqueId);

        // Complete final step
        traceManager.completeCurrentStep();

    } catch (error) {
        // Stop timer on error
        if (traceManager) {
            traceManager.completeCurrentStep();
        }
        errorHandler("chatting with the LLM", "request-FormattedPrompt()", String(error.message))
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
            requestFormattedPrompt();
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


