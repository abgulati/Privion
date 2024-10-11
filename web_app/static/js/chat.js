

function formatTabsAndSpaces(text, tabSize = 4) {
    // Replace each tab with tabSize number of &nbsp;
    text = text.replace(/\t/g, '&nbsp;'.repeat(tabSize));

    // Replace multiple spaces (2 or more) with equivalent number of &nbsp;
    text = text.replace(/ {2,}/g, (match) => '&nbsp;'.repeat(match.length));

    // Replace newlines with <br>
    text = text.replace(/\n/g, '<br>');

    return text;
    //return "test";
}


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

function scrollChatAreaToBottom() {
    const chatArea = document.getElementById('chat-area');
    chatArea.scrollTop = chatArea.scrollHeight;
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

function updateChatAreaWithUserInput(userInputForHtml) {
    document.getElementById('chat-area').innerHTML += `<div class="user-message glassmorphism">${userInputForHtml}</div>`;
    scrollChatAreaToBottom();
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


function setupLLMResponse(userInput, file_attached, current_chat_id) { //no need to async-await here as fetch() inherently returns a promise, so by returning fetch() directly we're providing the same Promise that the async function with await would return.
    return fetch('/setup_for_llama_cpp_response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({'user_query': userInput, 'chat_id': current_chat_id, 'file_attached': file_attached})
    });
}

function appendResponseContainerToChatArea(masterWrapperID, responseWrapperID, responseContentID) {
    const chatArea = document.getElementById('chat-area');
    chatArea.innerHTML += `
        <div class="response-and-viewer-container" id="${masterWrapperID}">
            <div class="llm-wrapper" style="display:none;" id="${responseWrapperID}">
                <div class="llm-response" id="${responseContentID}"></div>
            </div>
        </div>
    `;
    document.getElementById(responseWrapperID).style.display  = 'block';
    scrollChatAreaToBottom();
}

function handleSetupResponse(data) {
    const {do_rag, stream_session_id, formatted_user_prompt, sequence_id, server_type} = data;   // using object=destructuring (ES6-2015) to extract and set specific values from the data object in a single step!

    console.log("do_rag: ", do_rag);
    console.log("stream_session_id: ", stream_session_id);
    console.log("formatted_user_prompt: ", formatted_user_prompt);
    console.log("server_type: ", server_type);
    setSequenceId(sequence_id);

    const responseIDs = {
        responseWrapperID: `ResponseWrapper${stream_session_id}`,
        responseContentID: `ResponseContent${stream_session_id}`,
        masterWrapperID: `MasterWrapper${stream_session_id}`
    }

    appendResponseContainerToChatArea(responseIDs.masterWrapperID, responseIDs.responseWrapperID, responseIDs.responseContentID);
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
                        const jsonStr = message.slice(6);   // remove 6 chars to get rid of the 'data: ' prefix!
                        try {
                            const dataObj = JSON.parse(jsonStr);
                            //console.log(dataObj.content);   // Log only the content
                            let streamed_content = dataObj.content.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>'); // /g - global - replace throughout string, not just the first occurance

                            if (shouldAppendContent(streamed_content)) {
                                totalContent += streamed_content;
                                appendContentToResponse(responseContentID, streamed_content);
                            }

                            handleAutoScroll(chatContainer);

                            if (dataObj.stop) {
                                receivedComplete = true;
                            }
                        } catch (error) {
                            console.error('Error parsing JSON: ', error);
                        }
                    }
                });

                document.getElementById(responseContentID).innerHTML = totalContent;    // Even though we have appendContentToResponse(), we still need to update the responseContentID div's innerHTML to reflect the latest content because the streamed content may not have any newlines in it!
                if (receivedComplete) break;
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
    downloadLink.href = `http://localhost:9069/serve_uploaded_file/${filename}`;
    downloadLink.target = '_blank';
    downloadLink.download = filename;   // HTML5 attribute that specifies that the linked resource should be downloaded when a user clicks on the hyperlink
    downloadLink.innerHTML = `<i class="fa-solid fa-file-arrow-down"></i> ${filename} Appended to Conversation`;
    downloadLink.classList.add('plain-text-download-link-for-vision-appended-file');
    return downloadLink;
}


function createDownloadContainerForFile(filename) {
    const downloadContainer = document.createElement('div');
    downloadContainer.classList.add('download-container-for-vision-appended-file');

    const downloadLink = document.createElement('a');
    downloadLink.href = `http://localhost:9069/serve_uploaded_file/${filename}`;
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

    const vision = document.getElementById('hf_waitress_vision_yes').checked;
    if (vision) {
        console.log("Invoking vision_stream");
        url = "http://localhost:9069/vision_stream";

        hfwHeaders = new Headers();
        hfwHeaders.append("X-DPI", "300");
        hfwHeaders.append("X-Max-New-Tokens", document.getElementById('HfwMaxNewToks').value);

        formdata = new FormData();

        const parsedPrompt = JSON.parse(formattedPrompt);
        formdata.append("messages", JSON.stringify(parsedPrompt.messages));
        if (file) { formdata.append("file", file); }
    } else {
        console.log("Invoking completions_stream");
        url = "http://localhost:9069/completions_stream";

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

        const request_body = vision ? formdata : rawBodyJSONStringified;

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
                        const jsonStr = message.slice(7, -1);   // remove first 7 and last 1 chars to get rid of the 'data: "' prefix and " suffix!

                        // console.log("message: ", message);
                        try {
                            let dataObj = String(jsonStr);
                            if (dataObj == "null") {
                                dataObj = "";
                            }
                            //console.log("dataObj: ", dataObj);

                            dataObj = dataObj.replace(/\\u[\dA-F]{4}/gi, function(match) {
                                return String.fromCharCode(parseInt(match.replace(/\\u/g, ''), 16));
                            });
                            // Explanation:
                            // 0. This exists to handle the issue of unicode characters in the streamed response, which break the HTML.
                            // 1. The regular expression /\\u[\dA-F]{4}/gi matches a sequence of characters that starts with "\\u" followed by exactly four hexadecimal digits (\d for digits, A-F for uppercase letters).
                            // 2. The "gi" flags are used for global and case-insensitive matching.
                            // 3. The function(match) { ... } is an arrow function (An arrow function expression has a shorter syntax and lexically binds the 'this' value) that takes the matched string and converts it back to a character using the parseInt function.
                            // 4. parseInt(..., 16) uses replace to remove the "\\u" prefix and convert the remaining 4-digit hexadecimal string to a decimal number, using the 16 argument to specify base 16.
                            // 5. String.fromCharCode() converts the decimal number integer (now a Unicode code point) back to the corresponding character.

                            let streamed_content = dataObj;

                            // The robust parsing below is necessary as the LLM sees HTML <br> tags on subsequent questions, because it sees it's prior responses formatted as HTML!
                            // First, decode any HTML entities that might already be present - /g implies global: replace throughout string, not just the first occurance
                            streamed_content = streamed_content.replace(/&lt;/g, '<').replace(/&gt;/g, '>');

                            // Then, replace newlines with <br> tags
                            streamed_content = streamed_content.replace(/\\n\\n/g, '<br><br>')
                                                            .replace(/\\n/g, '<br>')
                                                            .replace(/\n\n/g, '<br><br>')
                                                            .replace(/\n/g, '<br>');

                            // Replace tabs with spaces
                            streamed_content = streamed_content.replace(/\\t/g, '    ');

                            // Finally, encode HTML special characters, but preserve <br> tags
                            streamed_content = streamed_content.replace(/&/g, '&amp;')
                                                            .replace(/</g, '&lt;')
                                                            .replace(/>/g, '&gt;')
                                                            .replace(/&lt;br&gt;/g, '<br>');

                            hfwTotalContent += streamed_content;
                            appendContentToResponse(responseContentID, streamed_content);

                            handleAutoScroll(chatContainer);

                        } catch (error) {
                            console.error('Error parsing message: ', error);
                        }
                    } else if (message.startsWith('event: END') || message.startsWith('data: null')) {
                        console.log("Received null message from hf-waitress - stream complete");
                        hfwReceivedComplete = true;
                    }
                });

                document.getElementById(responseContentID).innerHTML = hfwTotalContent;
                if (hfwReceivedComplete) break;
            }
        }

        await hfwProcessChunk();
        return hfwTotalContent;

    } catch (error) {
        errorHandler("fetching event-streaming response", "localhost:9069/completions_stream", String(error));
    }

}

async function fetchHfwDiffusersEventStream(formattedPrompt, responseContentID, chatContainer) {
    const url = "http://localhost:9069/completions";

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

            // create image div with source = data.image_name (local file path) and append to responseContentID.innerHTML
            const imageDiv = document.createElement('div');
            const img = document.createElement('img');
            img.alt = "Generated Image";
            img.src = `http://localhost:9069/serve_generated_image/${data.image_name}`;
            imageDiv.appendChild(img);
            document.getElementById(responseContentID).appendChild(imageDiv);

            handleAutoScroll(chatContainer);
            console.log("returning imageDiv.outerHTML: ", imageDiv.outerHTML);
            return imageDiv.outerHTML;
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    } catch (error) {
        errorHandler("fetching diffusers response", "localhost:9069/completions", String(error));
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


function handleFetchedReferencess(do_rag, data, responseContentID, masterWrapperID, stream_session_id) {
    const latest_sequence_id = getSequenceId();
    const current_chat_id = data.chat_id;

    const chatState_chat_id = document.getElementById('chatState').getAttribute('data-ongoing-chat-id');
    if (chatState_chat_id != current_chat_id) {
        console.log("Updating chatID in the InfoBox");
        setChatId(current_chat_id);
        setModelHeaderInfoBox(current_chat_id, getLlmModel());
    }

    if (parseInt(latest_sequence_id) == 1) { 
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

    if (do_rag) {
        document.getElementById(responseContentID).innerHTML += `
        </br> 
        ${data.response}
        ` 
    }
    
    document.getElementById(responseContentID).innerHTML += `
    <div class="star-rating" data-rated="False" rating-chat-id=${current_chat_id} rating-sequence-id=${latest_sequence_id}>
        <i class="far fa-star" data-rate="1"></i>
        <i class="far fa-star" data-rate="2"></i>
        <i class="far fa-star" data-rate="3"></i>
        <i class="far fa-star" data-rate="4"></i>
        <i class="far fa-star" data-rate="5"></i>
    </div>
    `
    if (do_rag) {
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

async function getReferences(do_rag, params, responseContentID, masterWrapperID) {
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

        handleFetchedReferencess(do_rag, data, responseContentID, masterWrapperID, params.stream_session_id);
    } catch (error) {
        errorHandler("fetching relevant reference material", "getReferences()", String(error))
    }
}


async function requestFormattedPrompt() {
    initializePromptRequest();

    const current_chat_id = getChatId();
    const {userInput, file} = getUserInput();
    const userInputForHtml = formatTabsAndSpaces(userInput);
    updateChatAreaWithUserInput(userInputForHtml);

    let file_attached = false;
    if (file) { file_attached = true; }

    try {
        // 1- setup_for_llama_cpp_response
        const response = await setupLLMResponse(userInput, file_attached, current_chat_id);
        const data = await response.json();
        const {do_rag, stream_session_id, formatted_user_prompt, responseIDs, server_type} = handleSetupResponse(data);

        // 2- fetchEventStream()
        const chatContainer = document.getElementById('chat-area');
        const totalContent = await fetchEventStream(server_type, formatted_user_prompt, responseIDs.responseContentID, chatContainer, file);

        console.log("llm_response post stream: ", totalContent);

        // 3- getReferences()
        if (do_rag) {
            displayProcessingStatus('Fetching any references...');
        }

        const getReferencesParams = {
            'chat_id':current_chat_id,
            'sequence_id': getSequenceId(),
            'stream_session_id': stream_session_id,
            'user_query': userInput,
            'llm_response': totalContent,
            'formatted_user_prompt': formatted_user_prompt
        };

        await getReferences(do_rag, getReferencesParams, responseIDs.responseContentID, responseIDs.masterWrapperID);

    } catch (error) {
        errorHandler("chatting with the LLM", "requestFormattedPrompt()", String(error.message))
    } finally {
        enableSendButton();
        displayProcessingStatus(false);
    }
}


// Call sendMessage() if the user presses the 'Enter' key
const maxRows = 5; // Replace this value with the maximum allowed number of rows you want
const inputTextAreaElement = document.getElementById("user-input");
var currentRows = inputTextAreaElement.rows;

inputTextAreaElement.addEventListener("keydown", function(event) {
    const sendButton = document.getElementById('sendButton');
    // Check if the Enter key was pressed and the input isn't empty
    if (!event.shiftKey && event.key == "Enter" && this.value.trim() !== "") {
        // Prevent the default action (i.e., adding a new line)
        event.preventDefault();
        // Call the send function
        // sendMessage()
        if(!sendButton.disabled) {  //Only trigger a send event if the button is not disabled, i.e. another stream is in progress!
            inputTextAreaElement.rows = 1;
            //sendMessageAndProcessResponseStream()
            requestFormattedPrompt();
        }
    } else if (event.shiftKey && event.key == "Enter") {
        if (currentRows <= maxRows) {
            inputTextAreaElement.rows += 1;
            currentRows += 1;
        }
    } else if (event.keyCode === 8 || event.keyCode === 46) { //8 is Backspace and 46 is
        //console.log("backspace or delete pressed")
        newlineCount = inputTextAreaElement.value.split("\n").length;
        if (newlineCount < currentRows) {
            //console.log("trimming rows")
            inputTextAreaElement.rows = newlineCount;
            currentRows = newlineCount;
        }
    }
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


function adjustTextareaRows() {
    newlineCount = inputTextAreaElement.value.split("\n").length;
    if (newlineCount < maxRows) {
        inputTextAreaElement.rows = newlineCount;
        currentRows = newlineCount;
    } else if (newlineCount >= maxRows) {
        inputTextAreaElement.rows = maxRows;
        currentRows = newlineCount;
    }
}
document.getElementById("user-input").addEventListener('input', adjustTextareaRows);
document.getElementById("user-input").addEventListener('change', adjustTextareaRows);


// Upload new files to VectorDB
document.getElementById('fileInput').addEventListener('change', function (event) {
    if (this.value) {    // Check if a file is selected
    
        document.getElementById('overlay').style.display = 'block';
        
        let newFile = document.getElementById('fileInput');
        let file = newFile.files[0]

        if (file) {
            let formData = new FormData();
            formData.append('file', file);

            // Make the AJAX request to the server
            fetch('/process_new_file', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error)});
                }
                return response
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    populateDocsLoadedTable();
                    document.getElementById('overlay').style.display = 'none';
                    document.getElementById('fileInput').value = "";  // Clear the input value
                } else {
                    throw new Error(`Internal Server Error: Check server-log and server command-line for more details.`);
                }
            })
            .catch(error => {
                errorHandler("processing file", "/process_new_file", String(error.message))
                document.getElementById('overlay').style.display = 'none';
                document.getElementById('fileInput').value = "";  // Clear the input value
            });
        }
    }    
});


// Store user rating & update UI - confirm the associated route does not contain print() statements as the stdout is redirected during response generation!
document.getElementById('chat-area').addEventListener('click', function(e) {
    if(e.target.classList.contains('fa-star')) {
        let starContainer = e.target.parentElement;
        if(starContainer.getAttribute('data-rated') === "False") {
            
            let rate = e.target.getAttribute('data-rate');
            let chat_id = starContainer.getAttribute('rating-chat-id');
            let sequence_id = starContainer.getAttribute('rating-sequence-id');

            for(let i = 0; i < rate; i++) {
                console.log("filling star-rating")
                starContainer.children[i].classList.remove('far');
                starContainer.children[i].classList.add('fas'); // This fills the star
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
        }
    }
});


