

function setChatId(id) {
    document.getElementById('chatState').setAttribute('data-ongoing-chat-id', id);
}

function setSequenceId(id) {
    document.getElementById('chatState').setAttribute('data-latest-sequence-id', id);
}

function setLlmModel(model) {
    document.getElementById('chatState').setAttribute('data-llm-model', model);
}

function setOldLlmModel(model) {
    document.getElementById('chatState').setAttribute('data-old-llm-model', model);
}

function setLlamaCppUrl(host, port) {
    document.getElementById('chatState').setAttribute('data-llama-cpp-url', `http://${host}:${port}`);
}

function setHfwUrl(host, port) {
    document.getElementById('chatState').setAttribute('data-hf-waitress-url', `http://${host}:${port}`);
}

function setHfwAsrAsgiHost(host) {
    document.getElementById('chatState').setAttribute('data-hf-asr-asgi-host', host);
}

function setHfwAsrAsgiPort(port) {
    document.getElementById('chatState').setAttribute('data-hf-asr-asgi-port', port);
}

function setHfwAsrUrl(host, port) {
    document.getElementById('chatState').setAttribute('data-hf-asr-url', `http://${host}:${port}`);
    setHfwAsrAsgiHost(host);
    setHfwAsrAsgiPort(port);
}

function setExl2(exl2) {
    document.getElementById('chatState').setAttribute('data-exl2', exl2);
}

function setExl3(exl3) {
    document.getElementById('chatState').setAttribute('data-exl3', exl3);
}

function setVision(vision) {
    document.getElementById('chatState').setAttribute('data-vision', vision);
}

function setAsr(asr) {
    document.getElementById('chatState').setAttribute('data-asr', asr);
}

function setAsrModel(asr_model) {
    document.getElementById('chatState').setAttribute('data-asr-model', asr_model);
}

function setTts(tts) {
    document.getElementById('chatState').setAttribute('data-tts', tts);
}

function setLegacyMode(legacy_mode) {
    document.getElementById('chatState').setAttribute('data-legacy-mode', legacy_mode);
}

function setServerType(server_type) {
    document.getElementById('chatState').setAttribute('data-server-type', server_type);
}

function getServerType() {
    return document.getElementById('chatState').getAttribute('data-server-type');
}

function getLegacyMode() {
    return document.getElementById('chatState').getAttribute('data-legacy-mode');
}

function getTts() {
    return document.getElementById('chatState').getAttribute('data-tts');
}

function getAsr() {
    return document.getElementById('chatState').getAttribute('data-asr');
}

function getAsrModel() {
    return document.getElementById('chatState').getAttribute('data-asr-model');
}

function getVision() {
    return document.getElementById('chatState').getAttribute('data-vision');
}

function getExl2() {
    return document.getElementById('chatState').getAttribute('data-exl2');
}

function getExl3() {
    return document.getElementById('chatState').getAttribute('data-exl3');
}

function getChatId() {
    return document.getElementById('chatState').getAttribute('data-ongoing-chat-id');
}

function getSequenceId() {
    return document.getElementById('chatState').getAttribute('data-latest-sequence-id');
}

function getLlmModel() {
    return document.getElementById('chatState').getAttribute('data-llm-model');
}

function getOldLlmModel() {
    return document.getElementById('chatState').getAttribute('data-old-llm-model');
}

function getLlamaCppUrl() {
    return document.getElementById('chatState').getAttribute('data-llama-cpp-url');
}

function getHfwUrl() {
    return document.getElementById('chatState').getAttribute('data-hf-waitress-url');
}

function getHfwAsrUrl() {
    return document.getElementById('chatState').getAttribute('data-hf-asr-url');
}

function getHfwAsrAsgiHost() {
    return document.getElementById('chatState').getAttribute('data-hf-asr-asgi-host');
}

function getHfwAsrAsgiPort() {
    return document.getElementById('chatState').getAttribute('data-hf-asr-asgi-port');
}

function getUniqueId() {
    // return crypto.randomUUID();

    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
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

function incrementSequenceId() {
    const updated_sequence_id = parseInt(getSequenceId()) + 1;
    setSequenceId(updated_sequence_id);
    return updated_sequence_id;
}

function decrementSequenceId() {
    const updated_sequence_id = parseInt(getSequenceId()) - 1;
    setSequenceId(updated_sequence_id);
    return updated_sequence_id;
}

function coerceToObject(maybeJson, label = "prompt") {
    // Regular mode: already an object
    if (maybeJson && typeof maybeJson === 'object') return maybeJson;

    // Legacy mode: JSON string
    if (typeof maybeJson === 'string') {
        const s = maybeJson.trim();
        try{
            return JSON.parse(s);
        } catch (e) {
            errorHandlerNoAlert("coercing to object", "coerce-ToObject()", `Error parsing ${label} as JSON: ${e.message}`);
            return null;
        }
    }
}

// Debounce function to limit how often it's called:
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) { // ...args allows the function to accept any number of arguments.
        clearTimeout(timeout);   // clearTimeout() cancels a timeout previously set with setTimeout(). We invoke it here as a precaution to ensure that any previous timeout is cleared before setting a new one.
        const later = () => {   // later is an arrow function (an arrow function expression has a shorter syntax and lexically binds the 'this' value) that clears the timeout and calls the original function with the arguments.
            func(...args);
        };
        timeout = setTimeout(later, wait);   // setTimeout() calls a function or evaluates an expression after a specified number of milliseconds.
    };
}

// event listener for javascript variable - use this to trace a troublesome variable and see exactly where it's being changed:
let _tracedVariable; // Can be equal to any variable, like inputTextAreaElement.rows;
Object.defineProperty(window, '_tracedVariable', {
    get: function() { 
        return _tracedVariable; 
    },
    set: function(value) {
        console.log(`_tracedVariable changed from ${_tracedVariable} to ${value}`);
        console.log('Stack trace:', new Error().stack);
        _tracedVariable = value;
    }
});

function scrollChatAreaToBottom() {
    const chatArea = document.getElementById('chat-area');
    chatArea.scrollTop = chatArea.scrollHeight;
}

function appendLoadingAnimation(regenrationUserMessage = null) {
    const chatArea = document.getElementById('chat-area');
    const lastUserMessage = regenrationUserMessage || chatArea.querySelector('.user-message:last-of-type');

    const loadingContainer = document.createElement('div');
    loadingContainer.className = 'loading-indicator-container';
    
    loadingContainer.innerHTML = `
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;

    if (lastUserMessage) {
        lastUserMessage.insertAdjacentElement('afterend', loadingContainer);
    } else {
        chatArea.appendChild(loadingContainer);
    }

    scrollChatAreaToBottom();
}

function removeLoadingAnimation() {
    const loadingContainer = document.querySelector('.loading-indicator-container');
    if (loadingContainer) {
        loadingContainer.remove();
    }
}

function showStreamSpinner() {
    document.getElementById('info_stream_spinner').style.display = 'inline-block';
}

function hideStreamSpinner() {
    document.getElementById('info_stream_spinner').style.display = 'none';
}

function scrollStreamInfoBoxToBottom() {
    const streamInfoBox = document.getElementById('stream_info_box');
    streamInfoBox.scrollTop = streamInfoBox.scrollHeight;
}

function appendStreamInfo(message, status='waiting') {
    const streamInfoBox = document.getElementById('stream_info_box');
    //streamInfoBox.innerHTML += `<h6 class="info-stream-content">${message}</h6>`;

    switch(status) {
        case 'waiting':
            streamInfoBox.innerHTML += `<div class="stream-info-row"><i class="fas fa-clock waiting" title="Waiting"></i><span class="info-stream-content">${message}</span></div>`;
            break;
        case 'success':
            streamInfoBox.innerHTML += `<div class="stream-info-row"><i class="fas fa-check-circle success" title="Success"></i><span class="info-stream-content">${message}</span></div>`;
            break;
        case 'failure':
            streamInfoBox.innerHTML += `<div class="stream-info-row"><i class="fas fa-times-circle failure" title="Failed"></i><span class="info-stream-content">${message}</span></div>`;
            break;
    }
    scrollStreamInfoBoxToBottom();
}

function updateUIForFile(row, status) {
    let statusCell = row.cells[row.cells.length - 1];

    statusCell.textContent = ''; // Clear existing content

    switch(status) {
        case 'loading':
            statusCell.innerHTML = '<div class="cell-loader"></div>';
            break;
        case 'waiting':
            statusCell.innerHTML = '<i class="fas fa-clock waiting" title="Waiting"></i>';
            break;
        case 'success':
            statusCell.innerHTML = '<i class="fas fa-check-circle success" title="Success"></i>';
            break;
        case 'failure':
            statusCell.innerHTML = '<i class="fas fa-times-circle failure" title="Failed"></i>';
            break;
    }
}


function readGGUF(model) {
    document.getElementById('GgufDetails').value = "Loading... Reading GGUF details for model: " + model + "...";
    fetch('/gguf_reader', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: model })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            // console.log("GGUF details read successfully:", data.gguf_details);
            document.getElementById('GgufDetails').value = data.gguf_details;
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        document.getElementById('GgufDetails').value = "Error reading GGUF details: " + String(error.message);
    }).finally(() => {
        return true;    // Return promise to resolve the fetch() call!
    });
}


function getUserMessageContent(element) {
    const childNodes = Array.from(element.childNodes);  //childNodes is a property that returns a live collection of all child nodes of an element, including text nodes, comment nodes, and element nodes
    const textNode = childNodes.find(node => node.nodeType === Node.TEXT_NODE);
    return textNode ? textNode.textContent.trim() : '';
}

function resetResponseAndViewerContainerWithStreamSessionId(streamSessionId) {
    const responseAndViewerContainer = document.querySelector(`.response-and-viewer-container[data-stream-session-id="${streamSessionId}"]`);
    if (responseAndViewerContainer) {
        responseAndViewerContainer.id = `MasterWrapper${streamSessionId}`;
        responseAndViewerContainer.innerHTML = `
            <div class="llm-wrapper" style="display:none;" id="ResponseWrapper${streamSessionId}">
                <div class="llm-response" id="ResponseContent${streamSessionId}"></div>
            </div>
        `;

        // Reset Markdown Rendering Pipeline's buffer otherwise the regenerated response will be appended to the previous response!
        const state = ``;
        streamState.set(streamSessionId, { buffer: state, scheduled: false });
        finalizeStreamRender(`ResponseContent${streamSessionId}`);
    }
    document.getElementById(`ResponseWrapper${streamSessionId}`).style.display  = 'block';
    scrollChatAreaToBottom();
}

function getUserMessageByStreamSessionId(stream_session_id) {
    const messageElement = document.querySelector(`.user-message[data-stream-session-id="${stream_session_id}"]`);
    if (!messageElement) return null;
    return messageElement.querySelector('.user-query-content').textContent.trim();
}
// ... existing code ...

function prepareAttributeForUserMessage(userMessageDiv) {
    // Get the stream session id and sequence id
    const streamSessionId = userMessageDiv.getAttribute('data-stream-session-id');
    const sequenceId = userMessageDiv.getAttribute('data-sequence-id');

    // query selector for value of user-message class with data-stream-session-id:
    const userMessage = getUserMessageByStreamSessionId(streamSessionId);
    document.getElementById('user-input').value = userMessage;

    return { streamSessionId, sequenceId };
}

function deleteChatAreaElements(currentElement) {
    const elementsToDelete = []
    while (currentElement && currentElement.nextElementSibling) {
        currentElement = currentElement.nextElementSibling;

        if (currentElement.matches('.user-message') || currentElement.matches('.response-and-viewer-container')) {
            elementsToDelete.push(currentElement);
        }
    }
    elementsToDelete.forEach(element => element.remove());
}

function clearToolChainToggle(streamSessionId) {
    const toolChainTrace = document.querySelector(`.user-message[data-stream-session-id="${streamSessionId}"] .tool-chain-trace`);
    if (toolChainTrace) {
        toolChainTrace.replaceChildren();
    }
}

function deleteChatAreaElementsFromCurrentElement(currentElement) {
    const elementsToDelete = []
    while (currentElement) {
        if (currentElement.matches('.user-message') || currentElement.matches('.response-and-viewer-container')) {
            elementsToDelete.push(currentElement);
        }
        if (currentElement.nextElementSibling) {
            currentElement = currentElement.nextElementSibling;
        } else {
            break;
        }
    }
    elementsToDelete.forEach(element => element.remove());
}

function setModelHeaderInfoBox(chat_id, model_id) {
    setLlmModel(model_id);
    chatID = " Chat ".concat(String(chat_id))
    display_chatid_and_model = String(chatID).concat(": ", String(model_id))
    document.getElementById('model_header').innerHTML = '';
    document.getElementById('model_header').innerHTML = display_chatid_and_model;
}

async function delete_messages(chatId, sequenceId, userMessageDiv) {
    console.log("Deleting messages & responses for chat with chat_id: ", chatId, " from sequence_id: ", sequenceId);
    let formData = new FormData();
    formData.append('chat_id', chatId);
    formData.append('sequence_id', sequenceId);

    const response = await fetch('/delete_messages', {
        method: 'POST',
        body: formData
    }) 
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            console.log("Messages & responses deleted from chat history successfully");
            deleteChatAreaElementsFromCurrentElement(userMessageDiv); // Delete everything from this user message onwards
            if (sequenceId == 1) {
               location.reload();
            } else {
                chatState.clear();
                chatState.hydrate(data.messages_list);
            }
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("deleting the specified messages & responses", "/delete_messages", String(error.message))
    });
}

function goToPage(iframeId, url) {
    var iframe = document.getElementById(iframeId)
    
    // Set src to blank and then to desired src to prevent the browser from ignoring clicks!
    iframe.src = 'about:blank';
    setTimeout(() => iframe.src = url, 100); // Set a slight delay to ensure the reload
}

function goToPageAndSwitchTab(iframeId, url, tabName, streamSessionId) {
    // First, switch to the correct tab
    openTab(null, tabName, streamSessionId);

    // Then, navigate to the correct page
    goToPage(iframeId, url);
}   //set as an onclick trigger in app.py!


function toggleInfo() {
    var infoBox = document.getElementById('info_box');
    if (infoBox.style.display === 'none') {
        infoBox.style.display = 'block';
    } else {
        infoBox.style.display = 'none';
    }
}


function toggleStreamInfo() {
    var streamInfoBox = document.getElementById('stream_info_box');
    if (streamInfoBox.style.display === 'none') {
        streamInfoBox.style.display = 'block';
    } else {
        streamInfoBox.style.display = 'none';
    }
}


function errorHandlerNoAlert(attempted_action, error_generator, error_message) {
    let error_alert_message = "There was an error when "  + attempted_action + " in the method " + error_generator + ", more details can be viewed in the browser's console. "
    let full_error_message =  error_alert_message + error_message;
    console.error(full_error_message);
    document.getElementById('ModelAndDBLoading').style.display = 'none';
    document.getElementById('SavingHfWaitressSettings').style.display = 'none';
    hideLoader();
    hideStreamSpinner();
    removeLoadingAnimation();
}


function errorHandler(attempted_action, error_generator, error_message) {
    let error_alert_message = "There was an error when "  + attempted_action + " in the method " + error_generator + ", more details can be viewed in the browser's console. "
    errorHandlerNoAlert(attempted_action, error_generator, error_message);
    alert(error_alert_message);
}


const forceEnableRagCheckbox = document.getElementById('force_enable_rag_checkbox');
const forceDisableRagCheckbox = document.getElementById('force_disable_rag_checkbox');
const defaultRag = document.getElementById('default_rag_checkbox');

function updateCheckboxes(checkedBox) {
    const checkboxes = [forceEnableRagCheckbox, forceDisableRagCheckbox, defaultRag];
    checkboxes.forEach(checkbox => {
        checkbox.checked = checkbox === checkedBox;
    });
}

//Listeners for Checkbox Changes
[forceEnableRagCheckbox, forceDisableRagCheckbox, defaultRag].forEach(checkbox => {
    checkbox.addEventListener('change', function() {
        if (this.checked) {
            updateCheckboxes(this);
        }
    });
});

function showLoader() {
    document.getElementById('loader').style.display = 'block';
}

function hideLoader() {
    document.getElementById('loader').style.display = 'none';
}

function openNav() {
    document.getElementById("sidenav").style.width = "290px";
    document.getElementById("sidenav").style.left = "0";
}

function closeNav() {
    document.getElementById("sidenav").style.width = "0";
    document.getElementById("sidenav").style.left = "-2px";
}

function setServerStatusIndicator(status) {
    document.getElementById('local_llm_server_status_indicator_text').innerHTML = `<i class="fa-solid fa-circle"></i> Server ${status}`;
    document.getElementById('local_llm_server_status_indicator_text').style.color = status === "Online" ? "green" : "red";
}

function attachWindowEvents() {
    const customSelect = document.getElementById('hf-waitress-llm-custom-select');
    const dropdownContent = document.getElementById('hf-waitress-llm-custom-dropdown-content');
    const addInput = document.getElementById('hf-waitress-llm-custom-dropdown-add-input');
    const addBtn = document.getElementById('hf-waitress-llm-custom-dropdown-add-btn');
    
    window.addEventListener('click', function(e) {
        if (customSelect && !customSelect.contains(e.target)) {
            dropdownContent.classList.remove('show');
            addInput.style.display = 'none';
            addBtn.style.display = 'block';
        }
    });
}


function sortNavItems() {
    
    // Get all nav-items
    const nodeList = document.querySelectorAll('.nav-item');

    // Reverse nav-items
    const reversedNodes = Array.prototype.slice.call(nodeList).reverse();
    // Explanation:
    // querySelectorAll returns a NodeList, which is different from an array and thus does not have array methods available to it, such as reverse()
    // To use reverse(), we must call array methods on our NodeList as if it were an array
    // To do so, we use 'Array.prototype' as all JS objects have a prototype, which is itself an object, and thus inherits all props and methods of the JS object, such as arrays in this case
    // The slice() method returns a shallow copy of a protion of an array. If no bounds are specified, it returns the entire array
    // Every function in JS has a call() method, allowing you to set the value of 'this' for the duration of the function's execution
    // This effectively let's you borrow a function, using it as if it belongs to a different object than the one it's actually attached to
    // This lets us call Array's reverse() on our NodeList!
    
    // Get parent <div> container
    const parent = document.getElementById('sidenav-content');

    // Reverse and append nav-items:
    reversedNodes.forEach(node => parent.appendChild(node));
    // When calling appendChild on an element that's already part of the DOM, a duplicate will not be created, it'll simply be moved to a new position
    // This allows us to reverse and reinsert without clearing all nav-items first!
    
}


function openTab(evt, tabName, streamSessionId) {
    var i, tabcontent, tabbuttons;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        if (tabcontent[i].getAttribute('stream-session-id') === streamSessionId) {
            tabcontent[i].style.display = "none";
        }
    }
    tabbuttons = document.getElementsByClassName("tab-button");
    for (i = 0; i < tabbuttons.length; i++) {
        if (tabbuttons[i].getAttribute('stream-session-id') === streamSessionId) {
            tabbuttons[i].className = tabbuttons[i].className.replace(" active", "")
        }
    }
    document.getElementById(tabName).style.display = "block";
    
    // Find the button that opens this tab and add the active class
    var activeButton = document.querySelector(`button[onclick="openTab(event, '${tabName}', '${streamSessionId}')"]`);
    if (activeButton) {
        activeButton.className += " active";
    }
}


async function renameChat(chat_id, new_chat_name) {
    console.log("Renaming chat with ID: ", chat_id);
    let formData = new FormData();
    formData.append('chat_id', chat_id);
    formData.append('new_chat_name', new_chat_name);

    const response = await fetch('/rename_chat', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) { 
            console.log("Chat renamed successfully");
            // Update the chat name in the sidebar
            let chatItem = document.querySelector(`[data-chat-id="${chat_id}"]`);
            if (chatItem) {
                chatItem.querySelector('.chat-title').textContent = new_chat_name;
                chatItem.setAttribute('data-chat-name', new_chat_name);
            }
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("renaming the specified chat", "/rename_chat", String(error.message))
    });
}


async function deleteChat(chat_id) {
    console.log("Deleting chat with ID: ", chat_id);
    let formData = new FormData();
    formData.append('chat_id', chat_id);

    const response = await fetch('/delete_chat', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            console.log("Chat deleted successfully");
            // Delete the nav-item from the sidebar
            let chatItem = document.querySelector(`[data-chat-id="${chat_id}"]`);
            if (chatItem) {
                chatItem.remove();
            }
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("deleting the specified chat", "/delete_chat", String(error.message))
    });
}


function createChatHistoryMenuItem(chat) {
    const newDiv = document.createElement('div');
    newDiv.className = 'nav-item';

    const attributes = ['chat_id', 'local_llm_server', 'chat_name', 'date_time', 'prompt_template_format'];
    attributes.forEach(attribute => {
        newDiv.setAttribute(`data-${attribute.replace('_', '-')}`, chat[attribute]);
    });
    
    let chatTitle = chat['date_time'];
    if (chat['chat_name'] != '' && chat['chat_name'] != null && chat['chat_name'] != "null") {
        chatTitle = chat['chat_name'];
    } else {
        newDiv.setAttribute('data-chat-name', chatTitle);
    }

    const rename_chat_title_button_class = `rename-chat-title-button-for-chat-id-${chat['chat_id']}`;
    const rename_chat_title_input_class = `rename-chat-title-input-for-chat-id-${chat['chat_id']}`;

    newDiv.innerHTML = `
        <span class="chat-title">${chatTitle}</span>
        <div class="three-dot-menu">
            <i class="fas fa-ellipsis-v"></i>
            <div class="three-dot-menu-options">
                <span class="three-dot-menu-option" onclick="deleteChat(${chat['chat_id']})">Delete Chat</span>
                <span class="three-dot-menu-option rename-option ${rename_chat_title_button_class}" data-chat-id="${chat['chat_id']}">Rename Chat</span>
                <input type="text" class="rename-chat-title-input ${rename_chat_title_input_class}" style="display: none;" placeholder="Enter new chat name...">
            </div>
        </div>
    `;

    newDiv.querySelector('.chat-title').addEventListener('click', function() {
        let chatID = this.parentElement.getAttribute('data-chat-id');
        let chatTitleAttribute = this.parentElement.getAttribute('data-chat-name');
        loadChatHistory(chatID, chatTitleAttribute);
    });

    // Event listener to rename the chat name/title:
    const addBtnQuerySelectorName = '.' + rename_chat_title_button_class;
    const addInputQuerySelectorName = '.' + rename_chat_title_input_class;
    const addBtn = newDiv.querySelector(addBtnQuerySelectorName);
    const addInput = newDiv.querySelector(addInputQuerySelectorName);

    addBtn.addEventListener('click', function() {
        let chatID = this.getAttribute('data-chat-id');
        
        addBtn.style.display = 'none';
        addInput.style.display = 'block';

        addInput.addEventListener('keypress', function(e) {
            // console.log("Key pressed: ", e.key);
            if (e.key === 'Enter' && this.value) {
                renameChat(chatID, this.value);
                addInput.value = '';
                addInput.style.display = 'none';
                addBtn.style.display = 'block';
            }
        });

        addInput.addEventListener('keydown', function(e) {
            // console.log("Key downed: ", e.key);
            if (e.key === 'Escape' || e.key === 'Esc') {
                addInput.value = '';
                addInput.style.display = 'none';
                addBtn.style.display = 'block';
            }
        });
    });
    
    return newDiv;
}


function loadChatHistory(chatID, chatTitle) {

    // console.log("Loading chat history")
    console.log("Loading chat history for chatID: ", chatID);

    setChatId(chatID);
    let current_llm_model = getLlmModel();

    history_chat_id = chatID
    setModelHeaderInfoBox(history_chat_id, current_llm_model);
    document.getElementById('chat_name_header').innerHTML = " " + String(chatTitle);
    document.getElementById('chat_name_header').style.display = 'block';

    document.getElementById('chat-area').innerHTML = '';

    let formData = new FormData();
    formData.append('chat_id', chatID);

    //Make a POST request to the server
    fetch('/load_chat_history', {
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
            
            old_chat_model = data.old_chat_model
            old_chat_model = " Last LLM on this chat: ".concat(String(old_chat_model))
            document.getElementById('old_model_header').innerHTML = old_chat_model;
            document.getElementById('old_model_header').style.display = 'block';

            chat_history_data = data.chat_history

            for (let i = 0; i < chat_history_data.length; i++) {
                
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = chat_history_data[i];
                const toolChainTrace = tempDiv.querySelector('.tool-chain-trace');
                const toolChainToggle = tempDiv.querySelector('.tool-chain-toggle');
                const responseAndViewerContainer = tempDiv.querySelector('.response-and-viewer-container');
                
                if (toolChainTrace) {   // user-message element with tool-trace history - must instantiate traceManager object
                    
                    // Apply styling changes as at the time of saving the user-message HTML (when the call to /get_references was made), the tool-chain-trace was expanded!
                    toolChainTrace.classList.remove('expanded');

                    if (toolChainToggle) {
                        toolChainToggle.classList.toggle('expanded');
                    }

                    // Initialize the traceManager and attach it to the element
                    const userMessageElement = tempDiv.firstElementChild;
                    const traceManager = new ToolChainTraceManager(userMessageElement);
                    userMessageElement._traceManager = traceManager;

                    // Append the fully constructed element to the chat area without destroying anything
                    document.getElementById('chat-area').appendChild(userMessageElement);
                } else {
                    document.getElementById('chat-area').insertAdjacentHTML('beforeend', chat_history_data[i]);

                    if (responseAndViewerContainer) {
                        const streamSessionId = responseAndViewerContainer.getAttribute('data-stream-session-id');
                        const llmResponseContainer = responseAndViewerContainer.querySelector('.llm-response');

                        if (llmResponseContainer) {
                            const state = llmResponseContainer.innerHTML; // textContent will lose the star-rating div!
                            const responseContentID = `ResponseContent${streamSessionId}`;
                            streamState.set(streamSessionId, { buffer: state, scheduled: false });
                            finalizeStreamRender(responseContentID);
                        }
                    }
                }
            }

            if (data.messages_list) {
                chatState.clear();
                chatState.hydrate(data.messages_list);
            } else {
                console.warn("Backend did not return messages_list - conversation history not loaded into state!");
                chatState.clear();
                chatState.init({
                    sysPrompt: getSysPromptConfig().base_template, 
                    skipSystemPrompt: getSysPromptConfig().skip_system_prompt 
                });
            }

            var defaultTabs = document.getElementsByClassName("defaultTabs");
            for (let i = 0; i < defaultTabs.length; i++) {
                defaultTabs[i].click();
            }

            setSequenceId(data.sequence_id);
            setOldLlmModel(data.old_chat_model);
            
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("fetching chat history data", "/load_chat_history", String(error.message))
    });
    closeNav();
}

function loadChatHistoryMenu() {
    appendStreamInfo("Loading chat history menu...", 'waiting');
    //Make a GET request to the server
    fetch('/load_chat_history_list')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {

            const historyList = data.history_list
            const sidenav = document.getElementById('sidenav-content');

            historyList.forEach(chat => {
                const newDiv = createChatHistoryMenuItem(chat);
                sidenav.appendChild(newDiv);
            });

        } else {
            throw new Error('Internal Server Error when loading the chat history menu: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("fetching the history-menu list", "/load_chat_history_list", String(error.message))
    });
}


function startPerplexica() {
    fetch("/start_perplexica")
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            console.log("Perplexica started successfully");
        } else {
            throw new Error('Internal Server Error when starting Perplexica: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        console.error("Error starting Perplexica, skipping: ", error);
    });
}


function startSearXNG() {
    fetch("/start_searxng")
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            console.log("SearXNG started successfully");
        } else {
            throw new Error('Internal Server Error when starting SearXNG: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        console.error("Error starting SearXNG, skipping: ", error);
    });
}


function startFalkorDB() {
    fetch("/start_falkordb")
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            console.log("FalkorDB started successfully");
        } else {
            throw new Error('Internal Server Error when starting FalkorDB: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        console.error("Error starting FalkorDB, skipping: ", error);
    });
}


async function updateHfModelList(newModelList) {
    console.log("Updating HF-Waitress model list");
    const hfWaitress_URL = getHfwUrl();
    const response = await fetch(hfWaitress_URL + '/hf_config_writer_api', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ config_updates: { model_list: newModelList } })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            console.log("HF-Waitress model list updated successfully");
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("updating the HF-Waitress model list", "updateHfModelList()", String(error.message))
    });
}


async function updateLarsCustomModelList(list_name, new_list) {
    console.log("Updating LARS-Enterprise custom model list: ", list_name);
    const response = await fetch('/config_writer_api', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ config_updates: { [list_name]: new_list } }) // Use bracket notation to dynamically set the list_name
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            console.log("LARS-Enterprise custom model list updated successfully");
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("updating the LARS-Enterprise custom model list", "updateLarsCustomModelList()", String(error.message))
    });
}


function setDefaultTemplate() {
    let defaultTemplate = `Answer the user's question in as much detail as possible. Be very helpful, witty and precise. Do not make up answers or fabricate false information!`;

    document.getElementById('templateContent').value = defaultTemplate;     // set prompt-textarea value
    document.getElementById('template_dropdown_default').setAttribute('data-template', defaultTemplate);    // set 'Default' list-item attribute value

    document.getElementById('templateDropdown').textContent = 'Default';    //Set default dropdown button text
}


function resetLlmAdvancedDefaults() {
    document.getElementById('UseGpu').checked = false;
    document.getElementById("NumbGpuLayers").disabled = true;
    document.getElementById('NumbGpuLayers').value = "0";
    document.getElementById('LlmCtxLgt').value = "4096";
    document.getElementById('UnifiedKvBuffer').checked = false;
    document.getElementById('DisableKvOffloading').checked = false;
    document.getElementById('KeyCacheDataType').value = "f16";
    document.getElementById('ValueCacheDataType').value = "f16";
    document.getElementById('NoOfSeqsToParDecode').value = "1";
    document.getElementById('OffloadToDevices').value = "none";
    document.getElementById('CpuOnlyMoe').checked = false;
    document.getElementById('Mlock').checked = false;
    document.getElementById('NoNmap').checked = false;
    document.getElementById('MaxNewToks').value = "-1";
    document.getElementById('tempSlider').value = "0.8";
    document.getElementById('tempSliderValue').textContent = "0.8"
    document.getElementById('topkSlider').value = "40";
    document.getElementById('topkSliderValue').textContent = "40";
    document.getElementById('toppSlider').value = "0.9";
    document.getElementById('toppSliderValue').textContent = "0.9";
    document.getElementById('minpSlider').value = "0.1";
    document.getElementById('minpSliderValue').textContent = "0.1";
    document.getElementById('nkeepSlider').value = "0";
}


function resetHfLlmAdvancedDefaults() {
    document.getElementById('hf_waitress_is_awq_yes').checked = false;
    document.getElementById('hf_waitress_is_awq_no').checked = true;
    document.getElementById('hf_waitress_trust_remote_code_yes').checked = true;
    document.getElementById('hf_waitress_trust_remote_code_no').checked = false;
    document.getElementById('hf_use_exl2_no_flash_attn_checkbox').checked = false;
    document.getElementById('hf_waitress_return_full_text_checkbox').checked = false;
    document.getElementById('update_hf_access_token').checked = false;
    document.getElementById('hf_access_token').disabled = true;
    document.getElementById('hf_waitress_torch_device_map_choice').value = 'auto';
    document.getElementById('hf_waitress_torch_dtype_choice').value = 'auto';
    document.getElementById('hf_waitress_pipeline_task_choice').value = 'text-generation';
    document.getElementById('hf_waitress_quantization_choice').value = 'quanto';
    document.getElementById('hf_waitress_quantization_level_choice').value = 'int4';
    document.getElementById('HfwHqqGroupSize').value = "64";
    document.getElementById('HfwServingUrl').value = "0.0.0.0";
    document.getElementById('HfwAccessUrl').value = "localhost";
    document.getElementById('HfwPort').value = "9069";
    document.getElementById('HfwMaxNewToks').value = "500";
    document.getElementById('HfwTempSlider').value = "0";
    document.getElementById('HfwTempSliderValue').textContent = "0.0"
    document.getElementById('HfwTopkSlider').value = "40";
    document.getElementById('HfwTopkSliderValue').textContent = "40";
    document.getElementById('HfwToppSlider').value = "0.95";
    document.getElementById('HfwToppSliderValue').textContent = "0.95";
    document.getElementById('HfwMinpSlider').value = "0.05";
    document.getElementById('HfwMinpSliderValue').textContent = "0.05";
}

            
function clearDocsLoadedTable() {
    var table = document.getElementById('docs_loaded_details_table');

    for (var i = table.rows.length - 1; i > 0; i--) {   // HTML tables start at row 0, so the header row is 0 and won't be deleted thanks to the i > 0 condition. The length-1 is to account for 0-indexing as length may be 10, but max row index will be 9. 
        table.deleteRow(i);
    }
}


function clearGoogleDriveTable() {
    var table = document.getElementById('google_drive_files_table');
    var tbody = table.querySelector('tbody');
    tbody.innerHTML = '';
    document.querySelectorAll('.sort-btn').forEach(button => {
        button.setAttribute('data-order', 'none');
        button.textContent = '↕';
    });
}


function populateDocsLoadedTable() {
    let formData = new FormData();
    // formData.append('selected_knowledge_domain', document.getElementById('hf-waitress-kb-custom-dropdown-selected-value').textContent);
    // formData.append('selected_embedding_model', document.getElementById('hf-waitress-embed-custom-dropdown-selected-value').textContent);
    formData.append('selected_knowledge_domain', CustomDropdown.registry.get('kb').getSelectedValue());
    formData.append('selected_embedding_model', CustomDropdown.registry.get('embed').getSelectedValue());

    fetch('/fetch_file_list_for_vector_db', {
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

            clearDocsLoadedTable(); // clear the table since we now have new data

            row_list = data.file_row_list;

            var table = document.getElementById('docs_loaded_details_table');

            document.getElementById('docs_in_kb_count_header').textContent = `Total Number of Documents: ${row_list.length}`;

            for (var i = 0; i < row_list.length; i++) {
                // var rowCount = table.rows.length;   // get the number of rows in the table
                // var row = table.insertRow(rowCount);    // since HTML tables are initialized with row 0, rowCount will specify the current position to insert the new row

                var row = table.insertRow(-1);    // Or simply, use -1 to auto-append at the end of the table!
                var unique_id = getUniqueId();
                row.id = `vector_list_row_${unique_id}`;

                // Now create and insert new cells into the row while simulataneously setting their content:
                var nameCell = row.insertCell(0);
                nameCell.innerHTML = row_list[i][0];   // Name
                nameCell.className = 'vector_list_doc_name';
                nameCell.setAttribute('data-vector-list-row-id', unique_id);

                row.insertCell(1).innerHTML = row_list[i][1];   // Knowledge Domain
                row.insertCell(2).innerHTML = row_list[i][2];   // Embedding Model
                row.insertCell(3).innerHTML = row_list[i][3];   // Chunk Size
            }
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.')
        }
    })
    .catch(error => {
        errorHandler("fetching file list for the selected vector database", "/fetch_file_list_for_vector_db", String(error.message))
    });
}


function resetVectorDBtoBlank() {

    // Ask for user confirmation:
    if (!confirm("Are you sure you want to reset the VectorDB? The database will not be deleted from disk, and can be reverted to by manually modifying config.json")) {
        return; // If cancelled by the user, exit the function!
    }

    let formData = new FormData();
    // formData.append('selected_embedding_model', document.getElementById('hf-waitress-embed-custom-dropdown-selected-value').textContent);
    // formData.append('selected_knowledge_domain', document.getElementById('hf-waitress-kb-custom-dropdown-selected-value').textContent);
    formData.append('selected_embedding_model', CustomDropdown.registry.get('embed').getSelectedValue());
    formData.append('selected_knowledge_domain', CustomDropdown.registry.get('kb').getSelectedValue());

    fetch('/reset_vector_db_on_disk', {
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
            clearDocsLoadedTable();
            document.getElementById('docs_in_kb_count_header').textContent = ``;
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("resetting the specified vector database", "/reset_vector_db_on_disk", String(error.message))
    });
}


function openImageInNewTab(imageUrl) {
    window.open(imageUrl, '_blank');
}

function attachClickListeners() {
    document.querySelectorAll('.gallery-thumbnail').forEach(function(iframe) {
        iframe,addEventListener('click', function() {
            console.log('Clicked element:', iframe);
            openImageInNewTab(iframe.src)
        });
    });
}

function openImageGalleryModal(modalId) {
    document.getElementById(modalId).style.display = 'block';
    //attachClickListeners();
}

function closeImageGalleryModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function sortSelected() {
    const tbody = document.querySelector('#google_drive_files_table tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a,b) => {    // JS's built-in sort() method for arrays takes a comparison function as an argument, determining the order for two rows in this case
        const aChecked = a.querySelector('input[type="checkbox"]').checked;
        const bChecked = b.querySelector('input[type="checkbox"]').checked;
        return bChecked - aChecked;     // Clever bit! In JS, true is treated as 1, and false 0, and in sorting a positive number comes first - thus putting checked rows first while maintaining the ordering when both rows are checked/unchecked!
    });
    tbody.innerHTML = '';
    tbody.append(...rows);
}

function sortByColumn(button) {
    const table = document.getElementById('google_drive_files_table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const column = button.dataset.sort;
    const index = column === 'name' ? 1 : 2;    //determines which column to sort: 1 for Name, 2 for Type
    
    // Get current order, default to 'none' if not set
    const currentOrder = button.getAttribute('data-order') || 'none';
    const newOrder = currentOrder === 'none' || currentOrder === 'desc' ? 'asc' : 'desc';

    // Sort rows
    rows.sort((a,b) => {
        const aValue = a.cells[index].textContent.trim();
        const bValue = b.cells[index].textContent.trim();
        const comparison = aValue.localeCompare(bValue, undefined, {numeric: true, sensitivity: 'base'});   //sorts the rows based on the text content of the cells in the chosen column -> localeCompare() is used for string comparison, which handles alphabetical sorting well; undefined: This is the locale parameter. By leaving it undefined, we're using the default locale of the browser; sensitivity: 'base': This setting makes the comparison insensitive to case and diacritics. For instance, "a" and "A" would be considered equal, as would "é" and "e".
        return newOrder === 'asc' ? comparison : -comparison    //localeCompare() returns a negative number if a should be sorted before b, and a positive number otherwise. If newOrder is 'asc', it returns `comparison` as is (ascending order) and if it's 'desc', then by negating the comparison result (-comparison), we reverse the sort order!
    });
    
    // Update button state
    button.setAttribute('data-order', newOrder);
    button.textContent = newOrder === 'asc' ? '↑' : '↓';

    // Reset other buttons as should be sorted basis one column at any given time!
    table.querySelectorAll('.sort-btn').forEach(btn => {
        if (btn !== button) {
            btn.setAttribute('data-order', 'none');
            btn.textContent = '↕';
        }
    });

    // Clear and re-append the sorted rows!
    tbody.innerHTML = '';
    tbody.append(...rows);
}


function filterByDocType(e) {
    const filterValue = e.target.value.toLowerCase();
    const searchWords = document.getElementById('gdrive-search-bar').value.toLowerCase();
    const tbody = document.querySelector('#google_drive_files_table tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.forEach(row => {
        const type = row.cells[2].textContent.toLowerCase();    // cell 2 is column 3 which is the document type column
        const name = row.cells[1].textContent.toLowerCase();    // cell 1 is column 2 which is the document name column
        if (name.includes(searchWords) || searchWords === '') {
            row.style.display = (filterValue === '' || type === filterValue) ? '' : 'none';   //If the filter value is empty (showing all types) OR the row's type matches the filter value, set the row's display style to an empty string (which means "display normally"). Otherwise, set the row's display style to 'none!
        }
    }); 
}


function selectAll(e) {
    const tbody = document.querySelector('#google_drive_files_table tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.forEach(row => {
        row.querySelector('input[type="checkbox"]').checked = e.target.checked; //`e.target` refers to the checkbox that triggered the event (the "Select All" checkbox) and `e.target.checked` is a boolean value indicating whether this checkbox is checked or not, basis which the checkbox for all other rows is set via the querySelector.
    });
}


function initSortingAndFiltering() {
    // Sort by selected
    document.getElementById('sortSelected').addEventListener('click', sortSelected);

    // Sort by column
    
    // Approach 1 (Event Delegation) - unnecessary in this case as:
    // 1) Only two buttons, not a large number of elements
    // 2) Static elements in the header, not dynamically changing content
    // 3) Buttons themselves are not removed/added dynamically
    // document.getElementById('google_drive_files_table').addEventListener('click', function(e) {
    //     const target = e.target;
    //     if (target.classList.contains('sort-btn')) {
    //         sortByColumn(target);
    //     }
    // });

        // Approach 2 (Direct Attachment):
    const table = document.getElementById('google_drive_files_table');
    table.querySelectorAll('.sort-btn').forEach(button => {
        button.addEventListener('click', () => sortByColumn(button));
    });

    // Filter by document type
    document.getElementById('filterDocType').addEventListener('change', filterByDocType);

    // Select all checkboxes
    document.getElementById('selectAll').addEventListener('change', selectAll);
}


function populateGoogleDriveTable(gdrive_files) {
    const gdriveTableBody = document.querySelector('#google_drive_files_table tbody');
    const filterSelect = document.getElementById('filterDocType');
    const docTypes = new Set();

    gdriveTableBody.innerHTML = '';
    filterSelect.innerHTML = '<option value="">All Document Types</option>';

    gdrive_files.forEach((file, index) => {
        const iconClass = getFileIconClass(file.type);
        const rowHTML = `
            <tr data-gdrive-file-id="${file.id}" data-gdrive-file-name="${file.name}" data-gdrive-mime-type="${file.mimeType}" data-gdrive-mime-type-category="${file.type}" id="gdrive_doc_row_${file.id}">
                <td class="checkbox-cell"><input type="checkbox" id="select-${parseInt(index)+1}"></td>
                <td style="text-align:left" data-gdrive-row-id="${file.id}" class="gdrive_doc_name">${file.name}</td>
                <td style="text-align:left">${file.type}</td>
                <td style="text-align:center"><i class="file-icon fa-solid ${iconClass}"></i></td>
                <td style="text-align:center">v${file.version}</td>
                <td style="text-align:center"><i class="fas fa-cloud-arrow-down"></i></td>
            </tr>
        `;
        gdriveTableBody.insertAdjacentHTML('beforeend', rowHTML);

        docTypes.add(file.type);
    });

    // Populate filter options
    docTypes.forEach(type => {
        filterSelect.insertAdjacentHTML('beforeend', `<option value="${type}">${type}</option>`);
    });
}

function googleDriveLogin() {
    showLoader();
    const currentUrl = encodeURIComponent(window.location.href);
    window.location.href = `/web_login_to_google_drive?redirect=${currentUrl}`; // Redirecting without fetch to avoid CORS issues!
}

function googleDriveLogout() {
    fetch('/logout_from_google_drive')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (data.success) {
            // console.log(data);
            document.getElementById('googleDriveUserName').style.display = 'none';
            document.getElementById('googleDriveUserName').textContent = '';
            clearGoogleDriveTable();
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("logging out from Google Drive", "/logout_from_google_drive", String(error.message))
    });
}


function checkLocalLLMServerStatus(asr_check = false) {
    if (!asr_check) { setServerStatusIndicator('Status Check In-Progress...'); }
    const server_to_check = asr_check ? 'asr-waitress' : document.getElementById('local_llm_server_select_dropdown').value;
    let formData = new FormData();
    formData.append('server_to_check', server_to_check);
    return fetch('/check_local_llm_server_status', {    // returning will make the function return the promise chain so Promise.all (and any .then) will wait!
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        if (!data.success) throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        if (!asr_check) { setServerStatusIndicator(data.server_online ? 'Online' : 'Offline'); }
        return data.server_online;
    })
    .catch(error => {
        //errorHandler("checking local LLM server status", "check-LocalLLMServerStatus()", String(error.message));
        appendStreamInfo('Error: Could not check local LLM server status (server likely offline)', 'failure');
        throw error;    // rethrow so promise.all rejects!
    });
}


function coreFilterFunction(search_words, items) {
    const searchQuery = search_words.toLowerCase();

    Array.from(items).forEach(item => {
        const text = item.textContent.toLowerCase();
        if (text.includes(searchQuery)) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    })
}


function filterChatHistoryItems(search_words) {
    const customDropdownList = document.getElementById('sidenav-content');
    const items = customDropdownList.getElementsByClassName('nav-item');
    coreFilterFunction(search_words, items);
}


function coreFilterGDriveRowsFunction(search_words, items, filterValue) {
    const searchQuery = search_words.toLowerCase();

    Array.from(items).forEach(item => {
        const text = item.textContent.toLowerCase();
        const row_id = item.getAttribute('data-gdrive-row-id');
        const row = document.getElementById(`gdrive_doc_row_${row_id}`);
        if (text.includes(searchQuery)) {
            const type = row.cells[2].textContent.toLowerCase();    // cell 2 is column 3 which is the document type column
            row.style.display = (filterValue === '' || type === filterValue) ? '' : 'none';
        } else {
            row.style.display = 'none';
        }
    })
}


function filterGoogleDriveTable(search_words) {
    const fullGdriveTable = document.getElementById('google_drive_files_table');
    const gdrive_doc_name_cells = fullGdriveTable.getElementsByClassName('gdrive_doc_name');
    const filterValue = document.getElementById('filterDocType').value.toLowerCase();
    coreFilterGDriveRowsFunction(search_words, gdrive_doc_name_cells, filterValue);
}


function filterVectorListTable(search_words) {
    const fullVectorListTable = document.getElementById('vector_embeddings_details');
    const vector_doc_name_cells = fullVectorListTable.getElementsByClassName('vector_list_doc_name');
    coreFilterVectorListRowsFunction(search_words, vector_doc_name_cells);
}





// Transformers-specific settings:
function showTransformersSamplingParams() {
    document.getElementById('hf_waitress_return_full_text').style.display = 'table-row';
    document.getElementById('hf_waitress_minp_value').style.display = 'table-row';
}

function hideTransformersSamplingParams() {
    document.getElementById('hf_waitress_return_full_text').style.display = 'none';
    document.getElementById('hf_waitress_minp_value').style.display = 'none';
}

function showAdvTransformersSettings() {
    document.getElementById('hf_waitress_use_flash_attention_2').style.display = 'table-row';
    document.getElementById('hf_waitress_pipeline_task').style.display = 'table-row';
    document.getElementById('hf_waitress_torch_device_map').style.display = 'table-row';
    document.getElementById('hf_waitress_torch_dtype').style.display = 'table-row';
    document.getElementById('hf_waitress_quantization').style.display = 'table-row';
    document.getElementById('hf_waitress_quantization_level').style.display = 'table-row';
    document.getElementById('hf_waitress_hqq_group_size').style.display = 'table-row';
}

function hideAdvTransformersSettings() {
    document.getElementById('hf_waitress_use_flash_attention_2').style.display = 'none';
    document.getElementById('hf_waitress_pipeline_task').style.display = 'none';
    document.getElementById('hf_waitress_torch_device_map').style.display = 'none';
    document.getElementById('hf_waitress_torch_dtype').style.display = 'none';
    document.getElementById('hf_waitress_quantization').style.display = 'none';
    document.getElementById('hf_waitress_quantization_level').style.display = 'none';
    document.getElementById('hf_waitress_hqq_group_size').style.display = 'none';
}


// Exl3-specific settings:
function hideExl3SamplingParams() {
    document.getElementById('hf_waitress_minp_value').style.display = 'none';
    document.getElementById('hf_waitress_repetition_penalty_value').style.display = 'none';
    document.getElementById('hf_waitress_presence_penalty_value').style.display = 'none';
    document.getElementById('hf_waitress_frequency_penalty_value').style.display = 'none';
}

function showExl3SamplingParams() {
    document.getElementById('hf_waitress_minp_value').style.display = 'table-row';
    document.getElementById('hf_waitress_repetition_penalty_value').style.display = 'table-row';
    document.getElementById('hf_waitress_presence_penalty_value').style.display = 'table-row';
    document.getElementById('hf_waitress_frequency_penalty_value').style.display = 'table-row';
}

function displayOnlyExl3Settings() {
    // Manage config blocks
    document.getElementById('hf-waitress-exl3-configuration-div').style.display = 'block';
    document.getElementById('hf-waitress-diffusers-configuration-div').style.display = 'none';
    document.getElementById('hf-waitress-exl2-configuration-div').style.display = 'none';

    // Hide toggles
    document.getElementById('hf_waitress_exl2').style.display = 'none';
    document.getElementById('hf_waitress_trust_remote_code').style.display = 'none';
    document.getElementById('hf_waitress_is_awq').style.display = 'none';
    document.getElementById('hf_waitress_diffusers').style.display = 'none';

    // Manage Advanced Settings and Sampling Params
    hideAdvTransformersSettings();
    hideTransformersSamplingParams();
    showExl3SamplingParams();
}

function resetExl3OnlyView() {
    // Hide config block
    document.getElementById('hf-waitress-exl3-configuration-div').style.display = 'none';

    // Hide Exl3 Sampling Params
    hideExl3SamplingParams();

    // Since toggling this switches the UI to Transformers mode, ensure none of the required advanced settings are hidden!
    showAdvTransformersSettings();
    showTransformersSamplingParams();

    // Show toggles
    document.getElementById('hf_waitress_exl2').style.display = 'block';
    document.getElementById('hf_waitress_trust_remote_code').style.display = 'block';
    document.getElementById('hf_waitress_is_awq').style.display = 'block';
    document.getElementById('hf_waitress_diffusers').style.display = 'block';
    document.getElementById('advancedHfLlmSettingsToggle').style.display = 'inline-block';
}


// Diffusers-specific settings:
function displayOnlyDiffusersSettings() {
    // Manage config blocks
    collapseAdvancedSettings('advancedHfLlmSettings');
    document.getElementById('hf-waitress-diffusers-configuration-div').style.display = 'block';
    document.getElementById('hf-waitress-exl2-configuration-div').style.display = 'none';
    document.getElementById('hf-waitress-exl3-configuration-div').style.display = 'none';

    // Hide toggles
    document.getElementById('hf_waitress_exl2').style.display = 'none';
    document.getElementById('hf_waitress_exl3').style.display = 'none';
    document.getElementById('hf_waitress_trust_remote_code').style.display = 'none';
    document.getElementById('hf_waitress_is_awq').style.display = 'none';
    document.getElementById('advancedHfLlmSettingsToggle').style.display = 'none';  // since we're hiding this whole div, no need to explicitly hide Exl3 Sampling Params etc.!
}

function resetDiffusersOnlyView() {
    // Hide config block
    document.getElementById('hf-waitress-diffusers-configuration-div').style.display = 'none';

    // Since toggling this switches the UI to Transformers mode, ensure none of the required advanced settings are hidden!
    showAdvTransformersSettings();
    showTransformersSamplingParams();

    // Show toggles
    document.getElementById('hf_waitress_exl2').style.display = 'block';
    document.getElementById('hf_waitress_exl3').style.display = 'block';
    document.getElementById('hf_waitress_trust_remote_code').style.display = 'block';
    document.getElementById('hf_waitress_is_awq').style.display = 'block';
    document.getElementById('advancedHfLlmSettingsToggle').style.display = 'inline-block';
}


// Exl2-specific settings:
function displayOnlyExl2Settings() {
    // Manage config blocks
    document.getElementById('hf-waitress-exl2-configuration-div').style.display = 'block';
    document.getElementById('hf-waitress-exl3-configuration-div').style.display = 'none';
    document.getElementById('hf-waitress-diffusers-configuration-div').style.display = 'none';

    // Hide toggles
    document.getElementById('hf_waitress_exl3').style.display = 'none';
    document.getElementById('hf_waitress_trust_remote_code').style.display = 'none';
    document.getElementById('hf_waitress_is_awq').style.display = 'none';
    document.getElementById('hf_waitress_diffusers').style.display = 'none';

    // Manage Advanced Settings and Sampling Params
    hideAdvTransformersSettings();
    hideTransformersSamplingParams();
    hideExl3SamplingParams();
}

function resetExl2OnlyView() {
    // Hide config block
    document.getElementById('hf-waitress-exl2-configuration-div').style.display = 'none';

    // Since toggling this switches the UI to Transformers mode, ensure none of the required advanced settings are hidden!
    showAdvTransformersSettings();
    showTransformersSamplingParams();

    // Show toggles
    document.getElementById('hf_waitress_exl3').style.display = 'block';
    document.getElementById('hf_waitress_trust_remote_code').style.display = 'block';
    document.getElementById('hf_waitress_is_awq').style.display = 'block';
    document.getElementById('hf_waitress_diffusers').style.display = 'block';
    document.getElementById('advancedHfLlmSettingsToggle').style.display = 'inline-block';
}


function stopGeneration() {
    console.log("Stop Generation button clicked");
    displayProcessingStatus('Stopping...');
    const hfWaitress_URL = getHfwUrl();
    fetch(hfWaitress_URL + '/stop_generation')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        console.log(data);
        displayProcessingStatus(false);
    })
    .catch(error => {
        errorHandler("stopping generation", "/stop_generation", String(error.message));
    });
}

function showStopGenerationButton() {
    document.getElementById('stopGenerationButton').style.display = 'block';
}

function hideStopGenerationButton() {
    document.getElementById('stopGenerationButton').style.display = 'none';
}


function secure_filename(filename) {
    // Remove non-ASCII characters
    filename = filename.replace(/[^\x00-\x7F]/g, '');
    
    // Replace spaces with underscores
    filename = filename.replace(/\s+/g, '_');
    
    // Remove any other potentially dangerous characters
    filename = filename.replace(/[^a-zA-Z0-9._-]/g, '');
    
    // Remove leading underscores (to match Python's secure_filename behavior)
    filename = filename.replace(/^_+/, '');
    
    // Ensure the filename isn't empty after cleaning
    if (!filename) {
        filename = 'file';
    }
    
    return filename;
}


function collapseAdvancedSettings(advancedSettingsId) {
    var advancedSettings = document.getElementById(advancedSettingsId);
    if (advancedSettings.classList.contains('show')) {
        var bsCollapse = new bootstrap.Collapse(advancedSettings, {
            toggle: false
        });
        bsCollapse.hide();
    }
}


function handleHfWaitressShutdown(server_to_shutdown) {
    if (!server_to_shutdown || server_to_shutdown == '') {
        throw new Error('Invalid server choice: ' + server_to_shutdown);
    }
    console.log(`HF-Waitress shutdown requested for server: ${server_to_shutdown}`);
    showStreamSpinner();
    appendStreamInfo(`Shutting down ${server_to_shutdown} server...`, 'waiting');
    fetch('/shutdown_local_llm_server', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({'server_to_shutdown': server_to_shutdown})
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    }).then(data => {
        console.log(data);
        if (!data.success) {
            throw new Error(`Failed to shutdown ${server_to_shutdown} Server. Check server-log and server command-line for more details.`);
        }
        appendStreamInfo(`${server_to_shutdown} Server successfully shutdown.`, 'success');
    }).catch(error => {
        appendStreamInfo(error.message);
    }).finally(() => {
        hideStreamSpinner();
    });
}