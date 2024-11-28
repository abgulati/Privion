

///////////////////////---------SaveConfig Block Begins---------////////////////////////////
function getLlmConfig() {
    const useLocalLlmRadioButton = document.getElementById('local_llm_radio_button');
    const useApiRadioButton = document.getElementById('api_llm_radio_button');
    const update_azure_gpt_config = document.getElementById('update_azure_gpt').checked;

    let config = {
        'use_local_llm': useLocalLlmRadioButton.checked,
        'use_azure_open_ai':false
    };

    if (useLocalLlmRadioButton.checked) {
        config.local_llm_server = document.getElementById('local_llm_server_select_dropdown').value;
        if (config.local_llm_server === "llama-cpp") {
            config.model_choice = document.getElementById('modelDropdown').value;
            config.local_llm_chat_template_format = document.getElementById('llmTemplateDropdown').value;
            config.use_gpu = document.getElementById('gpu_radio_yes').checked;
            config.local_llm_gpu_layers = parseInt(document.getElementById('NumbGpuLayers').value);
            config.local_llm_context_length = parseInt(document.getElementById('LlmCtxLgt').value);
            config.local_llm_max_new_tokens = parseInt(document.getElementById('MaxNewToks').value);
            config.local_llm_temperature = parseFloat(document.getElementById('tempSlider').value);
            config.local_llm_top_k = parseInt(document.getElementById('topkSlider').value);
            config.local_llm_top_p = parseFloat(document.getElementById('toppSlider').value);
            config.local_llm_min_p = parseFloat(document.getElementById('minpSlider').value);
            config.local_llm_n_keep = parseInt(document.getElementById('nkeepSlider').value);
        } else if (config.local_llm_server === "hf-waitress") {
            config.hf_waitress_serving_url = document.getElementById('HfwServingUrl').value;
            config.hf_waitress_access_url = document.getElementById('HfwAccessUrl').value;
            config.hf_waitress_server_port = parseInt(document.getElementById('HfwPort').value);
        }
    } else if (useApiRadioButton.checked) {
        config.model_choice = document.getElementById('llmApiDropdown').value;
        if (config.model_choice === "AzureOpenAI") {
            config.use_azure_open_ai = true;
            if (update_azure_gpt_config) {
                config.azure_openai_base_url = document.getElementById("azure_openai_api_url").value;
                config.azure_openai_api_key = document.getElementById("azure_openai_api_key").value;
                config.azure_openai_deployment_name = document.getElementById("azure_openai_deployment_name").value;
            }
        }
    }

    return config;
}


function getVectorEmbeddingsConfig() {
    const embedding_model_choice = document.getElementById('embedding_model_dropdown').value;
    const update_azure_ada_config = document.getElementById('update_azure_ada').checked;

    let config= {
        'embedding_model_choice': embedding_model_choice,
        'use_openai_embeddings': false,
        'use_bge_large_embeddings': false,
        'use_bge_base_embeddings': false,
        'use_sbert_embeddings': false
    };
    
    switch(embedding_model_choice) {
        case 'bge_large':
            config.use_bge_large_embeddings = true;
            break;
        case 'bge_base':
            config.use_bge_base_embeddings = true;
            break;
        case 'sbert_mpnet_base_v2':
            config.use_sbert_embeddings = true;
            break;
        case 'openai_text_ada':
            config.use_openai_embeddings = true;
            if (update_azure_ada_config) {
                config.azure_openai_text_ada_api_url = document.getElementById("azure_openai_text_ada_api_url").value;
                config.azure_openai_text_ada_api_key = document.getElementById("azure_openai_text_ada_api_key").value;
                config.azure_openai_text_ada_deployment_name = document.getElementById("azure_openai_text_ada_deployment_name").value;
            }
            break;
    }

    return config;
}


function getSysPromptConfig() {
    const skip_system_prompt = document.getElementById('skip_system_prompt_checkbox').checked;
    const selectedTemplateName = document.getElementById('templateDropdown').textContent.trim();
    let prompt_template;

    switch(selectedTemplateName) {
        case 'Default':
            prompt_template = document.getElementById('template_dropdown_default').getAttribute('data-template');
            break;
        case 'Custom':
            prompt_template = String(document.getElementById('templateContent').value);
            break;
    }

    return {'base_template': prompt_template, 'skip_system_prompt': skip_system_prompt};
}


function getRagConfig() {
    let force_enable_rag = false;
    let force_disable_rag = false;
    let llm_filter_citations = false;

    if (forceEnableRagCheckbox.checked) {
        force_enable_rag = true;
    } else if (forceDisableRagCheckbox.checked) {
        force_disable_rag = true;
    }

    if (document.getElementById('llm_filter_citations_checkbox').checked) { 
        llm_filter_citations = true;
    }

    return {'force_enable_rag': force_enable_rag, 'force_disable_rag': force_disable_rag, 'llm_filter_citations': llm_filter_citations};
}


function getOcrConfig() {
    const azure_ocr_form = document.getElementById("azure_vision_api_form");
    const azure_doc_ai_form = document.getElementById("azure_doc_ai_api_form");
    const local_vision_form = document.getElementById("local_vision_api_form");
    const kosmos_form = document.getElementById("kosmos_api_form");
    const update_azure_vision_config = document.getElementById('update_azure_vision').checked;
    const update_azure_doc_ai_config = document.getElementById('update_azure_doc_ai').checked;
    const update_local_vision_config = document.getElementById('update_local_vision_config').checked;
    const update_kosmos_url_config = document.getElementById('update_kosmos_url_config').checked;
    const force_extract_previously_extracted_text = document.getElementById('force_extract_previously_extracted_text_checkbox').checked;

    let config = {
        'ocr_service_choice': 'None',
        'use_ocr': false,
        'force_extract_previously_extracted_text': force_extract_previously_extracted_text
    };

    if (window.getComputedStyle(azure_ocr_form).display != 'none') {
        config.ocr_service_choice = 'AzureVision';
        config.use_ocr = true;
        if (update_azure_vision_config) {
            config.azure_cv_free_tier = document.getElementById('is_azure_cv_free_tier').checked;
            config.azure_ocr_endpoint = document.getElementById("azure_vision_api_url").value;
            config.azure_ocr_subscription_key = document.getElementById("azure_vision_api_key").value;
        }
    } else if (window.getComputedStyle(azure_doc_ai_form).display != 'none') {
        config.ocr_service_choice = 'AzureDocAi';
        config.use_ocr = true;
        if (update_azure_doc_ai_config) {
            config.azure_doc_ai_endpoint = document.getElementById("azure_doc_ai_api_url").value;
            config.azure_doc_ai_subscription_key = document.getElementById("azure_doc_ai_api_key").value;
        }
    } else if (window.getComputedStyle(local_vision_form).display != 'none') {
        config.ocr_service_choice = 'LocalVisionLLM';
        config.use_ocr = true;
        if (update_local_vision_config) {
            config.local_vision_endpoint = document.getElementById("local_vision_api_url").value;
        }
    } else if (window.getComputedStyle(kosmos_form).display != 'none') {
        config.ocr_service_choice = 'Kosmos';
        config.use_ocr = true;
        if (update_kosmos_url_config) {
            config.kosmos_local_url = document.getElementById("kosmos_api_url").value;
        }
        config.kosmos_task = document.getElementById("kosmos_task").value;
        config.kosmos_threshold = document.getElementById("kosmos_threshold").value;
    }

    return config;
}


function getGoogleDriveConfig() {
    config = {

    };
    return config;
}


function getHfWaitressConfig() {
    let hf_config = {
        'model_id': document.getElementById('hf-waitress-llm-custom-dropdown-selected-value').textContent,
        'torch_device_map': document.getElementById('hf_waitress_torch_device_map_choice').value,
        'torch_dtype': document.getElementById('hf_waitress_torch_dtype_choice').value,
        'use_flash_attention_2': document.getElementById('hf_waitress_use_flash_attention_2_yes').checked,
        'trust_remote_code': document.getElementById('hf_waitress_trust_remote_code_yes').checked,
        'flux_diffusers': document.getElementById('hf_waitress_diffusers_yes').checked,
        'flux_low_vram_optimizations': document.getElementById('hf_waitress_diffusers_low_vram_optimizations_yes').checked,
        'load_quantized_flux': document.getElementById('hf_waitress_diffusers_fp8_yes').checked,
        'vision': document.getElementById('hf_waitress_vision_yes').checked,
        'awq': document.getElementById('hf_waitress_is_awq_yes').checked,
        'pipeline_task': document.getElementById('hf_waitress_pipeline_task_choice').value,
        'max_new_tokens': parseInt(document.getElementById('HfwMaxNewToks').value),
        'top_k': parseInt(document.getElementById('HfwTopkSlider').value),
        'top_p': parseFloat(document.getElementById('HfwToppSlider').value),
        'min_p': parseFloat(document.getElementById('HfwMinpSlider').value),
        'return_full_text': document.getElementById('hf_waitress_return_full_text_yes').checked,
        'quantize': document.getElementById('hf_waitress_quantization_choice').value,
        'quant_level': document.getElementById('hf_waitress_quantization_level_choice').value,
        'hqq_group_size': parseInt(document.getElementById('HfwHqqGroupSize').value),
    };

    if (hf_config.model_id.toLowerCase().includes('vision-instruct')) {
        hf_config.vision = true;
        document.getElementById('hf_waitress_vision_yes').checked = true;
        document.getElementById('hf_waitress_vision_no').checked = false;
    } else {
        hf_config.vision = false;
        document.getElementById('hf_waitress_vision_yes').checked = false;
        document.getElementById('hf_waitress_vision_no').checked = true;
    }

    hfw_temperature = parseFloat(document.getElementById('HfwTempSlider').value);
    hf_config.do_sample = hfw_temperature > 0.0;
    hf_config.temperature = Math.max(0.0, hfw_temperature);

    if(document.getElementById('update_hf_access_token').checked) {
        hf_config.access_gated = true;
        hf_config.access_token = document.getElementById('hf_access_token').value;
    }

    return hf_config;
}


function saveConfigToServer(config) {
    return fetch('/config_writer_api', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            'config_updates': config
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        console.log(data);
        if (data.restart_required) {    // Refresh the page
            console.log("LARS config changes saved successfully, restart required.");
            return true;    // Return true to indicate that a restart is required
        }
        console.log("LARS config changes saved successfully, restart unnecessary.");
        return false;   // Return false to indicate that a restart is not required
    })
    .catch(error => {
        errorHandler("writing to config.json", "/config_writer_api", String(error.message));
        return false;   // Return false to indicate that a restart is not required
    });
}


async function fetchRestartEventStream() {                
    const hfWaitress_URL = getHfwUrl();
    const url = hfWaitress_URL + '/restart_server_stream';
    const responseRestartContentElement = document.getElementById('hf-model-loader-stream');

    try {
        const hfwResponse = await fetch(url, {
            method: 'GET',
            redirect: 'follow'
        });

        const hfwReader = hfwResponse.body.getReader();
        let hfwTotalContent = '';
        let hfwReceivedComplete = false;

        async function hfwProcessChunk() {
            let progressLine = '';
            let currentProgressLine = null;

            while (true) {
                const { done, value } = await hfwReader.read();
                if (done) {
                    console.log("HF-Waitress Restart Stream complete");
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
                            // console.log("dataObj: ", dataObj);
                            if (dataObj != "") {

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

                                // Escape HTML special characters
                                let streamed_content = dataObj.replace(/</g, '&lt;')
                                                                .replace(/>/g, '&gt;')
                                                                .replace(/\\t/g, '    ')
                                                                .replace(/\\n\\n/g, '<br><br>')
                                                                .replace(/\\n/g, '<br>') 
                                                                .replace(/\\r/g, '\r')
                                                                .replace(/\\n/g, '\n'); // /g - global - replace throughout string, not just the first occurance

                                
                                // let streamed_content = dataObj.replace(/\\r/g, '\r').replace(/\\n/g, '\n');

                                // Check if the streamed content is a progress-line - \r tags indicate a carriage return, used to replace text in a command-line
                                if (streamed_content.startsWith('\r')) {
                                    if (currentProgressLine) {
                                        currentProgressLine.innerHTML = streamed_content.trim().replace(/\r/g, '').replace(/\n/g, '<br>').replace(/([#+])/g, '');
                                    } else {
                                        currentProgressLine = document.createElement('div');
                                        currentProgressLine.className = 'progress-line';
                                        currentProgressLine.innerHTML = streamed_content.trim().replace(/\r/g, '').replace(/\n/g, '<br>').replace(/([#+])/g, '');
                                        responseRestartContentElement.appendChild(currentProgressLine);
                                    }
                                } else {
                                    responseRestartContentElement.innerHTML += streamed_content.replace(/\n/g, '<br>');
                                    currentProgressLine = null; // We've moved past the progress-line, so set it to null so a new one may be created should it be required
                                }
                            }

                        } catch (error) {
                            console.error('Error parsing message: ', error);
                        }
                    } else if (message.startsWith('event: END') || message.startsWith('data: null')) {
                        console.log("Received null message from hf-waitress restart - stream complete");
                        hfwReceivedComplete = true;
                    }
                });

                if (hfwReceivedComplete) break;
            }
        
        }

        await hfwProcessChunk();
        
    } catch (error) {
        errorHandler("fetching HF-Waitress Restart event-streaming response", "HF-Waitress/restart_server_stream", String(error))
    }
}


function handleHfWaitressChanges(hf_config) {
    return new Promise((resolve, reject) => {
        document.getElementById('hf-model-loader-stream').innerHTML = '';
        document.getElementById('SavingHfWaitressSettings').style.display = 'block';

        const hfWaitress_URL = getHfwUrl();
        fetch(hfWaitress_URL + '/hf_config_writer_api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                'config_updates': hf_config
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error) });
            }
            return response.json();
        })
        .then(data => {
            console.log(data);
            if (data.restart_required) {
                // Restart HF-Waitress Server
                console.log("HF-Waitress server changes saved successfully, restarting server.");
                return fetchRestartEventStream();   // Call to an async function will return a promise, so we need to return the promise...
            } else {
                console.log("HF-Waitress server changes saved successfully, restart unnecessary.");
                resolve();
            }
        })
        .then(() => {
            resolve();  // ...which is resolved here!
        })
        .catch(error => {
            errorHandler("handling HF-Waitress Server changes", "handleHfWaitressChanges", String(error.message));
            reject(error);  // Reject the promise if there's an error
        })
        .finally(() => {
            setTimeout(() => {
                document.getElementById('SavingHfWaitressSettings').style.display = 'none';
            }, 3000);
        });
    });
}


function handleSaveChanges() {
    
    const config = {
        ...getLlmConfig(),
        ...getVectorEmbeddingsConfig(),
        ...getSysPromptConfig(),
        ...getRagConfig(),
        ...getOcrConfig(),
        ...getGoogleDriveConfig()
    };

    let hfSavePromise;  // Declaring the variable that will hold the promise returned by handleHfWaitressChanges(). Defaulting will result in a race condition, so we'll need to handle the Promise chain manually.
    let needsReload = false;

    if (config.local_llm_server === "hf-waitress") {
        const hf_waitress_server_status = document.getElementById('local_llm_server_status_indicator_text').style.color;
        if (hf_waitress_server_status === "green") {
            const hf_config = {
                ...getHfWaitressConfig()
            };
            console.log("handleHfWaitressChanges hf_config: ", hf_config);
            hfSavePromise = handleHfWaitressChanges(hf_config)  // Added to the promise chain only if the conditions are met!
                .then(() => {
                    console.log("HF-Waitress server changes saved successfully.");
                    
                    let chatID = getChatId();
                    setModelHeaderInfoBox(chatID, hf_config.model_id);
                    if (hf_config.model_id.toLowerCase().includes('vision-instruct')) { // we have switched to a Vision-Instruct model...
                        console.log("Loaded up a Vision model");
                        document.getElementById('textAttachmentButton').disabled = false;
                        let current_sequence_id = getSequenceId();
                        if (current_sequence_id != null && parseInt(current_sequence_id) > 0 && !String(getOldLlmModel()).toLowerCase().includes('vision-instruct')) {  //...midway through a chat, or while browsing chat history, and the old LLM model was not a Vision-Instruct model...
                            console.log("Reloading the page to clear the previous model context");
                            needsReload = true;
                        }
                    } else {    //...we have switched to a non-Vision-Instruct model...
                        console.log("Loaded up a non-Vision model");
                        document.getElementById('textAttachmentButton').disabled = true;
                        if (String(getOldLlmModel()).toLowerCase().includes('vision-instruct') || String(getOldLlmModel()).toLowerCase().includes('gguf') || String(getOldLlmModel()).toLowerCase().includes('flux')) {    //...from a Vision-Instruct, FLUX, or a GGUF model...
                            console.log("Reloading the page to clear the previous model context");
                            needsReload = true;
                        }
                    }
                    return needsReload;
                })
                .catch(error => {
                    errorHandler("saving HF-Waitress settings", "handleSaveChanges()", String(error.message));  // Catching the error will resolve the Promise and allow the rest of the code to continue!
                    return false; // hfSavePromise is assigned to the entire promise chain including this catch, so we need only return here to resolve the Promise. False indicates that a restart is not required.
                });
        } else {
            console.log("HF-Waitress server is not running, skipping HF-Waitress changes.");
            hfSavePromise = Promise.resolve(false);
        }
    } else {
        console.log("Not saving HF-Waitress settings, as the LLM server is not HF-Waitress");
        hfSavePromise = Promise.resolve(false);
    }
    
    console.log("Saving LARS config: ", config);
    Promise.all([hfSavePromise, saveConfigToServer(config)])    // This promise will always resolve, as hfSavePromise is manually set to resolve to false when necessary, so we're not blocking the saveConfigToServer() promise. Promise.all() is used to handle both promises in parallel, waiting for both to resolve before proceeding.
        .then(([hfNeedsReload, configNeedsReload]) => {
            console.log("LARS config saved successfully.");
            if (hfNeedsReload || configNeedsReload) {
                console.log("A restart is required to apply the changes.");
                location.reload();  // centralized handling of effects that depend on multiple async operations.
            }
        })
        .catch(error => {
            errorHandler("saving LARS settings", "handleSaveChanges()", String(error.message));
        });
}

document.getElementById('saveChangesButton').addEventListener('click', handleSaveChanges);
//########### End SaveConfig block!