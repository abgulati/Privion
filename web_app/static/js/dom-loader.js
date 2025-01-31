

////////////////////////---------DOMContentLoaded Block Begins---------////////////////////////////
function initializeChatLink() {
    var currentUrl = window.location.href;
    var chatLink = document.getElementById('dynamicChatLink');
    chatLink.href = currentUrl;
}


function initializeScrollDownButton() {
    //Scroll all the way down button:
    const scrollDownButton = document.getElementById('scrollDownButton');
    const chatArea = document.getElementById('chat-area');

    scrollDownButton.addEventListener('click', () => {
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}

function initializeRegenerateResponseButton() {
    document.getElementById('chat-area').addEventListener('click', function(e) {
        if (e.target.classList.contains('regenerate-option')) {
            const userMessageDiv = e.target.closest('.user-message');
            const {streamSessionId, sequenceId} = prepareAttributeForUserMessage(userMessageDiv);
            requestFormattedPrompt(true, false, false, streamSessionId, sequenceId);    // Request the formatted prompt
            deleteChatAreaElements(userMessageDiv.nextElementSibling); // Delete subsequent user messages and response containers
        } else if (e.target.classList.contains('regenerate-with-citations-enabled-option')) {
            const userMessageDiv = e.target.closest('.user-message');
            const {streamSessionId, sequenceId} = prepareAttributeForUserMessage(userMessageDiv);
            requestFormattedPrompt(true, true, false, streamSessionId, sequenceId);    // Request the formatted prompt
            deleteChatAreaElements(userMessageDiv.nextElementSibling); // Delete subsequent user messages and response containers
        } else if (e.target.classList.contains('regenerate-with-citations-disabled-option')) {
            const userMessageDiv = e.target.closest('.user-message');
            const {streamSessionId, sequenceId} = prepareAttributeForUserMessage(userMessageDiv);
            requestFormattedPrompt(true, false, true, streamSessionId, sequenceId);    // Request the formatted prompt
            deleteChatAreaElements(userMessageDiv.nextElementSibling); // Delete subsequent user messages and response containers
        } else if (e.target.classList.contains('delete-option')) {
            const userMessageDiv = e.target.closest('.user-message');
            const chatId = userMessageDiv.getAttribute('data-chat-id');
            const sequenceId = userMessageDiv.getAttribute('data-sequence-id');
            delete_messages(chatId, sequenceId, userMessageDiv);
        }
    });
}


function initializeUI() {
    attachWindowEvents();
    initializeChatLink();
    initializeScrollDownButton();
    initializeRegenerateResponseButton();
}


function setUIValues(values) {
    document.getElementById('NumbGpuLayers').value = values.local_llm_gpu_layers;
    document.getElementById('LlmCtxLgt').value = values.local_llm_context_length;
    document.getElementById('MaxNewToks').value = values.local_llm_max_new_tokens;
    document.getElementById('tempSlider').value = values.local_llm_temperature;
    document.getElementById('tempSliderValue').textContent = values.local_llm_temperature;
    document.getElementById('topkSlider').value = values.local_llm_top_k;
    document.getElementById('topkSliderValue').textContent = values.local_llm_top_k;
    document.getElementById('toppSlider').value = values.local_llm_top_p;
    document.getElementById('toppSliderValue').textContent = values.local_llm_top_p;
    document.getElementById('minpSlider').value = values.local_llm_min_p;
    document.getElementById('minpSliderValue').textContent = values.local_llm_min_p;
    document.getElementById('nkeepSlider').value = values.local_llm_n_keep;
}


function loadCoreLarsConfig() {
    appendStreamInfo("Loading core LARS config...", 'waiting');
    const initKeysToRead = [
        'local_llm_server',
        'model_choice',
        'use_local_llm',
        'use_gpu',
        'embedding_model_choice',
        'embedding_models_list',
        'selected_embedding_model',
        'knowledge_domain_list',
        'selected_knowledge_domain',
        'use_ocr',
        'ocr_service_choice',
        'local_llm_chat_template_format',
        'exl2_prompt_template_format',
        'force_enable_rag',
        'force_disable_rag',
        'base_template',
        'local_llm_gpu_layers',
        'local_llm_context_length',
        'local_llm_max_new_tokens',
        'local_llm_temperature',
        'local_llm_top_k',
        'local_llm_top_p',
        'local_llm_min_p',
        'local_llm_n_keep',
        'azure_cv_free_tier',
        'skip_system_prompt',
        'force_extract_previously_extracted_text',
        'vision_llm_local_url',
        'kosmos_local_url',
        'kosmos_task',
        'kosmos_threshold',
        'llm_filter_citations',
        'hf_waitress_serving_url',
        'hf_waitress_access_url',
        'hf_waitress_server_port',
        'llama_cpp_serving_url',
        'llama_cpp_access_url',
        'llama_cpp_server_port'
    ]
    
    return fetch('/config_reader_api', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({keys: initKeysToRead})
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        setUIValues(data.values);
        return data.values;
    })
    .catch(error => {
        errorHandler("reading config.json", "/config_reader_api", String(error.message));
    });
}


function initializeLocalLLMServerDropdown(local_llm_server, exl2_prompt_template_format) {
    const llmServerDd = document.getElementById('local_llm_server_select_dropdown');

    for (let option of llmServerDd.options) {
        if (option.value == local_llm_server) {
            option.selected = true;
            break;
        }
    }

    document.getElementById('llama_cpp_div').style.display = local_llm_server === 'llama-cpp' ? 'block' : 'none';
    document.getElementById('hf_waitress_div').style.display = local_llm_server === 'hf-waitress' ? 'block' : 'none';

    const hfExl2PromptTemplateFormat = document.getElementById('hf_waitress_exl2_prompt_template_format_choice');

    for (let option of hfExl2PromptTemplateFormat.options) {
        if (option.value == exl2_prompt_template_format) {
            option.selected = true;
            break;
        }
    }
}


function loadCoreHfConfig() {
    appendStreamInfo("Loading core HF-Waitress config...", 'waiting');
    const initKeysToRead = [
        'model_id',
        'model_list',
        'access_gated',
        'trust_remote_code',
        'torch_device_map',
        'torch_dtype',
        'use_flash_attention_2',
        'pipeline_task',
        'max_new_tokens',
        'return_full_text',
        'temperature',
        'do_sample',
        'top_k',
        'top_p',
        'min_p',
        'n_keep',
        'quantize',
        'quant_level',
        'awq',
        'hqq_group_size',
        'flux_diffusers',
        'flux_low_vram_optimizations',
        'load_quantized_flux',
        'vision',
        'exl2',
        'exl2_bpw',
        'exl2_max_seq_len',
        'exl2_cache_type',
        'exl2_force_regenerate_measurement'
    ]
    
    const hfWaitress_URL = getHfwUrl();
    return fetch(hfWaitress_URL + '/hf_config_reader_api', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({keys: initKeysToRead})
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response.json()
    })
    .then(data => {
        return data.values;
    })
    .catch(error => {
        errorHandler("reading hf_config.json (likely means the HF-Waitress server is offline)", "loadCoreHfConfig()", String(error.message))
        appendStreamInfo(`Error: ${String(error.message)}`, 'failure');
    });
}


function initializeHfWaitressCustomDropdown(model_list, model_id) {
    const customDropdownList = document.getElementById('hf-waitress-llm-custom-dropdown-items-list');
    customDropdownList.innerHTML = '';

    const selectedValue = document.getElementById('hf-waitress-llm-custom-dropdown-selected-value');

    model_list.forEach(model => {
        if (model.toLowerCase() == model_id.toLowerCase()) {
            selectedValue.textContent = model;
            if (model_id.toLowerCase().includes('vision-instruct')) {
                document.getElementById('textAttachmentButton').disabled = false;
            }
        }
        const div = document.createElement('div');
        div.className = 'hf-waitress-llm-custom-dropdown-item';
        div.innerHTML = `
            <span>${model}</span>
            <span class="hf-waitress-llm-custom-dropdown-delete-btn">×</span>
        `;
        const deleteButton = div.querySelector('.hf-waitress-llm-custom-dropdown-delete-btn');
        deleteButton.addEventListener('click', (event) => { // instead of deleteButton.onclick = (event) => {
            event.stopPropagation();    //This prevents the click event from bubbling up to the parent div onclick event, which would immediately close the dropdown
            removeModelFromCustomDropdown(model);
        });
        div.addEventListener('click', () => {   // instead of div.onclick = () => {
            selectedValue.textContent = model;
            document.getElementById('hf-waitress-llm-custom-dropdown-content').classList.remove('show');
        });
        customDropdownList.appendChild(div);
    });
}



function initializeKnowledgeDomainCustomDropdown(model_list, model_id) {
    const customDropdownList = document.getElementById('hf-waitress-kb-custom-dropdown-items-list');
    customDropdownList.innerHTML = '';

    const selectedValue = document.getElementById('hf-waitress-kb-custom-dropdown-selected-value');

    model_list.forEach(model => {
        if (model.toLowerCase() == model_id.toLowerCase()) {
            selectedValue.textContent = model;
        }
        const div = document.createElement('div');
        div.className = 'hf-waitress-kb-custom-dropdown-item';
        div.innerHTML = `
            <span class="hf-waitress-kb-custom-dropdown-item-text">${model}</span>
            <span class="hf-waitress-kb-custom-dropdown-delete-btn">×</span>
        `;
        const deleteButton = div.querySelector('.hf-waitress-kb-custom-dropdown-delete-btn');
        deleteButton.addEventListener('click', (event) => { // instead of deleteButton.onclick = (event) => {
            event.stopPropagation();    //This prevents the click event from bubbling up to the parent div onclick event, which would immediately close the dropdown
            removeKnowledgeBaseFromCustomDropdown(model);
        });
        div.addEventListener('click', () => {   // instead of div.onclick = () => {
            selectedValue.textContent = model;
            document.getElementById('hf-waitress-kb-custom-dropdown-content').classList.remove('show');
        });
        customDropdownList.appendChild(div);
    });
}



function initializeEmbeddingCustomDropdown(model_list, model_id) {
    const customDropdownList = document.getElementById('hf-waitress-embed-custom-dropdown-items-list');
    customDropdownList.innerHTML = '';

    const selectedValue = document.getElementById('hf-waitress-embed-custom-dropdown-selected-value');

    model_list.forEach(model => {
        if (model.toLowerCase() == model_id.toLowerCase()) {
            selectedValue.textContent = model;
        }
        const div = document.createElement('div');
        div.className = 'hf-waitress-embed-custom-dropdown-item';
        div.innerHTML = `
            <span class="hf-waitress-embed-custom-dropdown-item-text">${model}</span>
            <span class="hf-waitress-embed-custom-dropdown-delete-btn">×</span>
        `;
        const deleteButton = div.querySelector('.hf-waitress-embed-custom-dropdown-delete-btn');
        deleteButton.addEventListener('click', (event) => { // instead of deleteButton.onclick = (event) => {
            event.stopPropagation();    //This prevents the click event from bubbling up to the parent div onclick event, which would immediately close the dropdown
            removeEmbeddingModelFromCustomDropdown(model);
        });
        div.addEventListener('click', () => {   // instead of div.onclick = () => {
            selectedValue.textContent = model;
            document.getElementById('hf-waitress-embed-custom-dropdown-content').classList.remove('show');
        });
        customDropdownList.appendChild(div);
    });
}



function initializeHfSettingsDropdowns(all_values) {
    const hfTorchDeviceMap = document.getElementById('hf_waitress_torch_device_map_choice');

    for (let option of hfTorchDeviceMap.options) {
        if (option.value == all_values.torch_device_map) {
            option.selected = true;
            break;
        }
    }

    const hfTorchDataType = document.getElementById('hf_waitress_torch_dtype_choice');

    for (let option of hfTorchDataType.options) {
        if (option.value == all_values.torch_dtype) {
            option.selected = true;
            break;
        }
    }

    const hfPipelineTask = document.getElementById('hf_waitress_pipeline_task_choice');

    for (let option of hfPipelineTask.options) {
        if (option.value == all_values.pipeline_task) {
            option.selected = true;
            break;
        }
    }

    const hfQuantMethod = document.getElementById('hf_waitress_quantization_choice');

    for (let option of hfQuantMethod.options) {
        if (option.value == all_values.quantize) {
            option.selected = true;
            break;
        }
    }

    toggleHfwQuantizationLevel();   // This is necessary to ensure the quantization level options are updated based on the selected quantization method.

    const hfQuantLevel = document.getElementById('hf_waitress_quantization_level_choice');

    for (let option of hfQuantLevel.options) {
        if (option.value == all_values.quant_level) {
            option.selected = true;
            break;
        }
    }

    const hfExl2CacheType = document.getElementById('hf_waitress_exl2_cache_type_choice');

    for (let option of hfExl2CacheType.options) {
        if (option.value == all_values.exl2_cache_type) {
            option.selected = true;
            break;
        }
    }
}


function initializeHfRadioButtons(all_values) {
    document.getElementById('hf_waitress_exl2_yes').checked = all_values.exl2;
    document.getElementById('hf_waitress_exl2_no').checked = !all_values.exl2;

    document.getElementById('hf_waitress_is_awq_yes').checked = all_values.awq;
    document.getElementById('hf_waitress_is_awq_no').checked = !all_values.awq;

    document.getElementById('hf_waitress_trust_remote_code_yes').checked = all_values.trust_remote_code;
    document.getElementById('hf_waitress_trust_remote_code_no').checked = !all_values.trust_remote_code;

    document.getElementById('hf_waitress_diffusers_yes').checked = all_values.flux_diffusers;
    document.getElementById('hf_waitress_diffusers_no').checked = !all_values.flux_diffusers;

    document.getElementById('hf_waitress_vision_yes').checked = all_values.vision;
    document.getElementById('hf_waitress_vision_no').checked = !all_values.vision;

    toggleHfwDiffusersConfig();
    toggleHfwExl2Config();

    document.getElementById('hf_waitress_diffusers_low_vram_optimizations_yes').checked = all_values.flux_low_vram_optimizations;
    document.getElementById('hf_waitress_diffusers_low_vram_optimizations_no').checked = !all_values.flux_low_vram_optimizations;

    document.getElementById('hf_waitress_diffusers_fp8_yes').checked = all_values.load_quantized_flux;
    document.getElementById('hf_waitress_diffusers_fp8_no').checked = !all_values.load_quantized_flux;

    document.getElementById('hf_waitress_use_flash_attention_2_yes').checked = all_values.use_flash_attention_2;
    document.getElementById('hf_waitress_use_flash_attention_2_no').checked = !all_values.use_flash_attention_2;

    document.getElementById('hf_waitress_return_full_text_yes').checked = all_values.return_full_text;
    document.getElementById('hf_waitress_return_full_text_no').checked = !all_values.return_full_text;
}


function setHfSlidersAndTextAreas(values) {
    document.getElementById('HfwHqqGroupSize').value = values.hqq_group_size;
    document.getElementById('HfwMaxNewToks').value = values.max_new_tokens;
    document.getElementById('HfwTempSlider').value = values.temperature;
    document.getElementById('HfwTempSliderValue').textContent = values.temperature;
    document.getElementById('HfwTopkSlider').value = values.top_k;
    document.getElementById('HfwTopkSliderValue').textContent = values.top_k;
    document.getElementById('HfwToppSlider').value = values.top_p;
    document.getElementById('HfwToppSliderValue').textContent = values.top_p;
    document.getElementById('HfwMinpSlider').value = values.min_p;
    document.getElementById('HfwMinpSliderValue').textContent = values.min_p;
    document.getElementById('HfwExl2Bpw').value = values.exl2_bpw;
    document.getElementById('HfwExl2MaxSeqLen').value = values.exl2_max_seq_len;
    document.getElementById('hf_waitress_exl2_force_regenerate_measurement').checked = values.exl2_force_regenerate_measurement;
}


function initializeHfwServerConfig() {
    loadCoreHfConfig()
        .then(hf_values => {
            setVision(hf_values.vision);
            setExl2(hf_values.exl2);
            initializeHfWaitressCustomDropdown(hf_values.model_list, hf_values.model_id);
            initializeHfSettingsDropdowns(hf_values);
            initializeHfRadioButtons(hf_values);
            setHfSlidersAndTextAreas(hf_values);
        })
        .catch(error => {
            errorHandler("initializing HF-Waitress Server config (likely means the HF-Waitress server is offline)", "initializeHfwServerConfig()", String(error.message));
        });
}


function initializeModelDropdown(model_choice) {
    fetch('/load_local_models')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            //console.log('model_choice:', model_choice);
            const dropdown = document.getElementById('modelDropdown');
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                //console.log('Model:', model);

                if (typeof model !== 'undefined' && typeof model_choice !== 'undefined') {
                    if (model.toLowerCase() == model_choice.toLowerCase()) {
                        option.selected = true;
                    }
                }
                dropdown.appendChild(option);
            });
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("fetching the model list", "/load_local_models", String(error.message))
    });
}


function initializeLLMTemplateDropdown(local_llm_chat_template_format) {
    const llmTempDd = document.getElementById('llmTemplateDropdown');

    for (let option of llmTempDd.options) {
        if (option.value == local_llm_chat_template_format) {
            option.selected = true;
            break;
        }
    }
}


function initializeGPURadioButtons(use_gpu) {
    document.getElementById('gpu_radio_yes').checked = use_gpu;
    document.getElementById('gpu_radio_no').checked = !use_gpu;
}


function initializeLLMRadioButtons(use_local_llm, model_choice) {
    var useLocalLlm = document.getElementById('local_llm_radio_button');
    var useApiLlm = document.getElementById('api_llm_radio_button');

    if (use_local_llm) {
        useLocalLlm.checked = true;
        useApiLlm.checked = false;
    } else {
        useLocalLlm.checked = false;
        
        // select API service from dropwdown
        var selectLlmApiForDropdown = document.getElementById('llmApiDropdown');

        for (var i = 0; i < selectLlmApiForDropdown.length; i++) {
            if (selectLlmApiForDropdown.options[i].value === model_choice) {
                selectLlmApiForDropdown.options[i].selected = true;
                break;
            }
        }
        useApiLlm.checked = true;
    }
}


function toggleLocalLlmSelection() {    // Show or hide local-LLM selection:
    const selection = document.querySelector('input[name="use_local_or_api_llm"]:checked').value;
    document.getElementById('localLlmDiv').style.display = selection === 'local' ? 'block' : 'none';
    document.getElementById('apiLlmDiv').style.display = selection === 'api' ? 'block' : 'none';

    if (selection === 'local') {
        document.getElementById('azure_openai_api_form').style.display = 'none'; //Hide API-Details-Form if local selected

        const local_llm_server_selected = document.getElementById('local_llm_server_select_dropdown');

        if (local_llm_server_selected === 'hf-waitress') {
            document.getElementById('hf_waitress_div').style.display = 'block';
            document.getElementById('llama_cpp_div').style.display = 'none';
        } else if (local_llm_server_selected === 'llama-cpp') {
            document.getElementById('hf_waitress_div').style.display = 'none';
            document.getElementById('llama_cpp_div').style.display = 'block';
        }
    } else {
        // Collapse Advanced Settings if it's expanded
        collapseAdvancedSettings('advancedLlmSettings');
        collapseAdvancedSettings('advancedHfLlmSettings');
    }
}


function toggleLocalLLMSettingsForms() {
    const selection = document.getElementById('local_llm_server_select_dropdown').value;

    if (selection === 'hf-waitress') {
        initializeHfwServerConfig();
    }

    document.getElementById('llama_cpp_div').style.display = selection === 'llama-cpp' ? 'block' : 'none';
    document.getElementById('hf_waitress_div').style.display = selection === 'hf-waitress' ? 'block' : 'none';
}


function toggleLlmApiForm() {   // Show or hide API form:
    var selection = document.getElementById('llmApiDropdown').value;
    document.getElementById('azure_openai_api_form').style.display = selection === 'AzureOpenAI' ? 'block' : 'none';
    //Add more API form selectors here
}


function toggleNGL() {  // Enable or disable ngl based on use_gpu
    var selection = document.querySelector('input[name="use_gpu"]:checked').value;
    if(selection === 'y') {
        document.getElementById("NumbGpuLayers").disabled = false;
    } else {
        document.getElementById('NumbGpuLayers').value = "0";
        document.getElementById("NumbGpuLayers").disabled = true;
    }
}


function toggleHfwDiffusersConfig() {
    var selection = document.querySelector('input[name="hf_use_diffusers"]:checked').value;
    if (selection === 'y') {
        document.getElementById('hf-waitress-diffusers-configuration-div').style.display = 'block';
        document.getElementById('hf-waitress-exl2-configuration-div').style.display = 'none';
        disableTransformersSettings();
        collapseAdvancedSettings('advancedHfLlmSettings');
    } else {
        document.getElementById('hf-waitress-diffusers-configuration-div').style.display = 'none';
        enableTransformersSettings();
    }
}


function toggleHfwExl2Config() {
    var selection = document.querySelector('input[name="hf_use_exl2"]:checked').value;
    if (selection === 'y') {
        document.getElementById('hf-waitress-exl2-configuration-div').style.display = 'block';
        document.getElementById('hf-waitress-diffusers-configuration-div').style.display = 'none';
        disableNonExl2Settings();
    } else {
        document.getElementById('hf-waitress-exl2-configuration-div').style.display = 'none';
        enableNonExl2Settings();
    }
}


function toggleHfwVisionConfig() {
    var selection = document.querySelector('input[name="hf_use_vision"]:checked').value;
    if (selection === 'y') {
        document.getElementById('hf_waitress_diffusers_no').checked = true;
        document.getElementById('hf_waitress_diffusers_yes').checked = false;
    }
}


function disableFluxQuantization() {
    var lv_selection = document.querySelector('input[name="hf_diffusers_use_low_vram_optimizations"]:checked').value;
    
    if (lv_selection === 'y') {
        document.getElementById('hf_waitress_diffusers_fp8_yes').checked = false;
        document.getElementById('hf_waitress_diffusers_fp8_no').checked = true;
    } 
}


function disableFluxLowVram() {
    var quant_selection = document.querySelector('input[name="hf_diffusers_use_fp8"]:checked').value;

    if (quant_selection === 'y') {
        document.getElementById('hf_waitress_diffusers_low_vram_optimizations_yes').checked = false;
        document.getElementById('hf_waitress_diffusers_low_vram_optimizations_no').checked = true;
    }
}


function toggleHfwQuantizationLevel() {
    const quantMethod = document.getElementById('hf_waitress_quantization_choice').value;
    const quantLevelSelection = document.getElementById('hf_waitress_quantization_level_choice');

    //Clear existing options:
    quantLevelSelection.innerHTML = "";

    // Define available options for each quantization method:
    const options = {
        quanto: [
            {value: "float8", title: "Float8 Quantization of Weights & Activations."},
            {value: "int8", title: "Int8 Quantization of Weights & Activations."},
            {value: "int4", title: "Int4 Quantization of Weights, Int8 Quantization of Activations."},
            {value: "int2", title: "Int2 Quantization of Weights, Int8 Quantization ofActivations."}
        ],
        bitsandbytes: [
            {value: "int8", title: "Int8 Quantization"},
            {value: "int4", title: "Int4 Quantization"}
        ],
        hqq: [
            {value: "int8", title: "Int8 Quantization"},
            {value: "int4", title: "Int4 Quantization"},
            {value: "int3", title: "Int3 Quantization"},
            {value: "int2", title: "Int2 Quantization"},
            {value: "int1", title: "Int1 Quantization"}
        ],
        n: []
    };

    // Add options based on quantization method:
    if (options[quantMethod]) {
        options[quantMethod].forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.value;
            option.title = opt.title;
            quantLevelSelection.appendChild(option);
        });
    }

    // Show or hide HQQ Group Size input based on quantization method:
    document.getElementById('hf_waitress_hqq_group_size').style.display = quantMethod === 'hqq' ? 'block' : 'none';

    // Disable the Selection of No Quantization methos is selected
    quantLevelSelection.disabled = quantMethod === 'n';

}


function initializeEventListenersForLLMTab() {
    // Event listener for Local LLM Server selection:
    document.getElementById('local_llm_server_select_dropdown').addEventListener('change', checkLocalLLMServerStatus);
    document.getElementById('local_llm_server_select_dropdown').addEventListener('change', toggleLocalLLMSettingsForms);

    // Event listener for LLM/API radio buttons:
    document.getElementById('local_llm_radio_button').addEventListener('change', toggleLlmApiForm);
    document.getElementById('local_llm_radio_button').addEventListener('change', toggleLocalLlmSelection);

    document.getElementById('api_llm_radio_button').addEventListener('change', toggleLlmApiForm);
    document.getElementById('api_llm_radio_button').addEventListener('change', toggleLocalLlmSelection);

    // Event listener for API dropdown:
    document.getElementById('llmApiDropdown').addEventListener('change', toggleLlmApiForm);

    // Event Listener for toggle GPU:
    document.getElementById('gpu_radio_yes').addEventListener('change', toggleNGL);
    document.getElementById('gpu_radio_no').addEventListener('change', toggleNGL);

    // Event listener for toggle Vision:
    document.getElementById('hf_waitress_vision_yes').addEventListener('change', toggleHfwVisionConfig);
    document.getElementById('hf_waitress_vision_no').addEventListener('change', toggleHfwVisionConfig);

    // Event listener for toggle Diffusers:
    document.getElementById('hf_waitress_diffusers_yes').addEventListener('change', toggleHfwDiffusersConfig);
    document.getElementById('hf_waitress_diffusers_no').addEventListener('change', toggleHfwDiffusersConfig);

    // Event listener for toggle Exl2:
    document.getElementById('hf_waitress_exl2_yes').addEventListener('change', toggleHfwExl2Config);
    document.getElementById('hf_waitress_exl2_no').addEventListener('change', toggleHfwExl2Config);

    // Event listener for toggle Flux Low Vram:
    document.getElementById('hf_waitress_diffusers_low_vram_optimizations_yes').addEventListener('change', disableFluxQuantization);

    // Event listener for toggle Flux Quantization:
    document.getElementById('hf_waitress_diffusers_fp8_yes').addEventListener('change', disableFluxLowVram);

    // Event listener for Reset Defaults button:
    document.getElementById('resetLlmAdvancedDefaults').addEventListener('click', resetLlmAdvancedDefaults);    //aparently, adding parenthesis () here will cause resetLlmAdvancedDefaults to run immediately when the script executes, not on button click. Removing them allows the function to be passed as a reference correctly on click.
    document.getElementById('resetHfLlmAdvancedDefaults').addEventListener('click', resetHfLlmAdvancedDefaults);

    // Event listener for toggle Quantization Level:
    document.getElementById('hf_waitress_quantization_choice').addEventListener('change', toggleHfwQuantizationLevel);

    // Initial state check:
    toggleLocalLlmSelection();
    toggleLlmApiForm();
    toggleNGL();

    document.getElementById('update_azure_gpt').addEventListener('change', function() {
        document.getElementById('azure_openai_api_url').disabled = !this.checked;   
        document.getElementById('azure_openai_api_key').disabled = !this.checked;   
        document.getElementById('azure_openai_deployment_name').disabled = !this.checked;
    });

    document.getElementById('update_hf_access_token').addEventListener('change', function() {
        document.getElementById('hf_access_token').disabled = !this.checked;   
    });

    // Event listener to toggle custom dropdown:
    document.getElementById('hf-waitress-llm-custom-select-header').addEventListener('click', function() {
        document.getElementById('hf-waitress-llm-custom-dropdown-content').classList.toggle('show');
    });

    // Event listener to filter custom dropdown:
    document.getElementById('hf-waitress-llm-custom-dropdown-search-input').addEventListener('input', function() {
        filterCustomDropdown(this.value);
    });

    // Event listener to add new item to custom dropdown:
    const addInput = document.getElementById('hf-waitress-llm-custom-dropdown-add-input');
    const addBtn = document.getElementById('hf-waitress-llm-custom-dropdown-add-btn');

    addBtn.addEventListener('click', function() {
        addBtn.style.display = 'none';
        addInput.style.display = 'block';
    });

    addInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && this.value) {
            addNewHfwModel(this.value);
            addInput.value = '';
            addInput.style.display = 'none';
            addBtn.style.display = 'block';
        }
    });
}


function initializeEventListenersForLLMTabSliders() {
    document.getElementById('tempSlider').addEventListener('input', function() {
        document.getElementById('tempSliderValue').textContent = this.value;
    });

    document.getElementById('topkSlider').addEventListener('input', function() {
        document.getElementById('topkSliderValue').textContent = this.value;
    });

    document.getElementById('toppSlider').addEventListener('input', function() {
        document.getElementById('toppSliderValue').textContent = this.value;
    });

    document.getElementById('minpSlider').addEventListener('input', function() {
        document.getElementById('minpSliderValue').textContent = this.value;
    });

    document.getElementById('HfwTempSlider').addEventListener('input', function() {
        document.getElementById('HfwTempSliderValue').textContent = this.value;
    });

    document.getElementById('HfwTopkSlider').addEventListener('input', function() {
        document.getElementById('HfwTopkSliderValue').textContent = this.value;
    });

    document.getElementById('HfwToppSlider').addEventListener('input', function() {
        document.getElementById('HfwToppSliderValue').textContent = this.value;
    });

    document.getElementById('HfwMinpSlider').addEventListener('input', function() {
        document.getElementById('HfwMinpSliderValue').textContent = this.value;
    });

    document.getElementById('HfDGuidanceScaleSlider').addEventListener('input', function() {
        document.getElementById('HfDGuidanceScaleSliderValue').textContent = this.value;
    });
}


function initializeHfwUrlComponents(hf_waitress_serving_url, hf_waitress_access_url, hf_waitress_server_port) {
    document.getElementById('HfwServingUrl').value = hf_waitress_serving_url;
    document.getElementById('HfwAccessUrl').value = hf_waitress_access_url;
    document.getElementById('HfwPort').value = hf_waitress_server_port;
    setHfwUrl(hf_waitress_access_url, hf_waitress_server_port);
}


function initializeLLMTabComponents(values) {
    initializeLocalLLMServerDropdown(values.local_llm_server, values.exl2_prompt_template_format);
    initializeModelDropdown(values.model_choice);   
    initializeLLMTemplateDropdown(values.local_llm_chat_template_format);
    initializeGPURadioButtons(values.use_gpu);
    initializeLLMRadioButtons(values.use_local_llm, values.model_choice);
    initializeHfwUrlComponents(values.hf_waitress_serving_url, values.hf_waitress_access_url, values.hf_waitress_server_port);
    initializeEventListenersForLLMTab();
    initializeEventListenersForLLMTabSliders();
}


function initializeEventListenersForEmbeddingModelTab() {

    // Add Event Listener for ResetDB button:
    document.getElementById('resetVectorDB').addEventListener('click', resetVectorDBtoBlank);

    // Event listener to toggle custom dropdown:
    document.getElementById('hf-waitress-kb-custom-select-header').addEventListener('click', function() {
        document.getElementById('hf-waitress-kb-custom-dropdown-content').classList.toggle('show');
    });

    document.getElementById('hf-waitress-embed-custom-select-header').addEventListener('click', function() {
        document.getElementById('hf-waitress-embed-custom-dropdown-content').classList.toggle('show');
    });

    // Event listener to filter custom dropdown:
    document.getElementById('hf-waitress-kb-custom-dropdown-search-input').addEventListener('input', function() {
        filterCustomKbDropdown(this.value);
    });

    document.getElementById('hf-waitress-embed-custom-dropdown-search-input').addEventListener('input', function() {
        filterCustomEmbedDropdown(this.value);
    });

    // Event listener to populate docs table when a different embedding model is selected:
    document.getElementById('hf-waitress-embed-custom-dropdown-items-list').addEventListener('click', function(e) {
        if (e.target.matches('.hf-waitress-embed-custom-dropdown-item') || e.target.matches('.hf-waitress-embed-custom-dropdown-item-text')) {
            clearDocsLoadedTable();
            populateDocsLoadedTable();
        }
    });

    // Event listener to populate docs table when a different knowledge domain is selected:
    document.getElementById('hf-waitress-kb-custom-dropdown-items-list').addEventListener('click', function(e) {
        if (e.target.matches('.hf-waitress-kb-custom-dropdown-item') || e.target.matches('.hf-waitress-kb-custom-dropdown-item-text')) {
            clearDocsLoadedTable();
            populateDocsLoadedTable();
        }
    });

    // Event listener to add new item to custom dropdown:
    const addKbInput = document.getElementById('hf-waitress-kb-custom-dropdown-add-input');
    const addKbBtn = document.getElementById('hf-waitress-kb-custom-dropdown-add-btn');

    addKbBtn.addEventListener('click', function() {
        addKbBtn.style.display = 'none';
        addKbInput.style.display = 'block';
    });

    addKbInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && this.value) {
            addNewKnowledgeDomain(this.value); //TODO: add function
            addKbInput.value = '';
            addKbInput.style.display = 'none';
            addKbBtn.style.display = 'block';
        }
    });

    // Event listener to add new item to custom dropdown:
    const addEmbedInput = document.getElementById('hf-waitress-embed-custom-dropdown-add-input');
    const addEmbedBtn = document.getElementById('hf-waitress-embed-custom-dropdown-add-btn');

    addEmbedBtn.addEventListener('click', function() {
        addEmbedBtn.style.display = 'none';
        addEmbedInput.style.display = 'block';
    });

    addEmbedInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && this.value) {
            addNewEmbeddingModel(this.value); //TODO: add function
            addEmbedInput.value = '';
            addEmbedInput.style.display = 'none';
            addEmbedBtn.style.display = 'block';
        }
    });


}


function initializeEmbeddingModelTabComponents(values) {
    initializeKnowledgeDomainCustomDropdown(values.knowledge_domain_list, values.selected_knowledge_domain);
    initializeEmbeddingCustomDropdown(values.embedding_models_list, values.selected_embedding_model);
    clearDocsLoadedTable();
    populateDocsLoadedTable();
    initializeEventListenersForEmbeddingModelTab();
}


function initializeSystemPromptTabListeners(base_template) {
    let default_system_template = document.getElementById('template_dropdown_default').getAttribute('data-template');

    if (base_template.toLowerCase().trim() != default_system_template.toLowerCase().trim()) {
        document.getElementById('template_dropdown_custom').setAttribute('data-template', base_template);
        document.getElementById('templateContent').value = base_template;
        document.getElementById('templateContent').readOnly = false;
        document.getElementById('templateDropdown').textContent = 'Custom';
    }

    // 3b. Set event-listeners for the template-choices:
    let dropdownItems = document.querySelectorAll('.dropdown-item');    // Get a reference to the dropdown items and the textarea
    let textarea = document.getElementById('templateContent');

    // 3c. Loop through each dropdown item
    dropdownItems.forEach(function(item) {
        item.addEventListener('click', function(event) {

            event.preventDefault();

            // Get the template content from the clicked dropdown item's data-template attribute
            let templateContent = event.target.getAttribute('data-template');
            let isReadonly = event.target.getAttribute('data-readonly') === 'true'; // Convert string to boolean via '===' strict equality check: if the data-readonly attribute is true, the boolean isReadonly will be true, else false!

            // Set the textarea value to the template content
            textarea.value = templateContent;

            // Set readonly attribute and placeholder
            textarea.readOnly = isReadonly;
            textarea.placeholder = isReadonly ? "" : "Use this space to enter your own prompt"; // No placeholder if readonly

            // Change the dropdown button text to reflect the chosen template
            document.getElementById('templateDropdown').textContent = event.target.textContent;

        });
    });
}


function initializeSystemPromptTabComponents(values) {
    setDefaultTemplate();   // Set the default template when the page loads
    initializeSystemPromptTabListeners(values.base_template);
    if (values.skip_system_prompt) {
        document.getElementById('skip_system_prompt_checkbox').checked = true;
    }
}


function initializeRAGTabCheckbox(force_enable_rag, force_disable_rag) {
    // Set RAG-Checkbox       
    if (force_enable_rag) {
        console.log("Force enabling RAG")
        updateCheckboxes(forceEnableRagCheckbox);
    } else if (force_disable_rag) {
        console.log("Force disabling RAG")
        updateCheckboxes(forceDisableRagCheckbox);
    } else {
        updateCheckboxes(defaultRag);
    }
}


function initializeRAGTabComponents(values) {
    initializeRAGTabCheckbox(values.force_enable_rag , values.force_disable_rag);
    if (values.llm_filter_citations) {
        document.getElementById('llm_filter_citations_checkbox').checked = true;
    }
}


function initializeOCRTabRadios(use_ocr, ocr_service_choice, azure_cv_free_tier) {
    var useOcr = document.getElementById('ocr_yes_radio_button');
    var usePypdf = document.getElementById('ocr_no_radio_button');

    if (use_ocr) {
        useOcr.checked = true;

        // select API service from dropwdown
        var selectOcrApiForDropdown = document.getElementById('ocrApiDropdown');

        for (var i = 0; i < selectOcrApiForDropdown.length; i++) {
            if (selectOcrApiForDropdown.options[i].value === ocr_service_choice) {
                selectOcrApiForDropdown.options[i].selected = true;
                break;
            }
        }
        
        if (azure_cv_free_tier){
            document.getElementById('is_azure_cv_free_tier').checked = true;
        } else {
            document.getElementById('is_azure_cv_free_tier').checked = false;
        }

        usePypdf.checked = false;
    } else {
        useOcr.checked = false;
        usePypdf.checked = true;
    }
}


function toggleOcrSelection() { //Show or hide OCR-Service selection:
    var selection = document.querySelector('input[name="specify_ocr_and_service"]:checked').value;
    document.getElementById('apiOcrSelection').style.display = selection === 'ocr' ? 'block' : 'none';
    if (selection === 'pypdf') {    //Hide API-Details-Form 
        document.getElementById('azure_vision_api_form').style.display = 'none'; 
        document.getElementById('azure_doc_ai_api_form').style.display = 'none';
        document.getElementById('local_vision_api_form').style.display = 'none';
        document.getElementById('kosmos_api_form').style.display = 'none';
    } else {
        if (document.getElementById('ocrApiDropdown').value != 'AzureVision') {
            document.getElementById('azure_vision_api_form').style.display = 'none'; //Hide API-Details-Form
        } else if (document.getElementById('ocrApiDropdown').value != 'AzureDocAi') {
            document.getElementById('azure_doc_ai_api_form').style.display = 'none';
        } else if (document.getElementById('ocrApiDropdown').value != 'LocalVisionLLM') {
            document.getElementById('local_vision_api_form').style.display = 'none';
        } else if (document.getElementById('ocrApiDropdown').value != 'Kosmos') {
            document.getElementById('kosmos_api_form').style.display = 'none';
        }
    }
}


function toggleOcrApiForm() {   //Show or hide API form:
    var selection = document.getElementById('ocrApiDropdown').value;
    document.getElementById('azure_vision_api_form').style.display = selection === 'AzureVision' ? 'block' : 'none';
    document.getElementById('azure_doc_ai_api_form').style.display = selection === 'AzureDocAi' ? 'block' : 'none';
    document.getElementById('local_vision_api_form').style.display = selection === 'LocalVisionLLM' ? 'block' : 'none';
    document.getElementById('kosmos_api_form').style.display = selection === 'Kosmos' ? 'block' : 'none';
    //Add more API form selectors here
}


function initializeOCRTabListeners() {
    document.getElementById('update_azure_vision').addEventListener('change', function() {
        document.getElementById('azure_vision_api_url').disabled = !this.checked;    
        document.getElementById('azure_vision_api_key').disabled = !this.checked;
        document.getElementById('is_azure_cv_free_tier').disabled = !this.checked;  
    });
    document.getElementById('update_azure_doc_ai').addEventListener('change', function() {
        document.getElementById('azure_doc_ai_api_url').disabled = !this.checked;   // !this.check -> logical NOT to flip boolean. this.checked will be true when the box is checked, so !this.checked will be false... 
        document.getElementById('azure_doc_ai_api_key').disabled = !this.checked;   // ...and so disabled will be false. So if Update-Config is checked, the fields are enabled.
    });
    document.getElementById('update_local_vision_config').addEventListener('change', function() {
        document.getElementById('local_vision_api_url').disabled = !this.checked;
    });
    document.getElementById('update_kosmos_url_config').addEventListener('change', function() {
        document.getElementById('kosmos_api_url').disabled = !this.checked;
    });

    // Event listener for radio buttons:
    document.getElementById('ocr_yes_radio_button').addEventListener('change', toggleOcrApiForm);
    document.getElementById('ocr_yes_radio_button').addEventListener('change', toggleOcrSelection);

    document.getElementById('ocr_no_radio_button').addEventListener('change', toggleOcrApiForm);
    document.getElementById('ocr_no_radio_button').addEventListener('change', toggleOcrSelection);

    // Initial state check:
    toggleOcrSelection();
    toggleOcrApiForm();

    // Event listener for API dropdown:
    document.getElementById('ocrApiDropdown').addEventListener('change', toggleOcrApiForm);
}


function initializeOCRTabComponents(values) {
    initializeOCRTabRadios(values.use_ocr, values.ocr_service_choice, values.azure_cv_free_tier);
    initializeOCRTabListeners();
    if (values.force_extract_previously_extracted_text) { document.getElementById('force_extract_previously_extracted_text_checkbox').checked = true; }
    if (values.kosmos_local_url) { document.getElementById('kosmos_api_url').value = values.kosmos_local_url; }
    if (values.kosmos_task) { document.getElementById('kosmos_task').value = values.kosmos_task; }
    if (values.kosmos_threshold) { document.getElementById('kosmos_threshold').value = values.kosmos_threshold; }
    if (values.vision_llm_local_url) { document.getElementById('local_vision_api_url').value = values.vision_llm_local_url; }
}


function getFileIconClass(fileType) {
    const iconMap = {
        'word': 'fa-file-word',
        'presentation': 'fa-file-powerpoint',
        'image': 'fa-file-image',
        'text': 'fa-file-alt',
        'folder': 'fa-folder',
        'pdf': 'fa-file-pdf',
        'other': 'fa-file'
    };
    return iconMap[fileType] || iconMap['other'];
}


function initializeGoogleDriveTabListeners() {
    document.getElementById('googleDriveLogin').addEventListener('click', googleDriveLogin);
    document.getElementById('googleDriveLogout').addEventListener('click', googleDriveLogout);
    initSortingAndFiltering();  // Inititalize sorting and filtering
    document.getElementById('googleDriveSyncAction').addEventListener('click', triggerSyncGoogleDrive);
}


function initializeGoogleDriveTabComponents(values) {
    initializeGoogleDriveTabListeners();
}


function initializeSettingsModalTabCycleListener() {
    // Now that all on-load setup for Modal-Setting's Tabs based on config.json is completed, set listeners to cycle between tabs:
    let options = document.querySelectorAll('[data-content]');

    options.forEach(function(option) {
        option.addEventListener('click', function(e) {
            e.preventDefault();

            // 1) Deavtivate all options: Make the clicked option active, by first marking all options as inactive
            options.forEach(function(opt) {
                opt.classList.remove('active');
            });
            option.classList.add('active'); //2) Activate the clicked option

            // 3) Hide all detail panels on the right
            // w-75 > div selects all div elements that are direct children of an element of class w-75. CSS selector syntax, with > denoting a direct child relationship
            let details = document.querySelectorAll('.w-75 > div');
            details.forEach(function(detail) {
                detail.classList.add('d-none');
            });

            // 4) Reveal details for clicked option, by first hiding the detail divs for all options
            let selectDetail = document.getElementById(option.getAttribute('data-content'));
            selectDetail.classList.remove('d-none');
        });
    });
}


function updateUIWithChatInfo(chat_id, llm_model) {
    curr_chat_id = chat_id
    setModelHeaderInfoBox(curr_chat_id, llm_model);
    document.getElementById('model_header').style.display = 'block';
    
    document.getElementById('ModelAndDBLoading').style.display = 'none';
    document.getElementById('ReadyToChat').style.display = 'block';
    
    var timeoutDelayInMilliseconds = 1500; //1.5 seconds
    setTimeout(function() {
        document.getElementById('ReadyToChat').style.display = 'none';
    }, timeoutDelayInMilliseconds);
}


function initChatHistoryDB(llm_model) {
    return fetch('/init_chat_history_db')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // If LLM, VectorDB and chat history DB initialized successfully, continue:
            setChatId(data.chat_id);    //chat_id is requred to determine current chat ID, but we don't want to set sequence_id here as DOM load begins a new chat!
            updateUIWithChatInfo(data.chat_id, llm_model)
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("initializing the chat history DB", "/init_chat_history_db", String(error.message))
    });
}


function loadVectorDB(llm_model) {
    // Once the llama.cpp server is up (the LLM is loaded), load the VectorDB
    return fetch('/load_vectordb')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            return initChatHistoryDB(llm_model);
        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("loading the VectorDB", "/load_vectordb", String(error.message))
    });
}


function startLLMServer() {
    // Finally, trigger the LLM & vectorDB starter!
    appendStreamInfo("Starting LLM Server...", 'waiting');
    return fetch('/local_llm_server_starter')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (data.other_server_running) {
                alert("Warning: Both HF-Waitress and llama.cpp servers appear to be running. Consider manually shutting one down to conserve memory!")
            }
            setServerStatusIndicator("Online");
            const llm_model = data.llm_model;
            setLlmModel(llm_model);
            // return loadVectorDB(llm_model);
            return initChatHistoryDB(llm_model);
        } else {
            setServerStatusIndicator("Error");
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        setServerStatusIndicator("Offline");
        errorHandler("loading the LLM", "/local_llm_server_starter", String(error.message))
    })
}


document.addEventListener("DOMContentLoaded", function() {
    initializeUI();
    document.getElementById('ModelAndDBLoading').style.display = 'block';   //Start initializing the chat session
    showStreamSpinner();

    let local_llm_server;

    loadCoreLarsConfig()
        .then(values => {
            initializeLLMTabComponents(values);
            initializeEmbeddingModelTabComponents(values);
            initializeSystemPromptTabComponents(values);
            initializeRAGTabComponents(values);
            initializeOCRTabComponents(values);
            initializeGoogleDriveTabComponents(values);
            initializeSettingsModalTabCycleListener();
            local_llm_server = values.local_llm_server;
            return startLLMServer();
        })
        .then(() => {
            if (local_llm_server === "hf-waitress") {
                initializeHfwServerConfig();
                showStopGenerationButton();
            }
            loadChatHistoryMenu();  // should be done regardless of errors
            handleGoogleDrivePostAuth();
            hideStreamSpinner();
            appendStreamInfo("All loaded up & ready to chat!", 'success');
        })
        .catch(error => {
            errorHandler("initializing the application", "DOMContentLoaded", String(error.message));
        })
        .finally(() => {

        });
});//########### End DOMContentLoaded Block!