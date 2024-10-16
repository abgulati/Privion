

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


function initializeUI() {
    attachWindowEvents();
    initializeChatLink();
    initializeScrollDownButton();
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


function loadAndSetCoreValues() {
    const initKeysToRead = ['local_llm_server', 'model_choice', 'use_local_llm', 'use_gpu', 'embedding_model_choice', 'use_ocr', 'ocr_service_choice', 'local_llm_chat_template_format', 'force_enable_rag', 'force_disable_rag', 'base_template', 'local_llm_gpu_layers', 'local_llm_context_length', 'local_llm_max_new_tokens', 'local_llm_temperature', 'local_llm_top_k', 'local_llm_top_p', 'local_llm_min_p', 'local_llm_n_keep', 'azure_cv_free_tier']
    
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


function initializeLocalLLMServerDropdown(local_llm_server) {
    const llmServerDd = document.getElementById('local_llm_server_select_dropdown');

    for (let option of llmServerDd.options) {
        if (option.value == local_llm_server) {
            option.selected = true;
            break;
        }
    }

    document.getElementById('llama_cpp_div').style.display = local_llm_server === 'llama-cpp' ? 'block' : 'none';
    document.getElementById('hf_waitress_div').style.display = local_llm_server === 'hf-waitress' ? 'block' : 'none';
}


function loadCoreHfValues() {
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
        'port',
        'awq',
        'hqq_group_size',
        'flux_diffusers',
        'flux_low_vram_optimizations',
        'load_quantized_flux',
        'vision'
    ]
    
    return fetch('http://localhost:9069/hf_config_reader_api', {
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
        errorHandler("reading hf_config.json (likely means the HF-Waitress server is offline)", "loadCoreHfValues()", String(error.message))
    });
}


// function initializeHfModelDropdown(model_list, model_id) {
//     const hfDropdown = document.getElementById('hfModelDropdown');
//     for (let i = 0; i < model_list.length; i++) {
//         const option = document.createElement('option');
//         option.value = model_list[i];
//         option.textContent = model_list[i];

//         if (model_list[i].toLowerCase() == model_id.toLowerCase()) {
//             option.selected = true;
//         }
//         hfDropdown.appendChild(option);
//     }
// }


function initializeHfWaitressCustomDropdown(model_list, model_id) {
    const customDropdownList = document.getElementById('hf-waitress-llm-custom-dropdown-items-list');
    customDropdownList.innerHTML = '';

    const selectedValue = document.getElementById('hf-waitress-llm-custom-dropdown-selected-value');

    model_list.forEach(model => {
        if (model.toLowerCase() == model_id.toLowerCase()) {
            selectedValue.textContent = model;
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

    const hfQuantLevel = document.getElementById('hf_waitress_quantization_level_choice');

    for (let option of hfQuantLevel.options) {
        if (option.value == all_values.quant_level) {
            option.selected = true;
            break;
        }
    }

}


function initializeHfRadioButtons(all_values) {
    document.getElementById('hf_waitress_is_awq_yes').checked = all_values.awq;
    document.getElementById('hf_waitress_is_awq_no').checked = !all_values.awq;

    document.getElementById('hf_waitress_trust_remote_code_yes').checked = all_values.trust_remote_code;
    document.getElementById('hf_waitress_trust_remote_code_no').checked = !all_values.trust_remote_code;

    document.getElementById('hf_waitress_diffusers_yes').checked = all_values.flux_diffusers;
    document.getElementById('hf_waitress_diffusers_no').checked = !all_values.flux_diffusers;

    document.getElementById('hf_waitress_vision_yes').checked = all_values.vision;
    document.getElementById('hf_waitress_vision_no').checked = !all_values.vision;

    toggleHfwDiffusersConfig();

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
    document.getElementById('HfwPort').value = values.port;
    document.getElementById('HfwMaxNewToks').value = values.max_new_tokens;
    document.getElementById('HfwTempSlider').value = values.temperature;
    document.getElementById('HfwTempSliderValue').textContent = values.temperature;
    document.getElementById('HfwTopkSlider').value = values.top_k;
    document.getElementById('HfwTopkSliderValue').textContent = values.top_k;
    document.getElementById('HfwToppSlider').value = values.top_p;
    document.getElementById('HfwToppSliderValue').textContent = values.top_p;
    document.getElementById('HfwMinpSlider').value = values.min_p;
    document.getElementById('HfwMinpSliderValue').textContent = values.min_p;
}


function initializeHfwServerConfig() {
    loadCoreHfValues()
        .then(hf_values => {
            // initializeHfModelDropdown(hf_values.model_list, hf_values.model_id);
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
        var advancedLlmSettings = document.getElementById('advancedLlmSettings');
        if (advancedLlmSettings.classList.contains('show')) {
            var bsCollapse = new bootstrap.Collapse(advancedLlmSettings, {
                toggle: false
            });
            bsCollapse.hide();
        }

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
        disableTransformersSettings();
    } else {
        document.getElementById('hf-waitress-diffusers-configuration-div').style.display = 'none';
        enableTransformersSettings();
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

    // Event listener for toggle Flux Low Vram:
    document.getElementById('hf_waitress_diffusers_low_vram_optimizations_yes').addEventListener('change', disableFluxQuantization);

    // Event listener for toggle Flux Quantization:
    document.getElementById('hf_waitress_diffusers_fp8_yes').addEventListener('change', disableFluxLowVram);

    // Event listener for Reset Defaults button:
    document.getElementById('resetLlmAdvancedDefaults').addEventListener('click', resetLlmAdvancedDefaults);    //aparently, adding parenthesis () here will cause resetLlmAdvancedDefaults to run immediately when the script executes, not on button click. Removing them allows the function to be passed as a reference correctly on click.
    document.getElementById('resetHfLlmAdvancedDefaults').addEventListener('click', resetHfLlmAdvancedDefaults);

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


function initializeLLMTabComponents(values) {
    initializeLocalLLMServerDropdown(values.local_llm_server);
    initializeModelDropdown(values.model_choice);   
    initializeLLMTemplateDropdown(values.local_llm_chat_template_format);
    initializeGPURadioButtons(values.use_gpu);
    initializeLLMRadioButtons(values.use_local_llm, values.model_choice);
    initializeEventListenersForLLMTab();
    initializeEventListenersForLLMTabSliders();
}


function initializeEmbeddingModelDropdown(embedding_model_choice) {
    var selectEmbedModelForDropdown = document.getElementById('embedding_model_dropdown');

    for (var i = 0; i < selectEmbedModelForDropdown.length; i++) {
        if (selectEmbedModelForDropdown.options[i].value === embedding_model_choice) {
            selectEmbedModelForDropdown.options[i].selected = true;
            clearDocsLoadedTable();
            populateDocsLoadedTable();
            break;
        }
    }
}


function initializeEventListenersForEmbeddingModelTab() {
    // Show or hide API form:
    function toggleAzureAdaApiForm() {
        var selection = document.getElementById('embedding_model_dropdown').value;
        document.getElementById('azure_openai_text_ada_api_form').style.display = selection === 'openai_text_ada' ? 'block' : 'none';
        //Add more API form selectors here
    }

    // Event listener for API dropdown:
    document.getElementById('embedding_model_dropdown').addEventListener('change', function() {
        populateDocsLoadedTable();
        toggleAzureAdaApiForm();
    });

    // Add Event Listener for ResetDB button:
    document.getElementById('resetVectorDB').addEventListener('click', resetVectorDBtoBlank);

    // Check init
    toggleAzureAdaApiForm();

    document.getElementById('update_azure_ada').addEventListener('change', function() {
        document.getElementById('azure_openai_text_ada_api_url').disabled = !this.checked;    
        document.getElementById('azure_openai_text_ada_api_key').disabled = !this.checked;   
        document.getElementById('azure_openai_text_ada_deployment_name').disabled = !this.checked;   
    });
}


function initializeEmbeddingModelTabComponents(values) {
    initializeEmbeddingModelDropdown(values.embedding_model_choice);
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
    } else {
        if (document.getElementById('ocrApiDropdown').value != 'AzureVision') {
            document.getElementById('azure_vision_api_form').style.display = 'none'; //Hide API-Details-Form
        } else if (document.getElementById('ocrApiDropdown').value != 'AzureDocAi') {
            document.getElementById('azure_doc_ai_api_form').style.display = 'none';
        } else if (document.getElementById('ocrApiDropdown').value != 'LocalVisionLLM') {
            document.getElementById('local_vision_api_form').style.display = 'none';
        }
    }
}


function toggleOcrApiForm() {   //Show or hide API form:
    var selection = document.getElementById('ocrApiDropdown').value;
    document.getElementById('azure_vision_api_form').style.display = selection === 'AzureVision' ? 'block' : 'none';
    document.getElementById('azure_doc_ai_api_form').style.display = selection === 'AzureDocAi' ? 'block' : 'none';
    document.getElementById('local_vision_api_form').style.display = selection === 'LocalVisionLLM' ? 'block' : 'none';
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
            setChatId(data.chat_id);
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


function startLLMAndVectorDB() {
    // Finally, trigger the LLM & vectorDB starter!
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
            return loadVectorDB(llm_model);
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
    let local_llm_server;

    loadAndSetCoreValues()
        .then(values => {
            initializeLLMTabComponents(values);
            initializeEmbeddingModelTabComponents(values);
            initializeSystemPromptTabComponents(values);
            initializeRAGTabComponents(values);
            initializeOCRTabComponents(values);
            initializeGoogleDriveTabComponents(values);
            initializeSettingsModalTabCycleListener();
            local_llm_server = values.local_llm_server;
            return startLLMAndVectorDB();
        })
        .then(() => {
            if (local_llm_server === "hf-waitress") {
                initializeHfwServerConfig();
                showStopGenerationButton();
            }
            loadChatHistoryMenu();  // should be done regardless of errors
        })
        .catch(error => {
            errorHandler("initializing the application", "DOMContentLoaded", String(error.message));
        })
        .finally(() => {

        });
});//########### End DOMContentLoaded Block!