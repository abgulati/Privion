

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
            clearToolChainToggle(streamSessionId);
            requestFormattedPrompt(true, false, false, streamSessionId, sequenceId);    // Request the formatted prompt
            deleteChatAreaElements(userMessageDiv.nextElementSibling); // Delete subsequent user messages and response containers
            appendLoadingAnimation(userMessageDiv);
        } else if (e.target.classList.contains('regenerate-with-citations-enabled-option')) {
            const userMessageDiv = e.target.closest('.user-message');
            const {streamSessionId, sequenceId} = prepareAttributeForUserMessage(userMessageDiv);
            clearToolChainToggle(streamSessionId);
            requestFormattedPrompt(true, true, false, streamSessionId, sequenceId);    // Request the formatted prompt
            deleteChatAreaElements(userMessageDiv.nextElementSibling); // Delete subsequent user messages and response containers
            appendLoadingAnimation(userMessageDiv);
        } else if (e.target.classList.contains('regenerate-with-citations-disabled-option')) {
            const userMessageDiv = e.target.closest('.user-message');
            const {streamSessionId, sequenceId} = prepareAttributeForUserMessage(userMessageDiv);
            clearToolChainToggle(streamSessionId);
            requestFormattedPrompt(true, false, true, streamSessionId, sequenceId);    // Request the formatted prompt
            deleteChatAreaElements(userMessageDiv.nextElementSibling); // Delete subsequent user messages and response containers
            appendLoadingAnimation(userMessageDiv);
        } else if (e.target.classList.contains('delete-option')) {
            const userMessageDiv = e.target.closest('.user-message');
            const chatId = userMessageDiv.getAttribute('data-chat-id');
            const sequenceId = userMessageDiv.getAttribute('data-sequence-id');
            delete_messages(chatId, sequenceId, userMessageDiv);
        }
    });
}


function initializeFileUploadButtons() {
    document.getElementById('filesInput').addEventListener('change', handleFileOrFolderSelection);
    document.getElementById('folderInput').addEventListener('change', handleFileOrFolderSelection);
}


function initializeUI() {
    attachWindowEvents();
    initializeChatLink();
    initializeScrollDownButton();
    initializeRegenerateResponseButton();
    initializeFileUploadButtons();
}


function loadCoreLarsConfig() {
    appendStreamInfo("Loading core LARS config...", 'waiting');
    const initKeysToRead = [
        'local_llm_server',
        'exclusive_server_mode',
        'model_choice',
        'use_local_llm',
        'embedding_models_list',
        'selected_embedding_model',
        'reranker_models_list',
        'selected_reranker_model',
        'use_embedding_model_for_reranking',
        'knowledge_domain_list',
        'selected_knowledge_domain',
        'force_ocr',
        'ocr_service_choice',
        'backup_ocr_service_choice',
        'docling_pipeline',
        'docling_vlm_model',
        'docling_do_ocr',
        'docling_force_full_page_ocr',
        'docling_ocr_model',
        'docling_do_code_enrichment',
        'docling_do_formula_enrichment',
        'docling_do_table_structure',
        'docling_do_picture_classification',
        'docling_do_picture_description',
        'docling_table_structure_mode',
        'docling_do_cell_matching',
        'docling_cuda_use_flash_attention_2',
        'docling_num_threads',
        'force_enable_rag',
        'force_disable_rag',
        'enable_graph_rag',
        'enable_butler_mode_selection',
        'upload_doc_to_graph_db',
        'graph_rag_context_length_limit_chars',
        'graph_chunk_size',
        'graph_chunk_overlap',
        'reuse_graph_extraction_cache_without_validation',
        'reuse_graph_extraction_cache_with_validation',
        'graph_generator_model',
        'graph_generator_model_list',
        'exl2_quantize_graph_model',
        'exl2_quantize_graph_model_bpw',
        'minimum_free_vram_for_graph_extraction_model',
        'graph_model_max_new_tokens',
        'graph_model_max_seq_len',
        'graph_model_temperature',
        'graph_model_do_sample',
        'graph_model_top_k',
        'graph_model_top_p',
        'graph_model_min_p',
        'graph_model_access_url',
        'graph_model_server_port',
        'reuse_graph_summary_cache_without_validation',
        'reuse_graph_summary_cache_with_validation',
        'graph_summarizer_model',
        'graph_summarizer_model_list',
        'exl2_quantize_graph_summarizer_model',
        'exl2_quantize_graph_summarizer_model_bpw',
        'minimum_free_vram_for_graph_summarizer_model',
        'graph_summarizer_max_new_tokens',
        'graph_summarizer_max_seq_len',
        'graph_summarizer_temperature',
        'graph_summarizer_do_sample',
        'graph_summarizer_top_k',
        'graph_summarizer_top_p',
        'graph_summarizer_min_p',
        'graph_summarizer_access_url',
        'graph_summarizer_server_port',
        'graph_db_server_host',
        'assign_host_port_to_graph_db_server',
        'assign_host_port_to_graph_db_ui',
        'launch_graph_db_with_ui',
        'apply_clustering_to_graph_db_on_doc_load',
        'fetch_top_k_results_from_whoosh',
        'fetch_top_k_results_from_vectordb',
        'filter_top_k_results_by_reranking',
        'min_semantic_similarity_threshold',
        'min_lexical_similarity_threshold',
        'base_template',
        'vision_ocr_prompt',
        'llama_cpp_use_gpu',
        'llama_cpp_gpu_layers',
        'llama_cpp_context_length',
        'llama_cpp_batch_size',
        'llama_cpp_ubatch_size',
        'llama_cpp_max_new_tokens',
        'llama_cpp_unified_kv_buffer',
        'llama_cpp_disable_kv_offloading',
        'llama_cpp_key_cache_data_type',
        'llama_cpp_value_cache_data_type',
        'llama_cpp_no_of_seqs_to_par_decode',
        'llama_cpp_offload_to_devices',
        'llama_cpp_cpu_only_moe',
        'llama_cpp_num_cpu_moe',
        'llama_cpp_mlock',
        'llama_cpp_no_nmap',
        'llama_cpp_split_mode',
        'llama_cpp_tensor_split',
        'llama_cpp_override_tensor',
        'llama_cpp_temperature',
        'llama_cpp_top_k',
        'llama_cpp_top_p',
        'llama_cpp_min_p',
        'llama_cpp_n_keep',
        'llama_cpp_repetition_penalty',
        'llama_cpp_presence_penalty',
        'llama_cpp_frequency_penalty',
        'azure_cv_free_tier',
        'skip_system_prompt',
        'force_re_extract',
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
        'llama_cpp_server_port',
        'enable_asr',
        'asr_wake_word',
        'asr_model',
        'asr_torch_device',
        'asr_temperature',
        'asr_max_new_tokens',
        'asr_samplerate',
        'asr_volume_threshold',
        'asr_silence_duration_s',
        'asr_min_chunk_duration_s',
        'asr_min_context_s',
        'asr_stale_buffer_timeout_s',
        'asr_min_meaningful_samples_factor',
        'asr_vad_model',
        'asr_vad_device',
        'asr_vad_threshold',
        'asr_vad_min_speech_ms',
        'asr_vad_min_silence_ms',
        'asr_vad_window_size_samples',
        'asr_vad_max_buffer_s',
        'asr_vad_speech_pad_ms',
        'asr_apply_normalization',
        'asr_apply_tts_padding',
        'asr_apply_zero_padding',
        'asr_apply_rms_dimming',
        'asr_apply_crossfade',
        'asr_waitress_serving_url',
        'asr_waitress_access_url',
        'asr_waitress_server_port',
        'enable_tts',
        'selected_tts',
        'selected_kokoro_voice',
        'tts_sample_rate'
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
        return data.values;
    })
    .catch(error => {
        errorHandler("reading config.json", "/config_reader_api", String(error.message));
    });
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
        'rep_p',
        'pres_p',
        'freq_p',
        'rep_sustain_range',
        'rep_decay_range',
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
        'exl2_no_flash_attn',
        'exl2_max_seq_len',
        'exl2_cache_type',
        'exl2_force_regenerate_measurement',
        'exl3',
        'exl3_bpw',
        'exl3_device',
        'exl3_resume_quant_job',
        'exl3_total_context',
        'exl3_tensor_parallel',
        'exl3_tp_output_device',
        'exl3_use_per_device',
        'exl3_max_chunk_size',
        'exl3_max_batch_size',
        'exl3_show_gen_visualizer'
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
        //errorHandler("reading hf_config.json (likely means the HF-Waitress server is offline)", "load-CoreHfConfig()", String(error.message))
        appendStreamInfo('Error: Could not read HF-Waitress config.json (likely means the HF-Waitress server is offline)', 'failure');
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


function initializeRerankerCustomDropdown(model_list, model_id) {
    const customDropdownList = document.getElementById('hf-waitress-reranker-custom-dropdown-items-list');
    customDropdownList.innerHTML = '';

    const selectedValue = document.getElementById('hf-waitress-reranker-custom-dropdown-selected-value');
    
    model_list.forEach(model => {
        if (model.toLowerCase() == model_id.toLowerCase()) {
            selectedValue.textContent = model;
        }
        const div = document.createElement('div');
        div.className = 'hf-waitress-reranker-custom-dropdown-item';
        div.innerHTML = `
            <span class="hf-waitress-reranker-custom-dropdown-item-text">${model}</span>
            <span class="hf-waitress-reranker-custom-dropdown-delete-btn">×</span>
        `;
        const deleteButton = div.querySelector('.hf-waitress-reranker-custom-dropdown-delete-btn');
        deleteButton.addEventListener('click', (event) => {
            event.stopPropagation();
            removeRerankerModelFromCustomDropdown(model);
        });
        div.addEventListener('click', () => {
            selectedValue.textContent = model;
            document.getElementById('hf-waitress-reranker-custom-dropdown-content').classList.remove('show');
        });
        customDropdownList.appendChild(div);
    });

}


function initializeGraphExtractorCustomDropdown(model_list, model_id) {
    const customDropdownList = document.getElementById('graph-extractor-custom-dropdown-items-list');
    customDropdownList.innerHTML = '';

    const selectedValue = document.getElementById('graph-extractor-custom-dropdown-selected-value');

    model_list.forEach(model => {
        if (model.toLowerCase() == model_id.toLowerCase()) {
            selectedValue.textContent = model;
        }
        const div = document.createElement('div');
        div.className = 'graph-extractor-custom-dropdown-item';
        div.innerHTML = `
            <span class="graph-extractor-custom-dropdown-item-text">${model}</span>
            <span class="graph-extractor-custom-dropdown-delete-btn">×</span>
        `;
        const deleteButton = div.querySelector('.graph-extractor-custom-dropdown-delete-btn');
        deleteButton.addEventListener('click', (event) => {
            event.stopPropagation();
            removeGraphExtractorModelFromCustomDropdown(model);
        });
        div.addEventListener('click', () => {
            selectedValue.textContent = model;
            document.getElementById('graph-extractor-custom-dropdown-content').classList.remove('show');
        });
        customDropdownList.appendChild(div);
    });
}


function initializeGraphSummarizerCustomDropdown(model_list, model_id) {
    const customDropdownList = document.getElementById('graph-summarizer-custom-dropdown-items-list');
    customDropdownList.innerHTML = '';

    const selectedValue = document.getElementById('graph-summarizer-custom-dropdown-selected-value');

    model_list.forEach(model => {
        if (model.toLowerCase() == model_id.toLowerCase()) {
            selectedValue.textContent = model;
        }
        const div = document.createElement('div');
        div.className = 'graph-summarizer-custom-dropdown-item';
        div.innerHTML = `
            <span class="graph-summarizer-custom-dropdown-item-text">${model}</span>
            <span class="graph-summarizer-custom-dropdown-delete-btn">×</span>
        `;
        const deleteButton = div.querySelector('.graph-summarizer-custom-dropdown-delete-btn');
        deleteButton.addEventListener('click', (event) => {
            event.stopPropagation();
            removeGraphSummarizerModelFromCustomDropdown(model);
        });
        div.addEventListener('click', () => {
            selectedValue.textContent = model;
            document.getElementById('graph-summarizer-custom-dropdown-content').classList.remove('show');
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

    document.getElementById('hf_use_exl2_no_flash_attn_checkbox').checked = all_values.exl2_no_flash_attn;

    document.getElementById('hf_waitress_exl3_yes').checked = all_values.exl3;
    document.getElementById('hf_waitress_exl3_no').checked = !all_values.exl3;

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
    toggleHfwExl3Config();
    if (getExl2() === "true") displayOnlyExl2Settings();  // as it might be taggled by the Exl3 toggle method! 

    document.getElementById('hf_waitress_diffusers_low_vram_optimizations_checkbox').checked = all_values.flux_low_vram_optimizations;
    document.getElementById('hf_waitress_diffusers_fp8_checkbox').checked = all_values.load_quantized_flux;
    document.getElementById('hf_waitress_use_flash_attention_2_checkbox').checked = all_values.use_flash_attention_2;
    document.getElementById('hf_waitress_return_full_text_checkbox').checked = all_values.return_full_text;
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
    document.getElementById('HfwRepetitionPenaltySlider').value = values.rep_p;
    document.getElementById('HfwRepetitionPenaltySliderValue').textContent = values.rep_p;
    document.getElementById('HfwPresencePenaltySlider').value = values.pres_p;
    document.getElementById('HfwPresencePenaltySliderValue').textContent = values.pres_p;
    document.getElementById('HfwFrequencyPenaltySlider').value = values.freq_p;
    document.getElementById('HfwFrequencyPenaltySliderValue').textContent = values.freq_p;
    document.getElementById('HfwExl2Bpw').value = parseFloat(values.exl2_bpw);
    document.getElementById('HfwExl2MaxSeqLen').value = values.exl2_max_seq_len;
    document.getElementById('hf_waitress_exl2_force_regenerate_measurement').checked = values.exl2_force_regenerate_measurement;
    document.getElementById('HfwExl3Bpw').value = parseFloat(values.exl3_bpw);
    document.getElementById('HfwExl3Device').value = values.exl3_device;
    document.getElementById('HfwExl3ResumeQuantJob').checked = values.exl3_resume_quant_job;
    document.getElementById('HfwExl3TotalContext').value = values.exl3_total_context;
    document.getElementById('HfwExl3TensorParallel').checked = values.exl3_tensor_parallel;
    document.getElementById('HfwExl3TpOutputDevice').value = values.exl3_tp_output_device;
    document.getElementById('HfwExl3UsePerDevice').value = values.exl3_use_per_device;
    document.getElementById('HfwExl3MaxChunkSize').value = values.exl3_max_chunk_size;
    document.getElementById('HfwExl3MaxBatchSize').value = values.exl3_max_batch_size;
    document.getElementById('HfwExl3ShowGenVisualizer').checked = values.exl3_show_gen_visualizer;
}


function initializeHfwServerConfig() {
    loadCoreHfConfig()
        .then(hf_values => {
            setVision(hf_values.vision);
            setExl2(hf_values.exl2);
            setExl3(hf_values.exl3);
            initializeHfWaitressCustomDropdown(hf_values.model_list, hf_values.model_id);
            initializeHfSettingsDropdowns(hf_values);
            initializeHfRadioButtons(hf_values);
            setHfSlidersAndTextAreas(hf_values);
        })
        .catch(error => {
            // errorHandler("initializing HF-Waitress Server config (likely means the HF-Waitress server is offline)", "initialize-HfwServerConfig()", String(error.message));
            appendStreamInfo('Error: Could not initialize HF-Waitress Server config (likely means the HF-Waitress server is offline)', 'failure');
        });
}


function setLlamaCppDropdowns(values) {
    const keyCacheDataType = document.getElementById('KeyCacheDataType');
    for (let option of keyCacheDataType.options) {
        if (option.value == values.llama_cpp_key_cache_data_type) {
            option.selected = true;
            break;
        }
    }
    const valueCacheDataType = document.getElementById('ValueCacheDataType');
    for (let option of valueCacheDataType.options) {
        if (option.value == values.llama_cpp_value_cache_data_type) {
            option.selected = true;
            break;
        }
    }

    const splitMode = document.getElementById('SplitMode');
    for (let option of splitMode.options) {
        if (option.value == values.llama_cpp_split_mode) {
            option.selected = true;
            break;
        }
    }
}


function setLlamaCppCheckboxes(values) {
    document.getElementById('UseGpu').checked = values.llama_cpp_use_gpu;
    document.getElementById('UnifiedKvBuffer').checked = values.llama_cpp_unified_kv_buffer;
    document.getElementById('DisableKvOffloading').checked = values.llama_cpp_disable_kv_offloading;
    document.getElementById('CpuOnlyMoe').checked = values.llama_cpp_cpu_only_moe;
    document.getElementById('Mlock').checked = values.llama_cpp_mlock;
    document.getElementById('NoNmap').checked = values.llama_cpp_no_nmap;
}


function setLlamaCppValues(values) {
    document.getElementById('NumbGpuLayers').value = values.llama_cpp_gpu_layers;
    document.getElementById('LlmCtxLgt').value = values.llama_cpp_context_length;
    document.getElementById('LlamaCppBatchSize').value = values.llama_cpp_batch_size;
    document.getElementById('LlamaCppUbatchSize').value = values.llama_cpp_ubatch_size;
    document.getElementById('NoOfSeqsToParDecode').value = values.llama_cpp_no_of_seqs_to_par_decode;
    document.getElementById('OffloadToDevices').value = values.llama_cpp_offload_to_devices;
    document.getElementById('TensorSplit').value = values.llama_cpp_tensor_split;
    document.getElementById('OverrideTensor').value = values.llama_cpp_override_tensor;
    document.getElementById('MaxNewToks').value = values.llama_cpp_max_new_tokens;
    document.getElementById('LlamaCppNumCpuMoe').value = values.llama_cpp_num_cpu_moe;
    document.getElementById('tempSlider').value = values.llama_cpp_temperature;
    document.getElementById('tempSliderValue').textContent = values.llama_cpp_temperature;
    document.getElementById('topkSlider').value = values.llama_cpp_top_k;
    document.getElementById('topkSliderValue').textContent = values.llama_cpp_top_k;
    document.getElementById('toppSlider').value = values.llama_cpp_top_p;
    document.getElementById('toppSliderValue').textContent = values.llama_cpp_top_p;
    document.getElementById('minpSlider').value = values.llama_cpp_min_p;
    document.getElementById('minpSliderValue').textContent = values.llama_cpp_min_p;
    document.getElementById('nkeepSlider').value = values.llama_cpp_n_keep;
    document.getElementById('repetitionPenaltySlider').value = values.llama_cpp_repetition_penalty;
    document.getElementById('repetitionPenaltySliderValue').textContent = values.llama_cpp_repetition_penalty;
    document.getElementById('presencePenaltySlider').value = values.llama_cpp_presence_penalty;
    document.getElementById('presencePenaltySliderValue').textContent = values.llama_cpp_presence_penalty;
    document.getElementById('frequencyPenaltySlider').value = values.llama_cpp_frequency_penalty;
    document.getElementById('frequencyPenaltySliderValue').textContent = values.llama_cpp_frequency_penalty;
}


function initializeLocalLLMServerDropdown(local_llm_server, exclusive_server_mode) {
    document.getElementById('exclusive_server_mode_checkbox').checked = exclusive_server_mode;

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
                        readGGUF(model);
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


function initializeHfwUrlComponents(hf_waitress_serving_url, hf_waitress_access_url, hf_waitress_server_port) {
    document.getElementById('HfwServingUrl').value = hf_waitress_serving_url;
    document.getElementById('HfwAccessUrl').value = hf_waitress_access_url;
    document.getElementById('HfwPort').value = hf_waitress_server_port;
    setHfwUrl(hf_waitress_access_url, hf_waitress_server_port);
}


function toggleLocalLlmSelection() {    // Show or hide local-LLM selection:
    const selection = document.querySelector('input[name="use_local_or_api_llm"]:checked').value;
    document.getElementById('localLlmDiv').style.display = selection === 'local' ? 'block' : 'none';
    //document.getElementById('apiLlmDiv').style.display = selection === 'api' ? 'block' : 'none';

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
    var selection = document.getElementById('UseGpu').checked;
    if(selection) {
        document.getElementById("NumbGpuLayers").disabled = false;
    } else {
        document.getElementById('NumbGpuLayers').value = "0";
        document.getElementById("NumbGpuLayers").disabled = true;
    }
}


function toggleLlamaCppCpuMoE() {
    var selection = document.getElementById('CpuOnlyMoe').checked;
    if (selection) {
        document.getElementById('LlamaCppNumCpuMoe').disabled = true;
        document.getElementById('LlamaCppNumCpuMoe').value = "0";
    } else {
        document.getElementById('LlamaCppNumCpuMoe').disabled = false;
    }
}


function toggleHfwDiffusersConfig() {
    var selection = document.querySelector('input[name="hf_use_diffusers"]:checked').value;
    selection === 'y' ? displayOnlyDiffusersSettings() : resetDiffusersOnlyView();
}


function toggleHfwExl2Config() {
    var selection = document.querySelector('input[name="hf_use_exl2"]:checked').value;
    selection === 'y' ? displayOnlyExl2Settings() : resetExl2OnlyView();
}


function toggleHfwExl3Config() {
    var selection = document.querySelector('input[name="hf_use_exl3"]:checked').value;
    selection === 'y' ? displayOnlyExl3Settings() : resetExl3OnlyView();
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
    document.getElementById('hf_waitress_hqq_group_size').style.display = quantMethod === 'hqq' ? 'table-row' : 'none';

    // Disable the Selection of No Quantization methos is selected
    quantLevelSelection.disabled = quantMethod === 'n';

}


function initializeEventListenersForLLMTab() {
    // Event listener for Local LLM Server selection:
    document.getElementById('local_llm_server_select_dropdown').addEventListener('change', () => checkLocalLLMServerStatus()); // Event handlers will pass 'this' as the first arg causing the default arg `asr_check = false` to be ignored! More below:
    /*
    When an event handler is called in JavaScript, the event object is automatically passed as the first argument. 
    This means asr_check is receiving the event object instead of undefined, so the default parameter `asr_check = false` is never used.
    In this case, the event object is truthy (it's a valid object), so asr_check evaluates to true when the dropdown changes, which is actually the opposite of what's required.
    The fix is to wrap the function call in an arrow function or use .bind() to prevent the event object from being passed!

    The former (above) wraps the function call in an arrow function that prevents the event object from being passed as the first parameter, 
    allowing the default parameter `asr_check = false` to work as intended.

    The .bind() method creates a new function with false hardcoded as the first argument, ensuring that asr_check will always be false regardless of what the event handler tries to pass. 
    The null is the this context (which doesn't matter here since the function doesn't use this):
    ```
    document.getElementById('local_llm_server_select_dropdown').addEventListener('change', checkLocalLLMServerStatus.bind(null, false));
    ```
    The .bind() method is a more explicit way to ensure the default parameter is always used, but the arrow function approach is simpler and more readable in this case.

    NOTE: Simply attempting `document.getElementById('local_llm_server_select_dropdown').addEventListener('change', (checkLocalLLMServerStatus(false)));` will NOT work!
    The issue with this is that `checkLocalLLMServerStatus(false)` executes immediately when this line runs, rather than waiting for the change event:
    The parentheses () cause the function to be called right away, and whatever it returns (likely undefined or a Promise) gets assigned as the event handler.
    This is different from the arrow function approach where the function is wrapped but not executed until the event fires.

    The key difference is that () => checkLocalLLMServerStatus(false) creates a function that will be called later, while (checkLocalLLMServerStatus(false)) calls the function immediately!
    */
    document.getElementById('local_llm_server_select_dropdown').addEventListener('change', toggleLocalLLMSettingsForms);

    // Event listener for LLM/API radio buttons:
    document.getElementById('local_llm_radio_button').addEventListener('change', toggleLlmApiForm);
    document.getElementById('local_llm_radio_button').addEventListener('change', toggleLocalLlmSelection);

    document.getElementById('api_llm_radio_button').addEventListener('change', toggleLlmApiForm);
    document.getElementById('api_llm_radio_button').addEventListener('change', toggleLocalLlmSelection);

    // Event listener for API dropdown:
    document.getElementById('llmApiDropdown').addEventListener('change', toggleLlmApiForm);

    // Event Listener for toggle GPU:
    document.getElementById('UseGpu').addEventListener('change', toggleNGL);

    // Event listener for CPU-MoE Toggle:
    document.getElementById('CpuOnlyMoe').addEventListener('change', toggleLlamaCppCpuMoE);

    // Event listener for toggle Vision:
    document.getElementById('hf_waitress_vision_yes').addEventListener('change', toggleHfwVisionConfig);
    document.getElementById('hf_waitress_vision_no').addEventListener('change', toggleHfwVisionConfig);

    // Event listener for toggle Diffusers:
    document.getElementById('hf_waitress_diffusers_yes').addEventListener('change', toggleHfwDiffusersConfig);
    document.getElementById('hf_waitress_diffusers_no').addEventListener('change', toggleHfwDiffusersConfig);

    // Event listener for toggle Exl2:
    document.getElementById('hf_waitress_exl2_yes').addEventListener('change', toggleHfwExl2Config);
    document.getElementById('hf_waitress_exl2_no').addEventListener('change', toggleHfwExl2Config);

    // Event listener for toggle Exl3:
    document.getElementById('hf_waitress_exl3_yes').addEventListener('change', toggleHfwExl3Config);
    document.getElementById('hf_waitress_exl3_no').addEventListener('change', toggleHfwExl3Config);

    // Event listener for toggle Flux Low Vram:
    // document.getElementById('hf_waitress_diffusers_low_vram_optimizations_yes').addEventListener('change', disableFluxQuantization);

    // Event listener for toggle Flux Quantization:
    // document.getElementById('hf_waitress_diffusers_fp8_yes').addEventListener('change', disableFluxLowVram);

    // Event listener for Reset Defaults button:
    document.getElementById('resetLlmAdvancedDefaults').addEventListener('click', resetLlmAdvancedDefaults);    //aparently, adding parenthesis () here will cause resetLlmAdvancedDefaults to run immediately when the script executes, not on button click. Removing them allows the function to be passed as a reference correctly on click.
    document.getElementById('resetHfLlmAdvancedDefaults').addEventListener('click', resetHfLlmAdvancedDefaults);

    // Event listener for toggle Quantization Level:
    document.getElementById('hf_waitress_quantization_choice').addEventListener('change', toggleHfwQuantizationLevel);

    // Initial state check:
    toggleLocalLlmSelection();
    toggleLlmApiForm();
    toggleNGL();
    toggleLlamaCppCpuMoE();

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

    document.getElementById('repetitionPenaltySlider').addEventListener('input', function() {
        document.getElementById('repetitionPenaltySliderValue').textContent = this.value;
    });

    document.getElementById('presencePenaltySlider').addEventListener('input', function() {
        document.getElementById('presencePenaltySliderValue').textContent = this.value;
    });

    document.getElementById('frequencyPenaltySlider').addEventListener('input', function() {
        document.getElementById('frequencyPenaltySliderValue').textContent = this.value;
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

    document.getElementById('HfwRepetitionPenaltySlider').addEventListener('input', function() {
        document.getElementById('HfwRepetitionPenaltySliderValue').textContent = this.value;
    });

    document.getElementById('HfwPresencePenaltySlider').addEventListener('input', function() {
        document.getElementById('HfwPresencePenaltySliderValue').textContent = this.value;
    });

    document.getElementById('HfwFrequencyPenaltySlider').addEventListener('input', function() {
        document.getElementById('HfwFrequencyPenaltySliderValue').textContent = this.value;
    });

    document.getElementById('HfDGuidanceScaleSlider').addEventListener('input', function() {
        document.getElementById('HfDGuidanceScaleSliderValue').textContent = this.value;
    });
}


function initializeLLMTabComponents(values) {
    setLlamaCppValues(values);
    setLlamaCppCheckboxes(values);
    setLlamaCppDropdowns(values);
    initializeLocalLLMServerDropdown(values.local_llm_server, values.exclusive_server_mode);
    initializeModelDropdown(values.model_choice);   
    //initializeLLMTemplateDropdown(values.local_llm_chat_template_format);
    initializeLLMRadioButtons(values.use_local_llm, values.model_choice);
    initializeHfwUrlComponents(values.hf_waitress_serving_url, values.hf_waitress_access_url, values.hf_waitress_server_port);
    initializeEventListenersForLLMTab();
    initializeEventListenersForLLMTabSliders();
}


function toggleRerankerDropdown() {
    document.getElementById('hf-waitress-reranker-custom-dropdown-content').classList.toggle('show');
}


function toggleRerankerDropdownEnabledState() {
    // disable the click event listener for 'hf-waitress-reranker-custom-select-header' and set its background-color to darkgray

    const header = document.getElementById('hf-waitress-reranker-custom-select-header');
    const rerankerCheckbox = document.getElementById('use_embedding_model_for_reranking');

    header.style.backgroundColor = rerankerCheckbox.checked ? 'darkgray' : 'white';
    
    // Remove or add the click event listener based on checkbox state
    if (rerankerCheckbox.checked) {
        header.removeEventListener('click', toggleRerankerDropdown);
        header.style.cursor = 'not-allowed';
    } else {
        header.addEventListener('click', toggleRerankerDropdown);
        header.style.cursor = 'pointer';
    }
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

    document.getElementById('hf-waitress-reranker-custom-select-header').addEventListener('click', toggleRerankerDropdown);

    // Event listener for use_embedding_model_for_reranking checkbox:
    document.getElementById('use_embedding_model_for_reranking').addEventListener('change', toggleRerankerDropdownEnabledState);

    // Event listener to filter custom dropdown:
    document.getElementById('hf-waitress-kb-custom-dropdown-search-input').addEventListener('input', function() {
        filterCustomKbDropdown(this.value);
    });

    document.getElementById('hf-waitress-embed-custom-dropdown-search-input').addEventListener('input', function() {
        filterCustomEmbedDropdown(this.value);
    });

    document.getElementById('hf-waitress-reranker-custom-dropdown-search-input').addEventListener('input', function() {
        filterCustomRerankerDropdown(this.value);
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

    // Event listener to add new item to KB-custom dropdown:
    const addKbInput = document.getElementById('hf-waitress-kb-custom-dropdown-add-input');
    const addKbBtn = document.getElementById('hf-waitress-kb-custom-dropdown-add-btn');

    addKbBtn.addEventListener('click', function() {
        addKbBtn.style.display = 'none';
        addKbInput.style.display = 'block';
    });

    addKbInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && this.value) {
            addNewKnowledgeDomain(this.value);
            addKbInput.value = '';
            addKbInput.style.display = 'none';
            addKbBtn.style.display = 'block';
        }
    });

    // Event listener to add new item to Embedding-custom dropdown:
    const addEmbedInput = document.getElementById('hf-waitress-embed-custom-dropdown-add-input');
    const addEmbedBtn = document.getElementById('hf-waitress-embed-custom-dropdown-add-btn');

    addEmbedBtn.addEventListener('click', function() {
        addEmbedBtn.style.display = 'none';
        addEmbedInput.style.display = 'block';
    });

    addEmbedInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && this.value) {
            addNewEmbeddingModel(this.value);
            addEmbedInput.value = '';
            addEmbedInput.style.display = 'none';
            addEmbedBtn.style.display = 'block';
        }
    });

    // Event listener to add new item to Reranker-custom dropdown:
    const addRerankerInput = document.getElementById('hf-waitress-reranker-custom-dropdown-add-input');
    const addRerankerBtn = document.getElementById('hf-waitress-reranker-custom-dropdown-add-btn');

    addRerankerBtn.addEventListener('click', function() {
        addRerankerBtn.style.display = 'none';
        addRerankerInput.style.display = 'block';
    });

    addRerankerInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && this.value) {
            addNewRerankerModel(this.value);
            addRerankerInput.value = '';
            addRerankerInput.style.display = 'none';
            addRerankerBtn.style.display = 'block';
        }
    });

}


function initializeEmbeddingModelTabComponents(values) {
    initializeKnowledgeDomainCustomDropdown(values.knowledge_domain_list, values.selected_knowledge_domain);
    initializeEmbeddingCustomDropdown(values.embedding_models_list, values.selected_embedding_model);
    initializeRerankerCustomDropdown(values.reranker_models_list, values.selected_reranker_model);
    document.getElementById('use_embedding_model_for_reranking').checked = values.use_embedding_model_for_reranking;
    initializeEventListenersForEmbeddingModelTab();
    toggleRerankerDropdownEnabledState();
    clearDocsLoadedTable();
    populateDocsLoadedTable();
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
    document.getElementById('fetch_top_k_results_from_whoosh').value = values.fetch_top_k_results_from_whoosh;
    document.getElementById('fetch_top_k_results_from_vectordb').value = values.fetch_top_k_results_from_vectordb;
    document.getElementById('filter_top_k_results_by_reranking').value = values.filter_top_k_results_by_reranking;
    document.getElementById('min_semantic_similarity_threshold').value = values.min_semantic_similarity_threshold;
    document.getElementById('min_lexical_similarity_threshold').value = values.min_lexical_similarity_threshold;
}


function initializeGraphRAGEventListeners() {

    const extractorReuseCacheCheckbox = document.getElementById('reuse_graph_extraction_cache_checkbox');
    const extractorValidateCacheCheckbox = document.getElementById('validate_graph_extraction_cache_checkbox');
    const summarizerReuseCacheCheckbox = document.getElementById('reuse_graph_summarization_cache_checkbox');
    const summarizerValidateCacheCheckbox = document.getElementById('validate_graph_summarization_cache_checkbox');

    extractorReuseCacheCheckbox.addEventListener('change', function() {
        if (this.checked) {
            extractorValidateCacheCheckbox.checked = true;
            extractorValidateCacheCheckbox.disabled = false;
        } else {
            extractorValidateCacheCheckbox.checked = false;
            extractorValidateCacheCheckbox.disabled = true;
        }
    });

    summarizerReuseCacheCheckbox.addEventListener('change', function() {
        if (this.checked) {
            summarizerValidateCacheCheckbox.checked = true;
            summarizerValidateCacheCheckbox.disabled = false;
        } else {
            summarizerValidateCacheCheckbox.checked = false;
            summarizerValidateCacheCheckbox.disabled = true;
        }
    });

}


function initializeGraphRAGCustomDropdowns(values) {
    // Extractor Server:
    initializeGraphExtractorCustomDropdown(values.graph_generator_model_list, values.graph_generator_model);
    
    // UNCOMMENT IN THE FUTURE IF MAKING THIS USER-SELECTABLE!
    // document.getElementById('graph-extractor-custom-select-header').addEventListener('click', function() {
    //     document.getElementById('graph-extractor-custom-dropdown-content').classList.toggle('show');
    // });

    document.getElementById('graph-extractor-custom-dropdown-search-input').addEventListener('input', function() {
        filterCustomGraphExtractorDropdown(this.value);
    });

    // Event listener to add new item to Extractor-custom dropdown:
    const addExtractorInput = document.getElementById('graph-extractor-custom-dropdown-add-input');
    const addExtractorBtn = document.getElementById('graph-extractor-custom-dropdown-add-btn');

    addExtractorBtn.addEventListener('click', function() {
        addExtractorBtn.style.display = 'none';
        addExtractorInput.style.display = 'block';
    });

    addExtractorInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && this.value) {
            addNewGraphExtractorModel(this.value); //TODO: add function
            addExtractorInput.value = '';
            addExtractorInput.style.display = 'none';
            addExtractorBtn.style.display = 'block';
        }
    });

    // Summarizer Server:
    initializeGraphSummarizerCustomDropdown(values.graph_summarizer_model_list, values.graph_summarizer_model);
    
    // UNCOMMENT IN THE FUTURE IF MAKING THIS USER-SELECTABLE!
    document.getElementById('graph-summarizer-custom-select-header').addEventListener('click', function() {
        document.getElementById('graph-summarizer-custom-dropdown-content').classList.toggle('show');
    });

    document.getElementById('graph-summarizer-custom-dropdown-search-input').addEventListener('input', function() {
        filterCustomGraphSummarizerDropdown(this.value);
    });

    // Event listener to add new item to Summarizer-custom dropdown:
    const addSummarizerInput = document.getElementById('graph-summarizer-custom-dropdown-add-input');
    const addSummarizerBtn = document.getElementById('graph-summarizer-custom-dropdown-add-btn');

    addSummarizerBtn.addEventListener('click', function() {
        addSummarizerBtn.style.display = 'none';
        addSummarizerInput.style.display = 'block';
    });

    addSummarizerInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && this.value) {
            addNewGraphSummarizerModel(this.value); //TODO: add function
            addSummarizerInput.value = '';
            addSummarizerInput.style.display = 'none';
            addSummarizerBtn.style.display = 'block';
        }
    });

}


function initializeGraphRAGTabComponents(values) {
    // General Settings:
    document.getElementById('enable_graph_rag_checkbox').checked = values.enable_graph_rag;
    document.getElementById('upload_doc_to_graph_db_checkbox').checked = values.upload_doc_to_graph_db;
    document.getElementById('graph_rag_context_length_limit_chars').value = values.graph_rag_context_length_limit_chars;
    document.getElementById('graph_rag_chunk_size').value = values.graph_chunk_size;
    document.getElementById('graph_rag_chunk_overlap').value = values.graph_chunk_overlap;
    
    // GraphDB Settings:
    document.getElementById('graph_db_server_host').value = values.graph_db_server_host;
    document.getElementById('graph_db_server_port').value = values.assign_host_port_to_graph_db_server;
    document.getElementById('launch_graph_db_with_ui_checkbox').checked = values.launch_graph_db_with_ui;
    document.getElementById('graph_db_ui_port').value = values.assign_host_port_to_graph_db_ui;
    document.getElementById('apply_clustering_to_graph_db_on_doc_load_checkbox').checked = values.apply_clustering_to_graph_db_on_doc_load;

    // GraphRAG Custom Dropdowns:
    initializeGraphRAGCustomDropdowns(values);
    initializeGraphRAGEventListeners();

    // Gearp-Extraction Model Settings:
    document.getElementById('reuse_graph_extraction_cache_checkbox').checked = values.reuse_graph_extraction_cache_with_validation;
    document.getElementById('validate_graph_extraction_cache_checkbox').checked = values.reuse_graph_extraction_cache_with_validation;
    document.getElementById('graph_extractor_exl2_yes').checked = String(values.exl2_quantize_graph_model).toLowerCase() === 'true';
    document.getElementById('graph_extractor_exl2_no').checked = String(values.exl2_quantize_graph_model).toLowerCase() === 'false';
    document.getElementById('graph_extractor_exl2_bpw').value = values.exl2_quantize_graph_model_bpw;
    document.getElementById('graph_extractor_memory_required').value = values.minimum_free_vram_for_graph_extraction_model;
    document.getElementById('graph_extractor_max_new_tokens').value = values.graph_model_max_new_tokens;
    document.getElementById('graph_extractor_max_seq_len').value = values.graph_model_max_seq_len;
    document.getElementById('graph_extractor_temperature').value = values.graph_model_temperature;
    document.getElementById('graph_extractor_top_k').value = values.graph_model_top_k;
    document.getElementById('graph_extractor_top_p').value = values.graph_model_top_p;
    document.getElementById('graph_extractor_min_p').value = values.graph_model_min_p;
    document.getElementById('graph_extractor_access_url').value = values.graph_model_access_url;
    document.getElementById('graph_extractor_server_port').value = values.graph_model_server_port;

    // Summarizer Model Settings:
    document.getElementById('reuse_graph_summarization_cache_checkbox').checked = values.reuse_graph_summary_cache_with_validation;
    document.getElementById('validate_graph_summarization_cache_checkbox').checked = values.reuse_graph_summary_cache_with_validation;
    document.getElementById('graph_summarizer_exl2_yes').checked = String(values.exl2_quantize_graph_summarizer_model).toLowerCase() === 'true';
    document.getElementById('graph_summarizer_exl2_no').checked = String(values.exl2_quantize_graph_summarizer_model).toLowerCase() === 'false';
    document.getElementById('graph_summarizer_exl2_bpw').value = values.exl2_quantize_graph_summarizer_model_bpw;
    document.getElementById('graph_summarizer_memory_required').value = values.minimum_free_vram_for_graph_summarizer_model;
    document.getElementById('graph_summarizer_max_new_tokens').value = values.graph_summarizer_max_new_tokens;
    document.getElementById('graph_summarizer_max_seq_len').value = values.graph_summarizer_max_seq_len;
    document.getElementById('graph_summarizer_temperature').value = values.graph_summarizer_temperature;
    document.getElementById('graph_summarizer_top_k').value = values.graph_summarizer_top_k;
    document.getElementById('graph_summarizer_top_p').value = values.graph_summarizer_top_p;
    document.getElementById('graph_summarizer_min_p').value = values.graph_summarizer_min_p;
    document.getElementById('graph_summarizer_access_url').value = values.graph_summarizer_access_url;
    document.getElementById('graph_summarizer_server_port').value = values.graph_summarizer_server_port;
}



function initializeOCRTabRadios(force_ocr, azure_cv_free_tier) {
    document.getElementById('ocr_yes_radio_button').checked = force_ocr;
    document.getElementById('ocr_no_radio_button').checked = !force_ocr;
    
    if (azure_cv_free_tier){
        document.getElementById('is_azure_cv_free_tier').checked = true;
    } else {
        document.getElementById('is_azure_cv_free_tier').checked = false;
    }
}


function toggleOcrSelection() { //Show or hide OCR-Service selection:
    var selection = document.querySelector('input[name="specify_ocr_and_service"]:checked').value;
    document.getElementById('apiOcrSelection').style.display = selection === 'ocr' ? 'block' : 'none';
    if (selection === 'pypdf') {    //Hide API-Details-Form
        document.getElementById('backupOcrSelection').style.display = 'block';
        document.getElementById('azure_vision_api_form').style.display = 'none'; 
        document.getElementById('azure_doc_ai_api_form').style.display = 'none';
        document.getElementById('local_vision_api_form').style.display = 'none';
        document.getElementById('kosmos_api_div').style.display = 'none';
        document.getElementById('docling_div').style.display = 'none';

        if (document.getElementById('backupOcrApiDropdown').value === 'Backup-Kosmos') {
            document.getElementById('kosmos_api_div').style.display = 'block';
        } 
        
        if (document.getElementById('backupOcrApiDropdown').value === 'Backup-Docling') {
            document.getElementById('docling_div').style.display = 'block';
        }
    } else {
        document.getElementById('backupOcrSelection').style.display = 'none';
        
        if (document.getElementById('ocrApiDropdown').value != 'AzureVision') {
            document.getElementById('azure_vision_api_form').style.display = 'none'; //Hide API-Details-Form
        } 
        
        if (document.getElementById('ocrApiDropdown').value != 'AzureDocAi') {
            document.getElementById('azure_doc_ai_api_form').style.display = 'none';
        } 
        
        if (document.getElementById('ocrApiDropdown').value != 'LocalVisionLLM') {
            document.getElementById('local_vision_api_form').style.display = 'none';
        } 
        
        if (document.getElementById('ocrApiDropdown').value != 'Kosmos') {
            document.getElementById('kosmos_api_div').style.display = 'none';
        } 
        
        if (document.getElementById('ocrApiDropdown').value != 'Docling') {
            document.getElementById('docling_div').style.display = 'none';
        }
    }
}


function toggleOcrApiForm() {   //Show or hide API form:
    var selection = document.getElementById('ocrApiDropdown').value;
    document.getElementById('azure_vision_api_form').style.display = selection === 'AzureVision' ? 'block' : 'none';
    document.getElementById('azure_doc_ai_api_form').style.display = selection === 'AzureDocAi' ? 'block' : 'none';
    document.getElementById('local_vision_api_form').style.display = selection === 'LocalVisionLLM' ? 'block' : 'none';
    document.getElementById('kosmos_api_div').style.display = selection === 'Kosmos' ? 'block' : 'none';
    document.getElementById('docling_div').style.display = selection === 'Docling' ? 'block' : 'none';
    //Add more API form selectors here
}


function toggleBackupOcrForms() {
    var selection = document.getElementById('backupOcrApiDropdown').value;
    document.getElementById('docling_div').style.display = selection === 'Backup-Docling' ? 'block' : 'none';
    document.getElementById('kosmos_api_div').style.display = selection === 'Backup-Kosmos' ? 'block' : 'none';
}


function toggleDoclingPipelineForm() {
    var selection = document.getElementById('docling_pipeline').value;
    document.getElementById('docling_vlm_options').style.display = selection === 'vlm' ? 'table-row' : 'none';
    document.getElementById('docling_standard_options').style.display = selection === 'standard' ? 'block' : 'none';
}


function toggleDoclingOcrOptions() {
    var selection = document.getElementById('docling_enable_ocr').checked;
    document.getElementById('docling_ocr_options').style.display = selection ? 'table-row' : 'none';
}


function toggleDoclingTableOptions() {
    var selection = document.getElementById('docling_do_table_structure').checked;
    document.getElementById('docling_table_options').style.display = selection ? 'table-row' : 'none';
}


function initializeOCRTabListeners() {

    // Event listeners for 'Update Config' checkboxes:
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
        document.getElementById('kosmos_task').disabled = !this.checked;
        document.getElementById('kosmos_threshold').disabled = !this.checked;
    });
    document.getElementById('update_docling_config').addEventListener('change', function() {
        document.getElementById('docling_pipeline').disabled = !this.checked;
        document.getElementById('docling_vlm_model').disabled = !this.checked;
        document.getElementById('docling_enable_ocr').disabled = !this.checked;
        document.getElementById('docling_force_full_page_ocr').disabled = !this.checked;
        document.getElementById('docling_ocr_model').disabled = !this.checked;
        document.getElementById('docling_do_code_enrichment').disabled = !this.checked;
        document.getElementById('docling_do_formula_enrichment').disabled = !this.checked;
        document.getElementById('docling_do_table_structure').disabled = !this.checked;
        document.getElementById('docling_table_structure_mode').disabled = !this.checked;
        document.getElementById('docling_do_picture_classification').disabled = !this.checked;
        document.getElementById('docling_do_picture_description').disabled = !this.checked;
        document.getElementById('docling_do_cell_matching').disabled = !this.checked;
        document.getElementById('docling_cuda_use_flash_attention_2').disabled = !this.checked;
        document.getElementById('docling_num_threads').disabled = !this.checked;
    });

    // Event listener for radio buttons and dropdowns:
    document.getElementById('ocr_yes_radio_button').addEventListener('change', toggleOcrApiForm);
    document.getElementById('ocr_yes_radio_button').addEventListener('change', toggleOcrSelection);

    document.getElementById('ocr_no_radio_button').addEventListener('change', toggleBackupOcrForms);
    document.getElementById('ocr_no_radio_button').addEventListener('change', toggleOcrSelection);

    document.getElementById('ocrApiDropdown').addEventListener('change', toggleOcrApiForm);

    document.getElementById('backupOcrApiDropdown').addEventListener('change', toggleBackupOcrForms);

    document.getElementById('docling_pipeline').addEventListener('change', toggleDoclingPipelineForm);
    document.getElementById('docling_enable_ocr').addEventListener('change', toggleDoclingOcrOptions);
    document.getElementById('docling_do_table_structure').addEventListener('change', toggleDoclingTableOptions);

    // Initial state check:
    toggleOcrSelection();
    if (document.getElementById('ocr_yes_radio_button').checked) { toggleOcrApiForm(); }
    if (document.getElementById('ocr_no_radio_button').checked) { toggleBackupOcrForms(); }
    toggleDoclingPipelineForm();
    toggleDoclingOcrOptions();
    toggleDoclingTableOptions();
}


function initializeOCRFormValues(values) {
    if (values.force_re_extract) { document.getElementById('force_extract_previously_extracted_text_checkbox').checked = true; }
    if (values.kosmos_local_url) { document.getElementById('kosmos_api_url').value = values.kosmos_local_url; }
    if (values.kosmos_task) { document.getElementById('kosmos_task').value = values.kosmos_task; }
    if (values.kosmos_threshold) { document.getElementById('kosmos_threshold').value = values.kosmos_threshold; }
    if (values.vision_llm_local_url) { document.getElementById('local_vision_api_url').value = values.vision_llm_local_url; }
    if (values.ocr_service_choice) { document.getElementById('ocrApiDropdown').value = values.ocr_service_choice; }
    if (values.backup_ocr_service_choice) { document.getElementById('backupOcrApiDropdown').value = values.backup_ocr_service_choice; }
    if (values.docling_pipeline) { document.getElementById('docling_pipeline').value = values.docling_pipeline; }
    if (values.docling_vlm_model) { document.getElementById('docling_vlm_model').value = values.docling_vlm_model; }
    if (values.docling_enable_ocr) { document.getElementById('docling_enable_ocr').checked = values.docling_enable_ocr; }
    if (values.docling_force_full_page_ocr) { document.getElementById('docling_force_full_page_ocr').checked = values.docling_force_full_page_ocr; }
    if (values.docling_ocr_model) { document.getElementById('docling_ocr_model').value = values.docling_ocr_model; }
    if (values.docling_do_code_enrichment) { document.getElementById('docling_do_code_enrichment').checked = values.docling_do_code_enrichment; }
    if (values.docling_do_formula_enrichment) { document.getElementById('docling_do_formula_enrichment').checked = values.docling_do_formula_enrichment; }
    if (values.docling_do_table_structure) { document.getElementById('docling_do_table_structure').checked = values.docling_do_table_structure; }
    if (values.docling_table_structure_mode) { document.getElementById('docling_table_structure_mode').value = values.docling_table_structure_mode; }
    if (values.docling_do_picture_classification) { document.getElementById('docling_do_picture_classification').checked = values.docling_do_picture_classification; }
    if (values.docling_do_picture_description) { document.getElementById('docling_do_picture_description').checked = values.docling_do_picture_description; }
    if (values.docling_do_cell_matching) { document.getElementById('docling_do_cell_matching').checked = values.docling_do_cell_matching; }
    if (values.docling_cuda_use_flash_attention_2) { document.getElementById('docling_cuda_use_flash_attention_2').checked = values.docling_cuda_use_flash_attention_2; }
    if (values.docling_num_threads) { document.getElementById('docling_num_threads').value = values.docling_num_threads; }
}


function initializeOCRTabComponents(values) {
    initializeOCRTabRadios(values.force_ocr, values.azure_cv_free_tier);
    initializeOCRFormValues(values);
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
    document.getElementById('googleDriveLogout').addEventListener('click', googleDriveLogout);
    initSortingAndFiltering();  // Inititalize sorting and filtering
    document.getElementById('googleDriveSyncAction').addEventListener('click', triggerSyncGoogleDrive);
}


function initializeGoogleDriveTabComponents(values) {
    initializeGoogleDriveTabListeners();
}


function initializeButlerTabCheckboxes(values) {
    document.getElementById('enable_butler_mode_selection').checked = values.enable_butler_mode_selection;
}


function initializeButlerTabComponents(values) {
    initializeButlerTabCheckboxes(values);
}


function initializeAsrTabCheckboxes(values) {
    document.getElementById('enable_asr').checked = values.enable_asr;
    setAsr(values.enable_asr);

    document.getElementById('asr_apply_normalization').checked = values.asr_apply_normalization;
    document.getElementById('asr_apply_tts_padding').checked = values.asr_apply_tts_padding;
    document.getElementById('asr_apply_zero_padding').checked = values.asr_apply_zero_padding;
    document.getElementById('asr_apply_rms_dimming').checked = values.asr_apply_rms_dimming;
    document.getElementById('asr_apply_crossfade').checked = values.asr_apply_crossfade;
}


function initializeAsrTabDropdowns(values) {
    const asrModel = document.getElementById('asr_model');

    for (let option of asrModel.options) {
        if (option.value == values.asr_model) {
            option.selected = true;
            break;
        }    
    }
    
    const asrVadModel = document.getElementById('asr_vad_model');
    for (let option of asrVadModel.options) {
        if (option.value == values.asr_vad_model) {
            option.selected = true;
            break;
        }
    }

    const asrVadDevice = document.getElementById('asr_vad_device');
    for (let option of asrVadDevice.options) {
        if (option.value == values.asr_vad_device) {
            option.selected = true;
            break;
        }
    }
}


function initializeAsrTabFields(values) {
    document.getElementById('asr_wake_word').value = values.asr_wake_word;
    document.getElementById('asr_samplerate').value = values.asr_samplerate;
    document.getElementById('asr_temperature').value = values.asr_temperature;
    document.getElementById('asr_max_new_tokens').value = values.asr_max_new_tokens;
    document.getElementById('asr_volume_threshold').value = values.asr_volume_threshold;
    document.getElementById('asr_silence_duration_s').value = values.asr_silence_duration_s;
    document.getElementById('asr_min_chunk_duration_s').value = values.asr_min_chunk_duration_s;
    document.getElementById('asr_min_context_s').value = values.asr_min_context_s;
    document.getElementById('asr_stale_buffer_timeout_s').value = values.asr_stale_buffer_timeout_s;
    document.getElementById('asr_min_meaningful_samples_factor').value = values.asr_min_meaningful_samples_factor;
    document.getElementById('asr_vad_threshold').value = values.asr_vad_threshold;
    document.getElementById('asr_vad_min_speech_ms').value = values.asr_vad_min_speech_ms;
    document.getElementById('asr_vad_min_silence_ms').value = values.asr_vad_min_silence_ms;
    document.getElementById('asr_vad_window_size_samples').value = values.asr_vad_window_size_samples;
    document.getElementById('asr_vad_max_buffer_s').value = values.asr_vad_max_buffer_s;
    document.getElementById('asr_vad_speech_pad_ms').value = values.asr_vad_speech_pad_ms;
    document.getElementById('asr_waitress_access_url').value = values.asr_waitress_access_url;
    document.getElementById('asr_waitress_server_port').value = values.asr_waitress_server_port;
    document.getElementById('asr_torch_device').value = values.asr_torch_device;
    setHfwAsrUrl(values.asr_waitress_access_url, values.asr_waitress_server_port);
}


function initializeAsrTabComponents(values) {
    initializeAsrTabCheckboxes(values);
    initializeAsrTabDropdowns(values);
    initializeAsrTabFields(values);
}


function initializeTtsTabCheckboxes(values) {
    document.getElementById('enable_tts').checked = values.enable_tts;
    setTts(values.enable_tts);
}


function initializeTtsTabDropdowns(values) {
    document.getElementById('selected_tts').value = values.selected_tts;
    document.getElementById('selected_kokoro_voice').value = values.selected_kokoro_voice;
    document.getElementById('tts_sample_rate').value = values.tts_sample_rate;
}


function initializeTTSTabComponents(values) {
    initializeTtsTabCheckboxes(values);
    initializeTtsTabDropdowns(values);
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
    
    // document.getElementById('ModelAndDBLoading').style.display = 'none';
    // document.getElementById('ReadyToChat').style.display = 'block';
    
    // var timeoutDelayInMilliseconds = 1500; //1.5 seconds
    // setTimeout(function() {
    //     document.getElementById('ReadyToChat').style.display = 'none';
    // }, timeoutDelayInMilliseconds);
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


function loadAsrPipeline() {
    return fetch('/asr_server_starter_endpoint', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({'hard_reboot_required': false})
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) throw new Error('ASR Start Failed - Internal Server Error: Check server-log and server command-line for more details.');
        setAsr(true);
        if (data.reboot_failed) { alert("The ASR server is online but the updated settings were not applied. Check application and browser logs for more details."); }
        return true;
    })
    .catch(error => {
        setAsr(false);
        console.error(error);
        throw error;    // rethrow so promise.all rejects!
    });
}


function loadTTSPipeline() {
    return fetch('/load_tts_pipeline')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        setTts(true);
        return response
    })
    .catch(error => {
        setTts(false);
        console.error(error);
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
            if (data.hf_waitress_server_running && data.llama_cpp_server_running) {
                alert("Warning: Both HF-Waitress and llama.cpp servers appear to be running. Consider manually shutting one down to conserve memory!")
            }
            setServerStatusIndicator("Online");
            const llm_model = data.llm_model;
            setLlmModel(llm_model);
            if (data.reboot_failed) { alert("The LLM server is online but the updated settings were not applied. Check application and browser logs for more details."); }
            return initChatHistoryDB(llm_model);
        } else {
            setServerStatusIndicator("Error");
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        setServerStatusIndicator("Offline");
        errorHandler("loading the LLM", "/local-llm_server_starter", String(error.message))
        throw error;    // rethrow so promise.all rejects!
    })
}


document.addEventListener("DOMContentLoaded", function() {
    initializeUI();
    document.getElementById('ModelAndDBLoading').style.display = 'block';   //Start initializing the chat session
    showStreamSpinner();

    let local_llm_server;

    loadCoreLarsConfig()
        .then(values => {
            startFalkorDB();
            initializeLLMTabComponents(values);
            initializeEmbeddingModelTabComponents(values);
            initializeSystemPromptTabComponents(values);
            initializeRAGTabComponents(values);
            initializeGraphRAGTabComponents(values);
            initializeOCRTabComponents(values);
            initializeGoogleDriveTabComponents(values);
            initializeButlerTabComponents(values);
            initializeAsrTabComponents(values);
            initializeTTSTabComponents(values);
            initializeSettingsModalTabCycleListener();
            local_llm_server = values.local_llm_server;
            if (getTts() === "true") { loadTTSPipeline(); }
            if (getAsr() === "true") { return Promise.all([startLLMServer(), loadAsrPipeline()]); }
            else { return startLLMServer(); }
        })
        .then(() => {
            document.getElementById('ModelAndDBLoading').style.display = 'none';
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
            document.getElementById('ModelAndDBLoading').style.display = 'none';
        });
});//########### End DOMContentLoaded Block!