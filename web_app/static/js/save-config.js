

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
            config.llama_cpp_use_gpu = document.getElementById('UseGpu').checked;
            config.llama_cpp_gpu_layers = parseInt(document.getElementById('NumbGpuLayers').value);
            config.llama_cpp_unified_kv_buffer = document.getElementById('UnifiedKvBuffer').checked;
            config.llama_cpp_disable_kv_offloading = document.getElementById('DisableKvOffloading').checked;
            config.llama_cpp_key_cache_data_type = document.getElementById('KeyCacheDataType').value;
            config.llama_cpp_value_cache_data_type = document.getElementById('ValueCacheDataType').value;
            config.llama_cpp_no_of_seqs_to_par_decode = parseInt(document.getElementById('NoOfSeqsToParDecode').value);
            config.llama_cpp_offload_to_devices = document.getElementById('OffloadToDevices').value;
            config.llama_cpp_cpu_only_moe = document.getElementById('CpuOnlyMoe').checked;
            config.llama_cpp_mlock = document.getElementById('Mlock').checked;
            config.llama_cpp_no_nmap = document.getElementById('NoNmap').checked;
            config.llama_cpp_context_length = parseInt(document.getElementById('LlmCtxLgt').value);
            config.llama_cpp_max_new_tokens = parseInt(document.getElementById('MaxNewToks').value);
            config.llama_cpp_temperature = parseFloat(document.getElementById('tempSlider').value);
            config.llama_cpp_top_k = parseInt(document.getElementById('topkSlider').value);
            config.llama_cpp_top_p = parseFloat(document.getElementById('toppSlider').value);
            config.llama_cpp_min_p = parseFloat(document.getElementById('minpSlider').value);
            config.llama_cpp_n_keep = parseInt(document.getElementById('nkeepSlider').value);
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
    return config = {
        'selected_embedding_model': document.getElementById('hf-waitress-embed-custom-dropdown-selected-value').textContent,
        'selected_knowledge_domain': document.getElementById('hf-waitress-kb-custom-dropdown-selected-value').textContent,
        'selected_reranker_model': document.getElementById('hf-waitress-reranker-custom-dropdown-selected-value').textContent,
        'use_embedding_model_for_reranking': document.getElementById('use_embedding_model_for_reranking').checked
    }
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

    return {
        'force_enable_rag': force_enable_rag, 
        'force_disable_rag': force_disable_rag, 
        'llm_filter_citations': llm_filter_citations,
        'fetch_top_k_results_from_whoosh': document.getElementById('fetch_top_k_results_from_whoosh').value,
        'fetch_top_k_results_from_vectordb': document.getElementById('fetch_top_k_results_from_vectordb').value,
        'filter_top_k_results_by_reranking': document.getElementById('filter_top_k_results_by_reranking').value,
        'min_semantic_similarity_threshold': document.getElementById('min_semantic_similarity_threshold').value,
        'min_lexical_similarity_threshold': document.getElementById('min_lexical_similarity_threshold').value
    };
}


function getGraphRAGConfig() {
    let reuse_graph_extraction_cache_without_validation = false;
    let reuse_graph_extraction_cache_with_validation = false;

    let reuse_graph_summary_cache_without_validation = false;
    let reuse_graph_summary_cache_with_validation = false;

    if (document.getElementById('reuse_graph_extraction_cache_checkbox').checked) {
        if (document.getElementById('validate_graph_extraction_cache_checkbox').checked) {
            reuse_graph_extraction_cache_without_validation = false;    // lars-cache not used - determine_graph_cache_reuse() resolves to False
            reuse_graph_extraction_cache_with_validation = true;    // waitress cache used - 'X-Reuse-Extraction-Cache' header set to True
        } else {
            reuse_graph_extraction_cache_without_validation = true;    // lars-cache used - determine_graph_cache_reuse() resolves to True
            reuse_graph_extraction_cache_with_validation = false;    // waitress cache not used - call never made as determine_graph_cache_reuse() resolves to True
        }
    } else {
        reuse_graph_extraction_cache_without_validation = false;    // lars-cache not used - determine_graph_cache_reuse() resolves to False
        reuse_graph_extraction_cache_with_validation = false;    // waitress cache not used - 'X-Reuse-Extraction-Cache' header set to False
    }

    if (document.getElementById('reuse_graph_summarization_cache_checkbox').checked) {
        if (document.getElementById('validate_graph_summarization_cache_checkbox').checked) {
            reuse_graph_summary_cache_without_validation = false;    // lars-cache not used - determine_graph_cache_reuse() resolves to False
            reuse_graph_summary_cache_with_validation = true;    // waitress cache used - 'X-Reuse-Summary-Cache' header set to True
        } else {
            reuse_graph_summary_cache_without_validation = true;    // lars-cache used - determine_graph_cache_reuse() resolves to True
            reuse_graph_summary_cache_with_validation = false;    // waitress cache not used - call never made as determine_graph_cache_reuse() resolves to True
        }
    } else {
        reuse_graph_summary_cache_without_validation = false;    // lars-cache not used - determine_graph_cache_reuse() resolves to False
        reuse_graph_summary_cache_with_validation = false;    // waitress cache not used - 'X-Reuse-Summary-Cache' header set to False
    }

    return {
        // Extractor Settings:
        'graph_generator_model': document.getElementById('graph-extractor-custom-dropdown-selected-value').textContent,
        'reuse_graph_extraction_cache_without_validation': reuse_graph_extraction_cache_without_validation,
        'reuse_graph_extraction_cache_with_validation': reuse_graph_extraction_cache_with_validation,
        'exl2_quantize_graph_model': document.getElementById('graph_extractor_exl2_yes').checked,
        'exl2_quantize_graph_model_bpw': document.getElementById('graph_extractor_exl2_bpw').value,
        'minimum_free_vram_for_graph_extraction_model': document.getElementById('graph_extractor_memory_required').value,
        'graph_model_max_new_tokens': document.getElementById('graph_extractor_max_new_tokens').value,
        'graph_model_max_seq_len': document.getElementById('graph_extractor_max_seq_len').value,
        'graph_model_temperature': document.getElementById('graph_extractor_temperature').value,
        'graph_model_do_sample': document.getElementById('graph_extractor_temperature').value > 0.0 ? true : false,
        'graph_model_top_k': document.getElementById('graph_extractor_top_k').value,
        'graph_model_top_p': document.getElementById('graph_extractor_top_p').value,
        'graph_model_min_p': document.getElementById('graph_extractor_min_p').value,
        'graph_model_access_url': document.getElementById('graph_extractor_access_url').value,
        'graph_model_server_port': document.getElementById('graph_extractor_server_port').value,

        // Summarizer Settings:
        'graph_summarizer_model': document.getElementById('graph-summarizer-custom-dropdown-selected-value').textContent,
        'reuse_graph_summary_cache_without_validation': reuse_graph_summary_cache_without_validation,
        'reuse_graph_summary_cache_with_validation': reuse_graph_summary_cache_with_validation,
        'exl2_quantize_graph_summarizer_model': document.getElementById('graph_summarizer_exl2_yes').checked,
        'exl2_quantize_graph_summarizer_model_bpw': document.getElementById('graph_summarizer_exl2_bpw').value,
        'minimum_free_vram_for_graph_summarizer_model': document.getElementById('graph_summarizer_memory_required').value,
        'graph_summarizer_max_new_tokens': document.getElementById('graph_summarizer_max_new_tokens').value,
        'graph_summarizer_max_seq_len': document.getElementById('graph_summarizer_max_seq_len').value,
        'graph_summarizer_temperature': document.getElementById('graph_summarizer_temperature').value,
        'graph_summarizer_do_sample': document.getElementById('graph_summarizer_temperature').value > 0.0 ? true : false,
        'graph_summarizer_top_k': document.getElementById('graph_summarizer_top_k').value,
        'graph_summarizer_top_p': document.getElementById('graph_summarizer_top_p').value,
        'graph_summarizer_min_p': document.getElementById('graph_summarizer_min_p').value,
        'graph_summarizer_access_url': document.getElementById('graph_summarizer_access_url').value,
        'graph_summarizer_server_port': document.getElementById('graph_summarizer_server_port').value,

        // General Settings:
        'enable_graph_rag': document.getElementById('enable_graph_rag_checkbox').checked,
        'upload_doc_to_graph_db': document.getElementById('upload_doc_to_graph_db_checkbox').checked,
        'graph_rag_context_length_limit_chars': document.getElementById('graph_rag_context_length_limit_chars').value,
        'graph_chunk_size': document.getElementById('graph_rag_chunk_size').value,
        'graph_chunk_overlap': document.getElementById('graph_rag_chunk_overlap').value,

        // GraphDB Settings:
        'graph_db_server_host': document.getElementById('graph_db_server_host').value,
        'assign_host_port_to_graph_db_server': document.getElementById('graph_db_server_port').value,
        'launch_graph_db_with_ui': document.getElementById('launch_graph_db_with_ui_checkbox').checked,
        'assign_host_port_to_graph_db_ui': document.getElementById('graph_db_ui_port').value,
        'apply_clustering_to_graph_db_on_doc_load': document.getElementById('apply_clustering_to_graph_db_on_doc_load_checkbox').checked
    };
}


function getOcrConfig() {
    const azure_ocr_form = document.getElementById("azure_vision_api_form");
    const azure_doc_ai_form = document.getElementById("azure_doc_ai_api_form");
    const local_vision_form = document.getElementById("local_vision_api_form");
    const kosmos_form = document.getElementById("kosmos_api_form");
    const docling_form = document.getElementById("docling_form");

    const ocr_yes_radio_button = document.getElementById('ocr_yes_radio_button').checked;
    const update_azure_vision_config = document.getElementById('update_azure_vision').checked;
    const update_azure_doc_ai_config = document.getElementById('update_azure_doc_ai').checked;
    const update_local_vision_config = document.getElementById('update_local_vision_config').checked;
    const update_kosmos_url_config = document.getElementById('update_kosmos_url_config').checked;
    const update_docling_config = document.getElementById('update_docling_config').checked;

    const force_extract_previously_extracted_text = document.getElementById('force_extract_previously_extracted_text_checkbox').checked;
    const backup_ocr_service_choice = document.getElementById('backupOcrApiDropdown').value;

    let config = {
        'ocr_service_choice': 'None',
        'force_ocr': false,
        'force_extract_previously_extracted_text': force_extract_previously_extracted_text,
        'backup_ocr_service_choice': backup_ocr_service_choice
    };

    if (window.getComputedStyle(azure_ocr_form).display != 'none') {
        config.ocr_service_choice = 'AzureVision';
        config.force_ocr = true;
        if (update_azure_vision_config) {
            config.azure_cv_free_tier = document.getElementById('is_azure_cv_free_tier').checked;
            config.azure_ocr_endpoint = document.getElementById("azure_vision_api_url").value;
            config.azure_ocr_subscription_key = document.getElementById("azure_vision_api_key").value;
        }
    } else if (window.getComputedStyle(azure_doc_ai_form).display != 'none') {
        config.ocr_service_choice = 'AzureDocAi';
        config.force_ocr = true;
        if (update_azure_doc_ai_config) {
            config.azure_doc_ai_endpoint = document.getElementById("azure_doc_ai_api_url").value;
            config.azure_doc_ai_subscription_key = document.getElementById("azure_doc_ai_api_key").value;
        }
    } else if (window.getComputedStyle(local_vision_form).display != 'none') {
        config.ocr_service_choice = 'LocalVisionLLM';
        config.force_ocr = true;
        if (update_local_vision_config) {
            config.local_vision_endpoint = document.getElementById("local_vision_api_url").value;
        }
    } else if (window.getComputedStyle(kosmos_form).display != 'none') {
        config.ocr_service_choice = 'Kosmos';
        config.force_ocr = ocr_yes_radio_button ? true : false;
        if (update_kosmos_url_config) {
            config.kosmos_local_url = document.getElementById("kosmos_api_url").value;
            config.kosmos_task = document.getElementById("kosmos_task").value;
            config.kosmos_threshold = document.getElementById("kosmos_threshold").value;
        }
    } else if (window.getComputedStyle(docling_form).display != 'none') {
        config.ocr_service_choice = 'Docling';
        config.force_ocr = ocr_yes_radio_button ? true : false;
        if (update_docling_config) {
            config.docling_pipeline = document.getElementById("docling_pipeline").value;
            if (config.docling_pipeline === 'vlm') {
                config.docling_vlm_model = document.getElementById("docling_vlm_model").value;
            } else if (config.docling_pipeline === 'standard') {
                
                config.docling_do_ocr = document.getElementById("docling_enable_ocr").checked;
                if (config.docling_do_ocr) {
                    config.docling_force_full_page_ocr = document.getElementById("docling_force_full_page_ocr").checked;
                    config.docling_ocr_model = document.getElementById("docling_ocr_model").value;
                }

                config.docling_do_table_structure = document.getElementById("docling_do_table_structure").checked;
                if (config.docling_do_table_structure) {
                    config.docling_table_structure_mode = document.getElementById("docling_table_structure_mode").value;
                }

                config.docling_do_code_enrichment = document.getElementById("docling_do_code_enrichment").checked;
                config.docling_do_formula_enrichment = document.getElementById("docling_do_formula_enrichment").checked;
                config.docling_do_picture_classification = document.getElementById("docling_do_picture_classification").checked;
                config.docling_do_picture_description = document.getElementById("docling_do_picture_description").checked;
                config.docling_do_cell_matching = document.getElementById("docling_do_cell_matching").checked;
                config.docling_cuda_use_flash_attention_2 = document.getElementById("docling_cuda_use_flash_attention_2").checked;
                config.docling_num_threads = document.getElementById("docling_num_threads").value;
            }
        }
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
        'exl2': document.getElementById('hf_waitress_exl2_yes').checked,
        'exl2_bpw': parseFloat(document.getElementById('HfwExl2Bpw').value),
        'exl2_max_seq_len': parseInt(document.getElementById('HfwExl2MaxSeqLen').value),
        'exl2_cache_type': document.getElementById('hf_waitress_exl2_cache_type_choice').value,
        'exl2_force_regenerate_measurement': document.getElementById('hf_waitress_exl2_force_regenerate_measurement').checked,
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
        errorHandler("writing to config.json", "/config-writer_api", String(error.message));
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


function requestHFWaitressHardReboot() {
    console.log("Hard-Reboot of HF-Waitress server requested. Restarting server...");
    return fetch('/hf_waitress_server_starter_endpoint', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({'hard_reboot_required': true})
    }).then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error) });
        }
        return response.json();
    }).then(data => {
        console.log("HF-Waitress server starter endpoint response: ", data);
    });
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
            console.log("HF-Waitress server starter endpoint response: ", data);
            if (data.restart_required) {
                // Restart HF-Waitress Server
                if (data.hard_reboot_required) {
                    return requestHFWaitressHardReboot();
                } else {
                    console.log("HF-Waitress server changes saved successfully, restarting server...");
                    return fetchRestartEventStream();   // Call to an async function will return a promise, so we need to return the promise...
                }
            } else {
                console.log("HF-Waitress server changes saved successfully, restart unnecessary.");
                resolve();
            }
        })
        .then(() => {
            resolve();  // ...which is resolved here!
        })
        .catch(error => {
            errorHandler("handling HF-Waitress Server changes", "handle-HfWaitressChanges", String(error.message));
            reject(error);  // Reject the promise if there's an error
        })
        .finally(() => {
            document.getElementById('SavingHfWaitressSettings').style.display = 'none';
            // setTimeout(() => {
            //     document.getElementById('SavingHfWaitressSettings').style.display = 'none';
            // }, 3000);
        });
    });
}


function handleSaveChanges() {
    
    const config = {
        ...getLlmConfig(),
        ...getVectorEmbeddingsConfig(),
        ...getSysPromptConfig(),
        ...getRagConfig(),
        ...getGraphRAGConfig(),
        ...getOcrConfig(),
        ...getGoogleDriveConfig()
    };

    let hfSavePromise;  // Declaring the variable that will hold the promise returned by handle-HfWaitressChanges(). Defaulting will result in a race condition, so we'll need to handle the Promise chain manually.
    let needsReload = false;

    if (config.local_llm_server === "hf-waitress") {
        const hf_waitress_server_status = document.getElementById('local_llm_server_status_indicator_text').style.color;
        if (hf_waitress_server_status === "green") {
            const hf_config = {
                ...getHfWaitressConfig()
            };
            console.log("handle-HfWaitressChanges hf_config: ", hf_config);
            hfSavePromise = handleHfWaitressChanges(hf_config)  // Added to the promise chain only if the conditions are met!
                .then(() => {
                    console.log("HF-Waitress server changes saved successfully.");
                    
                    let chatID = getChatId();
                    if (hf_config.model_id.toLowerCase().includes('vision-instruct')) { // we have switched to a Vision-Instruct model...
                        console.log("Loaded up a Vision model");
                        document.getElementById('textAttachmentButton').disabled = false;
                        if (!String(getLlmModel()).toLowerCase().includes('vision-instruct')) {  // First check the current model...
                            console.log("Reloading the page as we've switched to a Vision-Instruct model");
                            needsReload = true;
                        }
                        if (!needsReload && String(getOldLlmModel()) != null && !String(getOldLlmModel()).toLowerCase().includes('vision-instruct')) { //...midway through a chat, or while browsing chat history, and the old LLM model was not a Vision-Instruct model...
                            console.log("Reloading page as we've switched to a Vision-Instruct model in the middle of a chat");
                            needsReload = true;
                        }
                    } else {    //...we have loaded up a non-Vision-Instruct model...
                        console.log("Loaded up a non-Vision model");
                        document.getElementById('textAttachmentButton').disabled = true;
                        if (String(getLlmModel()).toLowerCase().includes('vision-instruct')) {  // First check the current model...
                            console.log("Reloading the page as we've switched to a non-Vision model");
                            needsReload = true;
                        }
                        if ((!needsReload && String(getOldLlmModel()) != null) && (String(getOldLlmModel()).toLowerCase().includes('vision-instruct') || String(getOldLlmModel()).toLowerCase().includes('gguf') || String(getOldLlmModel()).toLowerCase().includes('flux'))) {    //...from a Vision-Instruct, FLUX, or a GGUF model...
                            console.log("Reloading the page to clear the previous model context");
                            needsReload = true;
                        }
                    }
                    setModelHeaderInfoBox(chatID, hf_config.model_id);
                    return needsReload;
                })
                .catch(error => {
                    errorHandler("saving HF-Waitress settings", "handle-SaveChanges()", String(error.message));  // Catching the error will resolve the Promise and allow the rest of the code to continue!
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
    Promise.all([hfSavePromise, saveConfigToServer(config)])    // This promise will always resolve, as hfSavePromise is manually set to resolve to false when necessary, so we're not blocking the save-ConfigToServer() promise. Promise.all() is used to handle both promises in parallel, waiting for both to resolve before proceeding.
        .then(([hfNeedsReload, configNeedsReload]) => {
            console.log("LARS config saved successfully.");
            if (hfNeedsReload || configNeedsReload) {
                console.log("A restart is required to apply the changes.");
                location.reload();  // centralized handling of effects that depend on multiple async operations.
            }
        })
        .catch(error => {
            errorHandler("saving LARS settings", "handle-SaveChanges()", String(error.message));
        });
}

document.getElementById('saveChangesButton').addEventListener('click', handleSaveChanges);
//########### End SaveConfig block!