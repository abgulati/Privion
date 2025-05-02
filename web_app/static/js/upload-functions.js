

// Upload new files to VectorDB
// document.getElementById('fileInput').addEventListener('change', function (event) {
//     const confirmed = confirm('Make sure to verify that the following Settings pertaining to File Uploading are correct:\n\n- Text Extraction Method: ' 
//         + (document.getElementById('ocr_yes_radio_button').checked ? 'OCR' : 'Non-OCR (Plain-Text Extraction)') 
//         + '\n- OCR Service Choice: ' + (document.getElementById('ocr_yes_radio_button').checked ? document.getElementById('ocrApiDropdown').value : 'Not Applicable') 
//         + '\n- Embedding Model: ' + document.getElementById('hf-waitress-embed-custom-dropdown-selected-value').textContent
//         + '\n- Knowledge Domain: ' + document.getElementById('hf-waitress-kb-custom-dropdown-selected-value').textContent
//         + '\n\nIf unsure, click Cancel to abort the file upload process.');

//     if (!confirmed) {
//         document.getElementById('fileInput').value = "";  // Clear the input value
//         return;
//     }
    
//     if (this.value) {    // Check if a file is selected
    
//         document.getElementById('overlay').style.display = 'block';
        
//         let newFile = document.getElementById('fileInput');
//         let file = newFile.files[0]

//         if (file) {
//             let formData = new FormData();
//             formData.append('file', file);

//             // Make the AJAX request to the server
//             fetch('/process_new_file', {
//                 method: 'POST',
//                 body: formData
//             })
//             .then(response => {
//                 if (!response.ok) {
//                     return response.json().then(err => { throw new Error(err.error)});
//                 }
//                 return response
//             })
//             .then(response => response.json())
//             .then(data => {
//                 if (data.success) {
//                     populateDocsLoadedTable();
//                     document.getElementById('overlay').style.display = 'none';
//                     document.getElementById('fileInput').value = "";  // Clear the input value
//                 } else {
//                     throw new Error(`Internal Server Error: Check server-log and server command-line for more details.`);
//                 }
//             })
//             .catch(error => {
//                 errorHandler("processing file", "/process_new_file", String(error.message))
//                 document.getElementById('overlay').style.display = 'none';
//                 document.getElementById('fileInput').value = "";  // Clear the input value
//             });
//         }
//     }    
// });


async function bulk_response_processChunk(reader) {
    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            console.log("Bulk file upload stream complete");
            break;
        }
        const textChunk = new TextDecoder("utf-8").decode(value);
        const messages = textChunk.split('\n');

        messages.forEach(message => {
            if (message.startsWith('data: ')) {
                const jsonStr = message.slice(7, -1);
                try {
                    let dataObj = String(jsonStr);
                    if (dataObj == "null") dataObj = "";
                    if (dataObj != "") {
                        const string_and_status = dataObj.split('|');
                        appendStreamInfo(string_and_status[0].trim(), string_and_status[1] === undefined ? 'waiting' : string_and_status[1].trim());
                    }
                } catch (error) {
                    console.error('Error parsing message: ', error);
                }
            }
        });
    }
}

async function handleFileOrFolderSelection(event) {
    const inputElement = event.target;

    const confirmed = confirm('Make sure to verify that the following Settings pertaining to File Uploading are correct:\n\n- Text Extraction Method: ' 
        + (document.getElementById('ocr_yes_radio_button').checked ? 'OCR' : 'Non-OCR (Plain-Text Extraction)') 
        + '\n- OCR Service Choice: ' + (document.getElementById('ocr_yes_radio_button').checked ? document.getElementById('ocrApiDropdown').value : 'Not Applicable') 
        + '\n- Embedding Model: ' + document.getElementById('hf-waitress-embed-custom-dropdown-selected-value').textContent
        + '\n- Knowledge Domain: ' + document.getElementById('hf-waitress-kb-custom-dropdown-selected-value').textContent
        + '\n\nIf unsure, click Cancel to abort the file upload process.');
    
    if (!confirmed) {
        inputElement.value = "";  // Clear the input value
        return;
    }

    if (inputElement.files && inputElement.files.length > 0) {    // Check if a file is selected

        appendStreamInfo("File Upload In-Progress...", 'waiting');
        showStreamSpinner();

        let files = inputElement.files;

        appendStreamInfo("Transferring all selected files to LARS Server...", 'waiting');

        const allStagedFileInfo = [];
        const stagingPromises = [];
        const userId = "default_user"; // For future use, will be set to the user's ID when user is logged in.
        const uploadId = getUniqueId();
        const uploadInitiatedDatetime = new Date().toISOString();
        const selectedEmbeddingModel = document.getElementById('hf-waitress-embed-custom-dropdown-selected-value').textContent;
        const selectedKnowledgeDomain = document.getElementById('hf-waitress-kb-custom-dropdown-selected-value').textContent;
        const selectedTextExtractionMethod = document.getElementById('ocr_yes_radio_button').checked ? document.getElementById('ocrApiDropdown').value : 'Non-OCR (Plain-Text Extraction)';
        const source = "Local Drive";

        for (let i = 0; i < files.length; i++) {
            const file = files[i];

            const justFileName = file.webkitRelativePath ? file.webkitRelativePath.split('/').pop() : file.name;

            const fileTransferInfo = {
                user_id: userId,
                upload_id: uploadId,
                document_name_and_extension: justFileName,
                embedding_model: selectedEmbeddingModel,
                knowledge_domain: selectedKnowledgeDomain,
                source: source,
                text_extraction_method: selectedTextExtractionMethod,
                upload_initiated_datetime: uploadInitiatedDatetime
            };

            let singleFileFormData = new FormData();
            singleFileFormData.append('file_transfer_info', JSON.stringify(fileTransferInfo));
            singleFileFormData.append('file', file, justFileName);

            // Create a promise for each file transfer
            const stagingPromise = fetch('/file_transfer_to_staging', {
                method: 'POST',
                body: singleFileFormData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(`Failed to transfer file ${file.name} to Server. Encountered error: ${err.error}`)});
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    appendStreamInfo(`File ${file.name} transferred to server successfully!`, 'success');
                    allStagedFileInfo.push(data.staged_file_info);
                } else {
                    throw new Error(`File ${file.name} not transfered to server correctly - server side error. Check server logs for more details`)
                }
            })
            .catch(error => {
                appendStreamInfo(`Error transferring file ${file.name} to server, skipping. Check browser and server logs for more details`, 'failure');
                console.log("Error transferring file to server: ", String(error.message));
            });
            stagingPromises.push(stagingPromise);
        }   // End file-transfer for loop

        try {
            await Promise.all(stagingPromises); // Wait for all file transfers to staging area to complete (resolved either successfully or via the catch block)
            appendStreamInfo("Files transferred to server successfully!", 'success');
            console.log("File staging phase complete. Collected info:", allStagedFileInfo);

            if (allStagedFileInfo.length > 0) {
                appendStreamInfo(`Beginning file processing for ${allStagedFileInfo.length} files...`, 'waiting');

                let bulkUploadFormData = new FormData();
                bulkUploadFormData.append('docs_to_upload', JSON.stringify(allStagedFileInfo));

                const bulkResponse = await fetch('/bulk_upload_files', {
                    method: 'POST',
                    body: bulkUploadFormData,
                    redirect: 'follow'
                });

                if (!bulkResponse.ok) {
                    throw new Error('Failed to process transferred files');
                }

                const bulk_response_reader = bulkResponse.body.getReader();
                await bulk_response_processChunk(bulk_response_reader);

            } else {
                appendStreamInfo("No files were transferred to the server. Please check the server logs for more details.", 'failure');
            }

        } catch (error) {
            errorHandler("Processing uploaded files", "bulk_upload_files", String(error.message));
            appendStreamInfo(String(error.message), 'failure');
        } finally {
            inputElement.value = "";
            populateDocsLoadedTable();
            hideStreamSpinner();
            appendStreamInfo("File upload process completed!", 'success');
        }
    }
}


async function loadGoogleDriveDoc(file_id, file_mimeType) {
    try {
        let formData = new FormData();
        formData.append('file_id', file_id);
        formData.append('file_mimeType', file_mimeType);

        const gdrive_loader_response = await fetch('/google_drive_loader', {
            method: 'POST',
            body: formData,
            redirect: 'follow'
        });

        if (!gdrive_loader_response.ok) {
            throw new Error('Failed to load document from Google Drive');
        }

        const gdrive_loader_reader = gdrive_loader_response.body.getReader();
        let gdrive_loader_receivedComplete = false;

        async function gdrive_loader_processChunk() {
            while (true) {
                const { done, value } = await gdrive_loader_reader.read();
                if (done) {
                    console.log("Google Drive Loader stream complete");
                    break;
                }
                const textChunk = new TextDecoder("utf-8").decode(value);
                const messages = textChunk.split('\n');

                messages.forEach(message => {
                    if (message.startsWith('data: ')) {
                        const jsonStr = message.slice(7, -1);

                        try {
                            let dataObj = String(jsonStr);
                            if (dataObj == "null") {
                                dataObj = "";
                            }

                            if (dataObj != "") {
                                string_and_status = dataObj.split('|');
                                console.log(string_and_status);
                                appendStreamInfo(string_and_status[0].trim(), string_and_status[1] === undefined ? 'waiting' : string_and_status[1].trim());
                            }

                        } catch (error) {
                            console.error('Error parsing message: ', error);
                        }
                    }
                });

            }
        }

        await gdrive_loader_processChunk();
    } catch (error) {
        console.error("Error loading document from Google Drive in method loadGoogleDriveDoc(). Error details: ", String(error.message));
    }
}


// Clicking the 'googleDriveSyncAction' button triggers the following click-event response (see chat.html & DOM-Loader.js):
function triggerSyncGoogleDrive() {
    const confirmed = confirm('Make sure to verify that the following Settings pertaining to File Uploading are correct:\n\n- Text Extraction Method: ' 
        + (document.getElementById('ocr_yes_radio_button').checked ? 'OCR' : 'Non-OCR (Plain-Text Extraction)') 
        + '\n- OCR Service Choice: ' + (document.getElementById('ocr_yes_radio_button').checked ? document.getElementById('ocrApiDropdown').value : 'Not Applicable') 
        + '\n- Embedding Model: ' + document.getElementById('hf-waitress-embed-custom-dropdown-selected-value').textContent 
        + '\n- Knowledge Domain: ' + document.getElementById('hf-waitress-kb-custom-dropdown-selected-value').textContent 
        + '\n\nIf unsure, click Cancel to abort the file upload process.');

    if (!confirmed) {
        return;
    }

    const table = document.getElementById('google_drive_files_tables');
    const syncButton = document.getElementById('googleDriveSyncAction');
    const selectedFiles = [];

    if (!table || !syncButton) {
        console.error('Required elements not found');
        return;
    }

    syncButton.disabled = true;
    appendStreamInfo("Google Drive Synchronization In-Progress...", 'waiting');
    showStreamSpinner();

    for (let i = 1; i < table.rows.length; i++) {
        const checkbox = table.rows[i].cells[0].querySelector('input[type="checkbox"]');
        if (checkbox && checkbox.checked) {
            selectedFiles.push({
                id: table.rows[i].getAttribute('data-gdrive-file-id'),
                mimeType: table.rows[i].getAttribute('data-gdrive-mime-type'),
                row: table.rows[i]
            });
            // Add status cell if it doesn't exist
            if (table.rows[i].cells.length < 6) {
                table.rows[i].insertCell(-1);
            }
            updateUIForFile(table.rows[i], 'waiting');
        }
    }

    // Process files sequentially
    selectedFiles.reduce((promise, file, index) => {
        return promise.then(() => {
            updateUIForFile(file.row, 'loading');
            return loadGoogleDriveDoc(file.id, file.mimeType)
                .then(() => {
                    console.log(`File ${index + 1} loaded successfully`);
                    clearDocsLoadedTable();
                    populateDocsLoadedTable();
                    updateUIForFile(file.row, 'success');
                })
                .catch(error => {
                    console.error(`Error loading file ${index + 1}:`, error);
                    updateUIForFile(file.row, 'failure');
                });
        });
    }, Promise.resolve())
    .finally(() => {
        syncButton.disabled = false;
        appendStreamInfo("Google Drive Synchronization Completed!", 'success');
        hideStreamSpinner();
    });
}


// This function is triggered when the DOMContentLoaded event is fired (see DOM-Loader.js) - page refreshes on login, triggering this function!
function handleGoogleDrivePostAuth() {
    appendStreamInfo("Checking Google Drive Auth...", 'waiting');
    fetch('/check_gdrive_auth')
    .then(response => response.json())
    .then(data => {
        if (!data.is_authenticated) {
            console.log("User not authenticated to GDrive, skipping Google Drive sync.");
            return;
        }
        console.log("User authenticated to GDrive, proceeding with Google Drive sync.");
        showLoader();
        fetch('/get_google_drive_user')
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error)});
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                appendStreamInfo("Google Drive Logged In Successfully, proceeding to fetch file list...", 'success');
                document.getElementById('googleDriveUserName').textContent = "Logged in as: " + data.user_name;
                document.getElementById('googleDriveUserName').style.display = 'block';

                return fetch('/fetch_file_list_from_google_drive');
            } else {
                throw new Error("Failed to fetch Google Drive user.");
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to fetch Google Drive files.");
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                appendStreamInfo("Google Drive Files Fetched Successfully", 'success');
                hideLoader();
                console.log("GDrive Files Fetched");
                console.log(data.gdrive_files);
                if (data.gdrive_files.length > 0) {
                    clearGoogleDriveTable();
                    populateGoogleDriveTable(data.gdrive_files);
                    document.getElementById('googleDriveSyncAction').style.display = 'block';
                }
            } else {
                throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
            }
        })
    })
    .catch(error => {
        errorHandler("Google Auth Check Failed", "handleGoogleDrivePostAuth()", String(error.message));
    });
}


function handleUploadError(errorMessage) {
    console.error('Error uploading file:', errorMessage);
    alert("Error when trying to upload new LLM to models dir. Check console for more details.");
    cleanupUpload();
}

function cleanupUpload() {
    document.getElementById('UploadingLlmOverlay').style.display = 'none';
    document.getElementById('uploadLlm').value = "";  // Clear the input value
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('progressPercentage').textContent = '0%';
}

// Upload LLMs to '/models' dir
document.getElementById('uploadLlm').addEventListener('change', function (event) {
    if (this.value) {    // Check if a file is selected

        document.querySelector('.btn-close[data-bs-dismiss="modal"]').click();
        document.getElementById('UploadingLlmOverlay').style.display = 'block';
        document.getElementById('uploadProgress').style.display = 'block';
        
        let newFile = document.getElementById('uploadLlm');
        let file = newFile.files[0]

        if (file) {
            let formData = new FormData();
            formData.append('file', file);

            let xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload_new_llm', true);

            xhr.upload.onprogress = function(e) {
                if (e.lengthComputable) {
                    let percentComplete = (e.loaded / e.total) * 100;
                    document.getElementById('progressPercentage').textContent = percentComplete.toFixed(2) + '%';
                }
            };

            xhr.onload = function() {
                if (xhr.status === 200) {
                    let response;
                    try {
                        response = JSON.parse(xhr.responseText);
                    } catch(e) {
                        handleUploadError('Invalide JSON response from server');
                        return;
                    }
                    
                    if (response.success) {
                        
                        const initKeysToRead = ['model_choice']
                        fetch('/config_reader_api', {
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
                            return response
                        })
                        .then(response => response.json())
                        .then(data => {
                            var values = data.values;
                            var model_choice = values.model_choice;

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

                                    const dropdown = document.getElementById('modelDropdown');
                                    dropdown.innerHTML = '';
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

                                    cleanupUpload();
                                
                                } else {
                                    throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
                                }
                            })
                            .catch(error => {
                                handleUploadError(`There was an error in fetching the model list: ${error.message}`)
                            }); // END load_local_models

                        })
                        .catch(error => {
                            handleUploadError(`There was an error in reading config.json: ${error.message}`)
                        }); //END config_reader_api call

                    } else {
                        handleUploadError('Server-side response indicates failure')
                    }
                } else {
                        handleUploadError(`Non-200 response: ${xhr.status} ${xhr.statusText}`);
                    }
            };

            xhr.onerror = function() {
                handleUploadError('Network error occured - xhr.onerror() triggered')
            };

            xhr.send(formData);

        }
    }
});