

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
                        if (string_and_status[1].trim() == 'success') {
                            populateDocsLoadedTable();
                        }
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
        const selectedTextExtractionMethod = document.getElementById('ocr_yes_radio_button').checked ? document.getElementById('ocrApiDropdown').value : 'default';
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

    const table = document.getElementById('google_drive_files_table');
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
                name: table.rows[i].getAttribute('data-gdrive-file-name'),
                mimeType: table.rows[i].getAttribute('data-gdrive-mime-type'),
                mimeTypeCategory: table.rows[i].getAttribute('data-gdrive-mime-type-category'),
                row: table.rows[i]
            });
            // Add status cell if it doesn't exist
            if (table.rows[i].cells.length < 6) {
                table.rows[i].insertCell(-1);
            }
            updateUIForFile(table.rows[i], 'waiting');
        }
    }

    const allStagedFileInfo = [];
    /*
    Process files sequentially via array.reduce():

    The syntax for array.reduce is: array.reduce(callback(accumulator, currentValue, currentIndex, array), initialValue)
    'file' below is the currentValue, 'index' is the currentIndex, and 'selectedFiles' is the array.
    'promise' is the accumulator, and is initialized to Promise.resolve(), which is the 'initialValue' of the accumulator.
    The callback function is executed for each element in the array.
    Each .then() adds a new promise to the chain so we can ensure that the chain is executed sequentially.
    When you add a .then() to a promise (even a resolved one), it creates a new promise that waits for the callback inside .then() to complete.
    This creates a sequential chain:
        Promise.resolve()
            .then(process file 1)
            .then(process file 2)
            .then(process file 3)
        ...and so on
    */
    selectedFiles.reduce((promise, file, index) => {
        return promise.then(() => {
            updateUIForFile(file.row, 'loading');
            const userId = "default_user";
            const transfer_type = file.mimeTypeCategory == 'folder' ? 'Folder' : 'File';
            const transfer_name = file.name;

            let formData = new FormData();
            formData.append('file_id', file.id);
            formData.append('file_mimeType', file.mimeType);
            formData.append('user_id', userId);
    
            return fetch('/gdrive_file_transfer_to_staging', {
                method: 'POST',
                body: formData,
                redirect: 'follow'
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(`Failed to download file ${file.id} from Google Drive to Server. Encountered error: ${err.error}`)});
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    appendStreamInfo(`${transfer_type} ${transfer_name} transferred to server successfully!`, 'success');
                    if (data.staged_file_info_list) {
                        if (Array.isArray(data.staged_file_info_list)) {
                            for (let i = 0; i < data.staged_file_info_list.length; i++) {
                                allStagedFileInfo.push(data.staged_file_info_list[i]);
                            }
                        } else {
                            console.error("Server response 'staged_file_info_list' was not an array:", data.staged_file_info_list);
                            throw new Error(`File ${transfer_name} not transfered to server correctly - server side error: invalid response format. Check server logs for more details`)
                        }
                    }
                } else {
                    throw new Error(`File ${transfer_name} not transfered to server correctly - server side error. Check server logs for more details`)
                }
            })
            .catch(error => {
                appendStreamInfo(`Error transferring ${transfer_type} ${transfer_name} to server, skipping. Check browser and server logs for more details`, 'failure');
                console.log("Error transferring file to server: ", String(error.message));
                updateUIForFile(file.row, 'failure');
            });
        });
    }, Promise.resolve())
    .then(() => {
        appendStreamInfo("Files transferred to server successfully!", 'success');
        console.log("GDrive staging phase complete. Collected info:", allStagedFileInfo);

        if (allStagedFileInfo.length > 0) {
            appendStreamInfo(`Beginning file processing for ${allStagedFileInfo.length} files...`, 'waiting');

            let bulkUploadFormData = new FormData();
            bulkUploadFormData.append('docs_to_upload', JSON.stringify(allStagedFileInfo));

            return fetch('/bulk_upload_files', {
                method: 'POST',
                body: bulkUploadFormData,
                redirect: 'follow'
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to process transferred files');
                }
                return response.body.getReader();
            })
            .then(reader => bulk_response_processChunk(reader));
            /*
            Since bulk_response_processChunk is an async function and uses await reader.read(), the promise returned by 
            .then(reader => bulk_response_processChunk(reader)) will resolve only after the entire stream has been read and 
            processed by bulk_response_processChunk. This ensures that the subsequent .catch() and 
            .finally() blocks wait for the stream processing to complete.
            */
        } else {
            appendStreamInfo("No files were transferred to the server. Please check the server logs for more details.", 'failure');
        }
    })
    .catch(error => {
        errorHandler("Processing uploaded files", "bulk_upload_files", String(error.message));
        appendStreamInfo(String(error.message), 'failure');
    })
    .finally(() => {
        syncButton.disabled = false;
        appendStreamInfo("Google Drive Synchronization Completed", 'success');
        populateDocsLoadedTable();
        hideStreamSpinner();
        for (let i = 0; i < selectedFiles.length; i++) {    // Update status of GDrive table rows to success
            if (selectedFiles[i].row) { 
                let selectedFilesRow = selectedFiles[i].row;    // But only if the status is not failure!
                let statusCell = selectedFilesRow.cells[selectedFilesRow.cells.length - 1];
                if (statusCell.innerHTML.includes("failure") || statusCell.innerHTML.includes("Failed")) {
                    continue;
                } else {
                    updateUIForFile(selectedFiles[i].row, 'success');
                }
            }
        }
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