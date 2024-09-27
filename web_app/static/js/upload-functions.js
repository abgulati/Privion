

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