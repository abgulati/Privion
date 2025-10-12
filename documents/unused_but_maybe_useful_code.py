from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import TextLoader
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA


# JS: // let streamed_content = dataObj.replace(/(?<![A-Z]:|\/|[0-9]|[ivxlcdm])([.?!])(?=\s|$|[0-9])(?!\s*\/)/g, '$1<br><br>');

def preprocess_string(s):
    """
    This function removes all non-alphanumeric characters from the string, 
    converts it to lowercase, and trims whitespace.
    It's not used in the current implementation of LARS.
    """
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()




def PDFtoMSTrOCR(input_filepath):
    
    print("\n\nProcessing Document - PDF to MS TrOCR TXT\n\n")

    try:
        read_return = read_config(['model_dir'])
        model_directory = read_return['model_dir']
    except Exception as e:
        handle_local_error("Missing model_dir in config.json for PDFtoMSTrOCR. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_filepath, 300) # 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)
    
    # Set output path
    output_text_file_path = input_filepath.replace(".pdf","_ms_tr_ocr_cleaned.txt") 
    raw_output_text_file_path = input_filepath.replace(".pdf","_ms_tr_ocr_raw.txt") 

    # Init list for Whoosh indexing
    pdf_data = []

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
        raw_output_text_file = open(raw_output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)
    
    # Setting up Cleaner LLM:
    llm_name = 'openhermes-2.5-mistral-7b.Q8_0.gguf'
    llm_dir = model_directory + '/' + llm_name
    config = {'context_length': 8192, 'max_new_tokens': 8192, 'gpu_layers':50}
    cleaner_llm = CTransformers(model=llm_dir, model_type="llama", config=config)
    # cleanup_template = PromptTemplate(template="Correct the following text for any gramatical and formatting errors, otherwise leaving it unchanged: {input}", input_variables=["input"])
    conv_chain = ConversationChain(llm = cleaner_llm)

    #Load OCR TrOCR model:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    
    # Iterate over each page and apply OCR:
    print("\n\nBeginning image to MS TrOCR\n\n")
    for page_number, page_image in enumerate(pages, start = 1):

        rgb_image = page_image.convert("RGB")
        width, height = rgb_image.size

        page_text = ""

        block_no = 0

        # original_stdout = sys.stdout

        # Process the page in 240x71 blocks:
        # for y in range(0, height, 71):
        #     for x in range(0, width, 240):
                
        #         print(f"Processing block {block_no}")

        #         # long-term: discard output
        #         # f = open(os.devnull, 'w')
        #         # sys.stdout = f

        #         # Crop block:
        #         block = rgb_image.crop((x, y, x + 240, y + 71))

        #         # Process block with TrOCR:
        #         pixel_values = processor(images=block, return_tensors="pt").pixel_values

        #         generated_ids = model.generate(pixel_values)
        #         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        #         print(f"generated block text: {generated_text}")

        #         if str(generated_text) != "0.00":
        #             page_text += generated_text

        #         # Reset stdout to its original value
        #         # sys.stdout = original_stdout

        #         block_no += 1

        for y in range(0, height, 50):
                
            print(f"Processing block {block_no}")

            # long-term: discard output
            # f = open(os.devnull, 'w')
            # sys.stdout = f

            # Crop block:
            block = rgb_image.crop((0, y, width, y + 50))

            # Process block with TrOCR:
            pixel_values = processor(images=block, return_tensors="pt").pixel_values

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"generated block text: {generated_text}")

            if str(generated_text) != "0.00":
                page_text += generated_text

            # Reset stdout to its original value
            # sys.stdout = original_stdout

            block_no += 1

        # Save raw output of the above process for analysis:
        try:
            raw_output_text_file.write(page_text + '\n')
        except Exception as e:
            handle_local_error("Could not write to output text file, encountered error: ", e)
        
        # Clean Text:
        print(f"\n\nCleaning Page Text with {llm_name}\n\n")
        llm_input = f"Correct the following text for any gramatical and formatting errors, otherwise leaving it unchanged: {page_text}"
        try:
            clean_text = conv_chain.predict(input=llm_input)
        except Exception as e:
            handle_local_error("Could not clean text with LLM, encountered error: ", e)

        # Write the cleaned up text to the file
        try:
            output_text_file.write(clean_text + '\n')
        except Exception as e:
            handle_local_error("Could not write to output text file, encountered error: ", e)

        # Whoosh prep
        #whoosh_clean_text = preprocess_string(clean_text)
        whoosh_page_dict_entry = {"title": source_filename, "content": clean_text, "pagenumber":page_number+1}
        pdf_data.append(whoosh_page_dict_entry)

    # Close all files
    raw_output_text_file.close()
    output_text_file.close()

     # Create Whoosh Index; if error, log exception and proceed to returning output_text_file_path
    try:
        whoosh_indexer(pdf_data)
    except Exception as e:
        handle_error_no_return("Could not index file, encountered error: ", e)

    return output_text_file_path



#Local OCR using PyTesseract - Not used in LARS
def PDFtoOCRTXT(input_filepath):
    
    print("\n\nProcessing Document - PDF to OCR TXT\n\n")

    try:
        read_return = read_config(['base_directory'])
        app_base_directory = read_return['base_directory']
    except Exception as e:
        handle_local_error("Missing base_directory in config.json for PDFtoOCRTXT. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_filepath, 300) # 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)
    
    # Set output path
    output_text_file_path = input_filepath.replace(".pdf","_ocr_300.txt") 

    # Init list for Whoosh indexing
    pdf_data = []

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)
    
    # Iterate over each page and apply OCR:
    print("\n\nBeginning image to Text OCR\n\n")
    for page_number, page_image in enumerate(pages, start=1):

        try:
            custom_config = r'--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
            text = pytesseract.image_to_string(page_image, config='--psm 3')    # Page Segmentation Mode (PSM) 3 - Default; Fully Automatic Page Segmentation & OCR, but no Orientation and Script Detection (OSD). PSM 3,4 & 6 are common for docs. For a full list of PSMs, ask ChatGPT "Can you give me a walkthrough of all the different Page Segmentation Modes in Python's PyTesseract?" 
            #text = pytesseract.image_to_string(page_image, config=custom_config)
        except Exception as e:
            handle_error_no_return("Could not OCR text from page, encountered error: ", e)
            
        # Optionally save image for review
        try:
            ocr_img_directory = app_base_directory + '/OCR_IMAGES'
            if not os.path.exists(ocr_img_directory):
                os.makedirs(ocr_img_directory)
            ocr_img_filename = f'{ocr_img_directory}/{source_filename}_page_{page_number}.jpg'
            page_image.save(ocr_img_filename)
        except Exception as e:
            error_message = f"Could not save OCR image for {source_filename}_page_{page_number}, encountered error: "
            handle_error_no_return(error_message, e)

        # clean_text = text
        # Clean text
        clean_text = clean_text_string(text)
        
        # Optionally, you can include page numbers in the text file
        # output_text_file.write(f'\n\n--- Page {page_num + 1} ---\n\n')
        
        # Write the extracted text to the file
        try:
            output_text_file.write(clean_text + '\n')
        except Exception as e:
            handle_local_error("Could not write to output text file, encountered error: ", e)

        # Whoosh prep
        #whoosh_clean_text = preprocess_string(clean_text)
        whoosh_page_dict_entry = {"title": source_filename, "content": clean_text, "pagenumber":page_number+1}
        pdf_data.append(whoosh_page_dict_entry)

    # Close all files
    output_text_file.close()

    # Create Whoosh Index; if error, log exception and proceed to returning output_text_file_path
    try:
        whoosh_indexer(pdf_data)
    except Exception as e:
        handle_error_no_return("Could not index file, encountered error: ", e)

    return output_text_file_path




def PDFtoAzureOCRTXT_url(input_filepath):
    
    print("\n\nProcessing Document - PDF to Azure OCR TXT\n\n")

    try:
        read_return = read_config(['azure_ocr_endpoint', 'azure_ocr_subscription_key'])
        azure_ocr_endpoint = read_return['azure_ocr_endpoint']
        azure_ocr_subscription_key = read_return['azure_ocr_subscription_key']
    except Exception as e:
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for PDFtoAzureOCRTXT_url, please provide required API config. Error: ", e)
    
    try:
        os.environ["azure_ocr_endpoint"] = azure_ocr_endpoint
        os.environ["azure_ocr_subscription_key"] = azure_ocr_subscription_key
    except Exception as e:
        handle_local_error("Could not set OS environment variables for Azure OCR, encountered error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_filepath, 300) # 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)
    
    # Set output path
    output_text_file_path = input_filepath.replace(".pdf","_azure_ocr_300.txt") 

    # Init list for Whoosh indexing
    pdf_data = []

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)
    
    # Init Azure VisionServiceOptions
    service_options = sdk.VisionServiceOptions(os.environ["azure_ocr_endpoint"], os.environ["azure_ocr_subscription_key"])
    
    # Iterate over each page and apply OCR:
    print("\n\nBeginning image to Text OCR\n\n")
    for page_number, image in enumerate(pages, start = 1):
    #for image in pages:
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Save the image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image_file:
            image.save(temp_image_file, format='PNG')
            temp_image_path = temp_image_file.name

            # Setup cision source with byte array
            vision_source = sdk.VisionSource(url=temp_image_path)

            # Set analysis options:
            analysis_options = sdk.ImageAnalysisOptions()
            analysis_options.features = sdk.ImageAnalysisFeature.TEXT


            # Send to Azure OCR & analyze the image
            image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)
            result = image_analyzer.analyze()

            if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
                if result.text is not None:
                    # print("Text:")
                    for line in result.text.lines:
                        # print(f"Line: {line.content}")
                        clean_text = line.content

                        # Write the extracted text to the file:
                        try:
                            output_text_file.write(clean_text + '\n')
                        except Exception as e:
                            handle_local_error("Could not write to output text file, encountered error: ", e)

                        # Whoosh prep
                        #whoosh_clean_text = preprocess_string(clean_text)
                        whoosh_page_dict_entry = {"title": source_filename, "content": clean_text, "pagenumber":page_number+1}
                        pdf_data.append(whoosh_page_dict_entry)

            else:
                # Handle errors:
                error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
                print(" Analysis failed.")
                print("   Error reason: {}".format(error_details.reason))
                print("   Error code: {}".format(error_details.error_code))
                print("   Error message: {}".format(error_details.message))

    # Close all files
    output_text_file.close()

    # Create Whoosh Index; if error, log exception and proceed to returning output_text_file_path
    try:
        whoosh_indexer(pdf_data)
    except Exception as e:
        handle_error_no_return("Could not index file, encountered error: ", e)

    return output_text_file_path



def TxtCleaner(input_file):
    """
    Processes a text file by cleaning and subsequent indexing.

    :param input_file: Path to the input text file within the app's folder.
    :return: path to the cleaned output text file.
    """

    print("\nProcessing Text File")

    # Ensure the file is .txt:
    if not input_file.lower().endswith('.txt'):
        raise ValueError("File must be a .txt file")
    
    # Get filename
    try:
        source_filename = os.path.basename(input_file)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path:
    output_text_file_path = input_file.replace(".txt","_cleaned.txt")

    # Init list for Whoosh indexing
    text_data = []

    try:
        # Read and process the file, \ is a continuation char in Python used to split long lines of code for readibility!
        with open(input_file, 'r', encoding='utf-8') as input_file ,\
                open(output_text_file_path, 'w', encoding='utf-8') as output_text_file:
            
            # enumerate returns a tuple containing the count and value of an iterable such as a file or list. It starts at 0 but here we specify 1 as the start index:
            for line_num, line in enumerate(input_file, 1):
                
                # Clean text:
                clean_line = line.replace("►", "").replace("■", "").replace("▼", "")
                clean_line = clean_line.replace("Confidential Copy \n            for \n         DKPPU", "")
                clean_line = re.sub(r'\n(?=[a-z.])', '', clean_line)
                clean_line = re.sub(r'\n+', '\n', clean_line)
                clean_line = re.sub(r'[^\w\s]', '', clean_line)     # This regex substitutes anything that is not a word character or whitespace with an empty string.
                clean_line = re.sub(r'\s+', ' ', clean_line).strip()    # This regex substitutes any sequence of whitespace characters with a single space.

                # Write the cleaned text to the output file 
                output_text_file.write(clean_line + '\n')

                # Whoosh prep
                whoosh_page_dict_entry = {"title": source_filename, "content": clean_line, "pagenumber":line_num}
                text_data.append(whoosh_page_dict_entry)
    except Exception as e:
        handle_local_error("Could not create text file, encountered error: ", e)

    # Create Whoosh Index; if error, log exception and proceed to returning output_text_file_path
    try:
        whoosh_indexer(text_data)
    except Exception as e:
        handle_error_no_return("Could not index file, encountered error: ", e)

    return output_text_file_path



def find_text_in_pdf_dpr(pdf_path, target_text):
    print("pdf_path, target_text: ", pdf_path, ", ", target_text)

    page_numbers = []

    try:
        with open(pdf_path, 'rb') as file:

            reader = PyPDF2.PdfReader(file)

            for page_num in range(len(reader.pages)):

                page = reader.pages[page_num]

                content = page.extract_text()

                if target_text in content:
                    print("found match!")
                    page_numbers.append(page_num + 1)
    except Exception as e:
        handle_local_error("Could not find page numbers from PDF, encountered error: ", e)

    print("page numbers before returning: ")
    print(page_numbers)
    return page_numbers



def find_text_in_pdf(reference_pages):

    user_should_refer_pages_in_doc = {}
    docs_have_relevant_info = False

    for doc_path in reference_pages:

        source_filename = os.path.basename(doc_path)

        try:
            text = extract_text(doc_path)
            pages = text.split("\f")
            page_numbers = []

            for page_num, content in enumerate(pages):
                for target_text in reference_pages[doc_path]:
                    target_text = preprocess_string(target_text)
                    content = preprocess_string(content)
                    if target_text in content:
                        page_numbers.append(page_num + 1)
                        docs_have_relevant_info = True
            
            page_numbers = set(page_numbers)

            user_should_refer_pages_in_doc[source_filename] = page_numbers
        except Exception as e:
            handle_local_error("Could not find page numbers from PDF, encountered error: ", e)

    return docs_have_relevant_info, user_should_refer_pages_in_doc




def whoosh_text_in_pdf(reference_pages):

    print("Searching Index")

    try:
        read_return = read_config(['index_dir'])
        index_dir = read_return['index_dir']
    except Exception as e:
        handle_local_error("Missing index_dir in config.json for method whoosh_text_in_pdf. Error: ", e)

    user_should_refer_pages_in_doc = {}
    docs_have_relevant_info = False

    try:
        # Open the index
        ix = open_dir(index_dir)

        # Create a 'searcher' object
        with ix.searcher() as searcher:
            query_parser = QueryParser("content", ix.schema)

            for doc in reference_pages:
                
                source_filename = os.path.basename(doc)
                page_numbers = []
                
                for search_string in reference_pages[doc]:

                    # Only search for non-empty search strings
                    if search_string:

                        query = query_parser.parse(search_string)

                        results = searcher.search(query)

                        for hit in results:
                            print(f"Found in {hit['title']} on page {hit['pagenumber']}")
                            page_numbers.append(int(hit['pagenumber']))
                            docs_have_relevant_info = True

                page_numbers = set(page_numbers)
                user_should_refer_pages_in_doc[source_filename] = page_numbers

    except Exception as e:
        handle_error_no_return("Could not search Whoosh Index, encountered error: ", e)

    return docs_have_relevant_info, user_should_refer_pages_in_doc



# Route to handle the submission of the second form (file loading)
@app.route('/process_file', methods=['POST'])
def process_file():

    use_ocr = False
    try:
        read_return = read_config(['use_ocr', 'ocr_service_choice'])
        use_ocr = read_return['use_ocr']
        ocr_service_choice = read_return['ocr_service_choice']
    except Exception as e:
        handle_api_error("Could not determine use_ocr in config.json for process_new_file. Disabling OCR and proceeding. Error: ", e)

    try:
        load_new_file = request.form.get('load_new_file', 'n').lower()
    except Exception as e:
        handle_api_error("Server-side error - could not interpret user selection. Encountered error: ", e)

    if load_new_file ==  'y':

        try:
            input_file = request.files['input_file']
        except Exception as e:
            handle_api_error("Server-side error recieving file: ", e)

        # Ensure the filename is secure
        filename = secure_filename(input_file.filename)
        if "PDF" in filename:
            filename = filename.replace("PDF", "pdf")

        pdf_file = False
        txt_file = False

        if filename.endswith('.pdf'):
            pdf_file = True
        elif filename.endswith('.txt'):
            txt_file = True
        else:
            return jsonify(success=False, error="Invalid file format, expected a PDF or TXT file"), 400 #HTTP Bad Request


        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print("Loading new file - filename: ", filename)
            print("Loading new file - filepath: ", filepath)

            # Save the uploaded file to the specified path
            input_file.save(filepath)
        except Exception as e:
            handle_api_error("Failed to save document to app folder, encountered error: ", e)
        
        # print("input_file: ", input_file)
        
        if pdf_file:
            print("Processing PDF file")
            
            if use_ocr:
                try:
                    if ocr_service_choice == 'AzureVision':
                        input_file = PDFtoAzureOCRTXT(filepath)
                    elif ocr_service_choice == 'AzureDocAi':
                        input_file = PDFtoAzureDocAiTXT(filepath)
                except Exception as e:
                    handle_error_no_return("Failed to OCR text from PDF. Will now attempt to extract text via PyPDF2. Encountered error: ", e)
                    try:
                        input_file = PDFtoTXT(filepath)
                    except Exception as e:
                        handle_api_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)
            else:
                try:
                    input_file = PDFtoTXT(filepath)
                except Exception as e:
                    handle_api_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)

            try:
                images = extract_images_from_pdf(filepath)
            except Exception as e:
                handle_error_no_return("Failed to extract images from the PDF document, encountered error: ", e)

            try:
                store_images_to_db(images)
            except Exception as e:
                handle_error_no_return("Failed to save images to database, encountered error: ", e)

        if txt_file:
            print("Processing Text file")

            try:
                # Need to set to filepath as input_file just contains the file itself from the POST!
                input_file = TxtCleaner(filepath)
            except Exception as e:
                handle_api_error("Failed to extract text from PDF: ", e)
        
        try:
            LoadNewDocument(input_file)         
        except Exception as e:
            handle_api_error("Failed to extract text from PDF: ", e)


    # Don't get confused about not loading the VectorDB here! You'll notice that we're doing an additional VectorDB load step in the route method below, 'process_new_file()', but not here!
    # This is because this current route is triggered BEFORE the initital model and VectorDB loading occurs, so after this '/load_model_and_vectordb' triggers and loads the DB anyway!
    # However, when '/process_new_file' is invoked mid-chat, the VectorDB must be RE-LOADED! Hence the extra step in the route below. 
    
    #return "File processed (or not) and ready for chat!"
    #return redirect(url_for('load_model_and_vectordb'))
    return jsonify(success=True)


@app.route('/load_model_and_vectordb')
def load_model_and_vectordb():
    
    global LLM
    global VECTOR_STORE
    global LOADED_UP
    global LLM_CHANGE_RELOAD_TRIGGER_SET
    global HISTORY_SUMMARY
    global HISTORY_MEMORY_WITH_BUFFER
    global HF_BGE_EMBEDDINGS
    global AZURE_OPENAI_EMBEDDINGS

    try:
        read_return = read_config(['model_choice', 'use_gpu_for_embeddings', 'use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'vectordb_sbert_folder', 'vectordb_openai_folder', 'vectordb_bge_base_folder', 'vectordb_bge_large_folder', 'use_azure_open_ai'])
        model_choice = read_return['model_choice']
        use_gpu_for_embeddings = read_return['use_gpu_for_embeddings']
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        vectordb_sbert_folder = read_return['vectordb_sbert_folder']
        vectordb_openai_folder = read_return['vectordb_openai_folder']
        vectordb_bge_base_folder = read_return['vectordb_bge_base_folder']
        vectordb_bge_large_folder = read_return['vectordb_bge_large_folder']
        use_azure_open_ai = read_return['use_azure_open_ai']
    except Exception as e:
        handle_api_error("Missing values in config.json when attempting to load_model_and_vectordb. Error: ", e)


    # global CONVERSATION_RAG_CHAIN_WITH_SUMMARY_BUFFER

    if LOADED_UP and not LLM_CHANGE_RELOAD_TRIGGER_SET:
        print(f'\n\nAlready loaded! Clearing chat history and returning model choice: {model_choice}\n\n')
        HISTORY_MEMORY_WITH_BUFFER.chat_memory.clear()
        HISTORY_MEMORY_WITH_BUFFER = ConversationSummaryBufferMemory(llm=LLM, max_token_limit=300, return_messages=False)
        HISTORY_SUMMARY = {}
        return jsonify({'success': True, 'llm_model': model_choice})
    elif LLM_CHANGE_RELOAD_TRIGGER_SET:
        print('\n\nForce restarting app! Preserving chat history and proceeding to reload the VectorDB & LLM. Resetting reset flag too.\n\n')
        LLM_CHANGE_RELOAD_TRIGGER_SET = False
        

    ### 1 - Load VectorDB from disk
    print("\n\nLoading VectorDB: ChromaDB\n\n")
    try:
        if use_sbert_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_sbert_folder, embedding_function=HuggingFaceEmbeddings())
            # try:
            #     # chroma_client = VECTOR_STORE.PersistentClient
            #     # max_batch_size = chroma_client._producer.max_batch_size
            #     max_batch_size = VECTOR_STORE.max_batch_size
            #     print(f"max_batch_size: {max_batch_size}")
            # except Exception as e:
            #     print(f"Could not get max_batch_size. Error: {e}")
        
        elif use_openai_embeddings:

            try:
                read_return = read_config(['azure_openai_base_url', 'azure_openai_api_key', 'azure_openai_api_type', 'c'])
                azure_openai_base_url = read_return['azure_openai_base_url']
                azure_openai_api_key = read_return['azure_openai_api_key']
                azure_openai_api_type = read_return['azure_openai_api_type']
                azure_openai_api_version = read_return['azure_openai_api_version']
            except Exception as e:
                handle_error_no_return("Missing values for Azure OpenAI Embeddings in method load_model_and_vectordb in config.json. Error: ", e)
            
            try:
                os.environ["OPENAI_API_BASE"] = azure_openai_base_url
                os.environ["OPENAI_API_KEY"] = azure_openai_api_key
                os.environ["OPENAI_API_TYPE"] = azure_openai_api_type
                os.environ["OPENAI_API_VERSION"] = azure_openai_api_version
            except Exception as e:
                handle_error_no_return("Could not set OS environment variables for Azure OpenAI Embeddings in load_model_and_vectordb, encountered error: ", e)
            
            AZURE_OPENAI_EMBEDDINGS = OpenAIEmbeddings(deployment="openai-ada-embedding")
            VECTOR_STORE = Chroma(persist_directory=vectordb_openai_folder, embedding_function=AZURE_OPENAI_EMBEDDINGS)
        
        elif use_bge_base_embeddings:
            if HF_BGE_EMBEDDINGS is not None:
                VECTOR_STORE = Chroma(persist_directory=vectordb_bge_base_folder, embedding_function=HF_BGE_EMBEDDINGS)
            else:
                model_name = "BAAI/bge-base-en"
                model_kwargs = {}
                if use_gpu_for_embeddings:
                    model_kwargs.update({"device": "cuda"})
                else:
                    model_kwargs.update({"device": "cpu"})
                encode_kwargs = {"normalize_embeddings": True}
                HF_BGE_EMBEDDINGS = HuggingFaceBgeEmbeddings(
                    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
                )
                VECTOR_STORE = Chroma(persist_directory=vectordb_bge_base_folder, embedding_function=HF_BGE_EMBEDDINGS)
        
        elif use_bge_large_embeddings:
            if HF_BGE_EMBEDDINGS is not None:
                VECTOR_STORE = Chroma(persist_directory=vectordb_bge_large_folder, embedding_function=HF_BGE_EMBEDDINGS)
            else:
                model_name = "BAAI/bge-large-en"
                model_kwargs = {}
                if use_gpu_for_embeddings:
                    model_kwargs.update({"device": "cuda"})
                else:
                    model_kwargs.update({"device": "cpu"})
                encode_kwargs = {"normalize_embeddings": True}
                HF_BGE_EMBEDDINGS = HuggingFaceBgeEmbeddings(
                    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
                )
                VECTOR_STORE = Chroma(persist_directory=vectordb_bge_large_folder, embedding_function=HF_BGE_EMBEDDINGS)
        
        #VECTOR_STORE = Chroma(persist_directory=VECTORDB_SBERT_FOLDER, embedding_function=HuggingFaceEmbeddings())
    except Exception as e:
        handle_api_error("Could not load VectorDB, encountered error: ", e)


    ### 2 - Load LLM Model from config.json ###
    print("\n\nLoading LLM from config.json\n\n")
    try:

        if not use_azure_open_ai:

            try:
                read_return = read_config(['use_gpu', 'model_dir', 'local_llm_context_length', 'local_llm_max_new_tokens', 'local_llm_gpu_layers', 'local_llm_model_type', 'local_llm_temperature'])
                use_gpu = read_return['use_gpu']
                local_llm_context_length = read_return['local_llm_context_length']
                local_llm_max_new_tokens = read_return['local_llm_max_new_tokens']
                local_llm_gpu_layers = read_return['local_llm_gpu_layers']
                local_llm_model_type = read_return['local_llm_model_type']
                local_llm_temperature = read_return['local_llm_temperature']
                model_dir = read_return['model_dir']
            except Exception as e:
                handle_api_error("Missing values in config.json for setting-up local-LLM in method load_model_and_vectordb. Error: ", e)

            llm_model = model_dir + '/' + model_choice

            config = {'context_length': local_llm_context_length, 'max_new_tokens': local_llm_max_new_tokens, 'temperature': local_llm_temperature}
            
            if use_gpu:
                config.update({'gpu_layers':local_llm_gpu_layers})
            
            LLM = CTransformers(model=llm_model, model_type=local_llm_model_type, config=config, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

        else:

            try:
                read_return = read_config(['azure_openai_base_url', 'azure_openai_api_key', 'azure_openai_api_type', 'azure_openai_deployment_name', 'azure_openai_api_version', 'azure_openai_max_tokens', 'azure_openai_temperature'])
                azure_openai_base_url = read_return['azure_openai_base_url']
                azure_openai_api_key = read_return['azure_openai_api_key']
                azure_openai_api_type = read_return['azure_openai_api_type']
                azure_openai_deployment_name = read_return['azure_openai_deployment_name']
                azure_openai_api_version = read_return['azure_openai_api_version']
                azure_openai_max_tokens = read_return['azure_openai_max_tokens']
                azure_openai_temperature = read_return['azure_openai_temperature']
            except Exception as e:
                handle_api_error("Missing values in config.json for setting-up Azure-OpenAI-LLM in method load_model_and_vectordb. Error: ", e)
            
            LLM = AzureChatOpenAI(
                openai_api_base=azure_openai_base_url,
                openai_api_version=azure_openai_api_version,
                deployment_name=azure_openai_deployment_name,
                openai_api_key=azure_openai_api_key,
                openai_api_type=azure_openai_api_type,
                max_tokens=azure_openai_max_tokens, 
                temperature=azure_openai_temperature,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )

    except Exception as e:
        handle_api_error("Could not load LLM, encountered error: ", e)

    print("\n\n")


    ### 3 - Define History memory w/ buffer:
    try:
        HISTORY_MEMORY_WITH_BUFFER = ConversationSummaryBufferMemory(llm=LLM, max_token_limit=300, return_messages=False)
    except Exception as e:
        handle_api_error("Could not setup memory buffer for LLM, encountered error: ", e)
    
    LOADED_UP = True
    print(f'\n\nDone loading! Returning model choice: {model_choice}\n\n')
    return jsonify({'success': True, 'llm_model': model_choice})


 # Do not delete as vectorDB folder remains on disk
    # Once new VectorDB is created, proceed to update records DB:
    # try:
    #     read_return = read_config(['sqlite_docs_loaded_db'])
    #     sqlite_docs_loaded_db = read_return['sqlite_docs_loaded_db']
    # except Exception as e:
    #     handle_api_error("Missing sqlite_docs_loaded_db in config.json in method reset_vector_db_on_disk. Error: ", e)
    
    # try:
    #     conn = sqlite3.connect(sqlite_docs_loaded_db)
    #     c = conn.cursor()
    # except Exception as e:
    #     handle_api_error("Could not connect to sqlite_docs_loaded_db database to delete file list, encountered error: ", e)

    # try:
    #     c.execute("DELETE FROM document_records where embedding_model = ?", (selected_embedding_model_choice,))
    #     conn.commit()
    #     print(f"Deleted all records where embedding_model = {selected_embedding_model_choice}")
    # except Exception as e:
    #     handle_api_error("Could not delete document list from document_records db, encountered error: ", e)


    

@app.route('/setup_for_streaming_response', methods=['POST'])
def setup_for_streaming_response():

    print("\n\nSetting up to stream response\n\n")

    global QUERIES
    do_rag = True   # We will only return an internal server error in the events that do_rag cannot be written, the user_query cannot be read or if a unique stream_session_id cannot be established

    stream_session_id = ""
    # Generate a unique session ID using universally Unique Identifier via the uuid4() method, wherein the randomness of the result is dependent on the randomness of the underlying operating system's random number generator
    # UUI is a standard used for creating unique strings that have a very high likelihood of being unique across all time and space, for ex: f47ac10b-58cc-4372-a567-0e02b2c3d479
    try:
        stream_session_id = str(uuid.uuid4())
    except Exception as e:
        handle_api_error("Error creating unique stream_session_id when attempting to setup_for_streaming_response. Error: ", e)


    try:
        read_return = read_config(['use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'force_enable_rag', 'force_disable_rag'])
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        force_enable_rag = read_return['force_enable_rag']
        force_disable_rag = read_return['force_disable_rag']
    except Exception as e:
        handle_api_error("Missing values in config.json when attempting to setup_for_streaming_response. Error: ", e)


    # We do not modify the force_enable_rag or force_disable_rag flags in this method, we simply respond to them here. UI updates should handle those flags.
    if force_enable_rag:
        
        print("\n\nFORCE_ENABLE_RAG True, force enabling RAG and returning\n\n")
        
        do_rag = True
        
        try:
            write_config({'do_rag':do_rag})
        except Exception as e:
            handle_api_error("Could not force_enable_rag when attempting to setup_for_streaming_response, encountered error: ", e)
        
        return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag})
    
    if force_disable_rag:

        print("\n\nFORCE_DISABLE_RAG True, force disabling RAG and returning\n\n")

        do_rag = False

        try:
            write_config({'do_rag':do_rag})
        except Exception as e:
            handle_api_error("Could not force_disable_rag when attempting to setup_for_streaming_response, encountered error: ", e)

        return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag})

    try:
        # Attempt to get the user's query
        user_query = request.json['message']
        # Store the query associated with the ID
        QUERIES[stream_session_id] = user_query
    except KeyError:
        handle_api_error("Could not obtain and/or store user_query in setup_for_streaming_response, encountered error: ", e)


    # Perform similarity search on the vector DB
    print("\n\nPerforming similarity search to determine if RAG necessary\n\n")
    embedding_function = None
    try:
        if use_sbert_embeddings:
            embedding_function=HuggingFaceEmbeddings()
        elif use_openai_embeddings:
            embedding_function=AZURE_OPENAI_EMBEDDINGS
        elif use_bge_base_embeddings:
            embedding_function=HF_BGE_EMBEDDINGS
        elif use_bge_large_embeddings:
            embedding_function=HF_BGE_EMBEDDINGS
    except Exception as e:
        handle_error_no_return("Could not set embedding_function for similarity_search when attempting to setup_for_streaming_response, encountered error: ", e)
    
    try:
        docs = VECTOR_STORE.similarity_search(user_query, embedding_fn=embedding_function)
        # docs_with_relevance_score = VECTOR_STORE.similarity_search_with_relevance_scores(user_query, 10, embedding_fn=embedding_function)
        # docs_list_with_cosine_distance = VECTOR_STORE.similarity_search_with_score(user_query, 10, embedding_fn=embedding_function)
        # print(f'\n\nsimple similarity search results: \n {docs}\n\n')
        # print(f'\n\nRelevance Score similarity search results (range 0 to 1): \n {docs_with_relevance_score}\n\n')
        # print(f'\n\nDocs list most similar to query based on cosine distance: \n {docs_list_with_cosine_distance}\n\n')
    except Exception as e:
        handle_error_no_return("Could not perform similarity_search to determin do_rag when attempting to setup_for_streaming_response, encountered error: ", e)


    print("\n\nDetermining do_rag \n\n")
    try:
        page_contents, do_rag = filter_relevant_documents(user_query, docs)
    except Exception as e:
        handle_error_no_return("Force enabling RAG and returning: could not determine do_rag during setup_for_streaming_response, encountered error: ", e)
    
    print(f'Do RAG? {do_rag}')

    try:
        write_config({'do_rag':do_rag})
    except Exception as e:
        handle_api_error("Could not write do_rag during setup_for_streaming_response, encountered error: ", e)


    # Return the stream_session_id
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag})
    
    
    # if matched_images_found:
    #     images_iframe_html = "<br><h6>Refer to the images below:</h6>"
    #     for image_id, image_bytes_data in matched_images_in_bytes:
    #         #print(f"\n\nmatched image id: {image_id}")
    #         try:
    #             image_link_url = url_for('image_display', image_id=image_id)
    #             images_iframe_html += f'<br><iframe width="750" height="400" src="{image_link_url}" frameborder="0"></iframe><br>'
    #         except Exception as e:
    #             handle_error_no_return("Could not construct images_iframe_html, encountered error: ", e)


#############################################################################
##############---NOTES ON THE BELOW CUSTOM CLASS APPROACH---#################
#############################################################################

# class CustomStream(io.StringIO)
#   defines a new class 'CustomStream' that inherits the StringIO class from the io module.
#   'StringIO' is an in-memory, file-like object that can be used as a string buffer, essentially a file in-memory rather than on disk

# def __init__(self, callback=None)
#   initialization method for instances of 'CustomStream' accepting one optional argument that defaults to None if not provided

# super().__init__()
#   calls the 'init' method of the parent class 'StringIO', which here is also the super & base class!
#   necessary to ensure that parent/base/super class 'StringIO' is properly initialized for instances of 'CustomStream'
# 
# self.callback = callback
#   The passed 'callback' attribute is stored as an instance attrib, meaning each instance of 'CustomStream' will have its own 'callback' attrib
 
# def write(self, data)
#   Overwrites the 'write' method of parent class 'StringIO'; this method is called whenever data is written to our 'CustomStream'
 
# PRIMARY MOTIVATION FOR THIS CUSTOM CLASS!! If we have a callback, call it:
# if self.callback:
#   self.callback(data)
#
#   The method checks if a 'callback' function has been set for the instance, i.e. 'self.callback' is not 'None'
#   If there is a 'callback', it calls that function with the provided data, 
#   which allows us to "hook" into the write process & execute additional logic whenever data is written to the 'CustomStream'   

# return super().write(data)
#   Finally, this calls the 'write' method of the parent class 'StringIO' using the 'super()' function
#   This ensures that the actual writing of the data to the in-memory buffer, which is the primary function of 'StringIO' still happens!
#   The provided data is passed for this to the base method
 
# In summary, this 'CustomStream' class provides a custom implementation of 'StringIO' that supports a callback mechanism:
# Everytime data is written to this custom stream, the 'callback', if provided, is executed thus allowing for additional functionality during the write

# In this application, this mechanism is used to queue data for the streaming response! 
# It extends the 'StringIO' class by adding a new feature: the ability to trigger a 'callback' function whenever a 'write()' occurs!

# So to use this:

# 1. We define a queue: 
#       data_queue = queue.Queue()

# 2. We define a callback function that puts data in this queue:
#        def callback(data):
#           data_queue.put(data)

# 3. We create an instance of our custom stream passing this callback function:
#        custom_stream = CustomStream(callback=callback)

# 4. We redirect stdout to our custom stream temporarily
#       original_stdout = sys.stdout
#        sys.stdout = custom_stream

# 5. We start the llm_task() thread & the LLM() function now outputs here, finally resetting stdout and putting None into the queue: data_queue.put(None)

# 6. While the thread runs, we start a while loop that keeps yielding from the queue and stopping when None is read: yield f"data: {line}\n\n"

# 7. A final yield signals the end of the stream, to be handled at the client-side:  yield "event: END\ndata: null\n\n"

#############################################################################
#################---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---####################
#############################################################################


class CustomStream(io.StringIO):
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback

    def write(self, data):
        # If we have a callback, call it
        if self.callback:
            self.callback(data)

        return super().write(data)


@app.route('/stream/<stream_session_id>')
def stream(stream_session_id):

    print("stream route triggered")

    global QUERIES
    global HISTORY_MEMORY_WITH_BUFFER

    try:
        read_return = read_config(['do_rag', 'base_template'])
        do_rag = read_return['do_rag']
        base_template = read_return['base_template']
    except Exception as e:
        error_message = f"\n\Missing values in config.json in main method stream!. Error: {e}\n\n"
        if logger:
            logger.error(error_message)
            print(error_message)
        else:
            print(error_message)
        return jsonify(success=False, error=error_message), 500 # internal server error

    key_for_llm_result = "LlmResponseforQueryID_" + stream_session_id
    key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id

    user_query = request.args.get('input')
    #print(f"do_rag: {do_rag}")

    print(f'\n\nuser query passed to the LLM: {user_query}\n\n')

    if do_rag:
        ### 0 - If memory has been reset due to an old chat loading up, delete the additional key that added for a non-RAG resuming scenario:
        if 'has_been_reset' in HISTORY_SUMMARY:
            del HISTORY_SUMMARY['has_been_reset']

        ### 1 - Define Template:
        rag_prompt_template_variables = """

        Use the following context to answer the user's question:
        
        Context:{context}
        Question:{question}
        """
        history_summary_for_rag = re.sub(r"\{|\}", "", str(HISTORY_SUMMARY))    #search through the string str(HISTORY_SUMMARY) for all instances of { and } and replace them with an empty string "", effectively removing these characters from the string

        rag_history_prompt = "The conversation so far: " + history_summary_for_rag

        rag_prompt_template = rag_history_prompt + "\n" + base_template + "\n" + rag_prompt_template_variables

        print(f"\n\nrag_prompt_template: {rag_prompt_template}\n\n")

        rag_qa_chain_prompt = PromptTemplate.from_template(rag_prompt_template)

        ### 2 - Setup Chain
        qa_chain = RetrievalQA.from_chain_type(LLM, retriever=VECTOR_STORE.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt":rag_qa_chain_prompt})

    else:
        ### 0 - If memory has been reset due to an old chat loading up, delete the additional key that added for a non-RAG resuming scenario:
        if 'has_been_reset' in HISTORY_SUMMARY:
            del HISTORY_SUMMARY['has_been_reset']

        ### 1 - Define Template keping in mind if memory has been reset due to an an old chat loading up:
        non_rag_prompt_template_variables = "\nCurrent conversation:\n{history}\nHuman: {input}\nAI Assistant:"

        history_summary_for_non_rag = re.sub(r"\{|\}", "", str(HISTORY_SUMMARY))    #search through the string str(HISTORY_SUMMARY) for all instances of { and } and replace them with an empty string "", effectively removing these characters from the string
        
        non_rag_history_prompt = "The conversation so far: " + history_summary_for_non_rag
        
        non_rag_prompt_template = non_rag_history_prompt + "\n" + base_template + "\n" + non_rag_prompt_template_variables

        #print(f"\n\nnon_rag_prompt_template: {non_rag_prompt_template}\n\n")
        non_rag_qa_chain_prompt = PromptTemplate(template=non_rag_prompt_template, input_variables=["history","input"])
        print(f"\n\non_rag_qa_chain_prompt: {non_rag_qa_chain_prompt}\n\n")

        ### 2 - Setup Chain
        conversation_chain_with_summary_buffer = ConversationChain(
            llm = LLM,
            prompt=non_rag_qa_chain_prompt
        )

    if not user_query:
        return "Session not found", 404
    
    stop_thread = threading.Event()
    # Will be set() in llm_task() methods 'finally' block to stop the thread: Once the inferencing is complete in the 'try' block, the 'finally' block adds a 'None' object to the queue 
    # and sets the threading event below. The 'None' causes the yielding while loop to invoke join() on the llm_task() thread for synchronization, as it causes the invoking thread, 
    # in this case main thread, to wait until the ll_task thread object completes execution before resuming. Meanwhile setting the stop_thread causes the llm_task threads while() to complete!

    def generate():

        data_queue = queue.Queue()

        def callback(data):
            data_queue.put(data)

        custom_stream = CustomStream(callback=callback)

        # Redirect stdout to our custom stream temporarily
        original_stdout = sys.stdout
        sys.stdout = custom_stream

        def llm_task():
            global QUERIES
            global HISTORY_SUMMARY
            result = ""
            while not stop_thread.is_set():
                # Call LLM
                try:
                    
                    if do_rag:
                        result = qa_chain({"query": user_query})

                        # Experimental RAG chains here:

                    else:
                        result = conversation_chain_with_summary_buffer.predict(input=user_query)
                        
                        # Experimental chains here:
                finally:
                    # Reset stdout to its original value
                    sys.stdout = original_stdout

                    # experimental outputs here:
                    #print(f"history_buffer_result: {history_buffer_result}")

                    # Save the LLM's formatted response for reference searching 
                    formatted_llm_output = ""
                    
                    if do_rag:
                        print("\n\nStoring RAG-Context history:\n")

                        formatted_llm_output = str(result['result'])
                        QUERIES[key_for_llm_result] = formatted_llm_output
                        QUERIES[key_for_vector_results] = result

                        HISTORY_MEMORY_WITH_BUFFER.save_context({"input":user_query}, {"output":formatted_llm_output})
                    else:
                        HISTORY_MEMORY_WITH_BUFFER.save_context({"input":user_query}, {"output":result})

                    
                    HISTORY_SUMMARY = HISTORY_MEMORY_WITH_BUFFER.load_memory_variables({})
                    print(f"\n\nHISTORY_SUMMARY:{HISTORY_SUMMARY}\n\n")
                    print(f"\n\nHISTORY_MEMORY_WITH_BUFFER.chat_memory.messages: {HISTORY_MEMORY_WITH_BUFFER.chat_memory.messages}\n\n")


                    if not do_rag:
                        formatted_user_query = str(user_query).strip('\n')
                        formatted_llm_output = str(result)
                        formatted_llm_output = formatted_llm_output.strip('\n')
                        
                        # If RAG is done, get_references() method stores history. If not, we store history right here
                        print(f"\n\nStoring chat history with non-RAG LLM output: {formatted_llm_output}\n\n")
                        
                        # Storing to history DB as get_references() will not be invoked in non-RAG chains!
                        store_chat_history_to_db(formatted_user_query, formatted_llm_output, HISTORY_SUMMARY)

                    # Stop thread
                    data_queue.put(None)
                    stop_thread.set()

        # Start the LLM task in a separate thread
        thread = threading.Thread(target=llm_task)
        thread.start()

        i = 0
        # Continuously yield data as it becomes available
        while True:
            line = data_queue.get()
            if line is None:
                print("None read, breaking & stopping thread")
                thread.join()
                break
            if i == 0:
                line = line.strip('\n')
                i += 1
            line = line.replace('\n\n', '</br></br>')
            line = line.replace('\n', '</br>')
            #line = re.sub(r'\s{2,}', lambda match: '&nbsp;' * len(match.group()), line)
            yield f"data: {line}\n\n"

        # This part ensures that after LLM finishes, the stream is closed
        yield "event: END\ndata: null\n\n"

        print("LLM stream done")

    print("\n\nStarting inferencing!\n\n")
    return Response(generate(), content_type='text/event-stream')


@app.route('/lc_get_references', methods=['POST'])
def lc_get_references():

    print("\n\nGetting References\n\n")

    try:
        read_return = read_config(['do_rag', 'upload_folder'])
        do_rag = read_return['do_rag']
        upload_folder = read_return['upload_folder']
    except Exception as e:
        handle_api_error("Missing values in config.json when attempting to get_references. Error: ", e)

    if not do_rag:
        print("\n\nSkipping RAG and returning\n\n")
        return jsonify({'success': True, 'chat_id': CHAT_ID, 'sequence_id': SEQUENCE_ID})

    try:
        stream_session_id = request.json['stream_session_id']
        user_query = request.json['message']
    except Exception as e:
        handle_api_error("Could not read request content in method get_references, encountered error: ", e)
        
    try:
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        key_for_llm_result = "LlmResponseforQueryID_" + stream_session_id

        docs = QUERIES[key_for_vector_results]
        llm_response = QUERIES[key_for_llm_result]
    except Exception as e:
        handle_api_error("Could not obtain relevant data from QUERIES dict, encountered error: ", e)

    # Having obtained the relevant info, clear the QUERIES{} dict so as to not bloat it!
    try:
        del QUERIES[key_for_vector_results]
        del QUERIES[key_for_llm_result]
    except Exception as e:
        handle_error_no_return("Error clearing queries dict in method get_references: ", e)

    reference_response = ""

    all_sources = {}
    reference_pages = {}

    try:
        print(f"\n\ndocs['source_documents']: {docs['source_documents']}\n\n")
        print(f"\n\ndocs['result']: {docs['result']}\n\n")
    except Exception as e:
        handle_api_error("Could not parse vector DB search results during get_references() ops, encountered error: ", e)
    

    relevant_pages = "<br><br>Relevant Pages & Topics:<br><br>"

    for doc in docs['source_documents']:
        try:
            relevant_pages += str(doc.page_content)
            relevant_pages += "<br>In Source Document:<br>"
            relevant_pages += str(doc.metadata)
            relevant_pages += "<br><br>"

            relevant_page_text = str(doc.page_content)

            source_filepath = str(doc.metadata["source"])
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue
    
        relevant_page_text = relevant_page_text.split('\n', 1)[0]
        relevant_page_text = relevant_page_text.strip()
        relevant_page_text = re.sub(r'[\W_]+Page \d+[\W_]+', '', relevant_page_text)

        source_filepath = source_filepath.replace('\\', '/')
        
        try:
            source_filename = os.path.basename(source_filepath)
            _, file_extension = os.path.splitext(source_filepath)
        except Exception as e:
            handle_error_no_return("Could not parse path with OS lib, encountered error: ", e)
            continue

        # The source_filepath will likely always reference a TXT file because of how we're loading the VectorDB!
        # Check if the PDF version of the source doc exists
        if file_extension == '.txt':

            #print("\n\ntxt file\n\n")

            # Construct the path to the potential PDF version
            pdf_version_path = os.path.join(upload_folder, os.path.basename(source_filepath).replace('.txt', '.pdf'))   # not catching an error here as os.path.basename(source_filepath) has already been caught just above!

            # Check if PDF version of the source TXT exists!
            if os.path.exists(pdf_version_path):

                source_filename = source_filename.replace('.txt', '.pdf')
                
                if pdf_version_path in reference_pages:
                    reference_pages[pdf_version_path].extend([relevant_page_text])
                else:
                    reference_pages[pdf_version_path] = [relevant_page_text]

                # Add this file to our sources dictionary if it's not already present
                if source_filename not in all_sources:
                    source_filepath = pdf_version_path
                    all_sources.update({source_filename: source_filepath})

            # Else PDF does not exist, TXT is the source
            else:
                # Check if the TXT is already in the sources dict
                if source_filename not in all_sources:
                    try:
                        source_filepath = os.path.join(upload_folder, source_filename) # reconstructed path using the OS module just to be safe
                        all_sources.update({source_filename: source_filepath})
                    except Exception as e:
                        handle_error_no_return("Could not construct filepath for TXT file, encountered error: ", e)


        # If file is not a TXT file
        else:
            # Check if the TXT is already in the sources dict
            if source_filename not in all_sources:
                try:
                    source_filepath = os.path.join(upload_folder, source_filename) # reconstructed path using the OS module just to be safe
                    all_sources.update({source_filename: source_filepath})
                except Exception as e:
                    handle_error_no_return("Could not construct filepath for non-TXT file, encountered error: ", e)

    # print(f"\n\nreference_pages: {reference_pages}\n\n")

    try:
        docs_have_relevant_info, user_should_refer_pages_in_doc = whoosh_text_in_pdf_and_highlight(reference_pages, stream_session_id)
        # docs_have_relevant_info, user_should_refer_pages_in_doc = whoosh_text_in_pdf(reference_pages)
    except Exception as e:
        handle_error_no_return("Could not search Whoosh Index, encountered error: ", e)

    try:
        matched_images_found, matched_images_in_bytes = find_images_in_db(reference_pages)
    except Exception as e:
        handle_error_no_return("Could not search for images, encountered error: ", e)

    refer_pages_string = ""
    download_link_html = ""
    images_iframe_html = ""

    if docs_have_relevant_info:
        
        # refer_pages_string = "<br><br>Refer to the following pages in the mentioned docs:<br>"
        # for doc in user_should_refer_pages_in_doc:
        #     try:
        #         # Remove duplicates from reference_pages dict
        #         refer_pages_string += "<br>" + str(doc) + ": " + str(user_should_refer_pages_in_doc[doc]).replace("{", "").replace("}", "") + "<br>"
        #     except Exception as e:
        #         error_message = f"\n\nCould not construct refer_pages_string, encountered error: {e}\n\n"
        #         if logger:
        #             logger.error(error_message)
        #             print(error_message)
        #         else:
        #             print(error_message)


        refer_pages_string = "<br><br><h6>Refer to the following pages in the mentioned docs:</h6><br>"
        
        # for doc in user_should_refer_pages_in_doc:
        for index, doc in enumerate(user_should_refer_pages_in_doc, start=1):
            # pdf_iframe_id = str(doc) + "PdfViewer"
            pdf_iframe_id = "stream" + stream_session_id + "PdfViewer" + str(index)
            frame_doc_path = f"/pdf/{doc}"
            # frame_doc_path = upload_folder + f"/{doc}" 
            try:
                refer_pages_string += f"<br><h6>{doc}: "
                for page in user_should_refer_pages_in_doc[doc]:
                    frame_doc_path += "#page=" + str(page) 
                    refer_pages_string += f'<a href="javascript:void(0)" onclick="goToPage(\'{pdf_iframe_id}\', \'{frame_doc_path}\')">Page {page}</a>, '
                    frame_doc_path = f"/pdf/{doc}"
                refer_pages_string = refer_pages_string.strip(', ') + "</h6><br>"
            except Exception as e:
                handle_error_no_return("Could not construct refer_pages_string, encountered error: ", e)

        # download_link_html = "<br><h6>Refer to the source documents below:</h6>"
        pdf_right_pane_id = "stream" + stream_session_id + "PdfPane"
        download_link_html = f'<div class="pdf-viewer" id={pdf_right_pane_id}>'

        for index, source in enumerate(user_should_refer_pages_in_doc, start=1):
            try:
                # print("\n\nlooping sources\n\n")
                download_link_url = url_for('download_file', filename=source)
                pdf_iframe_id = "stream" + stream_session_id + "PdfViewer" + str(index)
                download_link_html += f'<br><h6><a href="{download_link_url}" target="_blank"><iframe id="{pdf_iframe_id}" src="{download_link_url}" width="100%" height="600"></iframe></a></h6><br>'
            except Exception as e:
                handle_error_no_return("Could not construct download_link_html, encountered error: ", e)

        download_link_html += "</div>"
        
        # print(f"\n\nall_sources: {all_sources}\n\n")
        # for source in all_sources:
        #     try:
        #         # print("\n\nlooping sources\n\n")
        #         download_link_url = url_for('download_file', filename=source)
        #         pdf_iframe_id = str(source) + "PdfViewer"
        #         download_link_html += f'<br><a href="{download_link_url}" target="_blank"><iframe id="{pdf_iframe_id}" src="{download_link_url}" width="600" height="400"></iframe></a><br>'
        #     except Exception as e:
        #         error_message = f"\n\nCould not construct download_link_html, encountered error: {e}\n\n"
        #         if logger:
        #             logger.error(error_message)
        #             print(error_message)
        #         else:
        #             print(error_message)
    
    if matched_images_found:
        images_iframe_html = "<br><h6>Refer to the images below:</h6>"
        for image_id, image_bytes_data in matched_images_in_bytes:
            #print(f"\n\nmatched image id: {image_id}")
            try:
                image_link_url = url_for('image_display', image_id=image_id)
                images_iframe_html += f'<br><iframe width="750" height="400" src="{image_link_url}" frameborder="0"></iframe><br>'
            except Exception as e:
                handle_error_no_return("Could not construct images_iframe_html, encountered error: ", e)

    
    # reference_response = refer_pages_string + download_link_html + images_iframe_html
    reference_response = refer_pages_string + images_iframe_html

    try:
        # model_response_for_history_db = str(llm_response) + refer_pages_string
        model_response_for_history_db = str(llm_response)
        model_response_for_history_db += f"\n\n{reference_response}"
        model_response_for_history_db += f"\n\npdf_pane_data={download_link_html}"
        model_response_for_history_db = model_response_for_history_db.strip('\n')

        formatted_user_query = str(user_query).strip('\n')

        user_query_for_history_db = formatted_user_query
    except Exception as e:
        handle_error_no_return("Could not prep data to store_chat_history_to_db in get_references(), encountered error: ", e)

    try:
        store_chat_history_to_db(user_query_for_history_db, model_response_for_history_db, HISTORY_SUMMARY)
    except Exception as e:
        handle_error_no_return("Could not store_chat_history_to_db in get_references(), encountered error: ", e)

    return jsonify({'success': True, 'response': reference_response, 'pdf_frame':download_link_html, 'chat_id': CHAT_ID, 'sequence_id': SEQUENCE_ID})



//Make a GET request to the server to load the LLM & vectorDB
                // fetch('/load_model_and_vectordb')
                // .then(response => {
                //     if (!response.ok) {
                //         return response.json().then(err => { throw new Error(err.error)});
                //     }
                //     return response
                // })
                // .then(response => response.json())
                // .then(data => {
                //     if (data.success) {

                //         llm_model = data.llm_model
                //         LLM_MODEL = String(data.llm_model)
                        
                //         // If LLM & VectorDB loaded successfully, init the chat history DB 
                //         fetch('/init_chat_history_db')
                //             .then(response => {
                //                 if (!response.ok) {
                //                     return response.json().then(err => { throw new Error(err.error)});
                //                 }
                //                 return response
                //             })
                //             .then(response => response.json())
                //             .then(data => {
                //                 if (data.success) {
                //                     // If LLM, VectorDB and chat history DB initialized successfully, continue

                //                     curr_chat_id = data.chat_id

                //                     curr_chat_id = " Chat ".concat(String(curr_chat_id))

                //                     display_chatid_and_model = String(curr_chat_id).concat(": ", String(llm_model))

                //                     document.getElementById('model_header').innerHTML = display_chatid_and_model;
                //                     document.getElementById('model_header').style.display = 'block';

                //                     // Load menu items for the chat history menu
                //                     loadChatHistoryMenu();
                //                     document.getElementById('ModelAndDBLoading').style.display = 'none';
                //                     document.getElementById('ReadyToChat').style.display = 'block';
                                    
                //                     var timeoutDelayInMilliseconds = 1500; //1.5 seconds
                //                     setTimeout(function() {
                //                         document.getElementById('ReadyToChat').style.display = 'none';
                //                     }, timeoutDelayInMilliseconds);
                                    
                //                 } else {
                //                     throw new Error('Error when initializing the chat history DB');
                //                 }
                //             })
                //             .catch(error => {
                //                 let full_error_message = "There was an error in initializing the chat history DB: " + String(error.message);
                //                 console.error(full_error_message);
                //                 alert(full_error_message);
                //             });
                //     } else {
                //         throw new Error('Data error when loading the model or vectorDB');
                //     }
                // })
                // .catch(error => {
                //     let full_error_message = "There was an error in loading the model or vectorDB: " + String(error.message);
                //     console.error(full_error_message);
                //     alert(full_error_message);
                // });
                

                // TEMPLATE: Make a GET request to the server
            //    fetch('/init_chat_history_db')
            //         .then(response => {
            //             if (!response.ok) {
            //                 throw new Error(`Server-side HTTP error! Status: ${response.status}`);
            //             }
            //             return response
            //         })
            //         .then(response => response.json())
            //         .then(data => {
            //             if (data.success) {

                            
            //             } else {
            //                 throw new Error('Data error when fetching history-menu list');
            //             }
            //         })
            //         .catch(error => {
            //             console.error("There was an error in fetching the history-menu list: ", error.message);
            //         });


            function sendMessage() {
    
                document.getElementById('processingQ').style.display = 'block';
    
                let userInput = document.getElementById('user-input').value;
    
                // Append user input to the chat area
                document.getElementById('chat-area').innerHTML += '<div class="user-message">' + userInput + '</div>';
    
                // Make AJAX call to the app.py server to get the models response
                fetch('/get_response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({'message': userInput})
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Server-side error! Status: ${response.status}`);
                    }
                    return response
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        CHAT_ID = data.chat_id;
                        SEQUENCE_ID = data.sequence_id;

                        document.getElementById('processingQ').style.display = 'none';
                        const responseAndRating = `
                        <div class="llm-wrapper">
                            <div class="llm-response">
                                ${data.response}
                            </div>
                            <div class="star-rating" data-rated="False" rating-chat-id=${data.chat_id} rating-sequence-id=${data.sequence_id}>
                                <i class="far fa-star" data-rate="1"></i>
                                <i class="far fa-star" data-rate="2"></i>
                                <i class="far fa-star" data-rate="3"></i>
                                <i class="far fa-star" data-rate="4"></i>
                                <i class="far fa-star" data-rate="5"></i>
                            </div>
                        </div>
                        `;
                        document.getElementById('chat-area').innerHTML += responseAndRating;
                        //document.getElementById('chat-area').innerHTML += '<div class="llm-response">' + data.response + '</div>';
                    } else {
                        throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
                    }
                })
                .catch(error => {
                    errorHandler("fetching response", "/get_response", String(error.message))
                });
    
                // Clear the input field
                document.getElementById('user-input').value = '';
            }





@app.route('/load_chat_history', methods=['POST'])
def load_chat_history():

    global HISTORY_SUMMARY
    global HISTORY_MEMORY_WITH_BUFFER

    print("loading chat history")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json in method load_chat_history. Error: ", e)

    # Clear chat history of current chat, prep for loading historical chat summary:
    # try:
    #     HISTORY_MEMORY_WITH_BUFFER.chat_memory.clear()
    #     HISTORY_MEMORY_WITH_BUFFER = ConversationSummaryBufferMemory(llm=LLM, max_token_limit=300, return_messages=False)
    #     HISTORY_SUMMARY = {}
    # except Exception as e:
    #     handle_error_no_return("Could not clear memory when loading chat history, encountered error: ", e)

    try:
        chat_id_for_history_search = request.form['chat_id']
        chat_id = request.form['chat_id']
    except Exception as e:
        return handle_api_error("Could not retrieve Chat ID from request form, encountered error: ", e)

    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to chat history database, encountered error: ", e)

    sequence_id_for_history_search = 1
    retrieve_history = True
    chat_history = []
    old_chat_model = ""

    while(retrieve_history):

        try:
            c.execute("SELECT user_query FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()
            
            user_message = str(result[0])

            user_message = user_message.strip('\n')
            regex_to_swap_multiple_spaces_with_newline = r' {2,}'
            user_message = re.sub(regex_to_swap_multiple_spaces_with_newline, '<br>', user_message)

            user_message = '<div class="user-message glassmorphism">' + user_message + '</div>'

            chat_history.append(user_message)

            c.execute("SELECT llm_response FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()

            result = str(result[0])
            result_parts = result.split("pdf_pane_data=",1)
            llm_response = '<div class="response-and-viewer-container"><div class="llm-wrapper"> <div class="llm-response">' + result_parts[0]

        except Exception as e:
            return handle_api_error("Could not retrieve chat history, encountered error: ", e)
        
        llm_response = llm_response.strip('\n')
        llm_response = llm_response.replace('\n\n', '<br><br>')
        llm_response = llm_response.replace('\n', '<br>')
        
        try:
            c.execute("SELECT user_rating FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()
        except Exception as e:
            handle_error_no_return("Could not fetch user rating, encountered error: ", e)

        response_rated = False
        user_rating_for_history_chat = None

        if result[0]:
            response_rated = True
            try:
                user_rating_for_history_chat = int(result[0])
                #print(f'rating exists: {user_rating_for_history_chat}')
            except Exception as e:
                handle_error_no_return("Could not retrieve integer value of user rating, encountered error: ", e)


        llm_rating = f'''<div class="star-rating" data-rated={response_rated} rating-chat-id={chat_id_for_history_search} rating-sequence-id={sequence_id_for_history_search}>
        <i class="far fa-star" data-rate="1"></i>
        <i class="far fa-star" data-rate="2"></i>
        <i class="far fa-star" data-rate="3"></i>
        <i class="far fa-star" data-rate="4"></i>
        <i class="far fa-star" data-rate="5"></i>
        </div>
        </div>
        </div>'''


        if user_rating_for_history_chat:
            rating_parts = llm_rating.split("far", user_rating_for_history_chat)
            if len(rating_parts) <= user_rating_for_history_chat:
                llm_rating = "fas".join(rating_parts)
            else:
                llm_rating = "fas".join(rating_parts[:-1]) + "fas" + "far".join(rating_parts[-1:])

        llm_response += llm_rating

        if len(result_parts) > 1:
            llm_response += result_parts[1]
            llm_response += "</div>"
            llm_response = llm_response.strip('\n')
            llm_response = llm_response.replace('\n\n', '<br><br>')
            llm_response = llm_response.replace('\n', '<br>')

        chat_history.append(llm_response)

        # Increment sequence ID for next iteration:
        sequence_id_for_history_search += 1

        # But first, check to see if next sequence exists!
        try:
            c.execute("SELECT EXISTS(SELECT 1 FROM chat_history WHERE chat_id = ? AND sequence_id = ?)", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            exists = c.fetchone()[0]
        except Exception as e:
            return handle_api_error("Could not determine if next sequence exists in chat history DB, encountered error: ", e)
            
        if not exists:
            sequence_id = sequence_id_for_history_search - 1
            retrieve_history = False
            try:
                c.execute("SELECT llm_model FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (chat_id, sequence_id))
                result = c.fetchone()
                old_chat_model = str(result[0])
            except Exception as e:
                handle_error_no_return("Could not determine previously used LLM in chat, encountered error: ", e)
            try:
                c.execute("SELECT history_summary FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (chat_id, sequence_id))
                result = c.fetchone()
                history_summary_dict = str(result[0])
            except Exception as e:
                handle_error_no_return("Could not fetch history summary of last chat, encountered error: ", e)
            c.close()

    # Convert History Summary and add a new key indicating it was recently cleared!
    if history_summary_dict is not None and history_summary_dict != "" and history_summary_dict != 'None':
        print(f"\n\history_summary_dict string from old chat: {history_summary_dict}\n\n")
        try:
            HISTORY_SUMMARY = ast.literal_eval(history_summary_dict)    #cast as dictionary
            HISTORY_SUMMARY["has_been_reset"] = True
        except Exception as e:
            handle_error_no_return("Could not cast history summary string from DB to dict and/or set has_been_reset boolean, encountered error: ", e)

    # Temp prints:
    # print(f"\n\nHISTORY_SUMMARY: {HISTORY_SUMMARY}\n\n")
    # print(f"\n\history_summary_dict: {history_summary_dict}\n\n")
    # print(f"\n\nHISTORY_MEMORY_WITH_BUFFER.summary: {HISTORY_MEMORY_WITH_BUFFER.summary}\n\n")
    # print(f"\n\nHISTORY_MEMORY_WITH_BUFFER.chat_memory.messages: {HISTORY_MEMORY_WITH_BUFFER.chat_memory.messages}\n\n")
    print(f'\n\nChat history loaded for chat with model: {old_chat_model}\n\n')

    return jsonify({'success': True, 'chat_history': chat_history, 'old_chat_model': old_chat_model})



# Use a pseudo-terminal to start the process in its own console
# master_fd, slave_fd = pty.openpty() # The pty() module creates a pseudo terminal, simulating a physical terminal device. pty.openpty() returns a pair of file descriptors master_fd (used to write to the pseudo-terminal) and slave_fd (used to read from the pseudo-terminal).
# command_list = [base_command, 'hf_waitress.py'] + launch_args.split()

# HF_WAITRESS_PROCESS = subprocess.Popen(
#     command_list,
#     stdin=slave_fd,
#     stdout=slave_fd,
#     stderr=subprocess.STDOUT,
#     text=True
# )
# os.close(slave_fd)  # Close the slave_fd as it's not needed in the parent process and so it won't remain open and blocking the pseudo-terminal.



# GDrive Experiments:

#app.py:

# add debuggin prints all over:
# @app.route('/login_to_google_drive')
# def login_to_google_drive():
#     print(f"\n\nlogin_to_google_drive()\n\n")
#     global GDRIVE_CREDS
#     if os.path.exists("gdrive_token.json"):
#         print(f"\n\nGDRIVE_CREDS exists\n\n")
#         GDRIVE_CREDS = Credentials.from_authorized_user_file("gdrive_token.json", GDRIVE_SCOPES)
#         if GDRIVE_CREDS and GDRIVE_CREDS.valid:
#             print(f"\n\nGDRIVE_CREDS is valid\n\n")
#             try:
#                 service = build('drive', 'v3', credentials=GDRIVE_CREDS)
#                 about_result = service.about().get(fields="user").execute()
#                 user_name = about_result.get('user', {}).get('emailAddress', 'Unknown User')
#                 print(f"\n\nreturning user_name: {user_name}\n\n")
#                 return jsonify(success=True, user_name=user_name)
#             except Exception as e:
#                 handle_error_no_return("Could not get user name from Google Drive, encountered error: ", e)
    
#     try:
#         print(f"\n\nTrying to get auth_url\n\n")
#         flow = InstalledAppFlow.from_client_secrets_file(
#             "gdrive_credentials.json", 
#             GDRIVE_SCOPES
#         )
        
#         # Start the local server in a separate thread
#         import threading
#         server_thread = threading.Thread(
#             target=flow.run_local_server,
#             kwargs={
#                 'port': 6003,
#                 'open_browser': False
#             }
#         )
#         server_thread.daemon = True
#         server_thread.start()
        
#         # Get the authorization URL
#         auth_url, _ = flow.authorization_url(
#             access_type='offline',
#             include_granted_scopes='true',
#             redirect_uri='http://localhost:6003'  # Add explicit redirect URI
#         )

#         print(f"\n\nreturning auth_url: {auth_url}\n\n")
#         return jsonify(
#             success=True, 
#             needs_auth=True,
#             auth_url=auth_url
#         )
#     except Exception as e:
#         return jsonify(success=False, error=str(e)), 500


# helper-functions.js:


// async function googleDriveLogin() {
//     try {
//         showLoader();
//         const response = await fetch('/login_to_google_drive');
//         const data = await response.json();
        
//         if (data.success) {
//             if (data.needs_auth) {
//                 // Open the authorization URL in a new window/tab
//                 console.log("data:", data);
//                 const authWindow = window.open(data.auth_url, '_blank');
                
//                 // Poll every 2 seconds to check if auth is complete
//                 const checkAuth = setInterval(async () => {
//                     try {
//                         const checkResponse = await fetch('/login_to_google_drive');
//                         const checkData = await checkResponse.json();
                        
//                         if (checkData.success && checkData.user_name) {
//                             // Auth is complete
//                             clearInterval(checkAuth);
//                             console.log('Logged in as:', checkData.user_name);
//                             // Handle successful login (e.g., update UI)
                            
//                             // Optional: close the auth window if it's still open
//                             if (authWindow && !authWindow.closed) {
//                                 authWindow.close();
//                             }
//                         }
//                     } catch (error) {
//                         clearInterval(checkAuth);
//                         // console.error('Error checking auth status:', error);
//                         errorHandler("checking auth status", "/login_to_google_drive", String(error.message));
//                     }
//                 }, 2000);

//                 // Stop polling after 5 minutes (optional timeout)
//                 setTimeout(() => {
//                     clearInterval(checkAuth);
//                 }, 300000);
                
//             } else if (data.user_name) {
//                 // Already authenticated
//                 console.log('Logged in as:', data.user_name);
//                 // Handle already logged in state
//             }
//         }
//     } catch (error) {
//         // console.error('Error:', error);
//         errorHandler("logging into Google Drive", "/login_to_google_drive", String(error.message));
//     }
// }



def is_citation_relevant(llm_response, source_filename):
    print(f"Checking if citation is relevant: {source_filename} in LLM response")
    llm_response = llm_response.lower()
    source_filename = source_filename.lower()   # Full source filename

    source_filename_no_extension = source_filename.split('.')[0]     # Source filename without extension

    source_filename_no_dashes_or_underscores = source_filename_no_extension.replace('_', ' ').replace('-', ' ') # Source filename without dashes or underscores: very unlikely that the LLM will deliberately output the document name with dashes or underscores but no extension!

    return source_filename in llm_response or source_filename_no_dashes_or_underscores in llm_response or source_filename_no_extension in llm_response



# legacy get_ref code from get_sources_and_pages_for_get_references():

if file_extension == '.txt':    # The source_filepath will likely always reference a TXT file because of how we're loading the VectorDB!

            #print("\n\ntxt file\n\n")

            pdf_version_path = os.path.join(upload_folder, os.path.basename(source_filepath).replace('.txt', '.pdf'))   # not catching an error here as os.path.basename(source_filepath) has already been caught just above! Construct the path to the potential PDF version.

            if os.path.exists(pdf_version_path):

                #print("\n\pdf exists\n\n")

                source_filename = source_filename.replace('.txt', '.pdf')
                
                if pdf_version_path in reference_pages:
                    reference_pages[pdf_version_path].extend([[relevant_page_text,relevant_page_number]])
                else:
                    reference_pages[pdf_version_path] = [[relevant_page_text,relevant_page_number]]

                if source_filename not in all_sources:  # Add this file to our sources dictionary if it's not already present
                    source_filepath = pdf_version_path
                    all_sources.update({source_filename: source_filepath})

            else:
                print("\n\nNo PDF source doc found (TXT Source) in the 'uploaded_pdfs' dir, RAG ACTIVE BUT REFERENCING WILL NOT DISPLAY!\n\n")
                if source_filename not in all_sources: # Do not duplicate if the TXT file is already in the sources dict
                    try:
                        source_filepath = os.path.join(upload_folder, source_filename) # reconstructed path using the OS module just to be safe
                        all_sources.update({source_filename: source_filepath})
                    except Exception as e:
                        handle_error_no_return("Could not construct filepath for TXT file, encountered error: ", e)


        # If by any odd chance the file is not a TXT file
        else:
            if source_filename not in all_sources: # Do not duplicate if the TXT file is already in the sources dict
                try:
                    source_filepath = os.path.join(upload_folder, source_filename) # reconstructed path using the OS module just to be safe
                    all_sources.update({source_filename: source_filepath})
                except Exception as e:
                    handle_error_no_return("Could not construct filepath for non-TXT file, encountered error: ", e)


async function requestFormattedPrompt(regeneration_request=false, stream_session_id=null, chat_id=null, sequence_id=null, user_query=null) {
    initializePromptRequest();

    let current_chat_id;
    let uniqueId;

    if (regeneration_request) { 
        current_chat_id = chat_id;
        sequence_id = sequence_id;
        user_query = user_query;
        uniqueId = getUniqueId();
    } else {
        current_chat_id = getChatId();
        const {userInput, file} = getUserInput();
        const userInputForHtml = formatTabsAndSpaces(userInput);
        uniqueId = updateChatAreaWithUserInput(userInputForHtml);
    }



def read_request_data_for_regenerate_response(request: Request) -> tuple[str, str, str, int]:
    stream_session_id = request.json.get('stream_session_id')
    user_query = request.json.get('user_query')
    chat_id = request.json.get('chat_id')
    sequence_id = request.json.get('sequence_id')
    return stream_session_id, user_query, chat_id, sequence_id

@app.route('/regenerate_response', methods=['POST'])
def regenerate_response():
    print("\n\nRegenerating Response\n\n")

    '''
    1. Read the request:
        - stream_session_id
        - user_query
        - chat_id
        - sequence_id
    2. Read config:
        - local_llm_server
        - local_llm_chat_template_format
    '''

    global QUERIES
    do_rag = True

    # Determine do_rag
    try:
        read_return = read_config_for_setup_for_local_llm_response()
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        force_enable_rag = read_return['force_enable_rag']
        force_disable_rag = read_return['force_disable_rag']
        local_llm_chat_template_format = read_return['local_llm_chat_template_format']
        base_template = read_return['base_template']
        local_llm_server = read_return['local_llm_server']
        fetch_top_k_results_from_vectordb = read_return['fetch_top_k_results_from_vectordb']
        filter_top_k_results_by_reranking = read_return['filter_top_k_results_by_reranking']
        skip_system_prompt = read_return['skip_system_prompt']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to setup_for_streaming_response. Error: ", e)

    try:
        stream_session_id, user_query, chat_id, sequence_id = read_request_data_for_regenerate_response(request)
        QUERIES[stream_session_id] = user_query     # Store the query associated with the ID
    except Exception as e:
        return handle_api_error("Could not obtain and/or store user_query in setup_for_streaming_response, encountered error: ", e)

    # Set defaults for regeneration:
    key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
    file_attached = False
    current_sequence_id = sequence_id

    try:
        formatted_prompt, current_sequence_id = get_formatted_prompt_from_history_db(chat_id, current_sequence_id)
    except Exception as e:
        return handle_api_error("Could not get formatted_prompt from history db in method regenerate_response, encountered error: ", e)

    if local_llm_server == 'hf-waitress':
        flux_diffusers, vision = determine_special_model_type_for_hf_waitress()
        if vision: local_llm_server = 'hfw-vision'

        elif flux_diffusers or file_attached:
            new_sequence_id = current_sequence_id
            reject_rag()
            try:
                response = prepare_special_model_response(
                    local_llm_server='hfw-diffusers' if flux_diffusers else 'hfw-vision',
                    stream_session_id=stream_session_id,
                    user_query=user_query,
                    current_sequence_id=current_sequence_id,
                    new_sequence_id=new_sequence_id,
                    formatted_prompt=formatted_prompt
                )
                print(f"Returning quick-return formatted_prompt: {response['formatted_user_prompt']}")
                return jsonify(response)
            except Exception as e:
                return handle_api_error("Could not prepare special model response in method regenerate_response, encountered error: ", e)
        
            
    # Perform similarity search on the vector DB
    print("\n\nPerforming similarity search to determine if RAG necessary\n\n")
    
    embedding_function = get_embedding_function(use_sbert_embeddings, use_openai_embeddings, use_bge_base_embeddings, use_bge_large_embeddings)
    
    try:
        docs_list_with_cosine_distance = VECTOR_STORE.similarity_search_with_score(user_query, fetch_top_k_results_from_vectordb, embedding_fn=embedding_function)
    except Exception as e:
        handle_error_no_return("Could not perform similarity_search to determine do_rag when attempting to regenerate_response, encountered error: ", e)

    try:
        whoosh_results = search_whoosh_index(user_query)
    except Exception as e:
        handle_error_no_return("Could not perform whoosh search to determine do_rag when attempting to regenerate_response, encountered error: ", e)

    filtered_docs = [doc for doc, score in docs_list_with_cosine_distance]  # the `doc,score` is crucial, as it ensure we select only the Document object, and not a tuple comprising of a Document object and a float score!

    if whoosh_results:  # Combine the whoosh and vector results
        combined_docs = combine_whoosh_and_vector_results(whoosh_results, filtered_docs)
    else:
        combined_docs = filtered_docs

    docs = []
    if combined_docs:
        docs = rerank_results_ml(user_query, combined_docs, top_n=filter_top_k_results_by_reranking)
        do_rag = determine_do_rag(user_query, docs, force_enable_rag, force_disable_rag)
    else:
        print("No documents for citations, setting do_rag to False")
        do_rag = False
    
    print(f'Do RAG? {do_rag}')

    try:
        write_config({'do_rag':do_rag})
    except Exception as e:
        handle_error_no_return("Could not write do_rag to config during regenerate_response, encountered error: ", e)

    if do_rag:  # add similarity search results for RAG if necessary!
        try:
            QUERIES[key_for_vector_results] = docs
            user_query += f"\n\nThe following context might be helpful in answering the user query above. If so, please reference it in your response by name and page number:\n{docs}"
        except Exception as e:
            reject_rag()
            handle_error_no_return("RAG Error: Could not update QUERIES dict and user_query during regenerate_response, proceeding without RAG. Encountered error: ", e)
    
    if local_llm_server == 'llama-cpp':
        formatted_prompt = format_prompt_for_llama_cpp(formatted_prompt, user_query, current_sequence_id, base_template, local_llm_chat_template_format)
    elif local_llm_server == 'hf-waitress':
        formatted_prompt = format_prompt_for_hf_waitress(formatted_prompt, user_query, current_sequence_id, base_template, False, False, skip_system_prompt)
    elif local_llm_server == 'hfw-vision':
        formatted_prompt = format_prompt_for_hf_waitress(formatted_prompt, user_query, current_sequence_id, "", False, True, True)  # No base_template for hfw-vision
    print("Returning formatted_prompt: ", formatted_prompt)

    new_sequence_id = int(current_sequence_id) + 1
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag, "formatted_user_prompt": formatted_prompt, "sequence_id":new_sequence_id, "server_type":local_llm_server})



@app.route('/setup_for_local_llm_response', methods=['POST'])
def setup_for_local_llm_response():

    print("\n\nSetting up for local LLM response\n\n")

    global QUERIES
    do_rag = True

    try:
        stream_session_id, key_for_vector_results = get_session_id_and_vector_key()
    except Exception as e:
        return handle_api_error("Could not get session_id and vector_key when attempting to setup_for_streaming_response, encountered error: ", e)

    # Determine do_rag
    try:
        read_return = read_config_for_setup_for_local_llm_response()
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        force_enable_rag = read_return['force_enable_rag']
        force_disable_rag = read_return['force_disable_rag']
        local_llm_chat_template_format = read_return['local_llm_chat_template_format']
        base_template = read_return['base_template']
        local_llm_server = read_return['local_llm_server']
        fetch_top_k_results_from_vectordb = read_return['fetch_top_k_results_from_vectordb']
        filter_top_k_results_by_reranking = read_return['filter_top_k_results_by_reranking']
        skip_system_prompt = read_return['skip_system_prompt']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to setup_for_streaming_response. Error: ", e)

    try:
        user_query, chat_id, file_attached = read_request_data_for_setup_for_local_llm_response(request)
        QUERIES[stream_session_id] = user_query     # Store the query associated with the ID
    except Exception as e:
        return handle_api_error("Could not obtain and/or store user_query in setup_for_streaming_response, encountered error: ", e)

    try:
        current_sequence_id = determine_sequence_id_for_chat(chat_id)
        print(f"Current Chat ID: {chat_id} & Sequence ID: {current_sequence_id}")
    except Exception as e:
        return handle_api_error("Could not determine current_sequence_id when attempting to setup_for_streaming_response, encountered error: ", e)

    try:
        formatted_prompt, current_sequence_id = get_formatted_prompt_for_setup_for_local_llm_response(chat_id, current_sequence_id)
    except Exception as e:
        return handle_api_error("Could not get formatted_prompt from history db in method setup_for_local_llm_response, encountered error: ", e)
    
    if local_llm_server == 'hf-waitress':
        flux_diffusers, vision = determine_special_model_type_for_hf_waitress()
        if vision: local_llm_server = 'hfw-vision'

        elif flux_diffusers or file_attached:
            new_sequence_id = prepare_for_quick_response(current_sequence_id)
            try:
                response = prepare_special_model_response(
                    local_llm_server='hfw-diffusers' if flux_diffusers else 'hfw-vision',
                    stream_session_id=stream_session_id,
                    user_query=user_query,
                    current_sequence_id=current_sequence_id,
                    new_sequence_id=new_sequence_id,
                    formatted_prompt=formatted_prompt
                )
                print(f"Returning quick-return formatted_prompt: {response['formatted_user_prompt']}")
                return jsonify(response)
            except Exception as e:
                return handle_api_error("Could not prepare special model response in method setup_for_local_llm_response, encountered error: ", e)
        
            
    # Perform similarity search on the vector DB
    print("\n\nPerforming similarity search to determine if RAG necessary\n\n")
    
    embedding_function = get_embedding_function(use_sbert_embeddings, use_openai_embeddings, use_bge_base_embeddings, use_bge_large_embeddings)
    
    try:
        docs_list_with_cosine_distance = VECTOR_STORE.similarity_search_with_score(user_query, fetch_top_k_results_from_vectordb, embedding_fn=embedding_function)
    except Exception as e:
        handle_error_no_return("Could not perform similarity_search to determine do_rag when attempting to setup_for_streaming_response, encountered error: ", e)

    try:
        whoosh_results = search_whoosh_index(user_query)
    except Exception as e:
        handle_error_no_return("Could not perform whoosh search to determine do_rag when attempting to setup_for_streaming_response, encountered error: ", e)

    filtered_docs = [doc for doc, score in docs_list_with_cosine_distance]  # the `doc,score` is crucial, as it ensure we select only the Document object, and not a tuple comprising of a Document object and a float score!

    if whoosh_results:  # Combine the whoosh and vector results
        combined_docs = combine_whoosh_and_vector_results(whoosh_results, filtered_docs)
    else:
        combined_docs = filtered_docs

    docs = []
    if combined_docs:
        docs = rerank_results_ml(user_query, combined_docs, top_n=filter_top_k_results_by_reranking)
        do_rag = determine_do_rag(user_query, docs, force_enable_rag, force_disable_rag)
    else:
        print("No documents for citations, setting do_rag to False")
        do_rag = False
    
    print(f'Do RAG? {do_rag}')

    try:
        write_config({'do_rag':do_rag})
    except Exception as e:
        handle_error_no_return("Could not write do_rag to config during setup_for_streaming_response, encountered error: ", e)

    if do_rag:  # add similarity search results for RAG if necessary!
        try:
            QUERIES[key_for_vector_results] = docs
            user_query += f"\n\nThe following context might be helpful in answering the user query above. If so, please reference it in your response by name and page number:\n{docs}"
        except Exception as e:
            reject_rag()
            handle_error_no_return("RAG Error: Could not update QUERIES dict and user_query during setup_for_streaming_response, proceeding without RAG. Encountered error: ", e)
    
    if local_llm_server == 'llama-cpp':
        formatted_prompt = format_prompt_for_llama_cpp(formatted_prompt, user_query, current_sequence_id, base_template, local_llm_chat_template_format)
    elif local_llm_server == 'hf-waitress':
        formatted_prompt = format_prompt_for_hf_waitress(formatted_prompt, user_query, current_sequence_id, base_template, False, False, skip_system_prompt)
    elif local_llm_server == 'hfw-vision':
        formatted_prompt = format_prompt_for_hf_waitress(formatted_prompt, user_query, current_sequence_id, "", False, True, True)  # No base_template for hfw-vision
    print("Returning formatted_prompt: ", formatted_prompt)

    new_sequence_id = int(current_sequence_id) + 1
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag, "formatted_user_prompt": formatted_prompt, "sequence_id":new_sequence_id, "server_type":local_llm_server})



@app.route('/setup_for_local_llm_response', methods=['POST'])
def setup_for_local_llm_response():
    print("\n\nSetting up for local LLM response\n\n")

    global QUERIES
    do_rag = True

    # Determine do_rag
    try:
        read_return = read_config_for_setup_for_local_llm_response()
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        force_enable_rag = read_return['force_enable_rag']
        force_disable_rag = read_return['force_disable_rag']
        local_llm_chat_template_format = read_return['local_llm_chat_template_format']
        base_template = read_return['base_template']
        local_llm_server = read_return['local_llm_server']
        fetch_top_k_results_from_vectordb = read_return['fetch_top_k_results_from_vectordb']
        filter_top_k_results_by_reranking = read_return['filter_top_k_results_by_reranking']
        skip_system_prompt = read_return['skip_system_prompt']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to setup_for_streaming_response. Error: ", e)
    
    try:
        stream_session_id, user_query, chat_id, sequence_id, file_attached, regeneration_request = read_request_data_for_response_setup(request)
        QUERIES[stream_session_id] = user_query     # Store the query associated with the ID
    except Exception as e:
        return handle_api_error("Could not obtain and/or store user_query in setup_for_streaming_response, encountered error: ", e)

    # Set defaults for regeneration:
    current_sequence_id = None
    if regeneration_request:
        print(f"\nSetting defaults for regeneration for request ID {stream_session_id}\n")
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        file_attached = False
        current_sequence_id = sequence_id
    else:
        try:
            current_sequence_id = determine_sequence_id_for_chat(chat_id)   # determine_sequence_id_for_chat() has error handling so not required here
            stream_session_id, key_for_vector_results = get_session_id_and_vector_key()
        except Exception as e:
            return handle_api_error("Error determining sequence_id and/or getting session_id and vector_key when attempting to setup_for_streaming_response, encountered error: ", e)
        print(f"Current Chat ID: {chat_id} & Sequence ID: {current_sequence_id}")

    try:
        formatted_history_prompt, current_sequence_id = get_formatted_prompt_for_setup_for_local_llm_response(chat_id, current_sequence_id)
    except Exception as e:
        return handle_api_error("Could not get formatted_history_prompt from history db in method setup_for_local_llm_response, encountered error: ", e)
    
    if local_llm_server == 'hf-waitress':
        flux_diffusers, vision = determine_special_model_type_for_hf_waitress()
        if vision: local_llm_server = 'hfw-vision'

        elif flux_diffusers or file_attached:
            new_sequence_id = prepare_for_quick_response(current_sequence_id)
            local_llm_server='hfw-diffusers' if flux_diffusers else 'hfw-vision'
            try:
                response = prepare_special_model_response(
                    local_llm_server=local_llm_server,
                    stream_session_id=stream_session_id,
                    user_query=user_query,
                    current_sequence_id=current_sequence_id,
                    new_sequence_id=new_sequence_id,
                    formatted_prompt=formatted_history_prompt
                )
                print(f"Returning quick-return formatted_user_prompt: {response['formatted_user_prompt']}")
                return jsonify(response)
            except Exception as e:
                return handle_api_error("Could not prepare special model response in method setup_for_local_llm_response, encountered error: ", e)
    
    if force_disable_rag:
        print(f"\nForce disabling RAG for request ID {stream_session_id}\n")
        reject_rag()
        try:
            formatted_updated_prompt = get_full_prompt_for_server(local_llm_server, formatted_history_prompt, user_query, current_sequence_id, base_template, local_llm_chat_template_format, skip_system_prompt)
        except Exception as e:
            return handle_api_error("Could not get formatted_updated_prompt in method setup_for_streaming_response, encountered error: ", e)
        if not regeneration_request: current_sequence_id = int(current_sequence_id) + 1
        return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": False, "formatted_user_prompt": formatted_updated_prompt, "sequence_id":current_sequence_id, "server_type":local_llm_server})
            
    # Perform similarity search on the vector DB
    print("\n\nPerforming similarity search to determine if RAG necessary\n\n")
    
    embedding_function = get_embedding_function(use_sbert_embeddings, use_openai_embeddings, use_bge_base_embeddings, use_bge_large_embeddings)
    
    try:
        docs_list_with_cosine_distance = VECTOR_STORE.similarity_search_with_score(user_query, fetch_top_k_results_from_vectordb, embedding_fn=embedding_function)
    except Exception as e:
        handle_error_no_return("Could not perform similarity_search to determine do_rag when attempting to setup_for_streaming_response, encountered error: ", e)

    try:
        whoosh_results = search_whoosh_index(user_query)
    except Exception as e:
        handle_error_no_return("Could not perform whoosh search to determine do_rag when attempting to setup_for_streaming_response, encountered error: ", e)

    filtered_docs = [doc for doc, score in docs_list_with_cosine_distance]  # the `doc,score` is crucial, as it ensure we select only the Document object, and not a tuple comprising of a Document object and a float score!

    if whoosh_results:  # Combine the whoosh and vector results
        combined_docs = combine_whoosh_and_vector_results(whoosh_results, filtered_docs)
    else:
        combined_docs = filtered_docs

    docs = []
    if combined_docs:
        docs = rerank_results_ml(user_query, combined_docs, top_n=filter_top_k_results_by_reranking)
        do_rag = determine_do_rag(user_query, docs, force_enable_rag, force_disable_rag)
    else:
        print("No documents for citations, setting do_rag to False")
        do_rag = False
    
    print(f'Do RAG? {do_rag}')

    try:
        write_config({'do_rag':do_rag})
    except Exception as e:
        handle_error_no_return("Could not write do_rag to config during setup_for_streaming_response, encountered error: ", e)

    if do_rag:  # add similarity search results for RAG if necessary!
        try:
            QUERIES[key_for_vector_results] = docs
            user_query += f"\n\nThe following context might be helpful in answering the user query above. If so, please reference it in your response by name and page number:\n{docs}"
        except Exception as e:
            reject_rag()
            handle_error_no_return("RAG Error: Could not update QUERIES dict and user_query during setup_for_streaming_response, proceeding without RAG. Encountered error: ", e)
    
    try:
        formatted_updated_prompt = get_full_prompt_for_server(local_llm_server, formatted_history_prompt, user_query, current_sequence_id, base_template, local_llm_chat_template_format, skip_system_prompt)
    except Exception as e:
        return handle_api_error("Could not get formatted_updated_prompt in method setup_for_streaming_response, encountered error: ", e)

    new_sequence_id = int(current_sequence_id) + 1
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag, "formatted_user_prompt": formatted_updated_prompt, "sequence_id":new_sequence_id, "server_type":local_llm_server})


    function xdrp_populateGoogleDriveTable(gdrive_files) {
    const gdriveTableBody = document.querySelector('#google_drive_files_table tbody');

    gdrive_files.forEach((file, index) => {
        const iconClass = getFileIconClass(file.type);
        const rowHTML = `
            <tr data-gdrive-file-id="${file.id}" data-gdrive-mime-type="${file.mimeType}">
                <td class="checkbox-cell"><input type="checkbox" id="select-${parseInt(index)+1}"></td>
                <td style="text-align:left">${file.name}</td>
                <td style="text-align:left">${file.type}</td>
                <td style="text-align:center"><i class="file-icon fa-solid ${iconClass}"></i></td>
                <td style="text-align:center">v${file.version}</td>
                <td style="text-align:center"><i class="fas fa-cloud-arrow-down"></i></td>
            </tr>
        `;
        gdriveTableBody.insertAdjacentHTML('beforeend', rowHTML);
    });
}


from transformers.utils import default_cache_path
print(f"Default cache directory: {default_cache_path}")

from huggingface_hub import scan_cache_dir
hf_cache_info = scan_cache_dir()
hf_cache_info

# To capture output (won't display in terminal):
result = subprocess.run(command, check=True, capture_output=True, text=True)
print(f"Output: {result.stdout}")
print(f"Errors: {result.stderr}")

# To pipe to terminal AND capture:
result = subprocess.run(command, check=True, text=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

# To suppress output entirely:
result = subprocess.run(command, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

# To redirect stderr to stdout:
result = subprocess.run(command, check=True,
                       stderr=subprocess.STDOUT)



### APPLY PROMPT TEMPLATE TO STREAM:
What's a better way to do this:

Option 1:

```
    try:
        print(f"\n\nApplying Chat Template for messages: {messages}\n\n")
        formatted_text = PIPE.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        handle_error_no_return("Could not apply chat template, encountered error: ", e)
        return False

    try:
        inputs = PIPE.tokenizer(formatted_text, return_tensors="pt")
    except Exception as e:
        handle_error_no_return("Could not tokenize formatted_text, encountered error: ", e)
        return False

    try:
        # Slice the tensor and decode only the output!
        decoded_inputs = PIPE.tokenizer.decode(inputs['input_ids'][0].tolist(), skip_special_tokens=True)    # Setting skip_special_tokens=True to remove: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.
        print(f"\n\ndecoded_inputs: {decoded_inputs}\n\n")
    except Exception as e:
        handle_error_no_return("Could not decode inputs, encountered error: ", e)

    PIPE(decoded_inputs, **generation_args)
```

OR Option 2:

```
    try:
        print(f"\n\nApplying Chat Template for messages: {messages}\n\n")
        inputs = PIPE.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    except Exception as e:
        handle_error_no_return("Could not apply chat template, encountered error: ", e)
        return False

    try:
        # Slice the tensor and decode only the output!
        decoded_inputs = PIPE.tokenizer.decode(inputs['input_ids'][0].tolist(), skip_special_tokens=True)    # Setting skip_special_tokens=True to remove: 1) Start and end special tokens (<s> and </s>) 2) <unk> tokens 3) <pad> tokens 4) [MASK] tokens 5) Input-formatting special tokens <|start_of_text|>, <|im_start|>, <|endoftext|>, etc.
        print(f"\n\ndecoded_inputs: {decoded_inputs}\n\n")
    except Exception as e:
        handle_error_no_return("Could not decode inputs, encountered error: ", e)

    PIPE(decoded_inputs, **generation_args)
```

Because both seem to work!

Option 2 is better for several reasons:
Efficiency: Option 2 performs the tokenization in a single step within apply_chat_template, while Option 1 requires two separate operations (template application and tokenization). This means Option 2 is more computationally efficient.
Cleaner Code: Option 2 is more concise and has fewer potential points of failure since it combines the template application and tokenization into a single operation.
Memory Usage: Option 2 avoids creating an intermediate string representation (formatted_text) that's only used once, making it more memory efficient.
Consistency: Option 2 uses the tokenizer's built-in functionality to handle the entire process in one go, which is more likely to maintain consistency with how the model expects the input to be formatted.


Here's a clean version of Option 2:

try:
    print(f\n\nApplying Chat Template for messages: {messages}\n\n")
   inputs = PIPE.tokenizer.apply_chat_template(
       messages, 
       add_generation_prompt=True, 
       return_dict=True, 
       return_tensors="pt"
   )
   
   # Decode and print for debugging
   decoded_inputs = PIPE.tokenizer.decode(
       inputs['input_ids'][0].tolist(), 
       skip_special_tokens=True
   )
   print(f"\n\ndecoded_inputs: {decoded_inputs}\n\n")
   
   # Process with pipeline
   PIPE(decoded_inputs, **generation_args)
   
except Exception as e:
   handle_error_no_return("Error in chat template processing: ", e)
   return False


function googleDriveLogin() {
    showLoader();
    fetch('/login_to_google_drive')
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error)});
        }
        return response
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            //alert("Google Drive Login Successful!")
            document.getElementById('googleDriveUserName').textContent = "Logged in as: " + data.user_name;
            document.getElementById('googleDriveUserName').style.display = 'block';

            fetch('/fetch_file_list_from_google_drive')
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error)});
                }
                return response.json()
            })
            .then(data => {
                if (data.success) {
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
            .catch(error => {
                errorHandler("fetching file list from Google Drive", "/fetch_file_list_from_google_drive", String(error.message))
            });

        } else {
            throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
        }
    })
    .catch(error => {
        errorHandler("logging into Google Drive", "/login_to_google_drive", String(error.message))
    });
}

@app.route('/login_to_google_drive')
def login_to_google_drive():
    global GDRIVE_CREDS
    if os.path.exists("gdrive_token.json"):
        GDRIVE_CREDS = Credentials.from_authorized_user_file("gdrive_token.json", GDRIVE_SCOPES)
    if not GDRIVE_CREDS or not GDRIVE_CREDS.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            "gdrive_credentials.json", GDRIVE_SCOPES
        )
        GDRIVE_CREDS = flow.run_local_server(port=6003)
        with open("gdrive_token.json", "w") as token:
            token.write(GDRIVE_CREDS.to_json())

    # Get name of the user
    try:
        service = build('drive', 'v3', credentials=GDRIVE_CREDS)
        about_result = service.about().get(fields="user").execute()
        user_name = about_result.get('user', {}).get('emailAddress', 'Unknown User')
        return jsonify(success=True, user_name=user_name)
    except Exception as e:
        handle_error_no_return("Could not get user name from Google Drive, encountered error: ", e)
        return jsonify(success=True)


####################
####################
####LANG-UNCHAINING:
####################
####################

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#From write_config() in app.py:
    if VECTORDB_LOADED_UP:
        vectordb_trigger_keys_for_app_restart = ['embedding_model_choice']

        for key in vectordb_trigger_keys_for_app_restart:
            if key in config_updates and config_updates[key] != config.get(key):
                global VECTORDB_CHANGE_RELOAD_TRIGGER_SET
                VECTORDB_CHANGE_RELOAD_TRIGGER_SET = True
                restart_required = True
                break


def reload_vector_store():
    global VECTOR_STORE
    print("\nRe-Loading VectorDB: ChromaDB")

    vectordb_used = ""

    try:
        read_return = read_config(['use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'vectordb_sbert_folder', 'vectordb_openai_folder', 'vectordb_bge_base_folder', 'vectordb_bge_large_folder', 'embedding_model_choice'])
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        vectordb_sbert_folder = read_return['vectordb_sbert_folder']
        vectordb_openai_folder = read_return['vectordb_openai_folder']
        vectordb_bge_base_folder = read_return['vectordb_bge_base_folder']
        vectordb_bge_large_folder = read_return['vectordb_bge_large_folder']
        embedding_model_choice = read_return['embedding_model_choice']
    except Exception as e:
        handle_local_error("Missing values in config.json when reloading VectorDB, could not fully complete process_new_file. Please try restarting the application. Error: ", e)

    try:
        if use_sbert_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_sbert_folder, embedding_function=HuggingFaceEmbeddings())
            vectordb_used = vectordb_sbert_folder
        elif use_openai_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_openai_folder, embedding_function=AZURE_OPENAI_EMBEDDINGS)
            vectordb_used = vectordb_openai_folder
        elif use_bge_base_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_bge_base_folder, embedding_function=HF_BGE_EMBEDDINGS)
            vectordb_used = vectordb_bge_base_folder
        elif use_bge_large_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_bge_large_folder, embedding_function=HF_BGE_EMBEDDINGS)
            vectordb_used = vectordb_bge_large_folder
    except Exception as e:
        handle_local_error("Could not reload VectorDB when trying to process_new_file. Please try restarting the application. Error: ", e)
    
    return embedding_model_choice, vectordb_used

# From reset_vector_db_on_disk() in app.py:
restart_required = True # Defined here in case try/except block below fails and the if/else fails to define it, thus failing the return!
    try:
        global VECTORDB_CHANGE_RELOAD_TRIGGER_SET
        read_return = read_config(['embedding_model_choice'])
        set_embedding_model_choice = read_return['embedding_model_choice']
        if set_embedding_model_choice != selected_embedding_model_choice:   # If the selected embedding model is different from the one currently set in config.json, then no restart is required
            restart_required = False
            VECTORDB_CHANGE_RELOAD_TRIGGER_SET = False
        else:
            VECTORDB_CHANGE_RELOAD_TRIGGER_SET = True
    except Exception as e:
        handle_error_no_return("Could not compare selected and set embedding models when determining if restart_required in reset_vector_db_on_disk(), encountered error: ", e)



# Route to handle the submission of the first form (LLM & embeddings model and GPU selection)
@app.route('/process_model', methods=['POST'])
def process_model():
    
    global HF_BGE_EMBEDDINGS

    ###---New config.json---###

    config_update_dict = {}

    use_azure_open_ai = 'use_azure' in request.form
    use_openai_embeddings = 'use_openai_embeddings' in request.form
    use_sbert_embeddings = 'use_sbert_embeddings' in request.form
    use_bge_large_embeddings = 'use_bge_large_embeddings' in request.form
    use_bge_base_embeddings = 'use_bge_base_embeddings' in request.form
    use_gpu_for_embeddings = request.form.get('use_gpu_for_embeds', False)    # default no
    model_choice = str(request.form['model_choice'])
    use_gpu = request.form.get('use_gpu', False)

    config_update_dict.update({'use_azure_open_ai':use_azure_open_ai, 'use_openai_embeddings':use_openai_embeddings, 'use_sbert_embeddings':use_sbert_embeddings, 'use_bge_large_embeddings':use_bge_large_embeddings, 'use_bge_base_embeddings':use_bge_base_embeddings, 'use_gpu_for_embeddings':use_gpu_for_embeddings, 'model_choice':model_choice, 'use_gpu':use_gpu})

    try:
        if use_bge_base_embeddings or use_bge_large_embeddings:
            model_name = ""
            if use_bge_base_embeddings:
                model_name = "BAAI/bge-base-en"
            elif use_bge_large_embeddings:
                model_name = "BAAI/bge-large-en"
            model_kwargs = {}
            if use_gpu_for_embeddings:
                model_kwargs.update({"device": "cuda"})
            else:
                model_kwargs.update({"device": "cpu"})
            encode_kwargs = {"normalize_embeddings": True}
            HF_BGE_EMBEDDINGS = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
    except Exception as e:
        return handle_api_error("Could not load BGE embeddings in process_model, encountered error: ", e)
    
    try:
        write_config(config_update_dict)
    except Exception as e:
        handle_local_error("Could not write updates to config.json, encountered error: ", e)

    # Redirect to the next step
    return redirect(url_for('load_file'))


def get_embedding_function(use_sbert_embeddings:bool, use_openai_embeddings:bool, use_bge_base_embeddings:bool, use_bge_large_embeddings:bool) -> HuggingFaceEmbeddings:
    try:
        if use_sbert_embeddings:
            return HuggingFaceEmbeddings()
        elif use_openai_embeddings:
            return AZURE_OPENAI_EMBEDDINGS
        elif use_bge_base_embeddings:
            return HF_BGE_EMBEDDINGS
        elif use_bge_large_embeddings:
            return HF_BGE_EMBEDDINGS
    except Exception as e:
        handle_error_no_return("Could not get embedding function in method get_embedding_function, encountered error: ", e)



@app.route('/load_vectordb')
def load_vectordb():

    global VECTOR_STORE
    global HF_BGE_EMBEDDINGS
    global AZURE_OPENAI_EMBEDDINGS
    global VECTORDB_CHANGE_RELOAD_TRIGGER_SET
    global VECTORDB_LOADED_UP

    if VECTORDB_LOADED_UP and not VECTORDB_CHANGE_RELOAD_TRIGGER_SET:
        print(f'\n\nVectorDB already loaded! Simply returning.\n\n')
        return jsonify({'success': True})
    elif VECTORDB_CHANGE_RELOAD_TRIGGER_SET:
        print('\n\nProceeding to reload VectorDB & resetting the VECTORDB_CHANGE_RELOAD_TRIGGER_SET flag.\n\n')
        VECTORDB_CHANGE_RELOAD_TRIGGER_SET = False

    try:
        read_return = read_config(['use_gpu_for_embeddings', 'use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'vectordb_sbert_folder', 'vectordb_openai_folder', 'vectordb_bge_base_folder', 'vectordb_bge_large_folder'])
        use_gpu_for_embeddings = read_return['use_gpu_for_embeddings']
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        vectordb_sbert_folder = read_return['vectordb_sbert_folder']
        vectordb_openai_folder = read_return['vectordb_openai_folder']
        vectordb_bge_base_folder = read_return['vectordb_bge_base_folder']
        vectordb_bge_large_folder = read_return['vectordb_bge_large_folder']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to load_vectordb. Error: ", e)
    
    
    ### 1 - Load VectorDB from disk
    print("\n\nLoading VectorDB: ChromaDB\n\n")
    try:
        if use_sbert_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_sbert_folder, embedding_function=HuggingFaceEmbeddings())
            # try:
            #     # chroma_client = VECTOR_STORE.PersistentClient
            #     # max_batch_size = chroma_client._producer.max_batch_size
            #     max_batch_size = VECTOR_STORE.max_batch_size
            #     print(f"max_batch_size: {max_batch_size}")
            # except Exception as e:
            #     print(f"Could not get max_batch_size. Error: {e}")
        
        elif use_openai_embeddings:

            try:
                read_return = read_config(['azure_openai_text_ada_api_url', 'azure_openai_text_ada_api_key', 'azure_openai_api_type', 'azure_openai_api_version', 'azure_openai_text_ada_deployment_name'])
                azure_openai_text_ada_api_url = read_return['azure_openai_text_ada_api_url']
                azure_openai_text_ada_api_key = read_return['azure_openai_text_ada_api_key']
                azure_openai_api_type = read_return['azure_openai_api_type']
                azure_openai_api_version = read_return['azure_openai_api_version']
                azure_openai_text_ada_deployment_name = read_return['azure_openai_text_ada_deployment_name']
            except Exception as e:
                return handle_api_error("Missing values for Azure OpenAI Embeddings in method load_model_and_vectordb in config.json. Error: ", e)
            
            try:
                os.environ["OPENAI_API_BASE"] = azure_openai_text_ada_api_url
                os.environ["OPENAI_API_KEY"] = azure_openai_text_ada_api_key
                os.environ["OPENAI_API_TYPE"] = azure_openai_api_type
                os.environ["OPENAI_API_VERSION"] = azure_openai_api_version
            except Exception as e:
                return handle_api_error("Could not set OS environment variables for Azure OpenAI Embeddings in load_model_and_vectordb, encountered error: ", e)

            
            AZURE_OPENAI_EMBEDDINGS = OpenAIEmbeddings(deployment=azure_openai_text_ada_deployment_name)
            VECTOR_STORE = Chroma(persist_directory=vectordb_openai_folder, embedding_function=AZURE_OPENAI_EMBEDDINGS)
        
        elif use_bge_base_embeddings:
            model_name = "BAAI/bge-base-en"
            model_kwargs = {}
            if use_gpu_for_embeddings:
                model_kwargs.update({"device": "cuda"})
            else:
                model_kwargs.update({"device": "cpu"})
            encode_kwargs = {"normalize_embeddings": True}
            HF_BGE_EMBEDDINGS = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
            VECTOR_STORE = Chroma(persist_directory=vectordb_bge_base_folder, embedding_function=HF_BGE_EMBEDDINGS)
                
        
        elif use_bge_large_embeddings:
            model_name = "BAAI/bge-large-en"
            model_kwargs = {}
            if use_gpu_for_embeddings:
                model_kwargs.update({"device": "cuda"})
            else:
                model_kwargs.update({"device": "cpu"})
            encode_kwargs = {"normalize_embeddings": True}
            HF_BGE_EMBEDDINGS = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
            VECTOR_STORE = Chroma(persist_directory=vectordb_bge_large_folder, embedding_function=HF_BGE_EMBEDDINGS)
        
        #VECTOR_STORE = Chroma(persist_directory=VECTORDB_SBERT_FOLDER, embedding_function=HuggingFaceEmbeddings())
    except Exception as e:
        return handle_api_error("Could not load VectorDB, encountered error: ", e)
    
    VECTORDB_LOADED_UP = True
    return jsonify(success=True)


##############################################################
# ALL HTML/JS CODE BELOW IS FOR THE OLD EMBEDDING MODEL STUFF:
##############################################################

<div id="embed_model_choice" style="display: none;">
    <label style="margin-bottom: 3px;">Select an Embedding Model:</label>
    <br>
    <select class="modal-selects" name="specify_embedding_model" id="embedding_model_dropdown" style="width: 100%;">
        <option value="bge_large">BGE-Large</option>
        <option value="bge_base">BGE-Base</option>
        <option value="sbert_mpnet_base_v2">SBERT: all-mpnet-base-v2</option>
        <option value="openai_text_ada">OpenAI: Text-Ada</option>
        <!-- Add more embedding models here!-->
    </select>
</div>

<br>

<form action="" class="api_form" id="azure_openai_text_ada_api_form" style="display: none;">
    <div class="form-group">
        <input type="checkbox" id="update_azure_ada" name="update_azure_ada">
        <label for="update_azure_ada">Update Configuration</label>
    </div>
    <div class="form-group">
        <label for="azure_openai_text_ada_api_url">Azure OpenAI Base URL:</label>
        <input type="text" id="azure_openai_text_ada_api_url" name="azure_openai_text_ada_api_url" disabled>
    </div>
    <div class="form-group">
        <label for="azure_openai_text_ada_api_key">Azure OpenAI Text-Ada Key:</label>
        <input type="text" id="azure_openai_text_ada_api_key" name="azure_openai_text_ada_api_key" disabled>
    </div>
    <div class="form-group">
        <label for="azure_openai_text_ada_deployment_name">Azure OpenAI Text-Ada Deployment Name:</label>
        <input type="text" id="azure_openai_text_ada_deployment_name" name="azure_openai_text_ada_deployment_name" disabled>
    </div>
    <br>
</form>


#DOMLoader:

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

#From initializeEventListenersForEmbeddingModelTab():
// Check init
toggleAzureAdaApiForm();

document.getElementById('update_azure_ada').addEventListener('change', function() {
    document.getElementById('azure_openai_text_ada_api_url').disabled = !this.checked;    
    document.getElementById('azure_openai_text_ada_api_key').disabled = !this.checked;   
    document.getElementById('azure_openai_text_ada_deployment_name').disabled = !this.checked;   
});

# From initializeEmbeddingModelTabComponents(values):
initializeEmbeddingModelDropdown(values.embedding_model_choice);

# save-config.js:

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

##############################################################
##############################################################
##############################################################




@app.route('/search_whoosh_api', methods=['POST'])
def search_whoosh_api():
    try:
        data = request.json
        query = data.get('query')
        results = search_whoosh_index(query)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500







if source_document:
    print(f"Checking if relationship {source} -> {target} relationship of type ({relationship_type}) from {source_document} exists in {selected_knowledge_domain} graph DB")
    check_query = f"""
        MATCH (s:{source} {{name:'%s'}})-[r:{relationship_type}]->(t:{target} {{name:'%s'}})
        WHERE '%s' IN r.source_documents
        RETURN count(r) as count
    """ % (relationship['source'].replace("'", ""), relationship['target'].replace("'", ""), source_document.replace("'", ""))

    result = graph.query(check_query)
    relationship_exists = False
    if hasattr(result, 'result_set') and result.result_set:
        relationship_exists = result.result_set[0][0] > 0

    if relationship_exists:
        print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type}) from {source_document} in {selected_knowledge_domain} graph DB")
        continue


if source_document: # Check if this node & type from this source document already exists
    print(f"Checking if node {name} of type {node_type} from {source_document} exists in {selected_knowledge_domain} graph DB")
    check_query = f"""
        MATCH (n:{node_name} {{name:'%s', type:'%s'}})
        WHERE '%s' IN n.source_documents
        RETURN count(n) as count
    """ % (name.replace("'", ""), node_type.replace("'", ""), source_document.replace("'", ""))

    result = graph.query(check_query)
    node_exists = False
    if hasattr(result, 'result_set') and result.result_set:
        node_exists = result.result_set[0][0] > 0

    if node_exists:
        print(f"Skipping duplicate node {name} of type {node_type} from {source_document} in {selected_knowledge_domain} graph DB")
        continue




def store_chunk_in_graph_db(chunk, source_document=None):
    selected_knowledge_domain = read_config(['selected_knowledge_domain'])['selected_knowledge_domain']
    skip_summary_generation = read_config(['skip_summary_generation'])['skip_summary_generation']
    
    client = get_graph_db_client()
    
    try:
        graph = client.select_graph(selected_knowledge_domain)  # Will create the graph if it doesn't exist
    except Exception as e:
        return handle_local_error("Could not select/create graph for {selected_knowledge_domain} domain in graph DB, encountered error: ", e)

    try:
        entities_and_relationships = extract_entities_and_relationships(chunk)
    except Exception as e:
        return handle_local_error("Could not extract entities and relationships from chunk, encountered error: ", e)

    if entities_and_relationships is None or entities_and_relationships == {}:
        print(f"No entities or relationships found in chunk, skipping storage to {selected_knowledge_domain} graph DB")
        return False

    add_nodes_to_graph(selected_knowledge_domain, entities_and_relationships, graph, chunk, source_document, skip_summary_generation)
    add_relationships_to_graph(selected_knowledge_domain, entities_and_relationships, graph, chunk, source_document, skip_summary_generation)

    print(f"\nSuccessfully processed {len(entities_and_relationships['nodes'])} nodes and {len(entities_and_relationships['relationships'])} relationships for {selected_knowledge_domain} GraphDB\n")
    return True


    def extract_entities_and_relationships(chunk):
    print("\nExtracting Entities and Relationships from Chunk\n")

    try:

        grapher_url, headers, exl2_quantize_graph_model = get_graphing_request_params()

        payload = get_graphing_request_payload(chunk, exl2_quantize_graph_model)

        if exl2_quantize_graph_model:
            return exl2_graphing_request_response_handler(grapher_url, headers, payload)
        else:
            return graphing_request_response_handler(grapher_url, headers, payload)
    
    except Exception as e:
        return handle_local_error("Could not extract entities and relationships from chunk, encountered error: ", e)






def generate_summary_for_node_or_relationship(chunk=None, name=None, node_type=None, summary=None, source=None, target=None, relationship=None, is_node=False, is_relationship=False):
    print("\nGenerating summary...\n")

    local_llm_server = read_config(['local_llm_server'])['local_llm_server']
    local_llm_chat_template_format = read_config(['local_llm_chat_template_format'])['local_llm_chat_template_format']

    if is_node:
        user_query = get_user_query_for_node_summary(name, node_type, summary, chunk)
    else:
        user_query = get_user_query_for_relationship_summary(source, target, relationship, summary, chunk)

    if local_llm_server == 'hf-waitress':
        formatted_prompt = format_prompt_for_hf_waitress(formatted_prompt="", user_query=user_query, current_sequence_id=0, base_template="", skip_system_prompt=True)
    else:
        formatted_prompt = format_prompt_for_llama_cpp(formatted_prompt="", user_query=user_query, current_sequence_id=0, base_template="", local_llm_chat_template_format=local_llm_chat_template_format, skip_system_prompt=True)

    endpoint_url, headers, payload, exl2 = get_request_params_for_local_llm_server(formatted_prompt)

    # print(f"\nProceeding with request to {local_llm_server} at url: {endpoint_url} with payload: {payload} and headers: {headers}\n")

    try:
        if local_llm_server == 'hf-waitress':
            if exl2:
                response = hf_waitress_streaming_request_response_handler(endpoint_url, headers, payload)
            else:
                response = hf_waitress_non_streaming_request_response_handler(endpoint_url, headers, payload)

            return trim_response(response, '"summary":', '}')

        else:   # TODO: response handler for local_llm_server == 'llama-cpp'
            pass
    except Exception as e:
        return handle_local_error("Could not generate summary for node, encountered error: ", e)


def generate_summaries_for_all_nodes(nodes: list, chunk_text: str, print_string: str = ""):
    processed_nodes = {}
    summarized_nodes = []

    for count, node in enumerate(nodes):
        print(f"Generating summary for entity(node) {count+1} of {len(nodes)} {print_string}...")
        try:
            name = str(node['name'])
            node_type = str(node['type'])
            existing_summary = str(node['summary'])
            
            node_key = (name, node_type)
            if node_key in processed_nodes:
                print(f"Skipping duplicate node {name} of type {node_type}")
                continue

            try:
                updated_summary = generate_summary_for_node_or_relationship(chunk=chunk_text, name=name, node_type=node_type, summary=existing_summary, is_node=True)
            except Exception as e:
                updated_summary = ""
                handle_error_no_return(f"Could not generate summary for node {name} of type {node_type}, skipping. Encountered error: ", e)

            # update node in chunk_entities dict:
            summarized_nodes.append({
                'name': name,
                'type': node_type,
                'summary': updated_summary
            })

            processed_nodes[node_key] = True

        except Exception as e:
            handle_error_no_return(f"Could not generate summary for node {name} of type {node_type}, skipping. Encountered error: ", e)
                    
    return summarized_nodes


def generate_summaries_for_all_relationships(relationships: list, chunk_text: str, print_string: str = ""):
    processed_relationships = {}
    summarized_relationships = []

    for count, relationship in enumerate(relationships):
        print(f"Generating summary for relationship {count+1} of {len(relationships)} {print_string}...")
        try:
            source = str(relationship['source'])
            target = str(relationship['target'])
            relationship_type = str(relationship['relationship'])
            existing_summary = str(relationship['summary'])
            
            relationship_key = (source, target, relationship_type)
            if relationship_key in processed_relationships:
                print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type})")
                continue

            try:
                updated_summary = generate_summary_for_node_or_relationship(chunk=chunk_text, source=source, target=target, relationship=relationship_type, summary=existing_summary, is_relationship=True)
            except Exception as e:
                updated_summary = ""
                handle_error_no_return(f"Could not generate summary for relationship {source} -> {target} ({relationship_type}), skipping. Encountered error: ", e)

            summarized_relationships.append({
                'source': source,
                'target': target,
                'relationship': relationship_type,
                'summary': updated_summary
            })

            processed_relationships[relationship_key] = True

        except Exception as e:
            handle_error_no_return(f"Could not generate summary for relationship {source} -> {target} ({relationship_type}), skipping. Encountered error: ", e)

    return summarized_relationships



    # From store_entities_and_relationships_in_graph_db() if skip_summary_generation is False:

        # b. Generate summaries for all nodes and relationships:
        for chunk_number, chunk_data in chunk_entities.items():
            print_string = f" in chunk {chunk_number} of total {len(chunk_entities)} chunks"
            print(f"\nGenerating summaries for all nodes and relationships {print_string}...\n")
            try:
                summarized_nodes = generate_summaries_for_all_nodes(nodes=chunk_data['entities_and_relationships']['nodes'], chunk_text=chunk_data['chunk_text'], print_string=print_string)
                chunk_entities[chunk_number]['entities_and_relationships']['nodes'] = summarized_nodes
            except Exception as e:
                handle_error_no_return(f"Error generating summaries for nodes, skipping chunk {chunk_number}. Encountered error: ", e)

            try:
                summarized_relationships = generate_summaries_for_all_relationships(relationships=chunk_data['entities_and_relationships']['relationships'], chunk_text=chunk_data['chunk_text'], print_string=print_string)
                chunk_entities[chunk_number]['entities_and_relationships']['relationships'] = summarized_relationships
            except Exception as e:
                handle_error_no_return(f"Error generating summaries for relationships, skipping chunk {chunk_number}. Encountered error: ", e)



def trim_response(response, start_substring, end_substring, include_start_substring=False, include_end_substring=False):    ### MOVED TO hf_waitress.py
    try:
        if start_substring in response and end_substring in response:
            start_index = response.rindex(start_substring)  # Sometimes the model re-gurgitates multiple copies of the same dict in it's response
            end_index = response.rindex(end_substring) # rindex() returns the index of the last occurrence of the substring
            
            if not include_start_substring:
                start_index += len(start_substring)
            if include_end_substring: end_index += 1
            
            response = response[start_index:end_index]           
            return response
        else:
            print(f"\nResponse does not contain start_substring: {start_substring} or end_substring: {end_substring}, returning unchanged response: {response}\n")
            return response
    except Exception as e:
        handle_error_no_return("Failed to trim response, encountered error: ", e)
        return response


def exl2_graphing_request_response_handler(grapher_url, headers, payload):
    print(f"\nHF-Waitress Streaming Graphing-Request Response Handler Invoked\n")
    try:
        full_response = hf_waitress_streaming_request_response_handler(grapher_url, headers, payload)
        print(f"\nExl2 Graphing Response (Entities and Relationships): {full_response}\n")

        try:
            return ast.literal_eval(full_response)
        except (ValueError, SyntaxError):
            # Sometimes additional text may be present so we need to strip it:
            full_response = trim_response(full_response, '{"nodes":', '}', include_start_substring=True, include_end_substring=True)
            print(f"\nTrimmed response to dictionary: {full_response}\n")
            try:
                return ast.literal_eval(full_response)
            except (ValueError, SyntaxError):
                raise # Re-raise the original exception if the second attempt also fails
        
    except Exception as e:
        return handle_local_error("Failed /exl2_stream request to extract entities and relationships from chunk, encountered error: ", e)


def hf_waitress_streaming_request_response_handler(endpoint_url, headers, payload):
    print(f"\nHF-Waitress Streaming Request Response Handler Invoked\n")
    try:
        response = requests.post(endpoint_url, headers=headers, data=payload, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes so we can catch them in the except block

        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data:"):
                    event_data = line[6:].strip()
                    try:
                        token = str(json.loads(event_data))
                        full_response += token
                    except json.JSONDecodeError as e:
                        handle_error_no_return(f"Failed to parse event data: {event_data}, encountered error: ", e)
                elif line.startswith("event: END"):
                    break
                else:
                    print(f"\nUnexpected Line Format: {line}\n")

        if not full_response:
            print("\nWarning: No response from exl2_stream / exl2_grapher request\n")
            return None

        print("\nCompleted, returning response\n")
        return full_response
        
    except Exception as e:
        return handle_local_error("Failed request to /exl2_stream or /exl2_grapher APIs, encountered error: ", e)











def get_user_query_for_comprehensive_summary(nodes_and_relationships, chunk):
    return f"""You are a precise knowledge graph analyst. Your task is to create a structured, comprehensive and sharp summary report (under 2000 words) based on the provided text and node/relationship information.

    Input:
    1. A text chunk containing detailed information
    2. A predefined list of nodes and relationships to focus on

    Instructions:
    1. First, analyze each node in the provided nodes list
    2. For each node, extract ONLY factual information from the text chunk
    3. Focus on these specific aspects for each node:
    - Definition/Description
    - Quantitative metrics (if any)
    - Important relationships with other nodes
    - Temporal information (dates, timelines)

    Required Structure:
    1. Primary Entities
    - List main entities from nodes list
    - Core attributes
    - Key relationships between entities

    2. Entity Details
    - For each major entity:
        * Key characteristics
        * Associated metrics
        * Relationships to other entities

    3. Quantitative Information
    - Any numerical data
    - Statistical information
    - Measurable outcomes

    4. Notable Events and Updates
    - Chronological developments
    - Significant changes
    - Important announcements

    Remember:
    - Include ONLY information present in the text
    - Use precise numbers and dates when available
    - Maintain a factual tone
    - Focus on relationships defined in the nodes list
    - Avoid assumptions or inferences not supported by the text

    <text_chunk>
    {chunk}
    </text_chunk>

    <nodes_and_relationships>
    {json.dumps(nodes_and_relationships)}
    </nodes_and_relationships>


    Output format:
    {{
        "summary": "Your structured, comprehensive and sharp summary report (under 2000 words) here"
    }}
    """









    try:
        # First, Get/Generate summaries for each node and relationship if applicable:
        if not skip_summary_generation:
            
            if not skip_check_for_exisiting_summaries:
                # a. Get summaries for all nodes and relationships:
                for chunk_number, chunk_data in chunk_entities.items():
                    print_string = f" in chunk {chunk_number} of total {len(chunk_entities)} chunks"
                    print(f"\nChecking for existing summaries for all nodes and relationships {print_string}...\n")

                    try:
                        nodes_with_existing_summaries = get_summaries_for_all_nodes(nodes=chunk_data['entities_and_relationships']['nodes'], graph=graph, print_string=print_string)
                        chunk_entities[chunk_number]['entities_and_relationships']['nodes'] = nodes_with_existing_summaries
                    except Exception as e:
                        handle_error_no_return(f"Error checking for existing summaries for nodes, skipping chunk {chunk_number}. Encountered error: ", e)

                    try:
                        relationships_with_existing_summaries = get_summaries_for_all_relationships(relationships=chunk_data['entities_and_relationships']['relationships'], graph=graph, print_string=print_string)
                        chunk_entities[chunk_number]['entities_and_relationships']['relationships'] = relationships_with_existing_summaries
                    except Exception as e:
                        handle_error_no_return(f"Error checking for existing summaries for relationships, skipping chunk {chunk_number}. Encountered error: ", e)

            else:
                print(f"\nNewly created or blank graph DB, skipping check for existing summaries for all nodes and relationships in {selected_knowledge_domain} graph DB\n")


skip_check_for_exisiting_summaries = is_graph_blank_or_newly_created(graph)


def process_nodes(nodes: list, chunk_text: str, print_string: str = "", exl2_prompt_template_format: str = "", requested_max_new_tokens: int = 1000, gen_settings = None):
    processed_nodes = {}
    summarized_nodes = []

    for count, node in enumerate(nodes):
        print(f"Generating summary for entity(node) {count+1} of {len(nodes)} {print_string}...")
        try:
            name = str(node['name'])
            node_type = str(node['type'])
            existing_summary = str(node.get('summary', '')) if node.get('summary') else ""   # dict .get() method is safer than `if node['summary']` because it provides a default value if the key doesn't exist and handles NoneType errors gracefully!
            
            node_key = (name, node_type)
            if node_key in processed_nodes:
                print(f"Skipping duplicate node {name} of type {node_type}")
                continue

            node_summary_request_prompt = get_user_query_for_node_summary(name, node_type, existing_summary, chunk_text)
            formatted_prompt = manually_format_prompt_with_prompt_template(
                formatted_prompt="",
                user_query=node_summary_request_prompt,
                current_sequence_id=0,
                base_template="",
                local_llm_chat_template_format=exl2_prompt_template_format,
                skip_system_prompt=True
            )
            full_response = create_and_execute_exl2_job(payload=formatted_prompt, max_new_tokens=requested_max_new_tokens, gen_settings=gen_settings)
            full_response = trim_response(full_response, '"summary":', '}')
            summarized_nodes.append({
                'name': name,
                'type': node_type,
                'summary':full_response
            })

            processed_nodes[node_key] = True

        except Exception as e:
            handle_error_no_return(f"Could not generate summary for node {name} of type {node_type}, skipping. Encountered error: ", e)

    return summarized_nodes


def process_relationships(relationships: list, chunk_text: str, print_string: str = "", exl2_prompt_template_format: str = "", requested_max_new_tokens: int = 1000, gen_settings = None):
    processed_relationships = {}
    summarized_relationships = []

    for count, relationship in enumerate(relationships):
        print(f"Generating summary for relationship {count+1} of {len(relationships)} {print_string}...")

        try:
            source = str(relationship['source'])
            target = str(relationship['target'])
            relationship_type = str(relationship['relationship'])
            existing_summary = str(relationship.get('summary', '')) if relationship.get('summary') else ""   # dict .get() method is safer than `if relationship['summary']` because it provides a default value if the key doesn't exist and handles NoneType errors gracefully!

            relationship_key = (source, target, relationship_type)
            if relationship_key in processed_relationships:
                print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type})")
                continue

            relationship_summary_request_prompt = get_user_query_for_relationship_summary(source, target, relationship_type, existing_summary, chunk_text)
            formatted_prompt = manually_format_prompt_with_prompt_template(
                formatted_prompt="",
                user_query=relationship_summary_request_prompt,
                current_sequence_id=0,
                base_template="",
                local_llm_chat_template_format=exl2_prompt_template_format,
                skip_system_prompt=True
            )
            full_response = create_and_execute_exl2_job(payload=formatted_prompt, max_new_tokens=requested_max_new_tokens, gen_settings=gen_settings)
            full_response = trim_response(full_response, '"summary":', '}')
            summarized_relationships.append({
                'source': source,
                'target': target,
                'relationship': relationship_type,
                'summary':full_response
            })

            processed_relationships[relationship_key] = True

        except Exception as e:
            handle_error_no_return(f"Could not generate summary for relationship {source} -> {target} ({relationship_type}), skipping. Encountered error: ", e)

    return summarized_relationships


def convert_doc_object_to_dictionary_list(docs: list[Document]) -> dict:
    parsed_documents = []
    
    for doc in docs:
        
        try:
            relevant_page_text = str(doc.page_content)
            relevant_page_number = str(doc.metadata.get('page_number'))
            source_filepath_full = str(doc.metadata.get('source'))
            source_filepath = os.path.basename(source_filepath_full)
        
            relevant_page_text = relevant_page_text.replace('\n', ' ')
            parsed_documents.append({
                'content': relevant_page_text.strip().replace("'", ""),
                'source': source_filepath,
                'page_number': relevant_page_number
            })
            
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue

    return parsed_documents


try:
    doc_object_as_dictionary_list = convert_doc_object_to_dictionary_list(docs)
except Exception as e:
    return handle_local_error("Could not get doc object dict, encountered error: ", e)



    if is_graph_rag:

        refer_pages_string = "<br><h6>Additional data may be found in the following documents:</h6>"
        
        for index, doc in enumerate(user_should_refer_pages_in_doc, start=1):
            pdf_iframe_id = f"stream{stream_session_id}PdfViewer{str(index)}"
            tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
            frame_doc_path = f"/pdf/{doc}"
            try:
                stream_id_string_to_remove = f"_{stream_session_id}"
                doc_name_without_stream_id = str(doc).replace(stream_id_string_to_remove, "")
                refer_pages_string += f"<br><h6>{doc_name_without_stream_id}</h6>"
            except Exception as e:
                handle_error_no_return("Could not construct refer_pages_string, encountered error: ", e)



// Call sendMessage() if the user presses the 'Enter' key
const maxRows = 11; // Replace this value with the maximum allowed number of rows you want
const inputTextAreaElement = document.getElementById("user-input");
//var currentRows = inputTextAreaElement.rows;


// event listener for javascript variable:
let _currentRows = inputTextAreaElement.rows;
Object.defineProperty(window, 'currentRows', {
    get: function() { 
        return _currentRows; 
    },
    set: function(value) {
        console.log(`currentRows changed from ${_currentRows} to ${value}`);
        console.log('Stack trace:', new Error().stack);
        _currentRows = value;
    }
});

// Function to automatically adjust textarea height:
// function autoAdjustHeight() {
//     // Store the current scroll position
//     const scrollPos = inputTextAreaElement.scrollTop;
    
//     // Temporarily set height to 'auto' to measure content
//     inputTextAreaElement.style.height = 'auto';
    
//     // Calculate new height within bounds
//     const lineHeight = parseInt(window.getComputedStyle(inputTextAreaElement).lineHeight);
//     const maxHeight = lineHeight * maxRows;
//     const newHeight = Math.min(inputTextAreaElement.scrollHeight, maxHeight);
    
//     // Set the new height directly
//     inputTextAreaElement.style.height = `${newHeight}px`;
    
//     // Restore scroll position
//     window.scrollTo(0, scrollPos);
// }

function autoAdjustHeight() {
    // Store the current scroll position
    const scrollPos = inputTextAreaElement.scrollTop;
    
    // Temporarily store the current height
    const currentHeight = inputTextAreaElement.style.height;
    
    // Set height to 'auto' to get the natural height based on content
    inputTextAreaElement.style.height = 'auto';
    
    // Get the actual content height
    const contentHeight = inputTextAreaElement.scrollHeight;
    const lineHeight = parseInt(window.getComputedStyle(inputTextAreaElement).lineHeight);
    const newRows = Math.min(Math.ceil(contentHeight / lineHeight), maxRows);
    
    // Restore the previous height before setting new rows to prevent visual jumping
    inputTextAreaElement.style.height = currentHeight;
    
    // Set the new rows if they're different
    if (newRows !== currentRows) {
        inputTextAreaElement.rows = newRows;
        currentRows = newRows;
    }
    
    // Restore scroll position
    window.scrollTo(0, scrollPos);
}

// Add input event listener for dynamic resizing:
// const debouncedAdjustHeight = debounce(autoAdjustHeight, 100);  // necessary to declare a const as this ensures only one instance of the debounced function is created, rather than creating a new instance on each call via `inputTextAreaElement.addEventListener('input', debounce(autoAdjustHeight, 70));`
// inputTextAreaElement.addEventListener('input', debouncedAdjustHeight);
//inputTextAreaElement.addEventListener('input', autoAdjustHeight);

// Also handle paste events explicitly:
inputTextAreaElement.addEventListener('paste', function() {
    setTimeout(autoAdjustHeight, 0); // Using timeout to let the paste complete before resizing
});

// Handle focus (clciking back into the textarea):
inputTextAreaElement.addEventListener('focus', function() {
    console.log("Focus event, restoring rows to: ", currentRows);
    if (this.value.trim()) {
        inputTextAreaElement.rows = currentRows;
    }
});

// Handle blur events (when the textarea is no longer focused):
inputTextAreaElement.addEventListener('blur', function() {
    console.log("Blur event, collapsing to 1 row")
    inputTextAreaElement.rows = 1;
});

// Handle SHIFT + ENTER for new lines and ENTER for sending messages:
inputTextAreaElement.addEventListener("keydown", function(event) {
    console.log("Keydown event, rows before: ", inputTextAreaElement.rows);
    const sendButton = document.getElementById('sendButton');

    if (!event.shiftKey && event.key == "Enter" && this.value.trim() !== "") {
        
        if(!sendButton.disabled) {  //Only trigger a send event if the button is not disabled, i.e. another stream is in progress!
            inputTextAreaElement.rows = 1;
            requestFormattedPrompt();
        }
    } 
    else if (event.shiftKey && event.key == "Enter") {
        if (currentRows < maxRows) {
            inputTextAreaElement.rows += 1;
            currentRows += 1;
        }
    } else if (event.keyCode === 8 || event.keyCode === 46) { //8 is Backspace and 46 is Delete
        console.log("backspace or delete pressed")
        newlineCount = inputTextAreaElement.value.split("\n").length;
        if (newlineCount < currentRows) {
            console.log("trimming rows")
            inputTextAreaElement.rows = newlineCount;
            currentRows = newlineCount;
        }
    }
    console.log("Keydown event, rows after: ", inputTextAreaElement.rows);
});

// function adjustTextareaRows() {
//     newlineCount = inputTextAreaElement.value.split("\n").length;
//     if (newlineCount < maxRows) {
//         inputTextAreaElement.rows = newlineCount;
//         currentRows = newlineCount;
//     } else if (newlineCount >= maxRows) {
//         inputTextAreaElement.rows = maxRows;
//         currentRows = newlineCount;
//     }
// }
// document.getElementById("user-input").addEventListener('input', adjustTextareaRows);
// document.getElementById("user-input").addEventListener('change', adjustTextareaRows);



def validate_entity_extraction_response(extraction_response: dict):
    '''
    This function attempts to validate the response from the entity extraction task.
    It first attempts to evaluate the response as a dict, which is the expected format.
    If this fails, it attempts to trim the response and then evaluate again.
    If this also fails, it attempts to trim the response again with simpler conditions and then evaluate again.
    If this also fails, it returns the unchanged response and a flag indicating that the response is invalid.
    '''
    try:
        extraction_response = ast.literal_eval(extraction_response) # Sometimes additional text may be present and need to be stripped, which we can test for by trying to evaluate the response as a dict
        return {'validated_response': extraction_response, 'is_valid': True}
    except Exception as e:
        print(f"Response invalid, attempting to trim. Encountered error: ", e)
        extraction_response = prompt_formatting_module.trim_response(extraction_response, '{"nodes":', '"}]}', include_start_substring=True, include_end_substring=True)
        try:
            extraction_response = ast.literal_eval(extraction_response)
            print("Success! Proceeding...")
            return {'validated_response': extraction_response, 'is_valid': True}
        except Exception as e:
            print(f"Response still invalid, re-attempting with minimal trimming. Encountered error: ", e)
            extraction_response = prompt_formatting_module.trim_response(extraction_response, '{"nodes":', '}', include_start_substring=True, include_end_substring=True)    # last-ditch effort!
            try:
                extraction_response = ast.literal_eval(extraction_response)
                print("Success! Proceeding...")
                return {'validated_response': extraction_response, 'is_valid': True}
            except Exception as e:
                print(f"Response still invalid, returning unchanged response. Encountered error: ", e)
                return {'validated_response': extraction_response, 'is_valid': False}


def parse_ocr_output(kosmos_response_output_json, threshold=20):
    print("\n\nParsing Kosmos-2.5 response output\n\n")
    print(f"\n\nKosmos-2.5 response output: {kosmos_response_output_json}\n\n")

    try:    # to extract the 'results' field from the JSON
        if isinstance(kosmos_response_output_json, str):
            print("\n\nKosmos-2.5 response output is a string\n\n")
            clean_json_str = kosmos_response_output_json.replace("Using flash_attn\n\n", "").replace("\ndone\n", "")
            parsed_data = ast.literal_eval(clean_json_str)
            results = parsed_data.get('results', '')
        else:
            results = kosmos_response_output_json.get('results', '')
    except Exception as e:
        return handle_local_error("Could not extract results, encountered error: ", e)

    # Sort the text elements by vertical position (y0) first - If y-positions are close (within threshold pixels), sort by horizontal x-position (x0):
    def sort_key(item):
        y = item['bounding box']['y0']
        x = item['bounding box']['x0']
        
        # Group items that are roughly on the same line of text together - Grouped items are sorted by their x-coordinate
        line_number = y // threshold    # Floor division to get the line number
        return (line_number, x)

    try:    # to sort the OCR elements by the sort_key
        sorted_elements = sorted(results, key=sort_key) # the sorted() function takes a list and a key function, and returns a new sorted list without modifying the original list
        # print(f"\n\nSorted elements: {sorted_elements}\n\n")
    except Exception as e:
        return handle_local_error("Could not sort OCR elements, encountered error: ", e)

    try:    # to extract just the text in order
        ordered_text = [elem['text'] for elem in sorted_elements]
    except Exception as e:
        return handle_local_error("Could not extract text in order, encountered error: ", e)

    full_text = ""
    for text in ordered_text:
        full_text += text + " "
    
    return sorted_elements, ordered_text, full_text


def parse_ocr_output(kosmos_response_output_json, threshold=20):
    print("\n\nParsing Kosmos-2.5 response output\n\n")
    print(f"\n\nKosmos-2.5 response output: {kosmos_response_output_json}\n\n")

    try:    # to extract the 'results' field from the JSON
        if isinstance(kosmos_response_output_json, str):
            print("\n\nKosmos-2.5 response output is a string\n\n")
            clean_json_str = kosmos_response_output_json.replace("Using flash_attn\n\n", "").replace("\ndone\n", "")
            parsed_data = ast.literal_eval(clean_json_str)
            results = parsed_data.get('results', '')
        else:
            results = kosmos_response_output_json.get('results', '')
    except Exception as e:
        return handle_local_error("Could not extract results, encountered error: ", e)

    def calculate_distances(item1, item2):
        """Calculate vertical and horizontal distances between two bounding boxes."""
        box1 = item1['bounding box']
        box2 = item2['bounding box']

        # Vertical distance between midpoints
        y1_mid = (box1['y0'] + box1['y1']) / 2 # y0 is the top of the bounding box, y1 is the bottom
        y2_mid = (box2['y0'] + box2['y1']) / 2
        vertical_distance = abs(y1_mid - y2_mid) # The absolute difference between the midpoints of the two bounding boxes

        # Horizontal distance (from end of first box to the start of the second)
        horizontal_dist = abs(box1['x1'] - box2['x0'])

        return vertical_distance, horizontal_dist
    
    def sort_elements(elements):

        if not elements:
            return []
        
        sort_elements = []
        remaining = elements.copy()

        # Start with the leftmost, topmost element
        current = min(remaining, key=lambda x: (x['bounding box']['y0'], x['bounding box']['x0']))
        '''
        The min() function can take either two items to compare directly, or a list and a key function that defines how to compare the items.
        We chose the latter approach here via the lambda function, which for each element x in the remaining list,
        creates a tuple of (y0, x0) coordinates from the bounding box of x, i.e. the top-vertical and left-horizontal start positions of the bounding box.
        The min() function then returns the element with the smallest y0 coordinate (topmost element), which explains why the tuple is constructed as (y0, x0):
            - First compare by y0 (vertical position) to ensure the topmost element is selected
            - If there are multiple elements with the same y0, then compare by x0 (horizontal position) to ensure the leftmost element is selected
        
        Edge case: if the very first word is smaller than the rest of the title/heading/text, then it will NOT be picked as the first, as it may not be topmost!
        An example is the word "Form" in the "W2_Elizabeth_Darling.pdf" document: it should be the first, but as it's not the topmost, it will not be picked as such.
        Thing is, titles and headings are often centered at the top so it would be unnatural to pick the first by topmost of the leftmost.
        There are some ways to handle this, but handling every edge case can lead to additional complexity and unpredicatable issues elsewhere.
        As this particular edge case does not affect the overall / core functionality of the OCR output, it is not handled.
        '''
        sort_elements.append(current)
        remaining.remove(current)

        while remaining:
            best_next = None
            min_score = float('inf')    # This initializes the minimum score to infinity, which is the highest possible score

            for candidate in remaining:
                vert_dist, horiz_dist = calculate_distances(current, candidate) # we're trying to find the word that comes after the current word

                # If elements are roughly on the same line (within threshold)
                if vert_dist < threshold:
                    # Prefer elements to the right (lower x0 means to the right in the coordinate system)
                    if candidate['bounding box']['x0'] > current['bounding box']['x0']:
                        score = horiz_dist
                    else:
                        # Heavily penalize elements to the left on the same line by shifting to the right by a large amount
                        score = horiz_dist + 10000
                else:
                    # For elements on different lines, prefer the leftmost element of the next line
                    score = vert_dist + candidate['bounding box']['x0'] / 10    

                if score < min_score:
                    min_score = score
                    best_next = candidate

            if best_next is None:
                # If no good next element found, take the topmost leftmost remaining element
                best_next = min(remaining, key=lambda x: (x['bounding box']['y0'], x['bounding box']['x0']))

            sort_elements.append(best_next)
            current = best_next
            remaining.remove(best_next)

        return sort_elements

    try:
        sorted_elements = sort_elements(results)
    except Exception as e:
        return handle_local_error("Could not sort OCR elements, encountered error: ", e)

    try:    # to extract just the text in order
        ordered_text = [elem['text'] for elem in sorted_elements]
    except Exception as e:
        return handle_local_error("Could not extract text in order, encountered error: ", e)

    full_text = ""
    for text in ordered_text:
        full_text += text + " "
    
    return sorted_elements, ordered_text, full_text



def is_citation_relevant(llm_response: str, source_filename: str) -> bool:
    print(f"\nChecking citation relevance: {source_filename} in LLM response?\n")
    try:
        if not llm_response or not source_filename:
            print("\nLLM response or source filename is empty, returning False\n")
            return False
        
        # Normalize inputs:
        llm_response_norm = llm_response.lower().strip()
        source_filename_norm = source_filename.lower().strip()

        # Variations of the filename:
        source_filename_no_extension, _ = os.path.splitext(source_filename_norm) # os.path.splitext() returns a tuple containing the path's name and extension. It handles edge cases and is platform-independent.
        
        # Clean by replacing common separators with spaces and collapsing multiple spaces into a single space
        # For filename: process both with and without extension initially for regex
        source_filename_cleaned_with_ext = re.sub(r'[-_+]', ' ', source_filename_norm)
        source_filename_cleaned_with_ext = re.sub(r' +', ' ', source_filename_cleaned_with_ext).strip()

        source_filename_cleaned_no_ext = re.sub(r'[-_+]', ' ', source_filename_no_extension)
        source_filename_cleaned_no_ext = re.sub(r' +', ' ', source_filename_cleaned_no_ext).strip()

        # Clean the LLM response similarly:
        llm_response_cleaned = re.sub(r'[-_+]', ' ', llm_response_norm)
        llm_response_cleaned = re.sub(r' +', ' ', llm_response_cleaned).strip()

        # --- Check 1: Regex matching variations of the *full* filename ---
        print("--- Running Check 1: Regex Checks ---")
        # These patterns try to match the filename more precisely, often as whole words.
        """
        re.escape() is used to escape special characters in the source filename, ensuring they are treated as literal characters in the regex pattern.
        \b is a word boundary, ensuring the pattern is a whole word. 
        rf'' is a raw f-string, allowing for the use of \b without it being interpreted as an escape character. This prevents partial matches, eg "doc1" matching on "doc123".
        """
        patterns = [
            rf'\b{re.escape(source_filename_norm)}\b', # Exact filename match with extension, case-insensitive due to prior normalization
            rf'\b{re.escape(source_filename_no_extension)}\b', # Filename without extension
            rf'\b{re.escape(source_filename_cleaned_no_ext)}\b', # Cleaned filename without extension (spaces for separators)
            rf'\b{re.escape(source_filename_cleaned_with_ext)}\b', # Cleaned filename with extension (less common, but possible)
        ]

        # Check original and cleaned LLm response against regex patterns
        # Using original llm_response_norm as well, in case cleaning removed crucial context for regex
        responses_to_check = [llm_response_norm, llm_response_cleaned]

        is_relevant = any(
            re.search(pattern, response) 
            for pattern in patterns
            for response in responses_to_check
        )

        # threshold = 90
        # if not is_relevant: # No exact matches found, LLM may have mentioned the filename just differently enough, so time to check if a Fuzzy match is found
        #     print(f"\nNo exact matches found, checking for fuzzy match with a {threshold}% or higher threshold\n")
        #     is_relevant = is_fuzzy_subset(llm_response_cleaned, source_filename_cleaned_no_ext, threshold)
        #     print(f"Fuzzy match result: {is_relevant} for {source_filename}\n")

        # print(f"\nCitation relevance check result: {is_relevant} for {source_filename}\n")
        # return is_relevant

        if is_relevant:
            print(f"RegEx match for {source_filename} in LLM response. Citation check result: True for {source_filename}")
            return True
                
        # --- Check 2: Simple Substring Checks ---
        if not is_relevant:
            print("--- Running Check 2: Simple Substring Checks ---")

            filename_variants = [
                v for v in [
                    source_filename_norm,
                    source_filename_no_extension,
                    source_filename_cleaned_no_ext,
                    source_filename_cleaned_with_ext
                ]
                if v # Filter out empty strings
            ]

            llm_response_variants = [
                v for v in [
                    llm_response_norm,
                    llm_response_cleaned
                ]
                if v # Filter out empty strings
            ]

            if not filename_variants or not llm_response_variants:
                print(f"No valid filename variants or LLM response variants, skipping substring checks. Citation check result: False for {source_filename}")
                return False
            
            check_A = any(
                fn_var in llm_var
                for fn_var in filename_variants
                for llm_var in llm_response_variants
            )
            
            if check_A:
                print(f"Found filename variant in LLM response. Citation check result: True for {source_filename}")
                return True
            
            check_B = any(
                llm_var in fn_var
                for llm_var in llm_response_variants
                for fn_var in filename_variants
            )

            if check_B:
                print(f"Found LLM response variant in filename variant. Citation check result: True for {source_filename}")
                return True
            
            print(f"No filename variant in LLM response or LLM response variant in filename variant. Citation check result: False for {source_filename}")
            return False
    
    except Exception as e:
        handle_error_no_return("Could not determine if citation is relevant in is_citation_relevant(), encountered error: ", e)
        return False




def highlight_text_on_page(highlight_list, stream_session_id):

    print("\nHighlighting Document\n")
    threshold = 80

    try:
        read_return = read_config(['upload_folder', 'highlighted_docs'])
        upload_folder = read_return['upload_folder']
        highlighted_pdfs_path = read_return['highlighted_docs']
    except Exception as e:
        handle_local_error("Missing upload_folder in config.json for method highlight_text_on_page. Error: ", e)
    
    for index, doc in enumerate(highlight_list, start=1):

        try:
            pdf_path = os.path.join(upload_folder, doc).replace("\\","/")
            output_file_extension = "_" + stream_session_id + '.pdf'
            output_file_name = doc.replace(".pdf",output_file_extension) 
            output_pdf_path = os.path.join(highlighted_pdfs_path, output_file_name).replace("\\","/")
            highlight_doc = fitz.open(pdf_path)
        except Exception as e:
            handle_error_no_return("Could not open doc for highlighting, encountered error: ", e)
            continue
        
        for target in highlight_list[doc]:
            try:
                text_to_highlight = str(target[1])
                text_to_highlight = re.sub(r'Row \d+, Column \d+: ', '', text_to_highlight)
                page_number = int(target[0])
                page = highlight_doc.load_page(page_number-1)
                page_text = page.get_text("text")

                # Split the page text into overlapping phrases
                words = page_text.split()
                phrases = [' '.join(words[i:i+len(text_to_highlight.split())]) for i in range(len(words))]

                # Find fuzzy matches
                good_matches = []
                for phrase in phrases:
                    score = fuzz.partial_ratio(text_to_highlight.lower(), phrase.lower())
                    if score >= threshold:
                        good_matches.append(phrase)

                for match in good_matches:
                    if len(str(match)) > 3:
                        text_instances = page.search_for(match)
                        for inst in text_instances:
                            try:
                                #print(f"HIGHLIGHTING inst {inst} in document {doc}")
                                page.add_highlight_annot(inst)
                            except Exception as e:
                                handle_error_no_return("Could not highlight text instance, encountered error: ", e)
                                continue

            except Exception as e:
                handle_error_no_return("Error loading page or searching for text to highlight, encountered error: ", e)
                continue
            
        try:
            highlight_doc.save(output_pdf_path, garbage=0, deflate=False, clean=False)
        except Exception as e:
            handle_error_no_return("Could not save highlighted doc, encountered error: ", e)
            continue

    return True


def highlighter_interface(reference_pages, stream_session_id):
    '''
    This function takes a dictionary of reference pages, and a stream session ID.
    It returns a tuple containing two elements:
    - A boolean indicating whether any relevant information was found in any of the documents
    - A dictionary containing the pages the user should refer to in the document
    '''
    try:
        user_should_refer_pages_in_doc = {}
        highlight_list = {}
        docs_have_relevant_info = False

        # print(f"\n\nreference_pages: {reference_pages}\n\n")

        for source_pdf_path, content in reference_pages.items():
            source_filename = os.path.basename(source_pdf_path)
            # print(f"\nsource_filename basename: {source_filename}\n")
            output_file_extension = "_" + stream_session_id + '.pdf'
            output_file_name = source_filename.replace(".pdf",output_file_extension) 
            page_numbers = set()
            highlight_strings = set()

            for item in content:
                # Each item in the list has two elements
                page_text, page_number = item
                
                if isinstance(page_number, int):
                    page_numbers.add(int(page_number))
                    highlight_strings.add((int(page_number), str(page_text[:50])))
                
                elif isinstance(page_number, str):
                    page_number_str = page_number.replace('[', '').replace(']', '')
                    str_to_page_number_list = page_number_str.split(',')
                    for page_number in str_to_page_number_list:
                        page_numbers.add(int(page_number.strip()))
                        highlight_strings.add((int(page_number.strip()), str(page_text[:50])))
                
                elif isinstance(page_number, list):
                    for page_number in page_number:
                        page_numbers.add(int(page_number))
                        highlight_strings.add((int(page_number), str(page_text[:50])))

                else:
                    handle_error_no_return("Could not handle page number type, encountered error: ", e)

            if page_numbers:
                user_should_refer_pages_in_doc[output_file_name] = page_numbers
                docs_have_relevant_info = True

            if highlight_strings:
                highlight_list[source_filename] = list(highlight_strings)

        if docs_have_relevant_info:
            try:
                highlight_text_on_page(highlight_list, str(stream_session_id))
            except Exception as e:
                handle_error_no_return("Could not highlight text, encountered error: ", e)

        return docs_have_relevant_info, user_should_refer_pages_in_doc
    except Exception as e:
        handle_error_no_return("Could not highlight text, encountered error: ", e)
        return False, {}



def is_fuzzy_subset(string1: str, string2: str, threshold: int) -> bool:
    score = fuzz.partial_ratio(string1, string2)
    return score >= threshold


def is_citation_relevant(llm_response: str, source_filename: str) -> bool:
    print(f"Checking citation relevance: {source_filename} in LLM response?")
    try:
        if not llm_response or not source_filename:
            print("LLM response or source filename is empty, returning False")
            return False
        
        # Normalize inputs:
        llm_response = llm_response.lower().strip()
        source_filename = source_filename.lower().strip()

        # Variations of the filename:
        source_filename_no_extension, _ = os.path.splitext(source_filename) # os.path.splitext() returns a tuple containing the path's name and extension. It handles edge cases and is platform-independent.
        source_filename_cleaned = re.sub(r'[-_+]', ' ', source_filename_no_extension)
        source_filename_cleaned = re.sub(r' +', ' ', source_filename_cleaned)

        llm_response_cleaned = re.sub(r'[-_+]', ' ', llm_response)
        llm_response_cleaned = re.sub(r' +', ' ', llm_response_cleaned)

        # Regex patterns for matching:
        """
        re.escape() is used to escape special characters in the source filename, ensuring they are treated as literal characters in the regex pattern.
        \b is a word boundary, ensuring the pattern is a whole word. 
        rf'' is a raw f-string, allowing for the use of \b without it being interpreted as an escape character. This prevents partial matches, eg "doc1" matching on "doc123".
        """
        patterns = [
            rf'\b{re.escape(source_filename)}\b', # Exact filename match with extension
            rf'\b{re.escape(source_filename_cleaned)}\b', # Filename with dashes or underscores replaced by spaces
            rf'\b{re.escape(source_filename_no_extension)}\b', # Filename without extension
        ]

        responses_to_check = [
            llm_response,
            llm_response_cleaned
        ]

        is_relevant = any(
            re.search(pattern, response) 
            for pattern in patterns
            for response in responses_to_check
        )

        threshold = 80
        if not is_relevant: # No exact matches found, LLM may have mentioned the filename just differently enough, so time to check if a Fuzzy match is found
            print(f"\nNo exact matches found, checking for fuzzy match with a {threshold}% or higher threshold\n")
            is_relevant = is_fuzzy_subset(llm_response_cleaned, source_filename_cleaned, threshold)
            print(f"Fuzzy match result: {is_relevant} for {source_filename}\n")

        print(f"Citation relevance check result: {is_relevant} for {source_filename}")
        return is_relevant
    
    except Exception as e:
        handle_error_no_return("Could not determine if citation is relevant in is_citation_relevant(), encountered error: ", e)
        return False


def filter_all_citations(docs: list[Document], llm_response: str, return_top_k: bool, user_query: str) -> list[Document]:
    print(f"Pre-filtering citations to determine if any are relevant to the LLM response")
    all_docs = []
    for doc in docs:
        
        try:
            relevant_page_text = str(doc.page_content)
            source_filepath = str(doc.metadata.get('source'))
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue
        
        relevant_page_text = relevant_page_text.replace('\n', ' ')
        
        try:
            source_filename = os.path.basename(source_filepath)
        except Exception as e:
            handle_error_no_return("Could not parse path with OS lib, encountered error: ", e)
            continue
        
        try:
            if is_citation_relevant(llm_response, source_filename):
                all_docs.append(doc)
            else:
                print(f"Citation {source_filename} is not relevant, skipping")
                continue
        except Exception as e:
            handle_error_no_return("Could not determine if citation is relevant in filter_all_citations(), encountered error: ", e)
            continue

    if all_docs == [] and return_top_k:
        print("No relevant citations found but top K requested, reranking all docs")
        all_docs = rerank_results_ml(user_query, docs, top_n=3)

    return all_docs


def read_config_for_get_references() -> tuple[str, str, str, bool, bool, str, bool]:
    try:
        read_return = read_config(['local_llm_server', 'upload_folder', 'local_llm_chat_template_format', 'llm_filter_citations', 'force_enable_rag', 'exl2_prompt_template_format', 'perform_graph_rag'])
        local_llm_server = read_return['local_llm_server']
        upload_folder = read_return['upload_folder']
        local_llm_chat_template_format = read_return['local_llm_chat_template_format']
        llm_filter_citations = read_return['llm_filter_citations']
        force_enable_rag = read_return['force_enable_rag']
        exl2_prompt_template_format = read_return['exl2_prompt_template_format']
        perform_graph_rag = str(read_return['perform_graph_rag']).lower() == 'true'
        return local_llm_server, upload_folder, local_llm_chat_template_format, llm_filter_citations, force_enable_rag, exl2_prompt_template_format, perform_graph_rag
    except Exception as e:
        return handle_local_error("Could not read config.json in method read_config_for_get_references(), encountered error: ", e)


def get_request_parameters_for_get_references(request: Request) -> tuple[str, str, str, str, str, str, bool, bool]:
    try:
        stream_session_id = request.json['stream_session_id']
        user_query = request.json['user_query']
        llm_response = request.json['llm_response']
        formatted_user_prompt = request.json['formatted_user_prompt']
        chat_id = request.json['chat_id']
        sequence_id = request.json['sequence_id']
        regeneration_request = request.json['regeneration_request']
        regenerate_with_citations_force_enabled = request.json['regenerate_with_citations_force_enabled']
        return stream_session_id, user_query, llm_response, formatted_user_prompt, chat_id, sequence_id, regeneration_request, regenerate_with_citations_force_enabled
    except Exception as e:
        return handle_local_error("Could not read request content in method get_request_parameters_for_get_references(), encountered error: ", e)


def get_vector_results_for_get_references(stream_session_id: str) -> tuple[list[Document], bool]:
    try:
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        docs = QUERIES.pop(key_for_vector_results, None)
        return docs, docs is not None
    except Exception as e:
        return handle_local_error("Could not get vector results for stream_session_id in method get_vector_results_for_get_references(), encountered error: ", e)


def determine_if_flux_diffusers_is_enabled() -> bool:
    try:
        hf_read_return = read_hf_config(['flux_diffusers'])
        flux_diffusers = str(hf_read_return['flux_diffusers']).lower() == 'true'
        return flux_diffusers
    except Exception as e:
        return False


def get_hf_waitress_formatted_user_prompt(formatted_user_prompt: str, llm_response: str) -> str:
    history_prompt_json = json.loads(formatted_user_prompt)
    new_response = {"role":"assistant", "content":llm_response}
    history_prompt_json['messages'].append(new_response)
    updated_history_prompt_json = json.dumps(history_prompt_json, indent=4)
    return str(updated_history_prompt_json)


def get_sources_and_pages_for_get_references(docs: list[Document], llm_response: str, llm_filter_citations: bool, upload_folder: str, force_enable_rag: bool, user_query: str) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    '''
    This function iterates through the list of Document objects, and for each Document object, it extracts the source filename, source filepath, and page number,a nd returns:

    reference_pages = {
        'source_pdf_path': [
            ['page_content', 'page_number'],
            ['page_content', 'page_number'],
            ...
        ],
        ...
    }
    '''
    if llm_filter_citations:
        try:
            docs = filter_all_citations(docs=docs, llm_response=llm_response, return_top_k=force_enable_rag, user_query=user_query)
        except Exception as e:
            handle_error_no_return("Could not pre-filter citations in get_sources_and_pages_for_get_references(), proceeding without pre-filtering. Encountered error: ", e)
    
    all_sources = {}
    reference_pages = {}
    for doc in docs:
        
        try:
            relevant_page_text = str(doc.page_content)
            relevant_page_number = str(doc.metadata.get('page_number'))
            source_filepath = str(doc.metadata.get('source'))
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue
        
        relevant_page_text = relevant_page_text.replace('\n', ' ')
        
        try:
            source_filename = os.path.basename(source_filepath)
            source_filename_without_extension = os.path.splitext(source_filename)[0]
            pdf_version_path = os.path.join(upload_folder, source_filename_without_extension + '.pdf')   # Construct the path to the potential PDF version.
        except Exception as e:
            handle_error_no_return("Could not parse source file path when getting sources and pages for get_references(), encountered error: ", e)
            continue

        if os.path.exists(pdf_version_path):
            #print("\n\pdf exists\n\n")
            source_filename = source_filename_without_extension + '.pdf'
            
            if pdf_version_path in reference_pages:
                reference_pages[pdf_version_path].extend([[relevant_page_text,relevant_page_number]])
            else:
                reference_pages[pdf_version_path] = [[relevant_page_text,relevant_page_number]]

            if source_filename not in all_sources:  # Add this file to our sources dictionary if it's not already present
                source_filepath = pdf_version_path
                all_sources.update({source_filename: source_filepath})

        else:
            print(f"\n\nCould not find source doc at {pdf_version_path}, RAG ACTIVE BUT REFERENCING WILL NOT DISPLAY!\n\n")
            if source_filename not in all_sources: # Do not duplicate if the TXT file is already in the sources dict
                try:
                    source_filepath = os.path.join(upload_folder, source_filename) # reconstructed path using the OS module just to be safe
                    all_sources.update({source_filename: source_filepath})
                except Exception as e:
                    handle_error_no_return("Could not construct filepath for TXT file, encountered error: ", e)

    return all_sources, reference_pages


def get_refer_pages_and_download_link_html(user_should_refer_pages_in_doc: dict[str, list[list[str]]], stream_session_id: str, is_graph_rag: bool) -> tuple[str, str]:
    refer_pages_string = ""
    if not is_graph_rag:
        refer_pages_string = "<br><h6>Additional data may be found in the following documents & pages:</h6>"
        
        for index, doc in enumerate(user_should_refer_pages_in_doc, start=1):
            pdf_iframe_id = f"stream{stream_session_id}PdfViewer{str(index)}"
            tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
            frame_doc_path = f"/pdf/{doc}"
            try:
                stream_id_string_to_remove = f"_{stream_session_id}"
                doc_name_without_stream_id = str(doc).replace(stream_id_string_to_remove, "")
                refer_pages_string += f"<br><h6>{doc_name_without_stream_id}: "
                for page in user_should_refer_pages_in_doc[doc]:
                    frame_doc_path += f"#page={str(page)}" 
                    refer_pages_string += f'<a href="javascript:void(0)" onclick="goToPageAndSwitchTab(\'{pdf_iframe_id}\', \'{frame_doc_path}\', \'tab{tab_name_string}\', \'{stream_session_id}\')">Page {page}</a>, '
                    frame_doc_path = f"/pdf/{doc}"
                refer_pages_string = refer_pages_string.strip(', ') + "</h6>"
            except Exception as e:
                handle_error_no_return("Could not construct refer_pages_string, encountered error: ", e)

    
    pdf_right_pane_id = f"stream{stream_session_id}PdfPane"
    download_link_html = f'<div class="pdf-viewer-container" id="{pdf_right_pane_id}">'

    # Add tab buttons
    download_link_html += '<div class="tab-buttons">'
    for index, source in enumerate(user_should_refer_pages_in_doc, start=1):
        tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
        stream_id_string_to_remove = f"_{stream_session_id}"
        doc_name_without_stream_id = str(source).replace(stream_id_string_to_remove, "")
        default_open = ' defaultTabs' if index == 1 else ''
        download_link_html += f'<button class="tab-button{default_open}" stream-session-id="{stream_session_id}" onclick="openTab(event, \'tab{tab_name_string}\', \'{stream_session_id}\')">{doc_name_without_stream_id}</button>'
    download_link_html += '</div>'

    # Add tab content
    for index, source in enumerate(user_should_refer_pages_in_doc, start=1):
        try:
            download_link_url = url_for('download_file', filename=source)
            pdf_iframe_id = f"stream{stream_session_id}PdfViewer{str(index)}"
            tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
            download_link_html += f'<div id="tab{tab_name_string}" class="tab-content" stream-session-id="{stream_session_id}">'
            download_link_html += f'<iframe class="citations-pdf-iframe" id="{pdf_iframe_id}" src="{download_link_url}"></iframe>'
            download_link_html += "</div>"
        except Exception as e:
            handle_error_no_return("Could not construct download_link_html, encountered error: ", e)

    download_link_html += "</div>"

    return refer_pages_string, download_link_html


def get_model_response_for_history_db_for_get_references(download_link_html: str, llm_response: str, reference_response: str) -> str:
    model_response_for_history_db = str(llm_response)
    model_response_for_history_db += f"\n\n{reference_response}"
    model_response_for_history_db += f"\n\npdf_pane_data={download_link_html}"
    model_response_for_history_db = model_response_for_history_db.strip('\n')
    return model_response_for_history_db



@app.route('/get_references', methods=['POST'])
def get_references():

    print("\n\nStoring History Post-Response -- Determining if Citations are Necessary\n\n")

    try:
        local_llm_server, upload_folder, local_llm_chat_template_format, llm_filter_citations, force_enable_rag, exl2_prompt_template_format, perform_graph_rag = read_config_for_get_references()
        exl2 = read_hf_config(['exl2'])['exl2']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to get_references. Error: ", e)

    try:
        stream_session_id, user_query, llm_response, formatted_user_prompt, chat_id, sequence_id, regeneration_request, regenerate_with_citations_force_enabled = get_request_parameters_for_get_references(request)
    except Exception as e:
        return handle_api_error("Could not read request content in method get_references, encountered error: ", e)

    do_rag = False
    try:
        docs, do_rag = get_vector_results_for_get_references(stream_session_id)
    except Exception as e:
        handle_error_no_return("Error determining if RAG was used in method get_references - Could not check the QUERIES dict. Proceeding without RAG. Encountered error: ", e)

    if local_llm_server == 'llama-cpp':
        formatted_user_prompt += append_eot_token_to_llm_response(local_llm_chat_template_format, llm_response)
    elif local_llm_server == 'hf-waitress':
        local_llm_chat_template_format = "hf-transformers"
        flux_diffusers = determine_if_flux_diffusers_is_enabled()
        if flux_diffusers:
            do_rag = False
        else:
            if exl2:
                if perform_graph_rag:
                    try:
                        last_context_index = formatted_user_prompt.rindex("The following context might be helpful in answering the user query above.")
                        formatted_user_prompt = formatted_user_prompt[:last_context_index]
                    except Exception as e:
                        handle_error_no_return("Trimming RAG context unnecessary, skipping. Encountered error: ", e)
                formatted_user_prompt += append_eot_token_to_llm_response(exl2_prompt_template_format, llm_response)
            else:
                formatted_user_prompt = get_hf_waitress_formatted_user_prompt(formatted_user_prompt, llm_response)

    if not do_rag:
        print("\n\nRAG Citations unnecessary, storing chat history and returning\n\n")
        try:
            if not regeneration_request:
                stored_datetime, chat_id = store_local_llm_chat_history_to_db(chat_id, sequence_id, stream_session_id, user_query, llm_response, formatted_user_prompt, local_llm_server, local_llm_chat_template_format)
            else:
                stored_datetime, chat_id = update_llm_response_in_history_db(chat_id, stream_session_id, user_query, llm_response)
        except Exception as e:
            handle_error_no_return("Could not store or update chat history DB in get_references(), encountered error: ", e)
        return jsonify({'success': True, 'stored_datetime':stored_datetime, 'local_llm_server':local_llm_server, 'local_llm_chat_template_format':local_llm_chat_template_format, 'chat_id':chat_id})
    

    print("\n\nFetching Citations\n\n")

    all_sources = {}
    reference_pages = {}
    try:
        all_sources, reference_pages = get_sources_and_pages_for_get_references(docs, llm_response, llm_filter_citations, upload_folder, (force_enable_rag or regenerate_with_citations_force_enabled), user_query)
    except Exception as e:
        return handle_api_error("Could not get sources and pages for get_references(), encountered error: ", e)
    
    try:
        docs_have_relevant_info, user_should_refer_pages_in_doc = highlighter_interface(reference_pages, stream_session_id)
    except Exception as e:
        handle_error_no_return("Could not complete highlighter_interface, encountered error: ", e)
    
    reference_response = ""
    download_link_html = ""
    if docs_have_relevant_info:
        try:
            reference_response, download_link_html = get_refer_pages_and_download_link_html(user_should_refer_pages_in_doc, stream_session_id, perform_graph_rag)
        except Exception as e:
            handle_error_no_return("Could not get refer_pages_string and download_link_html, encountered error: ", e)
    
    try:
        model_response_for_history_db = get_model_response_for_history_db_for_get_references(download_link_html, llm_response, reference_response)
    except Exception as e:
        handle_error_no_return("Could not prep data to store to chat history DB in get_references(), encountered error: ", e)

    try:
        if not regeneration_request:
            stored_datetime, chat_id = store_local_llm_chat_history_to_db(chat_id, sequence_id, stream_session_id, user_query, model_response_for_history_db, formatted_user_prompt, local_llm_server, local_llm_chat_template_format)
        else:
            stored_datetime, chat_id = update_llm_response_in_history_db(chat_id, stream_session_id, user_query, model_response_for_history_db)
    except Exception as e:
        handle_error_no_return("Could not store or update chat history DB in get_references(), encountered error: ", e)

    return jsonify({'success': True, 'response': reference_response, 'pdf_frame':download_link_html, 'stored_datetime':stored_datetime, 'local_llm_server':local_llm_server, 'local_llm_chat_template_format':local_llm_chat_template_format, 'chat_id':chat_id})




    # try: # 3. Apply UNIQUE constraint with a UNIQUE INDEX - works even if the table already exists!
    #     index_name = 'idx_upload_staging_unique_constraint'
    #     columns_for_uniqueness = [
    #         'upload_id',
    #         'user_id',
    #         'filepath',
    #         'embedding_model',
    #         'knowledge_domain',
    #         'source',
    #         'text_extraction_method',
    #         'upload_date'
    #     ]
    #     cursor.execute(f'''
    #         CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
    #         ON upload_staging ({", ".join(columns_for_uniqueness)})
    #     ''')
    # except Exception as e:
    #     return handle_local_error("Could not create unique index for upload_staging table, encountered error: ", e)



    '''
    The following JS from upload-functions.js is a good example of how to handle a bulk request from the client to the server.
    It's an alternative to the array.reduce approach, using a for loop to create an array of promises (stagingPromises[]) and then await Promise.all(stagingPromises) to wait for them all to finish.
    
    This fundamentally changes the behavior from sequential to parallel.
    
    Here's a breakdown of the differences, pros, and cons:
    1. Sequential Processing (reduce or an async/await loop):
    How it works: Starts the transfer for file 1. Waits for it to complete (success or caught failure). Then starts the transfer for file 2, waits, and so on.
    Pros:
        Lower Server Load: Sends only one request to /gdrive_file_transfer_to_staging at a time. This is gentler on the server.
        Predictable Order: Files are processed strictly one after another.
    Cons:
        Slower: The total time is the sum of all individual transfer times. If you have 10 files taking 1 second each, it takes ~10 seconds.
    
    2. Parallel Processing (Promise.all):
    How it works: The for loop starts all the file transfers almost simultaneously, pushing each resulting promise into stagingPromises[] & await Promise.all(stagingPromises[]) then waits for all of those 
    concurrently running transfers to finish (either resolve successfully or have their errors caught by the .catch within the loop).
    Pros:
        Much Faster (Potentially): If the server and network can handle it, transfers happen concurrently. If 10 files take 1 second each but can run in parallel, the total time might be closer to just 1-2 seconds (plus overhead).
        Simpler Loop: A standard for loop is often considered more straightforward than the reduce pattern for promises.
    Cons:
        Higher Server Load: Sends many requests to /gdrive_file_transfer_to_staging at roughly the same time. This could overload a server not designed for concurrency or hit API rate limits.
        Order Not Guaranteed: Files might finish transferring in any order, not necessarily the order they were listed.
    
    Which is "Better" or "More Correct"?
    Neither is inherently incorrect. Both are valid patterns for handling multiple asynchronous tasks.
    The "better" choice depends on your specific needs and your server's capabilities:
    If /gdrive_file_transfer_to_staging can efficiently handle many simultaneous requests and you want the fastest possible user experience for staging, the parallel (Promise.all) approach is usually preferred.
    If that endpoint might struggle with load, or if there's a strict need to process files sequentially (which doesn't seem apparent here), the sequential (reduce or async/await loop) approach is safer.
    Your current Promise.all implementation looks correct for achieving parallel transfers. It correctly initiates all fetches, waits for them all using Promise.all, and handles individual errors within the loop's 
    .catch so that one failure doesn't stop Promise.all from waiting for the others.

    The sequential approach might be better though, since the backend server needs to download the requested files off GDrive and making too many parallel download requests may be blocked or rejected by Google 
    as there's a risk that it might think someone's Drive API is compromised.
    
    That's an excellent point and a very strong argument for using the sequential approach (reduce or an async/await loop) instead of the parallel Promise.all approach.
    
    You are right to be concerned about potential rate limiting or blocking from Google Drive's API. Even though the calls are from your server to Google Drive, if your server suddenly makes dozens of simultaneous download requests
    for a single user's account, Google's systems might interpret that as unusual activity and could throttle or temporarily block the requests to protect the user's account or their own infrastructure.
    
    Pros of Sequential for GDrive:
        Rate Limit Avoidance: Significantly reduces the chance of hitting Google's API rate limits.
        Lower Risk of Blocking: Less likely to trigger Google's security heuristics for suspicious activity.
        Predictable Server Load: Easier on your own server resources as it only handles one download/staging process at a time.
    
    Cons of Sequential for GDrive:
        Slower Staging Phase: The staging step will take longer as files are downloaded one by one.
    
    Conclusion:
    Given the interaction with an external API (Google Drive) that likely has rate limits and security monitoring, the sequential approach is technically safer and likely more robust for the /gdrive_file_transfer_to_staging step. 
    The risk of failed transfers due to parallel requests outweighs the speed benefit in this context.
    Therefore, reverting to the reduce pattern you had originally, or using the equivalent async/await loop pattern I suggested earlier, would be the recommended course of action.
    
    PARALLEL CODE:
    
    const allStagedFileInfo = [];
    const stagingPromises = [];

    for (let i = 0; i < selectedFiles.length; i++) {
        const userId = "default_user";
        let formData = new FormData();
        formData.append('file_id', selectedFiles[i].id);
        formData.append('file_mimeType', selectedFiles[i].mimeType);
        formData.append('user_id', userId);

        const stagingPromise = fetch('/gdrive_file_transfer_to_staging', {
            method: 'POST',
            body: formData,
            redirect: 'follow'
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(`Failed to transfer file ${selectedFiles[i].id} to Server. Encountered error: ${err.error}`)});
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                appendStreamInfo(`File ${selectedFiles[i].id} transferred to server successfully!`, 'success');
                allStagedFileInfo.push(data.staged_file_info);
            } else {
                throw new Error(`File ${selectedFiles[i].id} not transfered to server correctly - server side error. Check server logs for more details`)
            }
        })
        .catch(error => {
            appendStreamInfo(`Error transferring file ${selectedFiles[i].id} to server, skipping. Check browser and server logs for more details`, 'failure');
            console.log("Error transferring file to server: ", String(error.message));
            updateUIForFile(selectedFiles[i].row, 'failure');
        });
        stagingPromises.push(stagingPromise);
    }

    try {
        await Promise.all(stagingPromises); // Wait for all file transfers to staging area to complete (resolved either successfully or via the catch block)
        appendStreamInfo("Files transferred to server successfully!", 'success');
        console.log("GDrive staging phase complete. Collected info:", allStagedFileInfo);

        if (allStagedFileInfo.length > 0) {
            appendStreamInfo(`Beginning file processing for ${allStagedFileInfo.length} files...`, 'waiting');

            let bulkUploadFormData = new FormData();
            bulkUploadFormData.append('docs_to_upload', JSON.stringify(allStagedFileInfo));

            const bulkResponse = await fetch('/bulk_upload_files', {
                method: 'POST',
                body: bulkUploadFormData,
                redirect: 'follow'
            })

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
        syncButton.disabled = false;
        appendStreamInfo("Google Drive Synchronization Completed!", 'success');
        populateDocsLoadedTable();
        hideStreamSpinner();
        for (let i = 0; i < selectedFiles.length; i++) {    // Update status of GDrive table rows to success
            if (selectedFiles[i].row) { 
                let selectedFilesRow = selectedFiles[i].row;    // But only if the status is not failure!
                let statusCell = selectedFilesRow.cells[selectedFilesRow.cells.length - 1];
                if (statusCell.innerHTML.includes("failure") || statusCell.innerHTML.includes("Failed")) {
                    continue;
                } else{
                    updateUIForFile(selectedFiles[i].row, 'success');
                }
            }
        }
    }

    '''



@app.route('/google_drive_loader', methods=['POST'])
def google_drive_loader():

    try:
        gdrive_file_id = str(request.form['file_id'])
        gdrive_file_mimeType = str(request.form['file_mimeType'])
    except Exception as e:
        return handle_api_error("Server-side error reading Google Drive file details for download: ", e)
    
    try:
        service = build("drive", "v3", credentials=GDRIVE_CREDS)
    except Exception as e:
        return handle_api_error("Could not create Google service handler, check credentials and re-try: ", e)

    data_queue = queue.Queue()
    stop_event = threading.Event()

    def sync_task():

        try:

            try:
                file_metadata = service.files().get(fileId=gdrive_file_id, fields='name, mimeType', supportsAllDrives=True).execute()   # includeItemsFromAllDrives=True not need here because it's a search and listing operation!
                original_filename = file_metadata.get('name', 'untitled')
                mime_type = file_metadata.get('mimeType', gdrive_file_mimeType)
                mime_type_category = categorize_mimetype(mime_type)
            except Exception as e:
                data_queue.put(f"Error fetching metadata for '{original_filename}' | failure")
                return handle_api_error(f"Could not read GoogleDrive file metadata for file: '{original_filename}', encountered error: ", e)
            
            data_queue.put(f"Fetched metadata for '{original_filename}', proceeding to download...")
            
            try:
                filename_with_extension, file_content = gdrive_downloader(service, gdrive_file_id, original_filename, mime_type, mime_type_category, data_queue)
            except Exception as e:
                data_queue.put(f"Error downloading {original_filename} from Google Drive | failure")
                return handle_api_error(f"Server-side error - could not download file: '{original_filename}' from Google Drive: ", e)

            if file_content is not None:    # gdrive downloader method downloaded & returned a single file
                try:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename_with_extension))

                    print(f"Saving {filename_with_extension} to {filepath}")
                    with open(filepath, 'wb') as f:
                        f.write(file_content)

                    data_queue.put(f"Vector Embedding & Indexing Document: '{filename_with_extension}'")
                    document_extractor_and_loader(filename_with_extension, filepath)
                    data_queue.put(f"Document '{filename_with_extension}' processed & uploaded successfully! | success")

                except Exception as e:
                    data_queue.put(f"Server-side error - could not process file: '{original_filename}' from Google Drive | failure")
                    return handle_api_error(f"Server-side error - could not process file: '{original_filename}' from Google Drive: ", e)
        
        finally:
            data_queue.put(None)
            print("\n\nGDrive-Sync stream done, breaking thread\n\n")
    
    def load_gdrive():

        try:
            thread = threading.Thread(target=sync_task)
            thread.start()
        except Exception as e:
            data_queue.put(f"Could not start sync process for Google Drive | failure")
            return handle_api_error("Could not start thread: ", e)

        while True:
            if stop_event.is_set(): #TODO: Add Cancel-Sync button to UI! Logic here will be simialr to STOP_GENERATION in hf_waitress.py
                print("\n\nStopping GDrive-Sync stream as requested by stop_event\n\n")
                thread.join()
                break
            output = data_queue.get()
            if output is None:
                print("\n\nNone read, breaking and stopping thread\n\n")
                thread.join()
                break
            yield f"data: {json.dumps(output)}\n\n"
        
        yield f"event: END\ndata: \"null\"\n\n"
    
    print("\n\nGoogle Drive Sync Begins!\n\n")
    return Response(load_gdrive(), content_type='text/event-stream')


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
        await bulk_response_processChunk(gdrive_loader_reader);
    } catch (error) {
        console.error("Error loading document from Google Drive in method loadGoogleDriveDoc. Error details: ", String(error.message));
    }
}


'''
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
'''


def document_extractor_and_loader(filename, filepath):

    print("Document Extraction and Loading to RAG & Records Databases in progress for single file...")

    try:
        pdf_filepath = get_pdf_filepath_for_upload(filename, filepath)
    except Exception as e:
        return handle_local_error("Could not get PDF filepath for upload, encountered error: ", e)
        
    try:
        txt_filepath = get_text_extract_from_pdf(pdf_filepath)
    except Exception as e:
        return handle_local_error("Failed to extract text from the PDF document, encountered error: ", e)
    
    try:
        upload_to_rag_and_records_databases(filename, txt_filepath)
    except Exception as e:
        return handle_local_error("Failed to upload to RAG & Records databases, encountered error: ", e)

    return True


def save_file_to_upload_dir(input_file):
    try:
        filename = secure_filename(input_file.filename)
        filename = filename.replace("PDF", "pdf") if "PDF" in filename else filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        input_file.save(filepath)
        print(f"\nSaved file {filename} to {filepath} successfully.\n")
        return filename, filepath
    except Exception as e:
        return handle_local_error("Could not save file to upload directory, encountered error: ", e)


# Route to handle the submission of the second form (file loading)
@app.route('/process_new_file', methods=['POST'])
def process_new_file():

    try:
        input_file = request.files['file']
    except Exception as e:
        return handle_api_error("Server-side error recieving file: ", e)

    try:
        filename, filepath = save_file_to_upload_dir(input_file)
    except Exception as e:
        return handle_api_error("Failed to save document to app folder, encountered error: ", e)

    try:
        document_extractor_and_loader(filename, filepath)
    except Exception as e:
        return handle_api_error("Failed to upload new document, encountered error: ", e)

    return jsonify(success=True)


# ===================================
# OLD DO_RAG LOGIC-------------------
# ===================================

def extract_significant_phrases(query):
    print("Extracting significant phrases")

    if not query:
        print("No query to extract significant phrases from")
        return []

    try:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        custom_stop_words = {"you", "me", "anything", "tell", "can", "could", "would", "should", "write", "writes", "wrote", "written", "read", "reads", "hi", "hello", "hey"}
        stop_words.update(custom_stop_words)
    except Exception as e:
        handle_error_no_return("Failed to download & set stopwords, encountered error: ", e)
    
    try:
        tokens = [token for token in query.lower().split() if token.isalnum() and token not in stop_words]  # isalnum() to remove punctuation and non-alphanumeric characters
    except Exception as e:
        handle_local_error("Could not extract significant tokens, encountered error: ", e)

    print(f"\nReturning tokens: {tokens}\n")
    return tokens


def calculate_relevance_score(phrases, document_content):
    #print("calculating relevance score")
    
    try:
        content_lower = document_content.lower()
    except Exception as e:
        handle_local_error("Could not read document_content in calculate relevance score, encountered error: ", e)
    
    #print(f"document content: {content_lower}")
    
    #score = sum(1 for phrase in phrases if phrase in content_lower)
    
    score = 0
    try:
        for phrase in phrases:
            if phrase in content_lower:
                print(f"Match found to enable RAG: {phrase}")
                score += 1
    except Exception as e:
        handle_local_error("Could not compare phrases in calculate relevance score, encountered error: ", e)
    
    return score


def filter_relevant_documents(query, search_results, threshold=1):

    print("Checking relevant docs to determin if RAG is required")

    do_rag = False
    page_contents = []

    try:
        significant_phrases = extract_significant_phrases(query)
    except Exception as e:
        handle_local_error("Could not extract significant phrases, encountered error: ", e)
    
    print(f"significant tokens: {significant_phrases}")
    #relevant_documents = []

    try:
        for document in search_results:
            # check for non-empty source field
            if document.page_content:
                page_contents.append(document.page_content)

            if not do_rag:  # if do_rag has already been set to true, why look?
                if document.metadata.get('source'):
                    score = calculate_relevance_score(significant_phrases, document.page_content)
                    if score >= threshold:
                        #relevant_documents.append(document)
                        print("Must do RAG!")
                        do_rag = True
    except Exception as e:
        handle_local_error("Could not read calculate relevance score, encountered error: ", e)

    #return relevant_documents
    return page_contents, do_rag


def determine_do_rag(query, docs, force_enable_rag, force_disable_rag):

    print("\n\nDetermining do_rag \n\n")

    do_rag = False
    
    # We do not modify the force_enable_rag or force_disable_rag flags in this method, we simply respond to them here. UI updates should handle those flags.
    if force_enable_rag:
        print("\n\nFORCE_ENABLE_RAG True, force enabling RAG and returning\n\n")
        try:
            do_rag = True
        except Exception as e:
            do_rag = False
            handle_error_no_return("Error force-enabling RAG, disabling RAG and continuing: could not filter relevant documents during determine do rag, encountered error: ", e)
    elif force_disable_rag:
        print("\n\nFORCE_DISABLE_RAG True, force disabling RAG and returning\n\n")
        do_rag = False
    else:
        try:
            _, do_rag = filter_relevant_documents(query, docs)
        except Exception as e:
            do_rag = True
            handle_error_no_return("Error determining if RAG is required, defaulting to enabling RAG and continuing: could not filter relevant documents during determine do rag, encountered error: ", e)

    return do_rag


'''
do_rag = determine_do_rag(user_query, docs, force_enable_rag, force_disable_rag)

Was called in search_knowledge_base() after `rerank_results_ml` and before `perform_graph_rag = read_config(['perform_graph_rag'])['perform_graph_rag']`
'''

# ===================================
# END OF OLD DO_RAG LOGIC-------------
# ===================================



def safe_empty_cuda_cache(timeout=5):
    import concurrent.futures

    '''
    - Demonstrates why we CANNOT use concurrent.futures's ThreadPoolExecutor nor ProcessPoolExecutor as neither gurantess termination and the former even risks holding the GIL hostage. Details:

    - ThreadPoolExecutor (Unsafe):
        
        - Mechanism: Launches a thread, which shares the main process's Global Interpretor Lock (GIL) and other resources such as memory.
        - Intended Use: I/O bound (background) tasks (network requests/downloads, disk access, etc.) that release the GIL while waiting.
        - Why it fails:
            - Failure Mode 1 -- GIL-Releasing Hang (stuck network calls, etc.): `TimeoutError` will trigger but will result in a zombie thread leading to resource leakage and unpredictable behavior (operations, logs, etc.)
            - Failure Mode 2 -- GIL-Holding Hang (deadlocks): Worker thread holds the GIL hostage. Python threads cannot be killed forcefully, so the main thread can never acquire the GIL. `TimeError` is never raised and the entire application freezes.
        - Key Takeaways:
            - Unrelaible because we cannot predict the failure mode.
            - Python threads cannot be forcefully killed as a fundamental design philosophy: Forcefully killing a thread is extremely dangerous because threads share memory and resources.
            - Testing with `time.sleep()` is misleading as it's a "polite hang" (GIL-releasing). A "rude hang" (`while True:`) is a more accurate way of testing for a deadlock.

    - ProcessPoolExecutor (Deceptively Flawed):

        - Mechanism: Launches a completely separate process with it's own GIL, memory & resources.
        - Why it fails: High-level API that lacks the control needed for guaranteed termination.
            - If using `with ProcessPoolExecutor(max_workers=1) as executor`: The `with` contains an implicit `executor.shutdown(wait=True)`, which aims for graceful shutdown (completion) of the task. If that task hangs, the main process also hangs indefinitely waiting for it.
            - If manually managing `TimeoutError` Exceptions with `executor.shutdown(wait=False)`: Fire's and forgets a SIGTERM, without awaiting termination, allowing the main process to continue. Child process might not terminate at all(hung/deadlocked) or may still have time to complete leading to unpredictable behavior (operations, logs, etc.)
    '''
    
    print(f"\nAttempting empty_cuda_cache with a {timeout}-second timeout\n")
    

    # --- Using ThreadPoolExecutor ---
    '''
    Test result (with `while True:`): Primary process and GIL held hostage, response slowed to a crawl. Application was non-responsive.
    CTRL + C termination failed and hung up too! Task manager had to be used to end the process.
    '''
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(empty_cuda_cache)
            try:
                future.result(timeout=timeout)
                print(f"\nCUDA cache successfully emptied\n")
            except TimeoutError:
                print(f"\nCUDA cache emptying timed out after {timeout} seconds, returning without emptying CUDA cache\n")
    except Exception:
        print("\nReturning without emptying CUDA cache\n")

    # --- Using ProcessPoolExecutor ---

    # Approach 1: Manually managing executor shutdown:
    '''
    Test result (with `while True:`): Application continued working but a number of never-resolving background processes were spawned that would eventually screw up the system.
    On termination, a number of Kernel errors were observed.
    '''
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)    # max_workers=1 ensures we only spin up one process
    future = executor.submit(empty_cuda_cache)
    
    try:
        result = future.result(timeout=timeout)
        if result:
            print(f"\nCUDA cache successfully emptied\n")
        else:
            print(f"\nCUDA cache could not be emptied within {timeout} seconds\n")

        # If successful, perform a clean shutdown
        executor.shutdown(wait=True)

    except TimeoutError:
        print(f"\nTimeout: `empty_cuda_cache` took longer than {timeout} seconds. Terminating operation and continuing...\n")

        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError: # For Python < 3.9 that doesn't have cancel_futures
            executor.shutdown(wait=False)

    except Exception as e:  # Ensure cleanup even on other errors
        handle_error_no_return("An exception occurred while emptying CUDA cache: ", e)
        executor.shutdown(wait=False)
    
    print(f"\nCUDA cache empty operation completed\n")


    # Approach 2: Using with statement (implicitly calls `executor.shutdown(wait=True)`):
    '''
    Test result (with `while True:`): Server hung up waiting on the child process to complete: even though as a child process it has an independent GIL, the implicit `wait=True` causes the main process to wait indefinitely!
    Similar Kernel errors were observed on CTRL + C termination.

    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:    
        try:
            future = executor.submit(empty_cuda_cache)
            # wait for the result for a max of timeout seconds
            result = future.result(timeout=timeout)
            if result:
                print(f"\nCUDA cache successfully emptied\n")
            else:
                print(f"\nCUDA cache could not be emptied within {timeout} seconds\n")
        
        except TimeoutError:
            print(f"\nTimeout: `empty_cuda_cache` took longer than {timeout} seconds. Terminating operation and continuing...\n")
        
        except Exception as e:
            handle_error_no_return("Could not empty CUDA cache, encountered error: ", e)


    # --- Using a Nested Thread ---
    '''
    Suffers the same main-process hanging issue on deadlock of the nested thread as the GIL is shared!
    '''

    def safe_empty_cuda_cache(timeout=10):

        def empty_cuda_cache():
            print("\n\nEmptying CUDA cache (in a separate process)\n\n")

            if torch.cuda.is_available():
                try:
                    print("Attempting to empty cuda cache")
                    torch.cuda.empty_cache()
                    print("CUDA cache successfully emptied")
                    return True
                except Exception as e:
                    print(f"Could not empty cuda cache, encountered error: {str(e)}")
                    return False
            else:
                print("\n\nCUDA is not available, skipping cache-emptying\n\n")
                return True

        try:
            cuda_cache_cleanup_thread = threading.Thread(target=empty_cuda_cache)
            cuda_cache_cleanup_thread.start()

            print(f"⏰ Starting cleanup with a {timeout}-second timeout...")
            cuda_cache_cleanup_thread.join(timeout=timeout)
            
            if cuda_cache_cleanup_thread.is_alive():
                print(f"⚠️  Cleanup timeout reached ({timeout} seconds) - proceeding with force shutdown")
                cuda_cache_cleanup_thread.terminate()
                cuda_cache_cleanup_thread.join()
            else:
                print("✅ Cleanup completed successfully")


        except Exception:
            print("\nReturning without emptying CUDA cache\n")



def remove_embedded_links_from_llm_response(llm_response: str) -> str:
    try:
        # print(f"llm_response before link-substitution: {llm_response}")
        # Multi-step:
        # embedded_link_start_pattern = r'\(*\[[^\]]+\]\(\s*'  # Matches embedded links in the format: ([link text](url))
        # embedded_link_end_pattern = r'\s*\){1,2}\.?'          # Matches the closing parenthesis and optional period
        # llm_response_without_embedded_links = re.sub(embedded_link_start_pattern, '', llm_response)
        # llm_response_without_embedded_links = re.sub(embedded_link_end_pattern, '', llm_response_without_embedded_links)

        # Single-step:
        embedded_link_complete_pattern = r'\(*\[[^\]]+\]\(\s*\){1,2}\.?'  # Matches embedded links in the format: ([link text](url))
        llm_response_without_embedded_links = re.sub(embedded_link_complete_pattern, '', llm_response)
        # print(f"llm_response after link-substitution: {llm_response_without_embedded_links}")

        return llm_response_without_embedded_links
    except Exception as e:
        handle_local_error("Could not remove embedded links from llm_response, encountered error: ", e)



<div class="loader2" id="overlay" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgb(0,0,0,1); z-index: 1000;">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 24px; font-weight: 500;">
        <img src="{{ url_for('static', filename='images/L-Flow-Logo.jpeg') }}" class="lars-logo" alt="LARS Logo">
        Processing New Document...
    </div>
</div>


function cleanStreamedContent(dataObj) {
    // console.log("dataObj: ", dataObj);

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
    // Important to process the less-than and greater-than entities so we can escape other HTML elements the LLM puts out - primarily <think>...</think> tags!
    streamed_content = streamed_content.replace(/</g, '&lt;')
                                    .replace(/>/g, '&gt;')
                                    .replace(/&lt;br&gt;/g, '<br>');

    // Replace <br> tags with <br>
    // streamed_content = streamed_content.replace(/&/g, '&amp;').replace(/&lt;br&gt;/g, '<br>');

    // console.log("streamed_content: ", streamed_content);
    return streamed_content;
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

                if (receivedComplete) {
                    document.getElementById(responseContentID).innerHTML = cleanStreamedContent(totalContent);  // Refresh innerHTML to ensure formatting of HTML tags is properly applied
                    break;
                }
            }
        }

        await processChunk();   // processChunk() is an async-Fn and thus returns a promise. Here, via await, we pause execution until the promise is resolved or rejected.
        return totalContent;

    } catch (error) {
        errorHandler("fetching llama.cpp event-streaming response", "localhost:8080/completions", String(error))
    }
}



async function fetchHfWaitressEventStream(formattedPrompt, responseContentID, chatContainer, file=null) {
    let url;
    let hfwHeaders = new Headers();
    let formdata = null;
    let rawBodyJSONStringified = null;

    const vision = getVision();
    const exl2 = getExl2();
    console.log("vision: ", vision, "typeof:", typeof vision);
    console.log("exl2: ", exl2, "typeof:", typeof exl2);
    if (vision === "true") {
        console.log("Invoking vision_stream");
        const hfWaitress_URL = getHfwUrl();
        url = `${hfWaitress_URL}/vision_stream`;

        hfwHeaders = new Headers();
        hfwHeaders.append("X-DPI", "300");
        hfwHeaders.append("X-Max-New-Tokens", document.getElementById('HfwMaxNewToks').value);

        formdata = new FormData();

        const parsedPrompt = JSON.parse(formattedPrompt);
        formdata.append("messages", JSON.stringify(parsedPrompt.messages));
        if (file) { formdata.append("file", file); }
    } else if (exl2 === "true") {
        console.log("Invoking exl2_stream");
        const hfWaitress_URL = getHfwUrl();
        url = `${hfWaitress_URL}/exl2_stream`;

        hfwHeaders = new Headers();
        hfwHeaders.append("Content-Type", "application/json");
        hfwHeaders.append("X-Max-New-Tokens", document.getElementById('HfwMaxNewToks').value);
        hfwHeaders.append("X-Temperature", document.getElementById('HfwTempSlider').value);
        hfwHeaders.append("X-Top-K", document.getElementById('HfwTopkSlider').value);
        hfwHeaders.append("X-Top-P", document.getElementById('HfwToppSlider').value);

        rawBodyJSONObj = JSON.parse(formattedPrompt);                                
        rawBodyJSONStringified = JSON.stringify(rawBodyJSONObj);
    } else {
        console.log("Invoking completions_stream");
        const hfWaitress_URL = getHfwUrl();
        url = `${hfWaitress_URL}/completions_stream`;

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

        const request_body = vision === "true" ? formdata : rawBodyJSONStringified;
        // console.log("request_body: ", request_body);

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
                            const streamed_content = cleanStreamedContent(dataObj);

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

                if (hfwReceivedComplete) {
                    document.getElementById(responseContentID).innerHTML = cleanStreamedContent(hfwTotalContent);   // Refresh innerHTML to ensure formatting of HTML tags is properly applied
                    break;
                }
            }
        }

        await hfwProcessChunk();
        return hfwTotalContent;

    } catch (error) {
        errorHandler("fetching event-streaming response", "HF-Waitress/completions_stream", String(error));
    }

}


def final_cleanup_of_llm_response(llm_response: str) -> str:
    # NOTE: This is no longer needed as the Markdown renderer handles this for us!
    pattern = r'\s*<br>\s+<br>\s*<br>\s*'    # Replace multiple breakline tags with a single one; \r?\n means optional carriage return and/or newline
    llm_response = re.sub(pattern, '<br><br>', llm_response)
    llm_response = llm_response.strip()
    return llm_response



##############---Old setup_for_local_llm_response Code ---##############

def search_knowledge_base(user_query:str, embedding_function:str, force_enable_rag:bool, filter_top_k_results_by_reranking:int, fetch_top_k_results_from_vectordb:int) -> tuple[list[Document], bool]:
    print("Searching knowledge base")

    try:
        llm_set_rag_config = determine_response_service(user_query, force_enable_rag)
        do_rag = llm_set_rag_config['do_rag']
    except Exception as e:
        handle_error_no_return("Could not determine response service, defaulting to use Naive RAG. Encountered error: ", e)
        safe_write_config(DEFAULT_CONFIG)   # update, at minimum, the do_rag and perform_graph_rag keys in config.json
        do_rag = DEFAULT_CONFIG['do_rag']

    if do_rag is False:
        print("Do RAG is False, returning...")
        return [], False, None

    print("Do RAG is True, continuing...")
    filtered_docs = []
    try:
        docs_list_with_cosine_distance = search_vector_db(user_query, embedding_function, int(fetch_top_k_results_from_vectordb))
        filtered_docs = [doc for doc, score in docs_list_with_cosine_distance]  # the `doc,score` is crucial, as it ensure we select only the Document object, and not a tuple comprising of a Document object and a float score!
    except Exception as e:
        handle_error_no_return("Could not perform vector search to determine do_rag when attempting to search-knowledge-base, encountered error: ", e)

    whoosh_results = []
    try:
        whoosh_results = search_whoosh_index(user_query)
    except Exception as e:
        handle_error_no_return("Could not perform whoosh search to determine do_rag when attempting to search-knowledge-base, encountered error: ", e)

    combined_docs = []
    try:
        combined_docs = combine_and_deduplicate_search_results(whoosh_results, filtered_docs)   # Combine the whoosh and vector results
    except Exception as e:
        handle_error_no_return("Could not combine and deduplicate search results, skipping. Encountered error: ", e)
        combined_docs = filtered_docs

    if not combined_docs:   # i.e. if blank
        print("No documents for citations, returning...")
        return [], False, None

    try:
        docs = rerank_results_ml(user_query, combined_docs, top_n=filter_top_k_results_by_reranking)
    except Exception as e:
        handle_error_no_return("Could not rerank search results, skipping. Encountered error: ", e)
        docs = combined_docs
    
    # do_rag = determine_do_rag(user_query, docs, force_enable_rag, force_disable_rag)
    
    perform_graph_rag = read_config(['perform_graph_rag'])['perform_graph_rag']
    enable_graph_rag = read_config(['enable_graph_rag'])['enable_graph_rag']

    graph_rag_context = None
    if perform_graph_rag and do_rag and enable_graph_rag:   # All conditions must be met for GraphRAG to be performed!
        try:
            graph_rag_context, reranked_summaries_list_descending = execute_graph_rag(user_query, docs)
            if reranked_summaries_list_descending != []:
                return reranked_summaries_list_descending, do_rag, graph_rag_context
        except Exception as e:
            handle_error_no_return("Could not execute graph RAG, encountered error: ", e)
    else:
        safe_write_config({'perform_graph_rag': False})  # In-case the LLM elected to use GraphRAG but the user has explicitly disabled it, we need to set perform-graph_rag to False to avoid any issues downstream!

    return docs, do_rag, graph_rag_context

@app.route('/setup_for_local_llm_response', methods=['POST'])
def setup_for_local_llm_response():
    print("\n\nSetting up for local LLM response\n\n")

    global QUERIES
    do_rag = True

    try:    # Read config and request data, determine base values while handling regeneration case
        config = read_config_for_llm_response_setup()
        stream_session_id, user_query, chat_id, sequence_id, file_attached, regeneration_request, regenerate_with_citations_force_enabled, regenerate_with_citations_force_disabled = read_request_data_for_setup_response(request) # stream_session_id = None if not regeneration_request, final value determined below!
        stream_session_id, key_for_vector_results, current_sequence_id = get_ids_for_llm_response_setup(stream_session_id, chat_id, sequence_id, regeneration_request)
    except Exception as e:
        return handle_api_error("Error getting base values for setup-for_local_llm_response, encountered error: ", e)

    try:    # Get full prompt including history from history-db
        current_sequence_id, full_prompt = get_full_llm_prompt_with_history(int(chat_id), int(current_sequence_id) if not regeneration_request else int(current_sequence_id) - 1) # if regeneration_request, then go back one sequence id!
    except Exception as e:
        return handle_api_error("Could not get full prompt from history db in method setup-for_local_llm_response, encountered error: ", e)
    
    try:
        local_llm_server, special_response = handle_special_model_case(config['local_llm_server'], current_sequence_id, file_attached, stream_session_id, user_query, full_prompt, regeneration_request)
    except Exception as e:
        return handle_api_error("Error determining appropriate model type and server for setup-for_local_llm_response: ", e)
    
    if special_response is not None:    # If a special model response is returned, quick-return here
        print(f"Returning special model response: {special_response}")
        return special_response
    
    if config['force_disable_rag'] or regenerate_with_citations_force_disabled:
        return handle_force_disabled_rag(local_llm_server, full_prompt, user_query, current_sequence_id, stream_session_id, regeneration_request, config['base_template'], config['local_llm_chat_template_format'], config['skip_system_prompt'])
    
    print("\n\nRAG Routine Begins: Performing semantic search on VectorDB, lexical search on Whoosh index, combining and reranking results and determining if RAG is necessary\n\n") 
            
    try:    # RAG Routine Begins: Perform semantic search on the vector DB, lexical search on the whoosh index, combine and rerank results and determine if RAG is necessary
        docs, do_rag, graph_rag_context = search_knowledge_base(user_query, config['selected_embedding_model'],
        (config['force_enable_rag'] or regenerate_with_citations_force_enabled), 
        int(config['filter_top_k_results_by_reranking']), int(config['fetch_top_k_results_from_vectordb']))
    except Exception as e:
        return handle_api_error("Could not process vector search in method setup-for_local_llm_response, encountered error: ", e)

    try:
        if do_rag:    # Add RAG results to user-query if necessary!
            QUERIES[key_for_vector_results] = docs
            user_query += f"\n\nThe following context might be helpful in answering the user query above. If so, please reference useful documents by name and specific page numbers in your response:\n"
            if graph_rag_context is not None:
                user_query += f"{graph_rag_context}"
            else:
                user_query += f"{docs}"
    except Exception as e:
        reject_rag()
        handle_error_no_return("Could not write do_rag or prepare RAG context during setup-for_local_llm_response, encountered error: ", e)

    try:    # Get full prompt for server
        formatted_updated_prompt = get_full_prompt_for_server(local_llm_server, full_prompt, user_query, current_sequence_id, config['base_template'], config['local_llm_chat_template_format'], config['skip_system_prompt'])
    except Exception as e:
        return handle_api_error("Could not get formatted_updated_prompt in method setup-for_local_llm_response, encountered error: ", e)

    if not regeneration_request or current_sequence_id == 0: current_sequence_id = int(current_sequence_id) + 1
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag, "formatted_user_prompt": formatted_updated_prompt, "sequence_id":current_sequence_id, "server_type":local_llm_server})


##############---End of Old setup_for_local_llm_response Code ---##############


def clean_text_string(text_to_be_cleaned:str) -> str:
    
    # Clean text
    # text_to_be_cleaned = text_to_be_cleaned.replace("►", "").replace("■", "").replace("▼", "")
    # text_to_be_cleaned = text_to_be_cleaned.replace("Confidential Copy \n            for \n         DKPPU", "")
    #clean_text = re.sub(r'\n(?=[a-z.])', ' ', text)     # replaces newline chars immediately followed by a small-letter or dot with a space as they're likely to be the same sentence split-up across lines.
    clean_text = re.sub(r'\n+', '\n', text_to_be_cleaned)

    # This regex substitutes anything that is not a word character or whitespace with an empty string.
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)

    # This regex substitutes any sequence of whitespace characters with a single space.
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


def prepare_prompt_for_jinja_auto_templating(formatted_prompt:str, user_query:str, current_sequence_id:int, system_prompt:str, skip_system_prompt:bool) -> str:

    print("\n\nFormatting prompt for hf-waitress\n\n")

    try:
        vision, flux_diffusers = read_config_for_hf_waitress_prompt_formatting()
    except Exception as e:
        handle_error_no_return("Could not read exl2 details from config.json / hf-config.json, encountered error: ", e)

    try:
    
        # double curly braces necessitated by Python's f-string syntax, to escape the inner curly braces in the JSON string
        if flux_diffusers:
            formatted_prompt = f'''
            {{
                "messages": [
                    {{"prompt": {json.dumps(user_query)}}}
                ]
            }}
            '''
        else:
            if current_sequence_id > 0:
                history_prompt_json = json.loads(formatted_prompt)
                new_message = {"role":"user", "content":user_query}
                history_prompt_json_without_think_tags = clean_think_tags_from_prompt(history_prompt_json)
                history_prompt_json_without_think_tags['messages'].append(new_message)
                updated_history_prompt_json = json.dumps(history_prompt_json_without_think_tags, indent=4)
                if vision:  
                    formatted_prompt = updated_history_prompt_json  # return json object
                else:
                    formatted_prompt = str(updated_history_prompt_json)
            else:   # first message in chat
                if vision:
                    formatted_prompt = {
                        "messages": [
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": user_query}
                                ]
                            }
                        ]
                    }
                    formatted_prompt = json.dumps(formatted_prompt) # Convert to a JSON string
                else:
                    if skip_system_prompt:
                        first_prompt_json = f'''
                        {{
                                "messages": [
                                    {{"role": "user", "content": {json.dumps(user_query)}}}
                                ]
                            }}
                        '''
                    else:
                        first_prompt_json = f'''
                        {{
                                "messages": [
                                    {{"role": "system", "content": {json.dumps(system_prompt)}}},
                                    {{"role": "user", "content": {json.dumps(user_query)}}}
                                ]
                            }}
                        '''                    

                    formatted_prompt = str(first_prompt_json)
    except Exception as e:
        handle_error_no_return("Could not format prompt for hf-waitress in method format-prompt_for_hf_waitress, encountered error: ", e)

    return formatted_prompt



# live whisper main backup:
if __name__ == "__main__":
    print("Starting... Generating padding audio first.")
    padding_audio = generate_padding_audio(PADDING_TEXT, sr=samplerate)

    print("Starting real-time transcription. Press Ctrl+C to stop.")
    
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()
    
    audio_buffer = np.array([], dtype=np.float32)
    last_speech_time = time.time()
    last_buffer_clear_time = time.time()

    try:
        while True:
            is_speech = False
            while not audio_queue.empty():
                chunk = audio_queue.get()
                volume_rms = np.sqrt(np.mean(chunk**2)) # Calculate the volume (Root Mean Square) of the chunk
                # print(f"RMS: {volume_rms:.4f}") # Print the volume level for debugging  - see the RMS values of your speech versus your room's silence to find the perfect value!
                
                if volume_rms > VOLUME_THRESHOLD:   # Only accumulate if above the volume threshold
                    # print("concatenated")
                    audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))
                    is_speech = True
            
            if is_speech:
                last_speech_time = time.time()

            current_time = time.time()
            # The trigger condition: buffer has content AND it's been quiet for a while since the last speech
            if len(audio_buffer) >= MIN_CHUNK_DURATION_S * samplerate and (current_time - last_speech_time) > SILENCE_DURATION_S:

                # if len(audio_buffer) >= MIN_CHUNK_DURATION_S * samplerate:
                print(f"Processing {len(audio_buffer)/samplerate:.2f}s of audio...")

                peak_volume = np.max(np.abs(audio_buffer))
                if peak_volume > 0: # normalize the spoken audio so it's not lost to the padding audio!
                    audio_buffer = audio_buffer / peak_volume

                audio_to_process = audio_buffer.copy()
                
                padding_applied = False
                if len(audio_to_process) < MIN_CONTEXT_S * samplerate:
                    print(f"Audio is shorter than {MIN_CONTEXT_S}s. Applying padding.")
                    audio_to_process = np.concatenate((audio_to_process, padding_audio))
                    padding_applied = True

                pipe = get_pipe()  # Get a fresh pipeline instance to avoid potential memory issues
                
                # Process the accumulated audio data
                result = pipe(audio_to_process, return_timestamps=True, generate_kwargs={
                    "task": "transcribe",
                    "language": "en",
                    "temperature": 0.0,        # fewer hallucinations
                    # Optional: "no_speech_threshold": 0.6, "compression_ratio_threshold": 2.4
                })
                print(f"\nRAW Result: {result}\n")
                transcription = result["text"].strip() if result else ""

                if padding_applied and transcription:
                    try:
                        #print(f"\ntranscription: {transcription}")
                        padding_start_index, padding_end_index = get_indices_of_substring(transcription.lower().strip(), start_substring="tony is quiet", end_substring="will be severely punished")
                        if padding_start_index is not None and padding_end_index is not None:
                            transcription = transcription[:padding_start_index] + transcription[padding_end_index:]
                            transcription = transcription.replace(" .", "").strip()
                            #print(f"\ntranscription after trimming: {transcription}")
                    except Exception as e:
                        print(f"Failed to trim response, encountered error: {e}")
                
                # Print the transcribed text if it's not empty
                if transcription:
                    print(f"\n\nTranscription: {transcription}\n\n")
            
                # After a SUCCESSFUL transcription, clear the buffer and reset the cleanup clock.
                audio_buffer = np.array([], dtype=np.float32)
                last_buffer_clear_time = time.time()

            # print(f"current_time - last_buffer_clear_time: {current_time - last_buffer_clear_time}")
            if (current_time - last_buffer_clear_time) > STALE_BUFFER_TIMEOUT_S and 0 < len(audio_buffer) < MIN_MEANINGFUL_SAMPLES:
                '''
                Clear the buffer and reset the cleanup clock, if there isn't at least 1 second of audio every STALE_BUFFER_TIMEOUT_S (eg 20 secs), then discard the buffer.
                `MIN_MEANINGFUL_SAMPLES` acts as a safety net: if a long sentence is being spoken, the buffer won't simply force-clear every STALE_BUFFER_TIMEOUT_S secs!
                '''
                print(f"\nDiscarding stale noise buffer of {len(audio_buffer)/samplerate:.2f}s (less than the {MIN_MEANINGFUL_SAMPLES/samplerate:.1f}s meaningful threshold).\n")
                audio_buffer = np.array([], dtype=np.float32)
                last_buffer_clear_time = time.time()
            
            # A short sleep to prevent the loop from running too fast
            time.sleep(0.05)


    except KeyboardInterrupt:
        print("\nStopping transcription.")
        stop_recording()
        recording_thread.join()