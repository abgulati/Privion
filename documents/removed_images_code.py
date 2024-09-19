

def extract_images_from_pdf(pdf_path):
    
    print("Extracting Images from PDF")

    try:
        source_filename = os.path.basename(pdf_path)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)
    
    with open(pdf_path, 'rb') as file:
        
        try:
            pdf_reader = PyPDF2.PdfReader(file)
        except Exception as e:
            handle_local_error("Could not read PDF, encountered error: ", e)

        images = []

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            text = page.extract_text().strip()
            if not text:
                print("Scanned page, skipping")
                continue

            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':

                        # Log details about the image object:
                        try:
                            image_obj = xObject[obj]
                            obj_details = {
                                'Object Reference': obj,
                                'Width': image_obj.get('/Width', 'Unknown'),
                                'Height': image_obj.get('/Height', 'Unknown'),
                                'Color Space': image_obj.get('/ColorSpace', 'Unkown'),
                                'Filter': image_obj.get('/Filter', 'Unknown'),
                                'Bits Per Component': image_obj.get('/BitsPerComponent', 'Unknown')
                            }
                            #print(f"\n\nImage Object Details: {obj_details}\n\n")   # Filter is indicative of format: '/DCTDecode': 'JPEG', '/FlateDecode': 'PNG or others','/JPXDecode': 'JPEG 2000', etc.

                            # data  = image_obj._data

                            if obj_details['Filter'] == '/FlateDecode':
                                #print("\n\nDecoding PNG!\n\n")
                                try:
                                    data  = image_obj._data
                                    decompressed_data = zlib.decompress(data)
                                except Exception as e:
                                    error_message = f"\n\nPNG decompression exception: {e}\n\n"
                                    if logger:
                                        logger.error(error_message)
                                        print(error_message)
                                    else:
                                        print(error_message)
                            else:
                                decompressed_data  = image_obj._data

                            text = page.extract_text()  
                            # clean_text = text

                            # Clean text
                            clean_text = clean_text_string(text)

                            try:
                                if obj_details['Filter'] == '/FlateDecode':
                                    # Determine Color Space:
                                    color_space = image_obj.get('/ColorSpace')

                                    if color_space == '/DeviceRGB':
                                        mode = 'RGB'
                                    elif color_space == '/DeviceCMYK':
                                        mode = 'CMYK'
                                    elif color_space == '/DeviceGray':
                                        mode = 'L'
                                    else:
                                        mode = 'L'  # Default to grayscale if unsure

                                    # Create image from bytes
                                    image = Image.frombytes(mode, ((obj_details['Width']), (obj_details['Height'])), decompressed_data) # 'L' for 8-bit pixels, black and white
                                    with io.BytesIO() as output:
                                        image.save(output, format='JPEG')
                                        binary_data = output.getvalue()
                                        format = "JPEG"
                                        images.append((binary_data,clean_text,format))

                                else:
                                    # Load image from bytes
                                    image = Image.open(io.BytesIO(decompressed_data))

                                    # Determine format (JPEG)
                                    format = image.format
                                    
                                    #print(f"\n\nImage format: {format}\n\n")  # This will print the format

                                    # If image loads, append image to images DB
                                    images.append((decompressed_data,clean_text,format))

                            except Exception as e:
                                error_message = f"\n\nEncountered unrecognized or invalid image data for object detailed below. Exception: {e}\n\n"
                                if logger:
                                    logger.error(error_message)
                                    logger.error(obj_details)
                                    print(error_message)
                                    print(f"\n\nImage Object Details: {obj_details}\n\n")   # Filter is indicative of format: '/DCTDecode': 'JPEG', '/FlateDecode': 'PNG or others','/JPXDecode': 'JPEG 2000', etc.

                                else:
                                    print(error_message)
                                    print(f"\n\nImage Object Details: {obj_details}\n\n")   # Filter is indicative of format: '/DCTDecode': 'JPEG', '/FlateDecode': 'PNG or others','/JPXDecode': 'JPEG 2000', etc.


                        except Exception as e:
                            handle_error_no_return("Could not process image object. Exception: ", e)

        # print("Images array:")
        # print(images)
        return images


def store_images_to_db(images):

    print("\n\nStoring Images to Database\n\n")

    try:
        read_return = read_config(['sqlite_images_db'])
        sqlite_images_db = read_return['sqlite_images_db']
    except Exception as e:
        handle_local_error("Missing sqlite_images_db in config.json for method store_images_to_db. Error: ", e)

    try:
        conn = sqlite3.connect(sqlite_images_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to Images DB, encountered error: ", e)
    
    # If the database does not currently exist...
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    image_data BLOB NOT NULL,
                    surrounding_text TEXT,
                    metadata TEXT,
                    format TEXT,
                    document_name TEXT,
                    page_number INTEGER
            )
        ''')

        conn.commit()
    except Exception as e:
        handle_local_error("Could not create Images DB, encountered error: ", e)

    try:
        add_column_if_not_exists(cursor, 'images', 'image_data', 'BLOB')
        add_column_if_not_exists(cursor, 'images', 'surrounding_text', 'TEXT')
        add_column_if_not_exists(cursor, 'images', 'metadata', 'TEXT')
        add_column_if_not_exists(cursor, 'images', 'format', 'TEXT')
        add_column_if_not_exists(cursor, 'images', 'document_name', 'TEXT')
        add_column_if_not_exists(cursor, 'images', 'page_number', 'INTEGER')
    except Exception as e:
        return handle_api_error("Could not add necessary columns to chat history db, encountered error: ", e)
    
    try:
        for image_data, surrounding_text, format in images:
            #print("surrounding_text: ", surrounding_text)
            # Check if the image_data already exists in the database:
            cursor.execute("SELECT COUNT(*) FROM images WHERE image_data = ?", (image_data,))
            if cursor.fetchone()[0] == 0:
                print("\nInserting new image into images DB\n")
                cursor.execute("INSERT INTO images (image_data, surrounding_text, format) VALUES (?, ?, ?)", (image_data, surrounding_text, format))
        conn.commit()
    except Exception as e:
        handle_local_error("Could not store images to Images DB, encountered error: ", e)
    finally:
        conn.close()


def vector_embed_filepath(filename, filepath):
    try:
        images = extract_images_from_pdf(filepath)
    except Exception as e:
        handle_error_no_return("Failed to extract images from the PDF document, encountered error: ", e)

    try:
        store_images_to_db(images)
    except Exception as e:
        handle_error_no_return("Failed to save images to database, encountered error: ", e)



def find_images_in_db(reference_pages):

    print("\nSearching for relevant Images\n")

    try:
        read_return = read_config(['sqlite_images_db'])
        sqlite_images_db = read_return['sqlite_images_db']
    except Exception as e:
        handle_local_error("Missing sqlite_images_db in config.json for method find_images_in_db. Error: ", e)

    matched_images = []
    matched_images_found = False

    try:
        conn = sqlite3.connect(sqlite_images_db)
        conn.row_factory = sqlite3.Row
        print("Database connected for image search")
    except Exception as e:
        handle_local_error("Could not connect to images DB for image search, encountered error: ", e)

    for file_path, content in reference_pages.items():

        for item in content:
            # Each item in the list has two elements
            page_text, page_number = item

            # Only search for non-empty search strings
            if page_text:
                try:
                    images = conn.execute('SELECT DISTINCT id, image_data FROM images WHERE surrounding_text LIKE ?', ('%' + page_text[:50] + '%',)).fetchall()
                except Exception as e:
                    handle_error_no_return("Could not select images from Images DB, encountered error: ", e)
                for row in images:
                    print("Matching image found!")
                    matched_images_found = True
                    image_id = row['id']
                    image_data = row['image_data']
                    matched_images.append((image_id, image_data))
        
    conn.close()
    matched_images = set(matched_images)
    return matched_images_found, matched_images


def fetch_image_from_db(image_id):

    try:
        read_return = read_config(['sqlite_images_db'])
        sqlite_images_db = read_return['sqlite_images_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json in method fetch_image_from_db. Error: ", e)
    
    # 1 - Connect to DB
    try:
        conn = sqlite3.connect(sqlite_images_db)
    except Exception as e:
        handle_local_error("Could not connect to images database, encountered error: ", e)
    
    # 2 - Get Images
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT image_data FROM images WHERE id = ?', (image_id,)).fetchone()
        images_bytes = row['image_data'] if row else None
    except Exception as e:
        handle_local_error("Could not fetch image from DB, encountered error: ", e)
    
    conn.close()
    
    return images_bytes


@app.route('/image_display/<int:image_id>')
def image_display(image_id):
    print(f"\n\nprepping image for display: {image_id}\n\n")

    try: 
        image_bytes = fetch_image_from_db(image_id)
    except Exception as e:
        handle_local_error("Could not fetch image for display, encountered error: ", e)
    
    try:
        encoded = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        handle_local_error("Could not encode image for display URI, encountered error: ", e)

    # Return an HTML response with the embedded image:
    data_uri = f"data:image/jpeg;base64,{encoded}"
    html_content = f'<img src="{data_uri}" alt="Image">'

    return html_content


@app.route('/get_references', methods=['POST'])
def get_references():

    images_iframe_html = ""
    matched_images_found = False    #temp over-ride

    matched_images_found = False
    # try:
    #     matched_images_found, matched_images_in_bytes = find_images_in_db(reference_pages)
    # except Exception as e:
    #     handle_error_no_return("Could not search for images, encountered error: ", e)

    if matched_images_found:
        image_gallery_id = f"image_gallery_for_stream_{stream_session_id}"
        images_iframe_html = f'''
        <h6>Browse a gallery of relevant images by clicking on the thumbnail below:</h6>
        <i class="fas fa-images thumbnail-icon" onclick="openImageGalleryModal('{image_gallery_id}')"></i>
        <div id="{image_gallery_id}" class="image-gallery-modal">
        <span class="image-gallery-close" onclick="closeImageGalleryModal('{image_gallery_id}')">&times;</span>
        <div class="image-gallery-content">
        '''
        for image_id, image_bytes_data in matched_images_in_bytes:
            #print(f"\n\nmatched image id: {image_id}")
            try:
                image_link_url = url_for('image_display', image_id=image_id)
                images_iframe_html += f'<iframe src="{image_link_url}" frameborder="0" class="gallery-thumbnail"></iframe>'
            except Exception as e:
                handle_error_no_return("Could not construct images_iframe_html, encountered error: ", e)
        
        images_iframe_html += f'</div></div>'

    reference_response = refer_pages_string + images_iframe_html
