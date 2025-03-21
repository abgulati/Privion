Copy the following files from the `web_app` directory into this dir:

	1. `hf_waitress.py`: The LLM Server Application - Will be used to load the Entity & Relationship Extraction Model for Graphing
	2. `prompt_formatting.py`: Module used by `hf_waitress.py`
	3. The entire `exllamav2` dir: Used by `hf_waitress.py` to Exl2 Quantize the Model on First Load

And that's it!