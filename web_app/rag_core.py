import privion_config_concierge as config_manager
import utils as privion_utils_module
import rag_support as rag_support_module

from sentence_transformers import SentenceTransformer, util
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, Or
from whoosh import scoring
from falkordb import FalkorDB

import chromadb
import datetime
import sqlite3
import pathlib
import torch
import uuid
import ast
import gc
import re


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self): # to provide string-representation of an object
        # return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"    # Does not truncate the actual page_content or even str(doc.page_content), rather it only comes into play for display purposes when we print the entire object as a string!
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


def init_and_connect_to_rag_context_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    try:
        read_return = config_manager.read_config(['rag_context_db'])
        rag_context_db = read_return['rag_context_db']
    except Exception as e:
        raise Exception(f"Could not read rag context db in method init_and_connect_to_rag_context_db, encountered error: {e}")
    
    try:
        conn = sqlite3.connect(rag_context_db)
        cursor = conn.cursor()
    except Exception as e:
        raise Exception(f"Could not connect to rag context db in method init_and_connect_to_rag_context_db, encountered error: {e}")
    
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_context (
                id INTEGER PRIMARY KEY,
                stream_session_id TEXT NOT NULL,
                rag_context TEXT NOT NULL
            )
        ''')
        conn.commit()
    except Exception as e:
        raise Exception(f"Could not create rag context table in method init_and_connect_to_rag_context_db, encountered error: {e}")
    
    try:
        privion_utils_module.add_column_if_not_exists(cursor, 'rag_context', 'stream_session_id', 'TEXT')
        privion_utils_module.add_column_if_not_exists(cursor, 'rag_context', 'rag_context', 'TEXT')
    except Exception as e:
        raise Exception(f"Could not add necessary columns to rag context table in method init_and_connect_to_rag_context_db, encountered error: {e}")
    
    return conn, cursor


def persist_rag_context(stream_session_id: str, docs: list[Document]):
    print(f"Persisting rag context for stream session {stream_session_id} to facilitate calls to get_references()")

    try:
        conn, cursor = init_and_connect_to_rag_context_db()
    except Exception as e:
        raise Exception(f"Could not initialize and connect to rag context db in method persist_rag_context, encountered error: {e}")
    
    try:
        cursor.execute("INSERT INTO rag_context (stream_session_id, rag_context) VALUES (?, ?)", (stream_session_id, str(docs)))
        conn.commit()
    except Exception as e:
        raise Exception(f"Could not persist rag context for stream session {stream_session_id} in method persist_rag_context, encountered error: {e}")
    finally:
        cursor.close()
        conn.close()
    
    print(f"Successfully persisted rag context for stream session {stream_session_id}")
    return True


def fetch_rag_context(stream_session_id: str, persist_in_db: bool = True) -> tuple[list[Document], bool]:
    print(f"Fetching rag context for stream session {stream_session_id}")
    
    try:
        conn, cursor = init_and_connect_to_rag_context_db()
    except Exception as e:
        raise Exception(f"Could not initialize and connect to rag context db in method fetch_rag_context, encountered error: {e}")
    
    try:
        cursor.execute("SELECT rag_context FROM rag_context WHERE stream_session_id = ?", (stream_session_id,))
        result = cursor.fetchone()
        rag_context = result[0] if result else None

        if rag_context and not persist_in_db:
            print(f"Deleting rag context for stream session {stream_session_id} from database as persist_in_db is False")
            cursor.execute("DELETE FROM rag_context WHERE stream_session_id = ?", (stream_session_id,))
            conn.commit()
    
    except Exception as e:
        raise Exception(f"Could not fetch rag context for stream session {stream_session_id} in method fetch_rag_context, encountered error: {e}")
    finally:
        cursor.close()
        conn.close()
    
    print(f"Successfully fetched rag context for stream session {stream_session_id}")
    return rag_context, rag_context is not None


def delete_rag_context(stream_session_id: str) -> bool:
    print(f"Deleting rag context for stream session {stream_session_id}")
    
    try:
        conn, cursor = init_and_connect_to_rag_context_db()
    except Exception as e:
        raise Exception(f"Could not initialize and connect to rag context db in method delete_rag_context, encountered error: {e}")
    
    try:
        cursor.execute("DELETE FROM rag_context WHERE stream_session_id = ?", (stream_session_id,))
        conn.commit()
    except Exception as e:
        raise Exception(f"Could not delete rag context for stream session {stream_session_id} in method delete_rag_context, encountered error: {e}")
    finally:
        cursor.close()
        conn.close()
    
    print(f"Successfully deleted rag context for stream session {stream_session_id}")
    return True


def extract_content_source_and_page_data_from_summary_text(summary_text: str) -> tuple[str, str, list]:
    '''
    Extracts content data, source document name and page numbers from a text string ending with the pattern:
    {Source Document Name: xxx}\n{Page Number(s): [y,z]}\n\n
    This pattern is established in the process_nodes_and_relationships method of hf_waitress.py
    
    Args:
        summary_text (str): The input text containing the metadata
    
    Returns:
        tuple[str, str, list]: (content_data, source_document_name, page_numbers_list)
    '''
    try:
        source_pattern = r'{Source Document Name: (.*?)}'   # () creates a capturing group and .*? matches any char except newline zero or more times, non-greedily
        source_match = re.search(source_pattern, summary_text)
        source_doc_name = source_match.group(1) if source_match else ""  # group(1) returns the first (and in this case, only) capturing group. 0 would return the entire match.
        
        page_pattern = r'{Page Number\(s\): \[(.*?)\]}'
        page_match = re.search(page_pattern, summary_text)
        if page_match:  # Convert string representation of list to actual list of integers
            pages_str = page_match.group(1)
            pages = [int(p.strip()) for p in pages_str.split(',')]
        else:
            pages = []
        
        content_data = summary_text[:source_match.start()].strip() if source_match else summary_text.strip()

        return content_data, source_doc_name, pages
    except Exception as e:
        print(f"Could not extract content data, source document name and page numbers from summary text, returning unchanged summary text. Encountered error: {e}")
        return summary_text, "", []


def get_summary_report(summarized_chunk_entities: dict, graph_rag_context_length_limit_chars: int, user_query: str) -> tuple[str, list[Document]]:
    print(f"\n\nGetting summary report\n\n")
    
    summary_report = set()
    summary_doc_objects = []
    
    try:
        
        for _, chunk_data in summarized_chunk_entities.items():
            source_doc_name = chunk_data['source_doc_name']
            
            if source_doc_name == 'user_query':
                print("\nSkipping user query chunk\n")
                continue
            
            try:
                for node in chunk_data['entities_and_relationships']['nodes']:

                    if not node.get('summary'):
                        continue    # Skip nodes with no summaries
                    
                    for summary in node.get('summary', []): # There may be multiple summaries for a single node, so we iterate over the list of summaries.
                        try:
                            if summary is not None and summary != '':
                                summary_preface_string = f"Summary for entity '{node['name']}' of type '{node['type']}'"

                            try:
                                content_data, source_doc_name, pages = extract_content_source_and_page_data_from_summary_text(summary)
                                source_link = f"http://llm-citations-database.net/source?doc_name={source_doc_name}&page_number={[pages[0]]}"
                                summary_doc_objects.append(Document(page_content=f"{summary_preface_string} -\nsource_link:{source_link}:\n{summary}\nsource_link:{source_link}\n\n", metadata={'page_number': pages, 'source': source_doc_name}))
                            except Exception as e:
                                print(f"Could not convert GraphRAG context to Document object, skipping. Encountered error: {e}")

                            entry = (
                                f"{summary_preface_string} - {summary}" #The summary, as generated in the process_nodes_and_relationships method of hf_waitress.py, contains metadata and newline spacing.
                            )
                            summary_report.add(entry)
                        except Exception as e:
                            print(f"Error processing a node's summary when adding to summary report. Skipping this summary. encountered error: {e}")
            
            except Exception as e:
                print(f"Error processing node in chunk_data when adding to summary report, likely a corrupt dict. Skipping node summaries for this chunk. encountered error: {e}")
            
            try:
                for relationship in chunk_data['entities_and_relationships']['relationships']:
                
                    if not relationship.get('summary'):
                        continue    # Skip relationships with no summaries

                    for summary in relationship.get('summary', []):
                        try:
                            if summary is not None and summary != '':
                                summary_preface_string = f"Summary for relationship '{relationship['relationship']}' between entities '{relationship['source']}' and '{relationship['target']}'"

                            try:
                                content_data, source_doc_name, pages = extract_content_source_and_page_data_from_summary_text(summary)
                                source_link = f"http://llm-citations-database.net/source?doc_name={source_doc_name}&page_number={[pages[0]]}"
                                summary_doc_objects.append(Document(page_content=f"{summary_preface_string} -\nsource_link:{source_link}:\n{summary}\nsource_link:{source_link}\n\n", metadata={'page_number': pages, 'source': source_doc_name}))
                            except Exception as e:
                                print(f"Could not convert GraphRAG context to Document object, skipping. Encountered error: {e}")

                                entry = (
                                    f"{summary_preface_string} - {summary}"
                                )
                                summary_report.add(entry)
                        except Exception as e:
                            print(f"Error processing a relationship's summary when adding to summary report. Skipping this summary. encountered error: {e}")
            
            except Exception as e:
                print(f"Error processing relationship in chunk_data when adding to summary report, likely a corrupt dict. Skipping relationship summaries for this chunk. encountered error: {e}")
    
    except Exception as e:
        print(f"Could not process summary report, skipping remaining items and exiting. Encountered error: {e}")
    
    textual_summary_report = ''.join(summary_report)

    if len(textual_summary_report) > graph_rag_context_length_limit_chars:
        try:
            textual_summary_report = ''
            try:
                reranked_summaries_list_ascending = rerank_results_ml(user_query, summary_doc_objects, top_n=len(summary_doc_objects))
            except Exception as e:
                print(f"Could not rerank search results, skipping. Encountered error: {e}")
                reranked_summaries_list_ascending = summary_doc_objects
            reranked_summaries_list_descending = reranked_summaries_list_ascending[::-1]    # The `rerank-results_ml` method returns a list of docs in ascending order of relevance, so we need to reverse it so we may iterate starting with the most relevant docs!
            for doc in reranked_summaries_list_descending:
                if len(textual_summary_report) + len(str(doc.page_content)) > graph_rag_context_length_limit_chars:
                    break
                textual_summary_report += str(doc.page_content)
            
            # print(f"\n\nReturning Textual summary report: {textual_summary_report}\n\n")
            return textual_summary_report, reranked_summaries_list_descending
        except Exception as e:
            raise Exception(f"Could not handle summary report that is too long, encountered error: {e}")
    else:
        return textual_summary_report, summary_doc_objects


def get_summary_and_source_documents_for_node(graph, name, node_type):
    # print(f"\nChecking if summary for node {name} of type {node_type} exists in graph\n")

    try:
        node_name = rag_support_module.sanitize_names(name)

        query = f"""
            MATCH (n:{node_name} {{name: '%s', type: '%s'}})
            RETURN n.summary AS summary, n.source_documents AS source_documents
        """ % (name.replace("'", ""), node_type.replace("'", ""))

        result = graph.query(query)
        
        if hasattr(result, 'result_set') and result.result_set:
            # print(f"\nExisting summary for node found: {result.result_set[0][0]}\n")
            summary_list = list(result.result_set[0][0]) if result.result_set[0][0] else []
            source_documents_list = list(result.result_set[0][1]) if result.result_set[0][1] else []
            return summary_list, source_documents_list
        else:
            # print(f"\nNo existing summary for node found...\n")
            return [], []   # If no summary is found, return an empty list:

    except Exception as e:
        print(f"Could not check if node {name} of type {node_type} exists in graph, returning empty list. Encountered error: {e}")
        return [], []


def get_summaries_for_all_nodes(nodes: list, graph: FalkorDB, get_source_documents: bool = False):
    nodes_with_existing_summaries = []
    processed_nodes = {}    # Will de-duplicate nodes!

    for _, node in enumerate(nodes):
        try:
            if not isinstance(node, dict):
                print(f"Skipping summary retrieval for node - Invalid Type: Expected a dict, got {type(node).__name__}")
                continue

            name = str(node.get('name', ''))
            node_type = str(node.get('type', ''))

            if name == '' or node_type == '':
                print(f"Skipping summary retrieval for node because it's missing required fields: name={name}, type={node_type}")
                continue

            node_key = (name, node_type)
            if node_key in processed_nodes:
                # print(f"Skipping duplicate node {name} of type {node_type} when checking for existing summaries in graph DB")
                continue

            try:
                existing_summary, existing_source_documents = get_summary_and_source_documents_for_node(graph, name, node_type)
            except Exception as e:
                existing_summary = []
                existing_source_documents = []
                print(f"Could not check existing summary for node {name} of type {node_type}, skipping. Encountered error: {e}")

            # update node in chunk_entities dict:
            if get_source_documents:
                nodes_with_existing_summaries.append({
                    'name': name,
                    'type': node_type,
                    'summary': existing_summary,
                    'source_documents': existing_source_documents
                })
            else:
                nodes_with_existing_summaries.append({
                    'name': name,
                    'type': node_type,
                    'summary': existing_summary
                })

            processed_nodes[node_key] = True

        except Exception as e:
            print(f"Could not get summary for node {name} of type {node_type}, skipping. Encountered error: {e}")
                    
    return nodes_with_existing_summaries


def get_summary_and_source_documents_for_relationship(graph, source, target, relationship_type):
    # print(f"\nChecking if summary for relationship {source} -> {target} ({relationship_type}) exists in graph\n")

    source_label = rag_support_module.sanitize_names(source)
    target_label = rag_support_module.sanitize_names(target)

    try:
        query = f"""
            MATCH (s:{source_label} {{name: '{source}'}})-[r:{relationship_type}]->(t:{target_label} {{name: '{target}'}})
            RETURN r.summary AS summary, r.source_documents AS source_documents
        """

        result = graph.query(query)

        if hasattr(result, 'result_set') and result.result_set:
            # print(f"\nExisting summary for relationship found: {result.result_set[0][0]}\n")
            summary_list = list(result.result_set[0][0]) if result.result_set[0][0] else []
            source_documents_list = list(result.result_set[0][1]) if result.result_set[0][1] else []
            return summary_list, source_documents_list
        else:
            # print(f"\nNo existing summary for relationship found...\n")
            return [], []   # If no summary is found, return an empty list:

    except Exception as e:
        print(f"Could not check if summary for relationship {source} -> {target} ({relationship_type}) exists in graph, returning empty list. Encountered error: {e}")
        return [], []


def get_summaries_for_all_relationships(relationships: list, graph: FalkorDB, get_source_documents: bool = False):

    relationships_with_existing_summaries = []
    processed_relationships = {}    # Will de-duplicate relationships!

    for _, relationship in enumerate(relationships):
        try:
            if not isinstance(relationship, dict):
                print(f"Skipping summary retrieval for relationship - Invalid Type: Expected a dict, got {type(relationship).__name__}")
                continue

            source = str(relationship.get('source', ''))
            target = str(relationship.get('target', ''))
            relationship_type = rag_support_module.sanitize_names(str(relationship.get('relationship', '')).upper())   # Added as sanitize_names().upper() hence formatting here too!

            if source == '' or target == '' or relationship_type == '':
                print(f"Skipping summary retrieval for relationship because it's missing required fields: source={source}, target={target}, relationship={relationship_type}")
                continue

            relationship_key = (source, target, relationship_type)
            if relationship_key in processed_relationships:
                # print(f"Skipping duplicate relationship {source} -> {target} ({relationship_type}) when checking for existing summaries in graph DB")
                continue

            try:
                existing_summary, existing_source_documents = get_summary_and_source_documents_for_relationship(graph, source.replace("'", ""), target.replace("'", ""), relationship_type)
            except Exception as e:
                existing_summary = []
                existing_source_documents = []
                print(f"Could not check existing summary for relationship {source} -> {target} ({relationship_type}), skipping. Encountered error: {e}")

            if get_source_documents:
                relationships_with_existing_summaries.append({
                    'source': source,
                    'target': target,
                    'relationship': str(relationship['relationship']),
                    'summary': existing_summary,
                    'source_documents': existing_source_documents
                })
            else:
                relationships_with_existing_summaries.append({
                    'source': source,
                    'target': target,
                    'relationship': str(relationship['relationship']),
                    'summary': existing_summary
                })

            processed_relationships[relationship_key] = True

        except Exception as e:
            print(f"Could not get summary for relationship {source} -> {target} ({relationship_type}), skipping. Encountered error: {e}")

    return relationships_with_existing_summaries


def get_summaries_from_graph_db(chunk_entities: dict, selected_knowledge_domain: str, graph: FalkorDB) -> dict:
    '''
    Receives a merged chunk_entities dict, which is the result of the merge-chunk_entities_for_graph_rag method:

    chunk_entities = {
        '0': {
            '<entities_and_relationships>': '<complete_entities_and_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_doc_name>': '<name>'
        }
    }

    And for each node and relationship in the 'entities_and_relationships' dict, checks for existing summaries in the GraphDB.
    '''

    print(f"\nStoring entities and relationships in {selected_knowledge_domain} graph DB\n")

    try:
        # Get summaries for all nodes and relationships:
        for chunk_number, chunk_data in chunk_entities.items():
            print(f"\nChecking for existing summaries for all nodes and relationships in chunk {chunk_number} of total {len(chunk_entities)} chunks...\n")

            try:
                nodes_with_existing_summaries = get_summaries_for_all_nodes(nodes=chunk_data['entities_and_relationships']['nodes'], graph=graph, get_source_documents=True)
                chunk_entities[chunk_number]['entities_and_relationships']['nodes'] = nodes_with_existing_summaries
            except Exception as e:
                print(f"Error checking for existing summaries for nodes, skipping chunk {chunk_number}. Encountered error: {e}")

            try:
                relationships_with_existing_summaries = get_summaries_for_all_relationships(relationships=chunk_data['entities_and_relationships']['relationships'], graph=graph, get_source_documents=True)
                chunk_entities[chunk_number]['entities_and_relationships']['relationships'] = relationships_with_existing_summaries
            except Exception as e:
                print(f"Error checking for existing summaries for relationships, skipping chunk {chunk_number}. Encountered error: {e}")
    
    except Exception as e:
        print(f"Could not get summaries from graph DB, encountered error: {e}")

    return chunk_entities


def merge_chunk_entities_for_graph_rag(chunk_entities: dict) -> dict:
    '''
    Receives a complete chunk_entities dict:

    chunk_entities = {
        '<graph_chunk_number_1>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_doc_name>': '<name>',
            '<page_number>': '<page_number>'
        },
        '<graph_chunk_number_2>': {
            '<entities_and_relationships>': '<node_relationships_dict>',
            '<chunk_text>': '<text>',
            '<source_doc_name>': '<name>',
            '<page_number>': '<page_number>'
        },
        ...
    }

    And returns a merged chunk_entities dict, because all entities and relationships are extracted from RAG context (user-query + semantic & lexical search results),
    and merging will allow for de-duplication of nodes and relationships in the get_summary step, which is all we need for querying the GraphDB: nodes & relationships.

    NOTE: Check docstring in assemble-chunks_for_graph_rag() for more details on the specific keys present! The 'page_number' key is not merged as that would be useless!
    '''

    # print(f"\n\nMerging chunk entities for graph RAG. Received chunk_entities: \n {chunk_entities}\n\n")
    print(f"\nMerging chunk entities for graph RAG.\n")

    chunk_entities_merged = {0: {
        'chunk_text': '',
        'entities_and_relationships': {
            'nodes': [],
            'relationships': []
        },
        'source_doc_name': ''
    }}
        
    for _, chunk_data in chunk_entities.items():
        try:
            chunk_entities_merged[0]['chunk_text'] += f"{chunk_data['chunk_text']} "
            chunk_entities_merged[0]['entities_and_relationships']['nodes'].extend(chunk_data['entities_and_relationships']['nodes'])   # extend() is used to add multiple elements to the end of the list...
            chunk_entities_merged[0]['entities_and_relationships']['relationships'].extend(chunk_data['entities_and_relationships']['relationships'])   # and we don't care about de-duplicating here as that'll happen anyways in the get_summaries step!
            chunk_entities_merged[0]['source_doc_name'] += f"{chunk_data['source_doc_name']} "

        except Exception as e:
            print(f"Could not merge chunk entities for graph RAG, proceeding with original chunk_entities dict. WARNING: Duplicates may be present, negatively impacting the context window! Encountered error: {e}")

    return chunk_entities_merged


def assemble_chunks_for_graph_rag(docs:list[Document], user_query:str=None) -> dict:
    '''
    Transforms docs, which is a list of Document objects:

        docs = [
            Document(
                page_content = '<page_content>',
                metadata = {
                    'source_link': '<source_link>',
                    'source': '<source_filepath>',
                    'page_number': '<page_number>',
                    'entities_and_relationships': '<entities_and_relationships>'
                }
            ),
            ...
        ]

    into:

        chunk_entities = {
            '<graph_chunk_number>': {
                'chunk_text': '<page_content>',
                'source_doc_name': '<source_filepath>',
                'page_number': '<page_number>',
                'entities_and_relationships': '<entities_and_relationships>'
            },
            ...
        }
    
    for the purposes of GraphRAG's query-response pipeline.
    
    NOTE: While 'source_doc_name' and 'page_number' data is being added here, it's unnecessary for the GraphRAG query-response pipeline, because this data is directly
    obtained from the GraphDB itself at a later step: In get-summaries_from_graph_db(), the get-summaries_for_all_nodes() and get-summaries_for_all_relationships()
    methods are used to obtain the 'source_doc_name' and 'page_number' data for each node and relationship respectively, by setting get_source_documents=True.

    The main source doc and page number data isobtained from the summary in the GraphDB, as summaries always end with the following pattern:

        {Source Document Name: AMD_Q4_and_FY_24_EarningsRelease_FINAL}{Page Number(s): [8]}  # For example...
    
    In fact, 'chunk_text' is also unnecessary as only the nodes and relationships are needed for GraphRAG, not the actual text!
    So they're all simply added here incase this data proves useful for some future use-case!
    '''
    try:
        chunk_entities = {}
        graph_chunk_count = 1   # Same init as in convert-doc_chunks_to_graph_entities()

        if user_query is not None:  # For GraphRAG response query-pipeline, we need to add the user query as a chunk
            user_query = user_query.replace("'", "").replace("<br>", "").replace("?", "")
            user_query_chunk_text = f"Do not attempt to answer any query that follows, simply proceed to extract nodes and relationships from the following text:\n{user_query}"
            chunk_entities[graph_chunk_count] = {
                'chunk_text': user_query_chunk_text,
                'source_doc_name': 'user_query'
            }
            graph_chunk_count += 1

        print("\nGenerating Graphing Chunks Dictionary...\n")
        
        for count, doc in enumerate(docs):

            try:
                source_filename = pathlib.Path(rf"{str(doc.metadata.get('source'))}").resolve().name

                page_number_list = []
                try:    # page numbers while useful are non-essential which is why I'm wrapping in a dedicated try-except block that does not raise an error!
                    page_number_list.append(int(doc.metadata.get('page_number')))
                    page_number_list = list(set(page_number_list))   # Remove duplicates
                except Exception as e:
                    print(f"Could not obtain page number from context document number {count} of {len(docs)} documents, skipping. Encountered error: {e}")

                chunk_entities[graph_chunk_count] = {
                    'chunk_text': str(doc.page_content).strip().replace("'", ""),
                    'source_doc_name': source_filename,
                    'page_number': page_number_list,
                    'entities_and_relationships': ast.literal_eval(doc.metadata.get('entities_and_relationships', '{}'))
                }   # check note in the docstring for more details on which keys are added here and why!
                graph_chunk_count += 1
            except Exception as e:
                print(f"Error processing context document number {count} of {len(docs)} documents in assemble-chunks_for_graph_db(), encountered error: {e}")

    except Exception as e:
        raise Exception(f"Could not assemble chunk_entities dictionary for GraphRAG, encountered error: {e}")

    return chunk_entities


def execute_graph_rag(user_query:str, docs_with_graph_entities: list[Document]) -> tuple[str, list[Document]]:
    '''
    Assembles document chunks into a dictionary of entities via convert-doc_chunks_to_graph_entities(); check append-graph_entities_to_chunks() for detailed
    documentation on the structure of docs and chunk_entities.

    These chunk_entities are then passed to the graphing model which will process each graph_chunk and append the `entities_and_relationships` key 
    to each chunk_entities dict:
        
        '<entities_and_relationships>': {"nodes": [{"type": "organization","name": "Intel"},{"type": "object","name": "Intel Products"},...], 
        "relationships": [{"source": "Intel","target": "Intel Products","relationship": "business unit"},...]}

    The various chunk_entities in the dict are then merged into a singular chunk_entity for querying the GraphDB to obtain summaries (as we're only interested
    in a de-duplicated list of nodes & relationships for GraphDB-queries). 
    The merge-chunk_entities_for_graph_rag and get-summaries_from_graph_db methods are respectively used for this purpose.

    The obtained summaries are deduplicated and formatted into a summary report via get-summary_report(), and finally re-ranked and trimmed to obtain the
    final graphRAG context, which is then returned.
    '''
    
    print(f"\n\nExecuting GraphRAG. Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    try:
        rag_support_module.bring_graph_db_online()
    except Exception as e:
        raise Exception(f"Could not bring graph DB or graphing model online, encountered error: {e}")
    
    try:
        selected_knowledge_domain = config_manager.read_config(['selected_knowledge_domain'])['selected_knowledge_domain']    
        client = rag_support_module.get_graph_db_client()
        graph = client.select_graph(selected_knowledge_domain)  # Will create the graph if it doesn't exist
    except Exception as e:
        raise Exception(f"Could not connect to / initialize graph for '{selected_knowledge_domain}' domain in graph DB, encountered error: {e}")

    try:
        complete_chunk_entities = assemble_chunks_for_graph_rag(docs_with_graph_entities, user_query=None)
        print(f"\n\nlen of complete_chunk_entities: \n {len(complete_chunk_entities.items())}\n\n")
        # for item in list(complete_chunk_entities.items()):
        #     print(f"\n\n{item}\n\n")
    except Exception as e:
        raise Exception(f"Could not assemble chunks for graph DB, encountered error: {e}")

    try:
        merged_graph_rag_entities_and_relationships_dict = merge_chunk_entities_for_graph_rag(complete_chunk_entities)
        print(f"\n\nlen of merged_graph_rag_entities_and_relationships_dict: \n {len(merged_graph_rag_entities_and_relationships_dict)}\n\n")
    except Exception as e:
        raise Exception(f"Fatal error merging chunk entities for GraphRAG: {e}")

    try:
        summarized_and_deduplicated_chunk_entities = get_summaries_from_graph_db(merged_graph_rag_entities_and_relationships_dict, selected_knowledge_domain, graph)
    except Exception as e:
        raise Exception(f"Could not fetch summaries for entities and relationships from graph DB, encountered error: {e}")

    try:
        graph_rag_context_length_limit_chars = int(config_manager.read_config(['graph_rag_context_length_limit_chars'])['graph_rag_context_length_limit_chars'])
        summary_report, reranked_summaries_list_descending = get_summary_report(summarized_and_deduplicated_chunk_entities, graph_rag_context_length_limit_chars, user_query)
    except Exception as e:
        raise Exception(f"Could not get summary report, encountered error: {e}")

    return summary_report, reranked_summaries_list_descending


def map_graph_entities_to_filtered_docs(combined_docs:list[Document], graph_entities_map:dict) -> list[Document]:
    print("\n\nMapping graph entities to filtered docs\n\n")
    for doc in combined_docs:
        try:
            doc.metadata['entities_and_relationships'] = graph_entities_map[doc.metadata['unique_id']]
        except Exception as e:
            print(f"Could not map graph entities to doc, skipping. Encountered error: {e}")
            doc.metadata['entities_and_relationships'] = {}
    return combined_docs


def rerank_results_ml(query:str, documents:list[Document], top_n:int=5) -> list[Document]:
    print("\n\nReranking Invoked\n\n")

    try:
        read_return = config_manager.read_config(['use_embedding_model_for_reranking', 'selected_embedding_model', 'selected_reranker_model'])
        use_embedding_model_for_reranking = str(read_return['use_embedding_model_for_reranking']).lower() == 'true'
        selected_embedding_model = str(read_return['selected_embedding_model'])
        selected_reranker_model = str(read_return['selected_reranker_model'])
    except Exception as e:
        use_embedding_model_for_reranking = True
        print(f"Could not read reranker configfrom config.json, encountered error: {e}")

    if use_embedding_model_for_reranking:
        selected_reranker_model = selected_embedding_model
    else:
        selected_reranker_model = selected_reranker_model

    print(f"\n\nSelected model for re-ranking: {selected_reranker_model}\n\n")

    model = None
    try:
        # Load pre-trained SBERT model
        model = SentenceTransformer(selected_reranker_model)
        
        # Encode the query
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Encode the documents
        doc_embeddings = model.encode([doc.page_content for doc in documents], convert_to_tensor=True)
    except Exception as e:
        raise Exception(f"Could not rerank results with {selected_reranker_model}, encountered error: {e}")
        return [doc.page_content for doc in documents]
    finally:
        if model is not None:
            del model
            if torch.cuda.is_available():
                print("Emptying CUDA cache")
                torch.cuda.empty_cache()
            print("Collecting garbage")
            gc.collect()

    try:
        # Compute cosine similarities
        cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    except Exception as e:
        raise Exception(f"Could not compute cosine similarities, encountered error: {e}")
        return [doc.page_content for doc in documents]
    
    try:
        # Create a list of (index, score) tuples
        indexed_scores = list(enumerate(cosine_scores))
        
        # Sort by score in descending order
        sorted_indexes = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        
        # Reorder the original documents based on the sorted indexes
        ranked_documents = [documents[idx] for idx, _ in sorted_indexes[:top_n]]

        # print(f"\n\nReturning Top {len(ranked_documents)} Ranked Documents: {ranked_documents}\n\n")

        '''
        Studies show that LLMs and Transformers in-general tend to perform better when the most relevant context is towards the beginning or end of the input, while important context in between tends to get 'lost in the middle'! 
        This can be a serious problem for a large multi-turn conversation, wherein extensive back-and-forth query-response history exists and grows with each prompt. 
        Therefore, the re-ranker method has been modified below to return a reversed context docs list, placing the most relevant docs at the end, so the list is now in ascending order of relevance. 
        This should be helpful right from query 1 especially when the system prompt is large!
        '''
        return ranked_documents[::-1]   #Slice to reverse the list, as `.reverse()` would return None because it creates an inplace change on the original list without returning anything

    except Exception as e:
        print(f"Could not reorder documents, encountered error: {e}")
        return [doc.page_content for doc in documents]



def combine_and_deduplicate_search_results(whoosh_results:list[dict], vector_results:list[Document]) -> tuple[list[Document], dict]:
    print("\n\nCombining whoosh and vector results\n\n")

    combined_results = []
    graph_entities_map = {}

    # Convert whoosh results to Document objects
    for result in whoosh_results:
        temp_unique_id = str(uuid.uuid4())
        combined_results.append(Document(
            page_content=result['content'].strip().replace('\n', ' '),
            metadata={
                'source_link': result['source_link'],
                'source': result['source'],
                'page_number': result['page_number'],
                'unique_id': temp_unique_id
            }
        ))
        graph_entities_map[temp_unique_id] = result['entities_and_relationships']

    # Add the vector results to the combined results - Unfortunately can't do in a single elegant line as we're also deciding whether to include graph entities or not!
    # combined_results.extend(vector_results)   # To keep this unchanged, we'd have to re-search the vectorDB to include or exclude graph entities, which is way worse!
    for doc in vector_results:
        temp_unique_id = str(uuid.uuid4())
        combined_results.append(Document(
            page_content=doc.page_content,
            metadata={
                'source_link': doc.metadata['source_link'],
                'source': doc.metadata['source'],
                'page_number': doc.metadata['page_number'],
                'unique_id': temp_unique_id
            }
        ))
        graph_entities_map[temp_unique_id] = doc.metadata['entities_and_relationships']

    # Filter out any duplicate documents based on page_content
    try:
        seen = {}
        unique_results = []
        for doc in combined_results:
            if doc.page_content not in seen:
                seen[doc.page_content] = True
                unique_results.append(doc)

        combined_results = unique_results
    except Exception as e:
        print(f"Could not filter out duplicate documents in method combine-and_deduplicate_search_results. Returning all results. Encountered error: {e}")
    
    return combined_results, graph_entities_map


def search_whoosh_index(query:str) -> list[dict]:

    print("Searching Whoosh Index")
    
    try:
        read_return = config_manager.read_config(['fetch_top_k_results_from_whoosh', 'whoosh_search_weighting', 'min_lexical_similarity_threshold'])
        fetch_top_k_results_from_whoosh = int(read_return['fetch_top_k_results_from_whoosh'])
        whoosh_search_weighting = read_return['whoosh_search_weighting']
        min_lexical_similarity_threshold = float(read_return['min_lexical_similarity_threshold'])  # Like semantic search with ChromaDB, higher scores indicate better matches but the range with Whoosh is different!
    except Exception as e:
        raise Exception(f"Missing whoosh config in config.json for method search-whoosh_index. Error: {e}")

    try:
        whoosh_index_folder = rag_support_module.determine_whoosh_index_folder()
    except Exception as e:
        raise Exception(f"Failed to determine Whoosh Index Folder, encountered error: {e}")

    try:
        ix = rag_support_module.get_whoosh_index_object_for_folder(whoosh_index_folder)
    except Exception as e:
        raise Exception(f"Failed to get Whoosh Index Object, encountered error: {e}")

    whoosh_weighting = scoring.BM25F()  # Rough ranges: 0.0: No Match; 1-2: Weak Match; 3-5: Moderate Match; 6+: Strong Match
    if whoosh_search_weighting == "TF-IDF":
        whoosh_weighting = scoring.TF_IDF()  # Rough ranges: 0.0: No Match; 1-4: Weak Match; 5-10: Moderate Match; 10+: Strong Match
    
    try:
        with ix.searcher(weighting=whoosh_weighting) as searcher:
            query_parser = QueryParser("content", schema=ix.schema, group=OrGroup)
            parsed_query = query_parser.parse(query)

            results = searcher.search(parsed_query, limit=fetch_top_k_results_from_whoosh)
            print(f"Whoosh Results: Number of results: {len(results)}")

            # Filter by score threshold
            filtered_results = [
                {
                    'content': result['content'],
                    'source_link': result['source_link'],
                    'source': result['source'],
                    'page_number': result['page_number'],
                    'entities_and_relationships': result['entities_and_relationships'],
                    'score': result.score
                }
                for result in results
                if result.score >= min_lexical_similarity_threshold
            ]
            print(f"Whoosh Results:Number of results after filtering by score threshold {min_lexical_similarity_threshold}: {len(filtered_results)}")

            # If no results, let's try a more lenient search:
            if len(filtered_results) == 0:
                print("No lexical results found after filtering by score threshold, trying a more lenient search...")
                terms = [Term("content", word) for word in query.lower().split()]
                or_query = Or(terms)
                lenient_results = searcher.search(or_query, limit=fetch_top_k_results_from_whoosh)
                print(f"number of results after very lenient search: {len(lenient_results)}")

                filtered_results = [
                    {
                        'content': result['content'],
                        'source_link': result['source_link'],
                        'source': result['source'],
                        'page_number': result['page_number'],
                        'entities_and_relationships': result['entities_and_relationships'],
                        'score': result.score
                    }
                    for result in lenient_results
                    if result.score >= min_lexical_similarity_threshold
                ]
                print(f"Whoosh Results: Number of results after filtering by score threshold {min_lexical_similarity_threshold} in very lenient search: {len(filtered_results)}")

            return filtered_results
            
            # return [{'content': result['content'], 'source': result['source'], 'page_number': result['page_number']} for result in results]

    except Exception as e:
        print(f"Failed to search Whoosh Index, encountered error: {e}")
        return []


def search_vector_db(user_query:str, embedding_function:str, fetch_top_k_results_from_vectordb: int) -> list[Document]:
    print("Searching vectorDB")

    min_semantic_similarity_threshold = float(config_manager.read_config(['min_semantic_similarity_threshold'])['min_semantic_similarity_threshold'])

    path_to_knowledge_domain = rag_support_module.get_path_to_knowledge_domain()
    vector_db_path = rag_support_module.create_vector_db_directory(path_to_knowledge_domain, embedding_function)

    print(f"Searching Knowledge Domain: {path_to_knowledge_domain} with embedding function: {embedding_function}")

     # Load Embedding Model
    embedding_model = None
    try:
        embedding_model = SentenceTransformer(embedding_function, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load embedding model for searching the vector database, encountered error: {e}")

    try:
        # Initialize Chroma Client and collection
        chroma_client = chromadb.PersistentClient(path=str(vector_db_path), settings=chromadb.Settings(allow_reset=True))
        collection = chroma_client.get_or_create_collection(name="knowledge_domain", metadata={"hnsw:space": "cosine"}) # By default, ChromaDB returns the L2 distance (lower is better), but we want cosine distance (higher is better)

        query_embedding = embedding_model.encode(user_query)

        # Perform the semantic search - 'results' is a dictionary with keys 'documents', 'metadatas', 'distances', whose values are lists of length = fetch_top_k_results_from_vectordb
        results = collection.query(
            query_embeddings=query_embedding.tolist(),  # Convert embeddings from NumPy arrays to list of lists
            n_results=fetch_top_k_results_from_vectordb,    # top-k here implies the top from the matched set, regardless of the actual similarity score!
            include=["documents", "metadatas", "distances"]
        )

        # Format similar to LangChain's Output so as to maintain consistency
        docs_list_with_cosine_distance = [
            (
                Document(
                    page_content=doc,
                    metadata=metadata
                ),
                distance
            )
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
            if distance >= min_semantic_similarity_threshold    # ChromaDB's score ranges from -1 (perfect dissimilarity) to 1 (perfect similarity), with 0.0 meaning no similarity.
        ]   # The zip() function combines multiple iterables (lists, tuples, etc.) element by element and helps iterate over multiple lists simultaneously

        print(f"Result of Semantic Search: Found {len(docs_list_with_cosine_distance)} documents of {len(results['documents'][0])} with a minimum semantic similarity threshold of {min_semantic_similarity_threshold}")
        return docs_list_with_cosine_distance
    except Exception as e:
        print(f"Could not perform similarity_search to determine do_rag when attempting to setup_for_streaming_response, encountered error: {e}")
        return []
    finally:
        if embedding_model is not None:
            del embedding_model
            if torch.cuda.is_available():
                print("Emptying CUDA cache")
                torch.cuda.empty_cache()
            print("Collecting garbage")
            gc.collect()


def legacy_execute_search_tools_on_query(
        user_query:str, 
        embedding_function:str, 
        llm_set_config:dict, 
        filter_top_k_results_by_reranking:int, 
        fetch_top_k_results_from_vectordb:int
    ) -> tuple[list[Document], bool]:
    print("Searching knowledge base")

    if not llm_set_config.get('do_rag', True) and not llm_set_config.get('perform_graph_rag', False):
        print("No RAG or GraphRAG to perform, returning...")
        return [], False, None

    filtered_docs = []
    try:
        docs_list_with_cosine_distance = search_vector_db(user_query, embedding_function, int(fetch_top_k_results_from_vectordb))
        filtered_docs = [doc for doc, score in docs_list_with_cosine_distance]  # the `doc, score` is crucial, as it ensure we select only the Document object, and not a tuple comprising of both the Document object and a float score!
    except Exception as e:
        print(f"Could not perform vector search to determine do_rag when attempting to search-knowledge-base, encountered error: {e}")

    whoosh_results = []
    try:
        whoosh_results = search_whoosh_index(user_query)
    except Exception as e:
        print(f"Could not perform whoosh search to determine do_rag when attempting to search-knowledge-base, encountered error: {e}")

    combined_docs = []
    try:
        combined_docs, graph_entities_map = combine_and_deduplicate_search_results(whoosh_results, filtered_docs)   # Combine the whoosh and vector results
    except Exception as e:
        print(f"Could not combine and deduplicate search results, skipping. Encountered error: {e}")
        combined_docs = filtered_docs

    if not combined_docs:   # i.e. if blank
        print("No documents for citations, returning...")
        return [], False, None

    try:
        docs = rerank_results_ml(user_query, combined_docs, top_n=filter_top_k_results_by_reranking)
    except Exception as e:
        print(f"Could not rerank search results, skipping. Encountered error: {e}")
        docs = combined_docs
        
    perform_graph_rag = llm_set_config.get('perform_graph_rag', False)
    enable_graph_rag = config_manager.read_config(['enable_graph_rag'])['enable_graph_rag']

    graph_rag_context = None
    if perform_graph_rag and llm_set_config.get('do_rag', True) and enable_graph_rag:   # All conditions must be met for GraphRAG to be performed!
        try:
            docs_with_graph_entities = map_graph_entities_to_filtered_docs(docs, graph_entities_map)
            graph_rag_context, reranked_summaries_list_descending = execute_graph_rag(user_query, docs_with_graph_entities)
            if reranked_summaries_list_descending != []:
                return reranked_summaries_list_descending, llm_set_config.get('do_rag', True), graph_rag_context
        except Exception as e:
            print(f"Could not execute graph RAG, encountered error: {e}")
    else:
        config_manager.safe_write_config({'perform_graph_rag': False})  # In-case the LLM elected to use GraphRAG but the user has explicitly disabled it, we need to set perform-graph_rag to False to avoid any issues downstream!

    return docs, llm_set_config.get('do_rag', True), graph_rag_context


def execute_full_search(query:str, stream_session_id: str = None) -> dict:
    print("Executing full search")

    try:
        rag_config = config_manager.read_config([
            'selected_embedding_model', 'filter_top_k_results_by_reranking', 
            'fetch_top_k_results_from_vectordb', 'enable_graph_rag',
            'force_disable_rag', 'enable_graph_rag'
        ])
    except Exception as e:
        raise Exception(f"Could not read rag config in method execute_full_search, encountered error: {e}")
    
    if rag_config['force_disable_rag']:
        print("Force disable rag is True, returning...")
        return {'success': False, 'message': 'RAG is Forcefully Disabled', 'docs': [], 'do_rag': False, 'graph_rag_context': None}
    
    filtered_docs = []
    try:
        docs_list_with_cosine_distance = search_vector_db(query, rag_config['selected_embedding_model'], int(rag_config['fetch_top_k_results_from_vectordb']))
        filtered_docs = [doc for doc, score in docs_list_with_cosine_distance]  # the `doc, score` is crucial, as it ensure we select only the Document object, and not a tuple comprising of both the Document object and a float score!
    except Exception as e:
        print(f"Could not perform vector search to determine do_rag when attempting to search-knowledge-base, encountered error: {e}")

    whoosh_results = []
    try:
        whoosh_results = search_whoosh_index(query)
    except Exception as e:
        print(f"Could not perform whoosh search to determine do_rag when attempting to search-knowledge-base, encountered error: {e}")

    combined_docs = []
    try:
        combined_docs, graph_entities_map = combine_and_deduplicate_search_results(whoosh_results, filtered_docs)   # Combine the whoosh and vector results
    except Exception as e:
        print(f"Could not combine and deduplicate search results, skipping. Encountered error: {e}")
        combined_docs = filtered_docs

    if not combined_docs:   # i.e. if blank
        print("No documents for citations, returning...")
        return {'success': False, 'message': 'No citation data found for query', 'docs': [], 'do_rag': False, 'graph_rag_context': None}

    try:
        docs = rerank_results_ml(query, combined_docs, top_n=int(rag_config['filter_top_k_results_by_reranking']))
    except Exception as e:
        print(f"Could not rerank search results, skipping. Encountered error: {e}")
        docs = combined_docs

    if rag_config['enable_graph_rag']:
        try:
            docs_with_graph_entities = map_graph_entities_to_filtered_docs(docs, graph_entities_map)
            graph_rag_context, reranked_summaries_list_descending = execute_graph_rag(query, docs_with_graph_entities)
            if len(reranked_summaries_list_descending) > 0:
                persist_rag_context(stream_session_id, reranked_summaries_list_descending)
                return {'success': True, 'message': graph_rag_context, 'docs': reranked_summaries_list_descending, 'do_rag': True, 'graph_rag_context': graph_rag_context}
        except Exception as e:
            print(f"Could not execute graph RAG, encountered error: {e}")
    else:
        persist_rag_context(stream_session_id, docs)
        return {'success': True, 'message': docs, 'docs': str(docs), 'do_rag': True, 'graph_rag_context': None}
    

