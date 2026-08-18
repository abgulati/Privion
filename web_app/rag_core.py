import privion_config_concierge as config_manager
import utils as privion_utils_module
import rag_support as rag_support_module

from sentence_transformers import SentenceTransformer, util
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, Or
from whoosh import scoring
from falkordb import FalkorDB

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from protego import Protego
import trafilatura

import requests
import chromadb
import datetime
import sqlite3
import pathlib
import torch
import time
import uuid
import ast
import gc
import re


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        '''Provides string-representation of an object'''
        # return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


def init_and_connect_to_rag_context_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    try:
        config = config_manager.read_config(['rag_context_db'])
        rag_context_db = config['rag_context_db']
    except Exception as e:
        raise Exception(f"Missing config value for rag_context_db: {e}")
    
    try:
        conn = sqlite3.connect(rag_context_db)
        cursor = conn.cursor()
    except Exception as e:
        raise Exception(f"Could not connect to rag context database: {e}")
    
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
        raise Exception(f"Could not create rag context table: {e}")
    
    try:
        privion_utils_module.add_column_if_not_exists(
            cursor,
            'rag_context',
            'stream_session_id',
            'TEXT'
        )
        privion_utils_module.add_column_if_not_exists(
            cursor,
            'rag_context',
            'rag_context',
            'TEXT'
        )
   
    except Exception as e:
        raise Exception(f"Could not add necessary columns to rag context table: {e}")
    
    return conn, cursor


def persist_rag_context(stream_session_id: str, docs: list[Document]):
    print(
        "Persisting rag context for sessionID "
        f"{stream_session_id} to facilitate calls to get_references()"
    )

    try:
        conn, cursor = init_and_connect_to_rag_context_db()
    except Exception as e:
        raise Exception(f"Could not initialize and connect to rag context database: {e}")
    
    try:
        cursor.execute(
            "INSERT INTO rag_context (stream_session_id, rag_context) VALUES (?, ?)",
            (stream_session_id, str(docs))
        )
   
        conn.commit()
    except Exception as e:
        raise Exception(
            "Could not persist rag context for stream session "
            f"{stream_session_id}: {e}"
        )
    finally:
        cursor.close()
        conn.close()
    
    print(f"Successfully persisted rag context for sessionID: {stream_session_id}")
    return True


def fetch_rag_context(
    stream_session_id: str, 
    persist_in_db: bool = True
) -> tuple[list[Document], bool]:
    print(f"Fetching rag context for sessionID: {stream_session_id}")
    
    try:
        conn, cursor = init_and_connect_to_rag_context_db()
    except Exception as e:
        raise Exception(f"Could not initialize and connect to rag context database: {e}")
    
    try:
        cursor.execute(
            "SELECT rag_context FROM rag_context WHERE stream_session_id = ?",
            (stream_session_id,)
        )
   
        result = cursor.fetchone()
        rag_context = result[0] if result else None

        if rag_context and not persist_in_db:
            print(
                "Deleting rag context for sessionID "
                f"{stream_session_id} from database as persist_in_db is False"
            )
            cursor.execute(
                "DELETE FROM rag_context WHERE stream_session_id = ?", 
                (stream_session_id,)
            )
       
            conn.commit()
    
    except Exception as e:
        raise Exception(f"Could not fetch rag context for sessionID: {stream_session_id}: {e}")
    finally:
        cursor.close()
        conn.close()
    
    print(f"Successfully fetched rag context for sessionID: {stream_session_id}")
    return rag_context, rag_context is not None


def delete_rag_context(stream_session_id: str) -> bool:
    print(f"Deleting rag context for sessionID: {stream_session_id}")
    
    try:
        conn, cursor = init_and_connect_to_rag_context_db()
    except Exception as e:
        raise Exception(f"Could not initialize and connect to rag context database: {e}")
    
    try:
        cursor.execute(
            "DELETE FROM rag_context WHERE stream_session_id = ?",
            (stream_session_id,)
        )
   
        conn.commit()
    except Exception as e:
        raise Exception(f"Could not delete rag context for sessionID: {stream_session_id}: {e}")
    finally:
        cursor.close()
        conn.close()
    
    print(f"Successfully deleted rag context for sessionID: {stream_session_id}")
    return True


def extract_content_source_and_page_data_from_summary_text(
    summary_text: str
) -> tuple[str, str, list]:
    '''
    Extracts content data, source document name and page numbers 
    from a text string ending with the pattern:
    `{Source Document Name: xxx}\n{Page Number(s): [y,z]}\n\n`
    
    This pattern is established in the `process_nodes_and_relationships()` 
    method of `hf_waitress.py`.
    
    Args:
        summary_text (str): The input text containing the metadata
    
    Returns:
        tuple[str, str, list]: (
            content_data, source_document_name, page_numbers_list
        )
    '''
    try:
        source_pattern = r'{Source Document Name: (.*?)}'
        '''
        () creates a capturing group and .*? matches any char 
        except newline zero or more times, non-greedily
        '''
        
        source_match = re.search(source_pattern, summary_text)
        source_doc_name = source_match.group(1) if source_match else ""
        '''
        group(1) returns the first (and in this case, only) 
        capturing group. 0 would return the entire match.
        '''

        page_pattern = r'{Page Number\(s\): \[(.*?)\]}'
        page_match = re.search(page_pattern, summary_text)
        if page_match:
            '''
            Convert a string representation of a list of 
            integers to an actual list of integers
            '''
            pages_str = page_match.group(1)
            if pages_str:
                pages = [
                    int(p.strip())
                    for p in pages_str.split(',')
                ]
            else:
                pages = [1]
       
        else:
            pages = [1]
        
        if source_match:
            content_data = summary_text[:source_match.start()].strip()
        else:
            content_data = summary_text.strip()
   

        return content_data, source_doc_name, pages
    except Exception as e:
        print(
            "Could not extract content data, source document name "
            "and page numbers from summary text, returning unchanged "
            f"summary text. Encountered error: {e}"
        )
        return summary_text, "", [1]


def get_summary_report(summarized_chunk_entities: dict, graph_rag_context_length_limit_chars: int, user_query: str) -> tuple[str, list[Document]]:
    '''
    Receives a dictionary of chunk entities complete with summaries for the comprising nodes and relationships, 
    the graph RAG context length limit in characters, and the user query.

    The user query is used to rerank the summaries if the provided limit is exceeded.

    Returns a summary report string and a list of Document objects, comprising the summaries as Document objects, which
    are assembled as follows:

        1. A summary-preface-string is constructed basis the node/relationship entities details
        2. For a given summary, the source-doc-name and page number(s) are extracted from the summary text.
        3. A source-link is constructed using the source-doc-name and page number(s)
        4. An entry is formatted as follows:
            {Summary Preface String} -
            source_link:{source_link}:
            {Summary Text}
            source_link:{source_link}
            \n\n
        5. The entry is added to the summary-report set and the summary-doc-objects list, with exceptions handled via a simplified entry fallback.

    The summary-report set is used to ensure that the same summary is not added multiple times to the summary report.
    The summary-doc-objects list is used to store the summaries as Document objects, which are then used to rerank the summaries if the provided limit is exceeded.

    '''
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
                            if not summary:
                                continue
                            
                            summary_preface_string = f"Summary for entity '{node['name']}' of type '{node['type']}'"

                            try:
                                content_data, source_doc_name, pages = extract_content_source_and_page_data_from_summary_text(summary)
                                source_link = f"http://llm-citations-database.net/source?doc_name={source_doc_name}&page_number={[pages[0]]}"
                                page_content_entry = f"{summary_preface_string} -\nsource_link:{source_link}:\n{summary}\nsource_link:{source_link}\n\n"
                                
                                summary_doc_objects.append(Document(page_content=page_content_entry, metadata={'page_number': pages, 'source': source_doc_name}))
                            
                            except Exception as e:
                                print(f"Could not convert GraphRAG context to Document object, skipping. Encountered error: {e}")
                                page_content_entry = (
                                    f"{summary_preface_string} - {summary}" #The summary, as generated in the process_nodes_and_relationships method of hf_waitress.py, contains metadata and newline spacing.
                                )
                            
                            summary_report.add(page_content_entry)
                        
                        except Exception as e:
                            print(f"Error processing a node's summary when adding to summary report. Skipping this summary. encountered error: {e}")
                            continue
            
            except Exception as e:
                print(f"Error processing node in chunk_data when adding to summary report, likely a corrupt dict. Skipping node summaries for this chunk. encountered error: {e}")
            
            try:
                for relationship in chunk_data['entities_and_relationships']['relationships']:
                
                    if not relationship.get('summary'):
                        continue    # Skip relationships with no summaries

                    for summary in relationship.get('summary', []):
                        try:
                            if not summary:
                                continue
                            
                            summary_preface_string = f"Summary for relationship '{relationship['relationship']}' between entities '{relationship['source']}' and '{relationship['target']}'"

                            try:
                                content_data, source_doc_name, pages = extract_content_source_and_page_data_from_summary_text(summary)
                                source_link = f"http://llm-citations-database.net/source?doc_name={source_doc_name}&page_number={[pages[0]]}"
                                page_content_entry = f"{summary_preface_string} -\nsource_link:{source_link}:\n{summary}\nsource_link:{source_link}\n\n"
                                
                                summary_doc_objects.append(Document(page_content=page_content_entry, metadata={'page_number': pages, 'source': source_doc_name}))
                            
                            except Exception as e:
                                print(f"Could not convert GraphRAG context to Document object, skipping. Encountered error: {e}")
                                page_content_entry = (
                                    f"{summary_preface_string} - {summary}"
                                )
                            
                            summary_report.add(page_content_entry)
                        
                        except Exception as e:
                            print(f"Error processing a relationship's summary when adding to summary report. Skipping this summary. encountered error: {e}")
                            continue
            
            except Exception as e:
                print(f"Error processing relationship in chunk_data when adding to summary report, likely a corrupt dict. Skipping relationship summaries for this chunk. encountered error: {e}")
    
    except Exception as e:
        print(f"Could not process summary report, skipping remaining items and exiting. Encountered error: {e}")
    
    textual_summary_report = ''.join(summary_report)

    if len(textual_summary_report) > graph_rag_context_length_limit_chars:
        trimmed_summary_report = ''
        try:
            
            reranked_summaries_list = rerank_results_ml(user_query, summary_doc_objects, top_n=len(summary_doc_objects), ascending_relevance_sort=False)
            
            for doc in reranked_summaries_list:
                
                if len(trimmed_summary_report) + len(str(doc.page_content)) > graph_rag_context_length_limit_chars:
                    break
                trimmed_summary_report += str(doc.page_content)
            
            return trimmed_summary_report, reranked_summaries_list

        except Exception as e:
            print(f"Could not rerank & trim GraphRAG summaries, returning original summary report. Encountered error: {e}")

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
        print(f"Could not get summaries from graph DB: {e}")

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

    The main source doc and page number data is obtained from the summary in the GraphDB, as summaries always end with the following pattern:

        {Source Document Name: AMD_Q4_and_FY_24_EarningsRelease_FINAL}{Page Number(s): [8]}  # For example...
    
    In fact, 'chunk_text' is also unnecessary as only the nodes and relationships are needed for GraphRAG, not the actual text!
    So they're all simply added here incase this data proves useful for some future downstream tasks.
    '''
    try:
        chunk_entities = {}
        graph_chunk_count = 1   # Same init as in convert-doc_chunks_to_graph_entities()

        if user_query is not None:  # For legacy-GraphRAG response query-pipeline, we need to add the user query as a chunk
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
                print(f"Error processing context document number {count} of {len(docs)} documents in assemble-chunks_for_graph_db(): {e}")

    except Exception as e:
        raise Exception(f"Could not assemble chunk_entities dictionary for GraphRAG: {e}")

    return chunk_entities


def execute_graph_rag(user_query:str, docs_with_graph_entities: list[Document]) -> tuple[str, list[Document]]:
    '''
    Receives a list of Document objects, comprising combined RAG results from local and Web sources, all containing 
    graph 'entities_and_relationships' in their metadata (set to empty dict for WebRAG / if no graph data is present).

    Will use append-graph_entities_to_chunks() to transform the list of Document objects into a dictionary of chunk_entities, 
    (see that method for detailed documentation on the structure of docs and chunk_entities), and then merges them into a 
    singular chunk_entity for querying the GraphDB to obtain summaries via merge-chunk_entities_for_graph_rag()
    (as we're only interested in the total list of nodes & relationships for GraphDB-queries).

    Inline comments in the merge method explain that deduplication of the list of nodes & relationships occurs naturally later. 
    
    Finally, the merged chunk_entities are used to fetch summaries from the GraphDB via the get-summary methods, and the summaries 
    are deduplicated and formatted into a summary report via get-summary_report().

    Returns: A summary report string and a list combining the summaries and Web Documents as Document objects.
    See method get-summary_report() for detailed documentation on the structure of the summary report and the list of Document objects.
    '''
    
    print(f"\n\nExecuting GraphRAG. Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # --- 1. SEPARATE WEB DOCS FROM GRAPH DOCS ---
    graph_docs = []
    web_docs = []

    for doc in docs_with_graph_entities:
        if doc.metadata.get('entities_and_relationships'):
            graph_docs.append(doc)
        else:   # blank dict will result in a False check, and the else block will execute
            web_docs.append(doc)

    print(f"Graph Pipeline: Processing {len(graph_docs)} Graph Docs and preserving {len(web_docs)} Web Docs.")

     # --- 2. SETUP GRAPH DB ---
    try:
        rag_support_module.bring_graph_db_online()
    except Exception as e:
        raise Exception(f"Could not bring graph DB or graphing model online: {e}")
    
    try:
        selected_knowledge_domain = config_manager.read_config(['selected_knowledge_domain'])['selected_knowledge_domain']    
        client = rag_support_module.get_graph_db_client()
        graph = client.select_graph(selected_knowledge_domain)  # Will create the graph if it doesn't exist
    except Exception as e:
        raise Exception(f"Could not connect to / initialize graph for '{selected_knowledge_domain}' domain in graph DB: {e}")

    # --- 3. EXECUTE GRAPH PIPELINE (ON GRAPH DOCS ONLY) ---
    summaries_list = []
    summary_report = ""

    if graph_docs:
        try:
            complete_chunk_entities = assemble_chunks_for_graph_rag(graph_docs, user_query=None)
            print(f"\n\nlen of complete_chunk_entities: \n {len(complete_chunk_entities.items())}\n\n")
            # for item in list(complete_chunk_entities.items()):
            #     print(f"\n\n{item}\n\n")
        except Exception as e:
            raise Exception(f"Could not assemble chunks for graph DB: {e}")

        try:
            merged_graph_rag_entities_and_relationships_dict = merge_chunk_entities_for_graph_rag(complete_chunk_entities)
            print(f"\n\nlen of merged_graph_rag_entities_and_relationships_dict: \n {len(merged_graph_rag_entities_and_relationships_dict)}\n\n")
        except Exception as e:
            raise Exception(f"Fatal error merging chunk entities for GraphRAG: {e}")

        try:
            summarized_and_deduplicated_chunk_entities = get_summaries_from_graph_db(merged_graph_rag_entities_and_relationships_dict, selected_knowledge_domain, graph)
        except Exception as e:
            raise Exception(f"Could not fetch summaries for entities and relationships from graph DB: {e}")

        try:
            graph_rag_context_length_limit_chars = int(config_manager.read_config(['graph_rag_context_length_limit_chars'])['graph_rag_context_length_limit_chars'])
            summary_report, summaries_list = get_summary_report(summarized_and_deduplicated_chunk_entities, graph_rag_context_length_limit_chars, user_query)
        except Exception as e:
            raise Exception(f"Could not get summary report: {e}")
    
    else:
        print("No Graph Docs found (Web Search only).")

    # --- 4. MERGE WEB CONTENT BACK IN ---
    
    # A. Append Web Content to the Text Report (String)
    if web_docs:
        summary_report += "\n\n" + "="*20 + "\n EXTERNAL WEB SEARCH RESULTS \n" + "="*20 + "\n\n"
        '''
        multiplying a string ("=" * 20) repeats it.
        So, that one line of code injects this exact block of text into your prompt:
        ====================
        EXTERNAL WEB SEARCH RESULTS 
        ====================
        '''
        for doc in web_docs:
            summary_report += f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            summary_report += f"source_link: {doc.metadata.get('source_link', '')}\n"
            summary_report += f"Content: {doc.page_content}\n\n"

    # B. Merge Document Lists
    # If both are empty, the returned list will be blank and execute-full_search will fallback to non-Graph docs list!
    final_docs_list = summaries_list + web_docs
    return summary_report, final_docs_list


def map_graph_entities_to_filtered_docs(combined_docs:list[Document], graph_entities_map:dict) -> list[Document]:
    '''
    Maps graph entities to the filtered docs, by assigning the graph entities to the metadata of the corresponding Document object.
    '''
    print("\n\nMapping graph entities to filtered docs\n\n")
    for doc in combined_docs:
        if doc.metadata.get('unique_id') and doc.metadata['unique_id'] in graph_entities_map:
            doc.metadata['entities_and_relationships'] = graph_entities_map[doc.metadata['unique_id']]
        else:
            # It is a Web Doc OR a Local Doc with no graph data -> Set empty dict
            doc.metadata['entities_and_relationships'] = {}
    return combined_docs


def rerank_results_ml(query:str, documents:list[Document], top_n:int=5, ascending_relevance_sort:bool=True) -> list[Document]:
    print("\n\nReranking Invoked\n\n")

    # 1. Safety Check: Don't crash if search returned nothing
    if not documents:
        print("No documents to rerank.")
        return []

    try:
        read_return = config_manager.read_config(
            [
                'use_embedding_model_for_reranking',
                'selected_embedding_model',
                'selected_reranker_model',
                'reranker_torch_device',
                'embedding_torch_device'
            ]
        )
        
        use_embedding_model_for_reranking = str(read_return['use_embedding_model_for_reranking']).lower() == 'true'
        selected_embedding_model = str(read_return['selected_embedding_model'])
        selected_reranker_model = str(read_return['selected_reranker_model'])
        reranker_torch_device = str(read_return['reranker_torch_device'])
        embedding_torch_device = str(read_return['embedding_torch_device'])
    except Exception as e:
        print(f"Could not read reranker config, using defaults. Error: {e}")
        # FALLBACKS defined here so the code doesn't crash later
        use_embedding_model_for_reranking = True 
        selected_embedding_model = 'all-MiniLM-L6-v2' 
        selected_reranker_model = 'all-MiniLM-L6-v2'
        reranker_torch_device = 'cpu'
        embedding_torch_device = 'cpu'
    
    if use_embedding_model_for_reranking:
        selected_reranker_model = selected_embedding_model
        reranker_torch_device = embedding_torch_device

    print(f"\n\nSelected model for re-ranking: {selected_reranker_model}\n\n")

    model = None
    try:
        model = SentenceTransformer(selected_reranker_model, trust_remote_code=True, device=reranker_torch_device)
        
        ## Bi-Encoder Logic (do NOT use cross-encoder Re-Ranking models here, stick to embedding models only for now!)
        query_embedding = model.encode(query, convert_to_tensor=True)
        doc_contents = [doc.page_content for doc in documents]
        doc_embeddings = model.encode(doc_contents, convert_to_tensor=True)
    
        cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

        ## Attach score to metadata and sort

        # We zip the documents with their scores, as opposed to enumerate `list(enumerate(cosine_scores))`,
        # which would pair each score with an index (0, 1, 2, etc.), which is not what we want!
        doc_score_pairs = list(zip(documents, cosine_scores))

        # Sort by score descending (sort paired items)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Slice top_n
        top_pairs = doc_score_pairs[:top_n]

        # Update metadata with scores (zip version is extensible and allows additions as below:)
        for doc, score in top_pairs:
            doc.metadata['relevance_score'] = float(score)

        # Extract just the docs
        top_docs = [doc for doc, score in top_pairs]

        if ascending_relevance_sort:
            # "Lost in the Middle" Optimization (Reverse list)
            # This puts the highest score LAST (closest to the LLM generation prompt)
            return top_docs[::-1]   
            # Slice to reverse the list, as `.reverse()` would return None since it 
            # creates an inplace change on the original list without returning anything!
        else:
            return top_docs

    except Exception as e:
        print(f"Reranking failed: {e}")
        return documents[:top_n]

    # finally:
    #     # if model is not None:
    #     #     del model
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     gc.collect()


def combine_and_deduplicate_search_results(whoosh_results:list[dict], vector_results:list[Document]) -> tuple[list[Document], dict]:
    '''
    Combines whoosh and vector results into a single list of Document objects, while building a dictionary of graph entities.
    A unique ID is assigned to the metadata of each Document object, which is also used as a key in the graph_entities_map to
    map graph 'entities_and_relationships' to the corresponding Document object.
    We keep the graph entities separate like this so as to keep the Document object itself lightweight, as it may be appended as 
    RAG context to the user prompt. These entities are only needed to extract summaries from the GraphDB IF GraphRAG is enabled,
    and including them in the Document object would serve no purpose except to bloat the RAG context unnecessarily.
    '''
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
        raise Exception(f"Failed to determine Whoosh Index Folder: {e}")

    try:
        ix = rag_support_module.get_whoosh_index_object_for_folder(whoosh_index_folder)
    except Exception as e:
        raise Exception(f"Failed to get Whoosh Index Object: {e}")

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
        print(f"Failed to search Whoosh Index: {e}")
        return []


def search_vector_db(user_query:str, embedding_function:str, fetch_top_k_results_from_vectordb: int) -> list[Document]:
    print("Searching vectorDB")

    torch_device = config_manager.read_config(['embedding_torch_device'])['embedding_torch_device']

    min_semantic_similarity_threshold = float(config_manager.read_config(['min_semantic_similarity_threshold'])['min_semantic_similarity_threshold'])

    path_to_knowledge_domain = rag_support_module.get_path_to_knowledge_domain()
    vector_db_path = rag_support_module.create_vector_db_directory(path_to_knowledge_domain, embedding_function)

    print(f"Searching Knowledge Domain: {path_to_knowledge_domain} with embedding function: {embedding_function}")

     # Load Embedding Model
    embedding_model = None
    try:
        embedding_model = SentenceTransformer(embedding_function, trust_remote_code=True, device=torch_device)
    except Exception as e:
        print(f"Could not load embedding model for searching the vector database: {e}")

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
        print(f"Could not perform similarity_search to determine do_rag when attempting to setup_for_streaming_response: {e}")
        return []
    # finally:
    #     # if embedding_model is not None:
    #     #     del embedding_model
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     gc.collect()


def search_searxng(user_query:str, category:str='general') -> list[Document]:
    '''
    Searches the web for information using SearXNG.
    '''
    try:

        print("\n\nSearching Web via local SearXNG server...\n\n")

        searxng_config = config_manager.read_config(['assign_host_port_to_searxng_server'])

        url = f"http://localhost:{searxng_config['assign_host_port_to_searxng_server']}/search"

        params = {
            'q': user_query,
            'format': 'json',
            'categories': category,
            'language': 'auto'
        }
        response = requests.get(url, params=params, timeout=20)
        results = response.json().get('results', [])

        if not results:
            return []

        documents = []
        print(f"Scraping {len(results)} URLs...")
        for res in results:

            try:
                snippet = res.get('content', '')

                if snippet:
                    # Create Document object - Does NOT contain unique_id and page_number, the latter for obvious reasons &
                    # the lack of the former is handled by map-graph_entities_to_filtered_docs() later.
                    doc = Document(
                        page_content=snippet,
                        metadata={
                            'source_link': res['url'],
                            'source': res['engine'],
                            'title': res['title'],  # instead of 'page_number'
                            'score': 0.0 # Placeholder - instead of 'unique_id'
                        }
                    )
                    documents.append(doc)
            except:
                continue

        print(f"Scraped {len(documents)} documents from SearXNG")
        return documents

    except Exception as e:
        print(f"Could not search SearXNG: {e}")
        return []


def _fetch_webpage_fallback(
    url: str,
    user_agent: str,
    max_chars: int = 11000,
    timeout: int = 10
) -> dict:
    
    '''Fallback using requests and BeautifulSoup.'''
    
    try:

        headers = {
            'User-agent': user_agent
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove boilerplate
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                            'aside', 'advertisement', 'ad', 'iframe', 'noscript']):
            element.decompose()
        
        # Find main content
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '.post', 
                        '#content', '#main', '.article']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        text_source = main_content if main_content else soup.body

        if not text_source:
            raise Exception("Could not parse content with bs4.")

        text = text_source.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines if line.strip()]
        text = '\n'.join(lines)

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated to top {max_chars} from total {len(text)} chars.]"

        print(f"\n\nSuccessfully extracted main content from URL using backup method (BS4): {url}\n\n{text}\n\n")

        return {
            'success': True,
            'message': f"Successfully extracted main content from URL: {url}\n\n{text}",
            'content': text,
            'url': url
        }

    except Exception as e:
        return {
            "success": False, 
            "message": f"Error processing URL: {url}. Extraction via trafilatura and backup bs4 failed. Error: {str(e)}"
        } 


def _get_robots_parser_protego(url: str, timeout: int = 10) -> bool:
    '''Checks robots.txt for a given url'''
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        response = requests.get(robots_url, timeout=timeout)
        if response.status_code == 200:
            return Protego.parse(response.text)
        return None # No robots.txt = allow all
    except Exception:
        return None # Network error = assume allowed


def fetch_webpage(url: str, **kwargs) -> dict:
    '''
    Fetches and extracts the main content from a webpage.
    Respects robots.txt before fetching
    '''

    # 0. Read config
    config_read = config_manager.read_config([
        'respect_robots', 
        'max_url_context_chars',
        'fetch_url_timeout_seconds'
    ])
    timeout = config_read['fetch_url_timeout_seconds']
    max_chars = int(config_read['max_url_context_chars'])

    # 1. Check robots.txt first
    user_agent = "Privion-LE-Research-Bot/1.0"

    if config_read['respect_robots']:
        parser = _get_robots_parser_protego(url, timeout)

        if parser and not parser.can_fetch(user_agent, url):
            return {"success": False, "message": f"Access to {url} denied by robots.txt"}

    # 2. Rate limiting - be polite!
    time.sleep(0.5)  # 500ms delay between requests

    # 3. Proceed with content extraction (same as before)
    try:

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise Exception(f"Could not download page content.")

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            deduplicate=True,
            output_format="markdown"
        )

        if not text:
            raise Exception(f"No meaningful content found.")

        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated to top {max_chars} from total {len(text)} chars.]"

        print(f"\n\nSuccessfully extracted main content from URL using Trafilatura: {url}\n\n{text}\n\n")

        return {
            'success': True,
            'message': f"Successfully extracted main content from URL: {url}\n\n{text}",
            'content': text,
            'url': url
        }

    except requests.exceptions.Timeout:
        return {"success": False, "message": f"Request timed out for URL: {url}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Failed to fetch URL: {url}. Error: {str(e)}"}
    except Exception as e:
        return _fetch_webpage_fallback(
            url, user_agent, max_chars, timeout
        )


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
        print(f"Could not perform vector search to determine do_rag when attempting to search-knowledge-base: {e}")

    whoosh_results = []
    try:
        whoosh_results = search_whoosh_index(user_query)
    except Exception as e:
        print(f"Could not perform whoosh search to determine do_rag when attempting to search-knowledge-base: {e}")

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
            graph_rag_context, summaries_list = execute_graph_rag(user_query, docs_with_graph_entities)
            if len(summaries_list) > 0:
                return summaries_list, llm_set_config.get('do_rag', True), graph_rag_context
        except Exception as e:
            print(f"Could not execute graph RAG: {e}")
    else:
        config_manager.safe_write_config({'perform_graph_rag': False})  # In-case the LLM elected to use GraphRAG but the user has explicitly disabled it, we need to set perform-graph_rag to False to avoid any issues downstream!

    return docs, llm_set_config.get('do_rag', True), graph_rag_context


def execute_full_search(query:str, category:str, stream_session_id: str = None) -> dict:
    print("Executing full search")

    try:
        rag_config = config_manager.read_config([
            'selected_embedding_model', 'filter_top_k_results_by_reranking', 
            'fetch_top_k_results_from_vectordb', 'enable_graph_rag',
            'force_disable_rag', 'enable_web_search'
        ])
    except Exception as e:
        raise Exception(f"Could not read rag config in method execute-full_search: {e}")
    
    if rag_config['force_disable_rag']:
        print("Force disable rag is True, returning...")
        return {'success': False, 'message': 'RAG is Forcefully Disabled', 'docs': [], 'do_rag': False, 'graph_rag_context': None}
    
    filtered_docs = []
    try:
        docs_list_with_cosine_distance = search_vector_db(query, rag_config['selected_embedding_model'], int(rag_config['fetch_top_k_results_from_vectordb']))
        filtered_docs = [doc for doc, score in docs_list_with_cosine_distance]  # the `doc, score` is crucial, as it ensure we select only the Document object, and not a tuple comprising of both the Document object and a float score!
    except Exception as e:
        print(f"Could not perform vector search to determine do_rag when attempting to search-knowledge-base: {e}")

    whoosh_results = []
    try:
        whoosh_results = search_whoosh_index(query)
    except Exception as e:
        print(f"Could not perform whoosh search to determine do_rag when attempting to search-knowledge-base: {e}")

    combined_docs = []
    graph_entities_map = {}
    try:
        combined_docs, graph_entities_map = combine_and_deduplicate_search_results(whoosh_results, filtered_docs)   # Combine the whoosh and vector results
    except Exception as e:
        print(f"Could not combine and deduplicate search results, skipping. Encountered error: {e}")
        combined_docs = filtered_docs

    web_docs = []
    if rag_config['enable_web_search']:
        try:
            print("Attempting Web Search via SearXNG...")
            web_docs = search_searxng(query, category)
            print(f"Web search returned {len(web_docs)} documents")
        except Exception as e:
            print(f"Could not perform web search: {e}")

    # We merge AFTER Document-RAG deduplication so we don't break existing GraphRAG logic, but BEFORE reranking so the AI can choose the best content.
    if web_docs:
        combined_docs.extend(web_docs)

    # --- Safety Check ---
    if not combined_docs:   # i.e. if blank
        print("No documents for citations, returning...")
        return {'success': False, 'message': 'No citation data found for query', 'docs': [], 'do_rag': False, 'graph_rag_context': None}

    docs = []
    try:
        docs = rerank_results_ml(query, combined_docs, top_n=int(rag_config['filter_top_k_results_by_reranking']))
    except Exception as e:
        print(f"Could not rerank search results, skipping. Encountered error: {e}")
        docs = combined_docs

    if rag_config['enable_graph_rag']:
        
        try:
            
            docs_with_graph_entities = map_graph_entities_to_filtered_docs(docs, graph_entities_map)
            graph_rag_context, summaries_list = execute_graph_rag(query, docs_with_graph_entities)
            
            if len(summaries_list) > 0 and len(graph_rag_context) > 0:  # GraphRAG results actually exist!
                persist_rag_context(stream_session_id, summaries_list)
                return {'success': True, 'message': graph_rag_context, 'docs': summaries_list, 'do_rag': True, 'graph_rag_context': graph_rag_context}                
        
        except Exception as e:
            print(f"Could not execute graph RAG: {e}")
    
    else:
        persist_rag_context(stream_session_id, docs)
    
    return {'success': True, 'message': docs, 'docs': str(docs), 'do_rag': True, 'graph_rag_context': None}
