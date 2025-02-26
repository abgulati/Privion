"""
Graph clustering algorithms for use with FalkorDB - This module implements community detection algorithms.
Since FalkorDB does not support clustering out of the box, we will need to custom implement them!
Neo4j does support quite a few with the GDS (Graph Data Science) library! Even if switching in the future, this module remains useful for reference!
"""

import os
import datetime


def apply_leiden_clustering(client, knowledge_domain: str, resolution: float = 1.0, beta: float = 0.01, n_iterations: int = 2) -> bool:
    """
    Apply the Leiden clustering algorithm to identify communities in the knowledge graph.

    Args:
        knowledge_domain (str): The name of the knowledge domain to cluster.
        resolution (float): The resolution parameter for Leiden clustering (higher values lead to more communities).
        beta (float): The beta parameter for Leiden clustering (controls the randomness of the clustering).
        n_iterations (int): The number of iterations for Leiden clustering.

    Returns:
        bool: True if clustering was applied successfully, False otherwise.
    """

    print(f"\n\nApplying Leiden clustering to knowledge domain: {knowledge_domain}\n\n")

    try:    # check dependencies first
        import igraph as ig
        from leidenalg import find_partition
        from leidenalg import ModularityVertexPartition
    except ImportError as e:
        print("Could not import required dependencies for Leiden clustering, encountered error: ", e)
        return False
    
    try:
        # Add a lock file to prevent concurrent clustering
        lock_file = f"clustering_{knowledge_domain}.lock"
        if os.path.exists(lock_file):
            print(f"Leiden clustering for {knowledge_domain} is already in progress. Please wait for it to complete.")
            return False

        with open(lock_file, 'w') as f:
            f.write(f"Leiden clustering for {knowledge_domain} is in progress. Operation started at {datetime.datetime.now().isoformat()}\n")

        try:
            print("Beginning Leiden clustering...")

            # Connect to the GraphDB and get the graph:
            graph = client.select_graph(knowledge_domain)

            # Estimate graph size:
            node_count_operation = graph.query("MATCH (n) RETURN count(n) as count")
            if hasattr(node_count_operation, 'result_set') and node_count_operation.result_set:
                node_count = node_count_operation.result_set[0][0]
            else:
                print("No nodes found in the graph")
                return False
            print(f"Number of nodes in the GraphDB: {node_count}")

            if node_count > 100000:
                print(f"Warning: Large graph with {node_count} nodes detected. This may require significant memory and time to process.")
            
            # Stream-load the graph in batches to reduce memory usage:
            batch_size = min(5000, max(1000, node_count // 10)) # Pick the smaller of 5000 or 10% of the nodes, or 1000 if less than 1000 nodes

            # Process in streaming fashion by creating an igraph object, which is a more memory-efficient way to handle large graphs:
            G = ig.Graph(directed=True) 
            """
            Relationships are directed in nature (e.g., "Person A" → "works at" → "Company B") so we load them as such to preserve the original directed structure of the graph.
            However for clustering, community detection algorithms like Leiden, we generally want to treat the graph as undirected because:
                1. They're designed for undirected graphs with symmetric relationships.
                2. When clustering, we're typically interested in which nodes are densely connected to each other, regardless of the direction of those connections.
                3. The clustering results are more consistent and meaningful when the graph is undirected.
                4. The computational complexity is lower for undirected graphs.
            That's why we'll convert it to undirected before applying the actual Leiden algorithm via `G_undirected = G.as_undirected()` below!
            """

            node_map = {}   # Map FalkorDB IDs to igraph indices

            # Process Nodes First: Populate the igraph object with nodes from the graph associated with the knowledge domain:
            offset = 0
            while True:
                nodes_query = f"MATCH (n) RETURN ID(n) AS id, n.name AS name, n.type AS type SKIP {offset} LIMIT {batch_size}"
                result = graph.query(nodes_query)

                if not result.result_set:
                    break

                # Get the current number of nodes in the igraph object to determine the start_idx for this iteration:
                start_idx = len(node_map)

                # The node IDs, names, and types for this batch:
                batch_ids = []
                batch_names = []
                batch_types = []

                for i, record in enumerate(result.result_set):
                    node_id = int(record[0])
                    name = record[1] if record[1] is not None else ""
                    node_type = record[2] if record[2] is not None else ""

                    batch_ids.append(node_id)
                    batch_names.append(name)
                    batch_types.append(node_type)
                    node_map[node_id] = start_idx + i   # Map the FalkorDB ID to the igraph index via the node_map dictionary

                G.add_vertices(len(batch_ids))  # Add the vertices (nodes) to the igraph object:

                idx_range = range(start_idx, start_idx + len(batch_ids))     # Set the indices for the vertices (nodes) in the igraph object. range() is used to create a sequence of integers.
                """
                Since idx_range is a sequence of integers, we can use it to set the attributes for each vertex (node) in the igraph object by using the G.vs[] syntax,
                wherein vs is short for "vertex sequence" and allows us to access and set the attributes of each vertex (node) in the igraph object.
                This is a more efficient way to set the attributes for multiple vertices (nodes) at once, rather than setting them one by one by iterating: G.vs[i]['name'] = batch_names[i]
                """
                G.vs[idx_range]['name'] = batch_names                        # Set the name for each vertex (node)
                G.vs[idx_range]['type'] = batch_types                        # Set the type for each vertex (node)
                G.vs[idx_range]['original_id'] = batch_ids                  # Set the original ID for each vertex (node)

                offset += len(result.result_set)
                print(f"Processed {offset}/{node_count} nodes")

                if len(result.result_set) < batch_size: # Fewer results than batch size means we've reached the end of the graph
                    break

            # Now Process Edges (Relationships) Next, again in batch fashion:
            edge_count_operation = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            if hasattr(edge_count_operation, 'result_set') and edge_count_operation.result_set:
                edge_count = edge_count_operation.result_set[0][0]
            else:
                print("No relationships found in the graph")
                edge_count = 0
            print(f"Number of edges(relationships) in the GraphDB: {edge_count}")

            edges = []
            offset = 0
            batch_size = min(10000, max(1000, edge_count // 5)) # Pick the smaller of 10000 or 5% of the edges, or 1000 if less than 1000 edges

            while True:
                edges_query = f"MATCH (s)-[r]->(t) RETURN ID(s) AS source, ID(t) AS target, TYPE(r) AS relationship SKIP {offset} LIMIT {batch_size}"
                result = graph.query(edges_query)

                if not result.result_set:
                    break

                batch_edges = []    # relationships to the processed in this batch
                for record in result.result_set:
                    source_id = int(record[0])
                    target_id = int(record[1])
                    if source_id in node_map and target_id in node_map:
                        batch_edges.append((node_map[source_id], node_map[target_id]))

                edges.extend(batch_edges)
                offset += len(result.result_set)
                print(f"Processed {offset}/{edge_count} edges(relationships)")

                if len(result.result_set) < batch_size:
                    break
            
            print(f"Total edges collected: {len(edges)}")
            G.add_edges(edges)
            print(f"Created igraph with {len(G.vs)} nodes and {len(G.get_edgelist())} edges(relationships)")

            # Run Leiden clustering algorithm with configurable parameters:
            print("Running Leiden clustering...")
            G_undirected = G.as_undirected()    # Converting to undirected as explained above!

            partition = find_partition(
                G_undirected,
                ModularityVertexPartition,  # `partition_type` parameter (optional); using this because it's the default and most commonly used partition type for community detection.
                #resolution_parameter = resolution,
                # beta = beta,
                n_iterations = n_iterations,
                seed = 42   # 42 because it's the answer to the ultimate question of life, the universe, and everything!
            )

            community_count = len(set(partition.membership))
            print(f"Identified {community_count} communities in the graph")

            # Done! Time to update the database!
            batch_size = 1000
            for i in range(0, len(G.vs), batch_size):
                batch_end = min(i + batch_size, len(G.vs))  # There may not be as many nodes as the batch size, so we use the minimum of the batch size and the number of nodes!

                # Prepare batch updates:
                updates = []
                for j in range(i, batch_end):
                    node_id = int(G.vs[j]['original_id'])
                    cluster_id = partition.membership[j]
                    updates.append(f"MATCH (n) WHERE ID(n) = {node_id} SET n.cluster = {cluster_id}")

                # Execute updates:
                for update in updates:
                    try:
                        graph.query(update)
                    except Exception as e:
                        print(f"Error updating cluster for node: {update}")
                        print(f"Error: {e}")

                print(f"Updated clusters for nodes {i+1}-{batch_end} of {len(G.vs)}")

            # Add metadata:
            graph.query(f"""
                MERGE (meta:ClusterMetadata {{algorithm: 'Leiden'}})
                SET meta.communityCount = {community_count},
                    meta.resolution = {resolution},
                    meta.beta = {beta},
                    meta.nIterations = {n_iterations},
                    meta.timestamp = '{datetime.datetime.now().isoformat()}'
            """)

            print("\n\nLeiden clustering completed successfully!\n\n")
            return True
            
        finally:
            if os.path.exists(lock_file):
                os.remove(lock_file)

    except Exception as e:
        print(f"Could not apply Leiden clustering to knowledge domain: {knowledge_domain}, encountered error: ", e)
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except:
            print(f"Could not remove lock file for knowledge domain: {knowledge_domain}, please remove manually.")
        return False
