MATCH (n)-[r]->(m) RETURN n,r,m

// View all nodes with their cluster assignments
MATCH (n) 
RETURN n.name, n.type, n.cluster
ORDER BY n.cluster

// View the graph with nodes colored by cluster
MATCH (n)-[r]->(m)
RETURN n, r, m

// Get count of nodes:
MATCH (n) RETURN count(n)

// View a specific cluster and its connections
MATCH (n)-[r]->(m)
WHERE n.cluster = 1
RETURN n, r, m
// Change this number to view different clusters

// In case the above fails
MATCH (n)-[r]->(m)
WHERE n.cluster = 1
RETURN n.name, n.type, TYPE(r), m.name, m.type

// Specific Query Syntax:
MATCH (n {name: 'Tremblant', type: 'location'})
RETURN n.summary AS summary, n.source_documents AS source_documents

// View connections between clusters
MATCH (n)-[r]->(m)
WHERE n.cluster <> m.cluster
RETURN n, r, m
// Shows only edges that cross between communities

// Get statistics about cluster sizes
MATCH (n)
RETURN n.cluster AS cluster, count(*) AS size
ORDER BY size DESC

// To get the total count of clusters in your graph
MATCH (n)
WHERE n.cluster IS NOT NULL
RETURN COUNT(DISTINCT n.cluster) AS total_clusters
// 1. Matches all nodes that have a cluster property assigned
// 2. Counts the distinct cluster values
// 3. Returns the total number of unique clusters

// Get more detailed information about your clusters
MATCH (n)
WHERE n.cluster IS NOT NULL
RETURN n.cluster AS cluster_id, COUNT(*) AS node_count
ORDER BY node_count DESC
// Will show each cluster ID and how many nodes belong to that cluster
// Sorted from largest to smallest cluster

// Find the largest or smallest clusters

// Largest cluster
MATCH (n)
WHERE n.cluster IS NOT NULL
WITH n.cluster AS cluster_id, COUNT(*) AS node_count
ORDER BY node_count DESC
LIMIT 1
RETURN cluster_id, node_count

// Smallest cluster
MATCH (n)
WHERE n.cluster IS NOT NULL
WITH n.cluster AS cluster_id, COUNT(*) AS node_count
ORDER BY node_count ASC
LIMIT 1
RETURN cluster_id, node_count

// Get all nodes in a specific cluster
MATCH (n)
WHERE n.cluster = 1
RETURN n.name, n.type, ID(n)
ORDER BY n.type, n.name
// Replace with the cluster ID you want to examine

        # Eg: MATCH (n:intel {name: 'Intel', type: 'organization'}) RETURN n.summary AS summary
        # MATCH (n:intel_foundry {name: 'Intel Foundry', type: 'object'}) RETURN n.summary AS summary


// To query for relationships with weight > 1, you can use:
MATCH (s)-[r]->(t)
WHERE r.weight > 1
RETURN s.name, type(r), t.name, r.weight
ORDER BY r.weight DESC

// If you want to see the source documents for these frequent relationships too, you can add r.source_documents to the RETURN clause:
MATCH (s)-[r]->(t)
WHERE r.weight > 1
RETURN s.name, type(r), t.name, r.weight, r.source_documents
ORDER BY r.weight DESC