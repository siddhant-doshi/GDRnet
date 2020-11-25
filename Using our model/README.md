Demo of how to use Dr-COVID - "Using_our_model.ipynb". All the pre-loaded variables and files required to run this code can be downloaded from https://ece.iisc.ac.in/~spchepuri/sw/drug_repurpose/

Variables -- 

A_tilda.p = Adjacency matrix of our graph

input_features.p = Input feature matrix associated with the graph

nodes_mapping.p = its a dictionary - {node_name:node_id}. node_name is the actual entity name (eg - "Compound::DB01234") and node_id is it's id in the graph.

DR_model = our pre-trained model

Disease_list and Drug_list are complete list of both the entities from our graph.
