"""Basic smoke tests for the LaMAGIC repository.

These tests exercise the self-contained, dependency-light graph utilities in
``topo_data_util`` so that a basic sanity check can be run without the heavy
training stack (PyTorch, transformers, a GPU, model checkpoints, etc.).

Run with::

    pytest tests/test_topo_graph.py
"""

from topo_data_util.topo_analysis.graphUtils import (
    adj_matrix_to_edges,
    adj_matrix_to_graph,
    graph_to_adjacency_matrix,
    indexed_graph_to_adjacency_matrix,
    nodes_and_edges_to_adjacency_matrix,
)
from topo_data_util.topo_analysis.topoGraph import HyperEdge, TopoGraph


def test_indexed_graph_to_adjacency_matrix():
    # Every node must appear as a key (the function sizes the matrix by the
    # number of keys in the graph).
    graph = {0: [1], 1: [0, 2], 2: [1]}
    adj = indexed_graph_to_adjacency_matrix(graph)
    assert adj == [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]


def test_graph_to_adjacency_matrix():
    node_list = ["A", "B", "C"]
    graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
    adj = graph_to_adjacency_matrix(graph, node_list)
    assert adj == [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]


def test_adj_matrix_to_graph_roundtrip():
    node_list = ["A", "B", "C"]
    graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
    adj = graph_to_adjacency_matrix(graph, node_list)
    recovered = adj_matrix_to_graph(node_list, adj)
    # Undirected graph: edges appear in both directions.
    assert set(recovered["A"]) == {"B"}
    assert set(recovered["B"]) == {"A", "C"}
    assert set(recovered["C"]) == {"B"}


def test_nodes_and_edges_to_adjacency_matrix():
    node_list = ["VIN", "VOUT", "GND"]
    edge_list = [("VIN", "VOUT"), ("VOUT", "GND")]
    adj = nodes_and_edges_to_adjacency_matrix(node_list, edge_list)
    assert adj == [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]


def test_adj_matrix_to_edges():
    node_list = ["VIN", "VOUT"]
    adj = [[0, 1], [1, 0]]
    edges = adj_matrix_to_edges(node_list, adj)
    assert ("VIN", "VOUT") in edges
    assert ("VOUT", "VIN") in edges


def test_hyper_edge_str():
    edge = HyperEdge(["VIN", "Sa0", "GND"])
    assert str(edge) == "(VIN, Sa0, GND)"


def test_hyper_edge_reduce_number():
    edge = HyperEdge(["VIN", "Sa0", "C1"])
    edge.reduce_number()
    assert edge.node_list == ["VIN", "Sa", "C"]


def _build_simple_topo():
    # Connection nodes are integers; device/terminal nodes are strings.
    node_list = ["VIN", "VOUT", "GND", 0, 1]
    edge_list = [("VIN", 0), ("VOUT", 0), ("GND", 1), ("VOUT", 1)]
    return TopoGraph(node_list=node_list, edge_list=edge_list)


def test_topograph_adjacency_matrix_from_edges():
    graph = _build_simple_topo()
    assert graph.get_adj_matrix() == [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
    ]


def test_topograph_find_path_vin_vout():
    graph = _build_simple_topo()
    paths = graph.find_path_Vin_Vout()
    # VIN (0) -> connection node 0 (index 3) -> VOUT (1)
    assert paths == [[0, 3, 1]]


def test_topograph_find_end_points_paths_as_str():
    graph = _build_simple_topo()
    paths = graph.find_end_points_paths_as_str()
    assert "VIN - VOUT" in paths
    assert "VOUT - GND" in paths
