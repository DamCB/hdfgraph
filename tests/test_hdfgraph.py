# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import graph_tool.all as gt
import pandas as pd
import hdfgraph
import tempfile


def make_graph():

    graph, pos = gt.triangulation(np.random.random((10, 2)) * 4, type="delaunay")
    x, y = gt.ungroup_vector_property(pos, [0, 1])
    graph.vertex_properties['x'] = x
    graph.vertex_properties['y'] = y

    dist = graph.new_edge_property('double')
    x_srce = gt.edge_endpoint_property(graph, x, 'source')
    x_trgt = gt.edge_endpoint_property(graph, x, 'target')
    y_srce = gt.edge_endpoint_property(graph, y, 'source')
    y_trgt = gt.edge_endpoint_property(graph, y, 'target')


    dist.a = np.hypot(x_srce.a - x_trgt.a,
                      y_srce.a - y_trgt.a)

    graph.edge_properties['dist'] = dist

    short_edges = graph.new_edge_property('bool')
    short_edges.a = dist.a < dist.a.mean()
    graph.edge_properties['short_edges'] = short_edges
    num = graph.new_edge_property('long')
    num.a = np.random.randint(10, size=graph.num_edges())

    graph.edge_properties['num'] = num
    return graph

def test_graph_to_hdf():

    graph = make_graph()
    vertex_df, edge_df = hdfgraph.graph_to_dataframes(graph)
    assert(vertex_df.shape == (graph.num_vertices(), 2))
    assert(edge_df.shape == (graph.num_edges(), 3))
    assert(vertex_df.loc[5, 'x'] == graph.vertex_properties['x'][graph.vertex(5)])
    np.testing.assert_almost_equal(vertex_df.x, graph.vertex_properties['x'].a)
    assert(set(edge_df.columns) == {'dist', 'short_edges', 'num'})


def test_hdf_to_graph():

    graph_in = make_graph()
    vertex_df, edge_df = hdfgraph.graph_to_dataframes(graph_in)
    graph = hdfgraph.graph_from_dataframes(vertex_df, edge_df)
    assert(vertex_df.shape == (graph.num_vertices(), 2))
    assert(edge_df.shape == (graph.num_edges(), 3))
    assert(vertex_df.loc[5, 'x'] == graph.vertex_properties['x'][graph.vertex(5)])
    np.testing.assert_almost_equal(vertex_df.x, graph.vertex_properties['x'].a)
    assert(set(edge_df.columns) == {'dist', 'short_edges', 'num'})

def test_graph_to_hdf():

    graph = make_graph()
    tmp = tempfile.mktemp(suffix='.h5')
    hdfgraph.graph_to_hdf(graph, tmp)
    with pd.get_store(tmp) as store:
        assert('/vertices' in store.keys())
        assert('/edges' in store.keys())

def test_update_dframes():

    graph = make_graph()
    vertex_df, edge_df = hdfgraph.graph_to_dataframes(graph)

    long_edges = graph.new_edge_property('bool')
    dist = graph.edge_properties['dist']
    long_edges.a = dist.a > dist.a.mean()
    graph.edge_properties['long_edges'] = long_edges
    norm = graph.new_vertex_property('float')
    norm.a = np.hypot(graph.vertex_properties['x'].a,
                      graph.vertex_properties['y'].a)
    graph.vertex_properties['norm'] = norm

    hdfgraph.update_dframes(graph, vertex_df, edge_df)
    np.testing.assert_almost_equal(vertex_df.norm,
                                   graph.vertex_properties['norm'].a)
    np.testing.assert_almost_equal(edge_df.long_edges,
                                   graph.edge_properties['long_edges'].a)

def test_complete_pmaps():

    graph = make_graph()
    vertex_df, edge_df = hdfgraph.graph_to_dataframes(graph)
    vertex_df['norm']= np.hypot(vertex_df.x,
                                vertex_df.y)
    edge_df['long_edges'] = edge_df.dist > edge_df.dist.mean()
    hdfgraph.complete_pmaps(graph, vertex_df, edge_df)

    assert 'long_edges' in graph.edge_properties
    assert 'norm' in graph.vertex_properties

def test_update_pmaps():

    graph = make_graph()
    vertex_df, edge_df = hdfgraph.graph_to_dataframes(graph)
    new_val = vertex_df.x.max() + 1
    vertex_df.loc[2, 'x'] = new_val
    hdfgraph.update_pmaps(graph, vertex_df, edge_df)
    v = graph.vertex(2)
    assert graph.vertex_properties['x'][v] == new_val