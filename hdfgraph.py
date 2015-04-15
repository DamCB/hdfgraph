#!/usr/bin/env python -*- coding: utf-8 -*-

'''This module allows to import and export
[http://graph-tool.skewed.de](graph-tool) Graph objects to HDF5 files using pandas
'''

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import pandas as pd

import logging
log = logging.getLogger(__name__)


try:
    from graph_tool import Graph
except ImportError:
    log.warning('''graph-tool is not available, you won't be able to load
                data as a graph. You can still read the tables though''')

TYPES_WHITELIST = ['bool', 'uint8_t', 'double', 'float',
                   'int16_t', 'int32_t', 'int64_t',
                   'short', 'int', 'long', 'long long',
                   'long double']
TYPES_BLACKLIST = ['string', 'vector<bool>', 'vector<uint8_t>',
                   'vector<int16_t>', 'vector<short>',
                   'vector<int32_t>', 'vector<int>',
                   'vector<int64_t>', 'vector<long>',
                   'vector<int64_t>', 'vector<long>',
                   'vector<long long>', 'vector<double>', 'vector<float>',
                   'vector<long double>', 'vector<string>',
                   'python::object', 'object']

ALIASES = {'float64':'double',
           'bool':'bool',
           'int8':'bool',
           'int32':'int',
           'int64':'long',
           'uint8':'bool',
           'uint16':'long'}



def graph_to_dataframes(graph, stamp=None):
    '''Function packs the vertex and edge propery maps of
    an **unfiltered** Graph object in two dataframes.

    parameters:
    ===========

    graph: a `Graph` object whose 'internal' property maps will be output.

    stamp: an integer or a time stamp, default `None`. If `stamp` is
        not None, it will be used as a supplementary index, this is
        usefull to store a dynmical graph history.

    returns:
    ========

    vertex_df : a pd.DataFrame containing the graph's vertex
        PropertyMaps as columns.  If `stamp` is `None`, the DataFrame
        index corresponds to the graph's vertices indices, else it
        is a pd.MultiIndex with (stamp, index) pairs.

    edge_df : a pd.DataFrame containing the graph's edge PropertyMaps
        as columns. If `stamp` is `None`, the DataFrame index is a
        pd.MultiIndex with `(source, target)` pairs, where `source`
        and `target` are the source and target vertices of a given edge.

    note:
    =====

    only the following value types are supported: 'bool', 'uint8_t',
    'double', 'float', 'int16_t', 'int32_t', 'int64_t', 'short',
    'int', 'long', 'long long', 'long double'
    The type 'string' might work but is untested. There might be a way to store
    vectors using `Pannels` object, but this is also untested.
    '''

    ## TODO: print out which propery maps were not ported
    if stamp is not None:
        vertex_index = pd.MultiIndex.from_tuples([(stamp, graph.vertex_index[v])
                                                  for v in graph.vertices()],
                                                 names=('stamp',
                                                        'vertex_index'))
        edge_index = pd.MultiIndex.from_tuples([(stamp,
                                                 graph.vertex_index[source],
                                                 graph.vertex_index[target])
                                                for source, target
                                                in graph.edges()],
                                               names=('stamp',
                                                      'source',
                                                      'target'))
    else:
        vertex_index = pd.Index([graph.vertex_index[v]
                                 for v in graph.vertices()],
                                name='vertex_index')
        edge_index = pd.MultiIndex.from_tuples([(graph.vertex_index[source],
                                                 graph.vertex_index[target])
                                                for source, target
                                                in graph.edges()],
                                                names=('source', 'target'))
    vertex_df = pd.DataFrame({key: np.array(prop.fa, dtype=prop.fa.dtype)
                              for key, prop in graph.vertex_properties.items()
                              if prop.value_type() in TYPES_WHITELIST},
                             index=vertex_index)
    graph.set_edge_filter(None)
    edge_df = pd.DataFrame({key: np.array(prop.fa, dtype=prop.fa.dtype)
                            for key, prop in graph.edge_properties.items()
                            if prop.value_type() in TYPES_WHITELIST},
                           index=edge_index)
    return vertex_df, edge_df

def frames_to_hdf(vertex_df, edge_df, fname, reset=False,
                  vert_kwargs={}, edge_kwargs={}):
    '''
    Records the two  DataFrame in the hdf file filename
    '''
    with pd.get_store(fname) as store:
        if not len(store.keys()):
            store.put('edges', edge_df,
                      format='table', **vert_kwargs)
            store.put('vertices', vertex_df,
                      format='table', **edge_kwargs)
        ## FIXME: should only remove the matching index
        elif reset:
            try:
                store.remove('vertices')
            except KeyError:
                pass
            try:
                store.remove('edges')
            except KeyError:
                pass
            store.put('vertices', vertex_df,
                      format='table', **vert_kwargs)
            store.put('edges', edge_df,
                      format='table', **edge_kwargs)
        else:
            store.append('vertices', vertex_df,
                         format='table', **vert_kwargs)
            store.append('edges', edge_df,
                         format='table', **edge_kwargs)

def graph_to_hdf(graph, fname, stamp=None, reset=False, **kwargs):
    '''

    '''
    vertex_df, edge_df = graph_to_dataframes(graph, stamp=stamp)
    frames_to_hdf(vertex_df, edge_df, fname, reset=reset, **kwargs)

def graph_from_dataframes(vertex_df, edge_df):
    '''Re-creates a Graph object with PropertyMaps taken
    from the vertex_df and edge_df DataFrames

    Paramters:
    ==========
    verex_df: a DataFrame with an index named 'vertex_index'
    edge_df: a DataFrame with a multi-index named ('source', 'target')

    Returns:
    ========
    graph: a grah-tool Graph with PropertyMaps copied
        from the columns of the input DataFrames
    '''

    graph = Graph(directed=True)

    vertex_index = vertex_df.index.get_level_values(level='vertex_index')
    vertices = graph.add_vertex(n=vertex_index.shape[0])
    for col in vertex_df.columns:
        in_type = vertex_df[col].dtype.name
        try:
            dtype = ALIASES[in_type]
        except KeyError:
            log.info('Data type {} not supported'.format(in_type))
            continue
        prop = graph.new_vertex_property(dtype)
        prop.fa = vertex_df[col]
        graph.vertex_properties[col] = prop

    src = edge_df.index.names.index('source')
    trgt = edge_df.index.names.index('target')
    ### TODO: use the list edge creation
    for tup in edge_df.index:
        source, target = tup[src], tup[trgt]
        try:
            edge = graph.add_edge(source, target)
        except ValueError:
            log.info('Invalid vertex in (source: {}, target: {})'.format(source, target))
    for col in edge_df.columns:
        in_type = edge_df[col].dtype.name
        try:
            dtype = ALIASES[in_type]
        except KeyError:
            log.info('Data type {} not supported'.format(in_type))
            continue
        prop = graph.new_edge_property(dtype)
        prop.fa = edge_df[col]
        graph.edge_properties[col] = prop

    return graph

def table_from_hdf(fname, table, stamp=None, **kwargs):
    with pd.get_store(fname) as store:
        if stamp == -1:

            edge_df = store.select(table,
                                   **kwargs)
            stamps = edge_df.index.get_level_values('stamp').unique()
            last = stamps.max()
            edge_df = edge_df.xs(last, level='stamp')

        elif stamp is not None:
            edge_df = store.select(table,
                                   where="'stamp'={}".format(stamp),
                                   **kwargs)
        else:
            edge_df = store.select(table,
                                   **kwargs)
    return edge_df

def table_slice(fname, table, bounds):
    '''
    Small utility function to get the data only between bounds

    Parameters:
    -----------

    fname: str, path to the HDF file
    bounds: dict
        key, value pairs of the form 'indexable':(min_value, max_value)
        if one of those values is None, it will not be taken into account
        (ie take the absolute min or max) as for range(), start is included and stop excluded)
    Each of the bounds will be joined to the 'where' keywarg (by logical and)

    Note:
    -----
    This will only work from indexable columns, aka data columns, which must be set
    explicetely when recording the table. See the corresponding error when you don't
    use one of those. See
    http://pandas.pydata.org/pandas-docs/stable/io.html#query-via-data-columns

    '''
    conditions = []
    for column, (min_value, max_value) in bounds.items():
        start = "{} >= {}".format(column, min_value) if min_value is not None else ""
        stop = "{} < {}".format(column, max_value) if max_value is not None else ""
        conditions.extend([start, stop])
    where = " & ".join([cnd for cnd in conditions if len(cnd)])
    with pd.get_store(fname) as store:
        if len(where):
            return store.select(table, where=where)
        else:
            return store.select(table)

def edges_time_slice(fname, start_stamp, stop_stamp):
    return table_slice(fname, 'edges', {'stamp':(start_stamp, stop_stamp)})

def vertices_time_slice(fname, start_stamp, stop_stamp):
    return table_slice(fname, 'vertices', {'stamp':(start_stamp, stop_stamp)})



def frames_from_hdf(fname, stamp=None,
                    vertex_kwargs={}, edge_kwargs={}):

    with pd.get_store(fname) as store:
        if stamp == -1:
            vertex_df = store.select('vertices',
                                     #where="'stamp'=stamp",
                                     **vertex_kwargs)
            stamps = vertex_df.index.get_level_values('stamp').unique()
            last = stamps.max()
            vertex_df = vertex_df.xs(last, level='stamp')
            edge_df = store.select('edges',
                                   where="'stamp'={}".format(last),
                                   **edge_kwargs)
        elif stamp is not None:
            vertex_df = store.select('vertices',
                                     where="'stamp'={}".format(stamp),
                                     **vertex_kwargs)
            edge_df = store.select('edges',
                                   where="'stamp'={}".format(stamp),
                                   **edge_kwargs)
        else:
            vertex_df = store.select('vertices',
                                     **vertex_kwargs)
            edge_df = store.select('edges',
                                   **edge_kwargs)
    return vertex_df, edge_df

def graph_from_hdf(fname, stamp=None):

    vertex_df, edge_df = frames_from_hdf(fname, stamp=stamp)
    return graph_from_dataframes(vertex_df, edge_df)

def _get_valid_mask(graph):
    '''Mask over valid edges '''
    valid = graph.new_edge_property('bool')
    valid.a[:] = 0
    for edge in graph.edges():
        valid[edge] = 1
    return valid


def slice_data(vertices_df, edge_df, v_bounds={}):

    for col, (v_min, v_max) in v_bounds.items():
        vertices_df = vertices_df[vertices_df[col] >= v_min]
        vertices_df = vertices_df[vertices_df[col] <= v_max]
    vertex_index = set(vertices_df.index.get_level_values('vertex_index'))

    src_index = set(edge_df.index.get_level_values('source'))
    trgt_index = set(edge_df.index.get_level_values('target'))
    srcs = src_index.intersection(vertex_index)
    trgts = trgt_index.intersection(vertex_index)
    edge_df['keep'] = True
    keep = edge_df.groupby(level='source', group_keys=True).apply(select_idx_,
                                                                   'source', srcs,)
    assert keep.shape[0] == edge_df.shape[0]
    edge_df = edge_df[keep.values]
    keep = edge_df.groupby(level='target', group_keys=True).apply(select_idx_,
                                                                   'target', trgts)
    edge_df = edge_df[keep.values]
    return vertices_df, edge_df

def update_dframes(graph, vertex_df, edge_df, vcols=None, ecols=None):
    if vcols is not None:
        vitems = {col: graph.vertex_properties[col] for col in vcols}.items()
    else:
        vitems = graph.vertex_properties.items()
    if ecols is not None:
        eitems = {col: graph.edge_properties[col] for col in vcols}.items()
    else:
        eitems = graph.edge_properties.items()

    for col, prop in vitems:
        try:
            vertex_df[col] = prop.fa
        except KeyError:
            log.info('Property {} not in vertex dataframe'.format(col))
    for col, prop in eitems:
        try:
            edge_df[col] = prop.fa
        except KeyError:
            log.info('Property {} not in edge dataframe'.format(col))


def complete_pmaps(graph, vertex_df, edge_df, vcols=None, ecols=None):
    '''
    Completes the vertex and edge (internalized) property maps
    with the columns of the vertex_df and edge_df DataFrames.
    Warning: this only initializes the properymaps, values are NOT set


    '''
    if vcols is None:
        vdf_cols = set(vertex_df.columns)
    else:
        vdf_cols = set(vcols)
    vp_cols = set(graph.vertex_properties.keys())
    missing_vps = vdf_cols.difference(vp_cols)

    for new in missing_vps:
        try:
            dtype = ALIASES[vertex_df.dtypes[new].name]
            new_vp = graph.new_vertex_property(dtype)
            graph.vertex_properties[new] = new_vp
        except ValueError:
            log.warning(
                "Data type not supported for column {}, "
                "it won't be passed as a vector property".format(new))


    if ecols is None:
        edf_cols = set(edge_df.columns)
    else:
        edf_cols = ecols
    ep_cols = set(graph.edge_properties.keys())
    missing_eps = edf_cols.difference(ep_cols)

    for new in missing_eps:
        try:
            dtype = ALIASES[edge_df.dtypes[new].name]
            new_vp = graph.new_edge_property(dtype)
            graph.edge_properties[new] = new_vp
        except ValueError:
            log.warning(
                "Data type not supported for column {}, "
                "it won't be passed as a vector property".format(new))

def update_pmaps(graph, vertex_df, edge_df, vcols=None, ecols=None):


    if vcols is not None:
        vitems = {col: graph.vertex_properties[col] for col in vcols}.items()
    else:
        vitems = graph.vertex_properties.items()
    if ecols is not None:
        eitems = {col: graph.edge_properties[col] for col in vcols}.items()
    else:
        eitems = graph.edge_properties.items()

    v_idx = np.asarray(graph.vertex_index.copy().fa)
    e_idx = np.asarray(graph.edge_index.copy().fa)
    for col, prop in vitems:
        try:
            prop.fa = vertex_df.loc[v_idx, col]
        except KeyError:
            log.info('Property {} not in vertex dataframe'.format(col))
    for col, prop in eitems:
        try:
            print('Property map: {}'.format(prop.fa[:5]))
            print('DataFrame: {}'.format(edge_df.iloc[e_idx[:5], col]))
            prop.fa = edge_df.iloc[e_idx, col]
        except KeyError:
            log.info('Property {} not in edge dataframe'.format(col))

def _guess_dtype(col, df):

    if 'float' in df.dtypes[col].name:
        return 'float'
    elif 'int' in df.dtypes[col].name:
        if set(df[col].unique()).issubset({0, 1}) :
            return 'bool'
        else:
            return 'long'
    else:
        raise ValueError(
            'Unsuported data type {}'.format(df.dtypes[col].name))


def select_idx_(df, idx_name, valids):
    idx = df.index.get_level_values(idx_name)[0]
    keep = df.keep.copy()
    if idx not in valids:
        keep.loc[:] = False
    return keep
