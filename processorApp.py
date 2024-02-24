#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:37:51 2023

@author: wrichter
"""

import base64
from dash import Dash, dash_table, html, dcc, callback
from dash.dependencies import Input, Output, State #, Event
import plotly.graph_objs as go
from flask import Flask, redirect, render_template, request, session, abort, url_for, json, make_response
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import pandas as pd
import numpy as np
from numpy import random

from nvidia import nvidiaLoader, nvidiaHeader, nvidiaSelect
from arm import armLoader, armSelect # ArmDataProcessor
from dashConf import dashConfig as conf, dashStyles as style


df = nvidiaLoader('nvidia', columns=nvidiaSelect)
df[' index'] = range(1, len(df) + 1)
nHeader = nvidiaHeader('nvidia')

armTest = armLoader('ARM')
armDf = armLoader('ARM', columns=armSelect)
# arm.process()
# armDf = arm.df['merged']
armDf[' index'] = range(1, len(armDf) + 1)
armHeader = list(armDf.keys())

app = Dash(__name__, title='Processors', suppress_callback_exceptions=True)

PAGE_SIZE = 10


def tableTemplate(tableDf, className, header):
    return html.Div(
        dash_table.DataTable(
            data=tableDf.to_dict('records'),
            id=f'{className}-datatable-interactivity',
            columns=[
                {"name": i, "id": i, "deletable": False, "selectable": True} for i in sorted(tableDf.columns)
            ],
            style_table=style['table'],
            fixed_rows={'headers': True, 'data': 0},
            fixed_columns={'headers': True, 'data': 1},
            style_cell=style['cell'],
            style_data=style['data'],
            style_data_conditional=style['data_conditional'],
            style_header=style['header'],
            page_current=0,
            page_size=conf['psize'],
            page_action=conf['paction'],

            column_selectable=conf['col_select'],
            row_selectable=conf["row_select"],
            row_deletable=conf['row_delete'],
            selected_columns=[],
            selected_rows=[],

            filter_action=conf['fil_action'],
            filter_query='',

            sort_action=conf['sort_action'],
            sort_mode=conf['sort_mode'],
            sort_by=[]
        ),
        style=style['style'],
        className=f'{className}-datatable-interactivity-container'
    )

table1 = tableTemplate(df, 'one', nHeader)
table2 = tableTemplate(df, 'two', armHeader)



app.layout = dcc.Loading(html.Div([
    html.H1('Processor Info'),
    dcc.Tabs(
        id="tabs-with-classes",
        value='tab-1',
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children=[
            dcc.Tab(
                label='Nvidia GPUs',
                value='tab-1',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Tab two',
                value='tab-2',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
        ]),
    html.Div(id='tabs-content-classes')
]))

@callback(Output('tabs-content-classes', 'children'),
              Input('tabs-with-classes', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Nvidia'),
            table1,
            html.Div(id='one-datatable-interactivity-container')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('ARM'),
            table2,
            html.Div(id='two-datatable-interactivity-container')
        ])


# app.layout = html.Div(
#     className="row",
#     children=[
#         html.H1('Processor Info'),
#         table1,
#         html.Div(id='datatable-interactivity-container')
#     ]
# )

operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3

@callback(
    Output('one-datatable-interactivity', 'style_data_conditional'),
    Output('two-datatable-interactivity', 'style_data_conditional'),
    Input('one-datatable-interactivity', 'selected_columns'),
    Input('two-datatable-interactivity', 'selected_columns')
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

@callback(
    Output('two-datatable-interactivity-container', "children"),
    Output('one-datatable-interactivity-container', "children"),
    Input('one-datatable-interactivity', "derived_virtual_data"),
    Input('two-datatable-interactivity', "derived_virtual_data"),
    Input('one-datatable-interactivity', "derived_virtual_selected_rows"),
    Input('two-datatable-interactivity', "derived_virtual_selected_rows"))
def update_graphs(rows, derived_virtual_selected_rows):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncrasy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dff))]

    return [
        dcc.Graph(
            id=column,
            figure={
                "data": [
                    {
                        "x": dff["country"],
                        "y": dff[column],
                        "type": "bar",
                        "marker": {"color": colors},
                    }
                ],
                "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": {
                        "automargin": True,
                        "title": {"text": column}
                    },
                    "height": 250,
                    "margin": {"t": 10, "l": 10, "r": 10},
                },
            },
        )
        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        for column in ["pop", "lifeExp", "gdpPercap"] if column in dff
    ]



if __name__ == '__main__':
    app.run(debug=True)
