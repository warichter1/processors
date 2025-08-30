#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:01:54 2024

@author: wrichter
"""

from dash import dash_table, html, dcc
import dash_bootstrap_components as dbc
import dash_ag_grid as dag


dashStyles = {'cell': {
                      'fontFamily': 'Open Sans',
                      'textAlign': 'left',
                      'width': '100px',
                      'minWidth': '100px',
                      'maxWidth': '200px',
                      'whiteSpace': 'no-wrap',
                      'overflow': 'hidden',
                      'textOverflow': 'ellipsis',
                      'backgroundColor': 'Rgb(230,230,250)'
                      },
              'data': {
                      'color': 'black',
                      'backgroundColor': 'white',
                      'border': '1px solid black'
                      },
              'data_conditional': [
                                   {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(220, 220, 220)',
                                   }
                                  ],
              'header': {
                         'backgroundColor': 'rgb(210, 210, 210)',
                         'border': 'thin lightgrey solid',
                         'color': 'black',
                         'fontWeight': 'bold',
                        },
              'style':  {"margin": 20, 'height': 750, 'overflowY': 'scroll'},
              'table':  {'minWidth': '100%', 'overflowX': 'auto', 'overflowY': 'scroll'},
              'rows':   {'headers': True, 'data': 0},
              'columns': {'headers': True, 'data': 1},
              }

dashConfig = {'pcurrent': 0, 'psize': 20, 'paction': 'custom',
              'col_select': "single", 'row_select': "single", 'row_delete': False,
              'fil_action': 'custom', 'fil_query': '', 'sort_action': 'custom', 'sort_mode': 'multi',
              }


style = dashStyles
conf = dashConfig


def pageination(df, className):
    return html.Div(
        [
            dcc.Markdown("Setting Page Size.  Enter number of rows"),
            dcc.Input(id="input-page-size", type="number", min=1, max=len(df), value=10, debounce=True),
            dag.AgGrid(
                id="grid-page-size",
                columnDefs=[
                    {"name": i, "id": i, "deletable": False, "selectable": True} for i in sorted(df.columns)
                ],
                rowData=df.to_dict("records"),
                columnSize="sizeToFit",
                defaultColDef={"filter": True},
                dashGridOptions={"pagination": True, "paginationPageSizeSelector": False, "animateRows": False},
            ),
        ],
        style=style['style'],
        className=f'{className}-datatable-interactivity-container'
        )


def pageLen(className):
    return html.Div([
            dcc.Dropdown(
                id=f'select_page_size_{className}',
                options=[{'label': '10', 'value': 10}, {'label': '25', 'value': 25},
                         {'label': '50', 'value': 50}, {'label': '100', 'value': 100}],
                value=5
        ),
    ], style={"width": "10%"})


pagination = html.Div(
    [
        dbc.Pagination(max_value=5, first_last=True, previous_next=True),
    ]
)


def tableTemplate(tableDf, className):
    return html.Div(
        pageLen(className),
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


def dbcTemplate(df, tabId, PAGE_SIZE=10):
    return html.Div(
        [
            # pageLen(tabId),
            dbc.Row(
                [
                    dbc.Col([
                        dash_table.DataTable(
                            id=f'table-dropdown{tabId}',
                            data=df.to_dict('records'),
                            # the contents of the table
                            columns=[{"name": i, "id": i, "deletable": False,
                                      "selectable": True} for i in sorted(df.columns)
                                     ],
                            editable=True,
                            persistence=True,
                            is_focused=True,
                            persisted_props=["page_current", 'data'],
                            style_table=style['table'],
                            fixed_rows={'headers': True, 'data': 0},
                            fixed_columns={'headers': True, 'data': 1},
                            style_cell=style['cell'],
                            style_data=style['data'],
                            style_data_conditional=style['data_conditional'],
                            style_header=style['header'],
                            # page_current=0,
                            # page_size=conf['psize'],
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
                            sort_by=[],

                        )

                    ])

                ]),
            # pagination,
        ],
        hidden=False, id=tabId, style={'overflowY': 'scroll'}

    )
