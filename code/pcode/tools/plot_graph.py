#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:45:07 2020

@author: aransil
"""

import math
import plotly.graph_objects as go
import networkx as nx

# Start and end are lists defining start and end points
# Edge x and y are lists used to construct the graph
# arrowAngle and arrowLength define properties of the arrowhead
# arrowPos is None, 'middle' or 'end' based on where on the edge you want the arrow to appear
# arrowLength is the length of the arrowhead
# arrowAngle is the angle in degrees that the arrowhead makes with the edge
# dotSize is the plotly scatter dot size you are using (used to even out line spacing when you have a mix of edge lengths)


def add_edge(
    start,
    end,
    edge_x,
    edge_y,
    lengthFrac=1,
    arrowPos=None,
    arrowLength=0.025,
    arrowAngle=30,
    dotSize=20,
):

    # Get start and end cartesian coordinates
    x0, y0 = start
    x1, y1 = end

    # Incorporate the fraction of this segment covered by a dot into total reduction
    length = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    dotSizeConversion = 0.0565 / 20  # length units per dot size
    convertedDotDiameter = dotSize * dotSizeConversion
    lengthFracReduction = convertedDotDiameter / length
    lengthFrac = lengthFrac - lengthFracReduction

    # If the line segment should not cover the entire distance, get actual start and end coords
    skipX = (x1 - x0) * (1 - lengthFrac)
    skipY = (y1 - y0) * (1 - lengthFrac)
    x0 = x0 + skipX / 2
    x1 = x1 - skipX / 2
    y0 = y0 + skipY / 2
    y1 = y1 - skipY / 2

    # Append line corresponding to the edge
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(
        None
    )  # Prevents a line being drawn from end of this edge to start of next edge
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

    # Draw arrow
    if not arrowPos == None:

        # Find the point of the arrow; assume is at end unless told middle
        pointx = x1
        pointy = y1
        eta = math.degrees(math.atan((x1 - x0) / (y1 - y0)))

        if arrowPos == "middle" or arrowPos == "mid":
            pointx = x0 + (x1 - x0) / 2
            pointy = y0 + (y1 - y0) / 2

        # Find the directions the arrows are pointing
        signx = (x1 - x0) / abs(x1 - x0)
        signy = (y1 - y0) / abs(y1 - y0)

        # Append first arrowhead
        dx = arrowLength * math.sin(math.radians(eta + arrowAngle))
        dy = arrowLength * math.cos(math.radians(eta + arrowAngle))
        edge_x.append(pointx)
        edge_x.append(pointx - signx ** 2 * signy * dx)
        edge_x.append(None)
        edge_y.append(pointy)
        edge_y.append(pointy - signx ** 2 * signy * dy)
        edge_y.append(None)

        # And second arrowhead
        dx = arrowLength * math.sin(math.radians(eta - arrowAngle))
        dy = arrowLength * math.cos(math.radians(eta - arrowAngle))
        edge_x.append(pointx)
        edge_x.append(pointx - signx ** 2 * signy * dx)
        edge_x.append(None)
        edge_y.append(pointy)
        edge_y.append(pointy - signx ** 2 * signy * dy)
        edge_y.append(None)

    return edge_x, edge_y


def ploty_draw_graph(G):
    nodeColor = "Blue"
    nodeSize = 20
    lineWidth = 2
    lineColor = "#000000"

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    for node in G.nodes:
        G.nodes[node]["pos"] = list(pos[node])

    # Make list of nodes for plotly
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)

    # Make a list of edges for plotly, including line segments that result in arrowheads
    edge_x = []
    edge_y = []
    for edge in G.edges():
        start = G.nodes[edge[0]]["pos"]
        end = G.nodes[edge[1]]["pos"]

        if start[0] != end[0] and start[1] != end[1]:
            edge_x, edge_y = add_edge(
                start,
                end,
                edge_x,
                edge_y,
                lengthFrac=0.95,
                arrowPos="end",
                arrowLength=0.025,
                arrowAngle=30,
                dotSize=nodeSize,
            )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=lineWidth, color=lineColor),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="RdBu",
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=10,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line=dict(width=0),
        ),
    )

    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace["marker"]["color"] += tuple([len(adjacencies[1])])
        node_info = (
            str(adjacencies[0]) + " # of connections: " + str(len(adjacencies[1]))
        )
        node_trace["text"] += tuple([node_info])

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="",
            titlefont=dict(size=16),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper")],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.show()
