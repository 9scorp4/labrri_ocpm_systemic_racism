from plotly import graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from loguru import logger
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import colorcet as cc

class TopicAnalysisVisualizer:
    """Enhanced visualization for topic analysis with improved aesthetics and interactivity."""
    
    def __init__(self):
        """Initialize visualizer with custom color schemes."""
        # Use perceptually uniform color schemes
        self.node_colorscale = cc.fire  # Better for coherence scores
        self.edge_colorscale = cc.blues  # Better for similarity values
        self.network_layouts = {
            'spring': self._spring_layout,
            'circular': nx.circular_layout,
            'spectral': nx.spectral_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }

    def create_topic_network(
        self,
        topics: List[Dict],
        similarity_matrix: np.ndarray,
        min_similarity: float = 0.3,
        layout: str = 'spring',
        node_size_factor: float = 40,
        edge_width_factor: float = 2,
        show_labels: bool = True
    ) -> go.Figure:
        """Create enhanced network visualization of topic relationships.
        
        Args:
            topics: List of topic dictionaries
            similarity_matrix: Topic similarity matrix
            min_similarity: Minimum similarity threshold for edges
            layout: Network layout algorithm
            node_size_factor: Base factor for node sizes
            edge_width_factor: Base factor for edge widths
            show_labels: Whether to show topic labels
            
        Returns:
            Plotly figure object
        """
        try:
            if not topics:
                return self._create_empty_figure("No topics to visualize")

            # Create network graph
            G = nx.Graph()
            
            # Add nodes with metadata
            for i, topic in enumerate(topics):
                # Calculate node size based on topic importance
                size = len(topic['words']) * node_size_factor
                
                # Add node with attributes
                G.add_node(
                    i,
                    label=topic.get('label', f'Topic {i+1}'),
                    size=size,
                    coherence=topic.get('coherence_score', 0),
                    words=', '.join(topic['words'][:5]) + '...'  # Top words for hover
                )

            # Add edges above similarity threshold
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    if similarity_matrix[i, j] >= min_similarity:
                        G.add_edge(
                            i, j,
                            weight=similarity_matrix[i, j],
                            width=similarity_matrix[i, j] * edge_width_factor
                        )

            # Get layout positions
            layout_func = self.network_layouts.get(layout, self._spring_layout)
            pos = layout_func(G)

            # Create figure
            fig = go.Figure()

            # Add edges with gradient colors
            edge_traces = self._create_edge_traces(G, pos)
            for trace in edge_traces:
                fig.add_trace(trace)

            # Add nodes
            node_trace = self._create_node_trace(G, pos, show_labels)
            fig.add_trace(node_trace)

            # Update layout
            fig.update_layout(
                title={
                    'text': 'Topic Similarity Network',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                paper_bgcolor='white',
                annotations=[
                    dict(
                        text=f"Node size: Topic size<br>Color: Coherence score<br>"
                             f"Edge thickness: Similarity",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0, y=-0.1,
                        align='left'
                    )
                ]
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating topic network: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")

    def create_coherence_plot(
        self,
        topics: List[Dict],
        show_threshold: bool = True,
        threshold: float = 0.3
    ) -> go.Figure:
        """Create enhanced coherence distribution visualization.
        
        Args:
            topics: List of topic dictionaries
            show_threshold: Whether to show threshold line
            threshold: Coherence threshold value
            
        Returns:
            Plotly figure object
        """
        try:
            if not topics:
                return self._create_empty_figure("No topics to visualize")

            # Sort topics by coherence
            sorted_topics = sorted(
                topics,
                key=lambda x: x.get('coherence_score', 0),
                reverse=True
            )

            # Create bar chart
            fig = go.Figure()

            # Add bars
            fig.add_trace(go.Bar(
                x=[t.get('label', f'Topic {i+1}') for i, t in enumerate(sorted_topics)],
                y=[t.get('coherence_score', 0) for t in sorted_topics],
                marker=dict(
                    color=[t.get('coherence_score', 0) for t in sorted_topics],
                    colorscale=self.node_colorscale,
                    showscale=True,
                    colorbar=dict(
                        title='Coherence Score',
                        thickness=15,
                        len=0.7
                    )
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Coherence: %{y:.3f}<br>" +
                    "<extra></extra>"
                )
            ))

            # Add threshold line if requested
            if show_threshold:
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {threshold:.2f}",
                    annotation_position="top right"
                )

            # Update layout
            fig.update_layout(
                title={
                    'text': 'Topic Coherence Distribution',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Topics",
                yaxis_title="Coherence Score",
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    tickangle=45,
                    gridcolor='lightgray',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    range=[0, 1]
                ),
                margin=dict(b=150, l=50, r=50, t=50),
                showlegend=False
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating coherence plot: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")

    def _create_edge_traces(self, G: nx.Graph, pos: Dict) -> List[go.Scatter]:
        """Create edge traces with gradients based on similarity."""
        traces = []
        
        # Group edges by similarity for different colors
        similarity_groups = {}
        for edge in G.edges(data=True):
            weight = edge[2]['weight']
            group = round(weight * 10) / 10  # Round to nearest 0.1
            if group not in similarity_groups:
                similarity_groups[group] = []
            similarity_groups[group].append(edge)

        # Create trace for each similarity group
        for similarity, edges in similarity_groups.items():
            edge_x = []
            edge_y = []
            
            for edge in edges:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            traces.append(go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(
                    width=max(0.5, similarity * 5),
                    color=f'rgba(150,150,150,{similarity})'
                ),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))

        return traces

    def _create_node_trace(
        self,
        G: nx.Graph,
        pos: Dict,
        show_labels: bool
    ) -> go.Scatter:
        """Create enhanced node trace with labels and hover info."""
        node_x = []
        node_y = []
        node_sizes = []
        node_colors = []
        node_labels = []
        hover_texts = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            size = G.nodes[node]['size']
            coherence = G.nodes[node]['coherence']
            label = G.nodes[node]['label']
            words = G.nodes[node]['words']
            
            node_sizes.append(size)
            node_colors.append(coherence)
            node_labels.append(label if show_labels else '')
            
            # Create hover text
            hover_text = (
                f"<b>{label}</b><br>"
                f"Coherence: {coherence:.3f}<br>"
                f"Top words: {words}"
            )
            hover_texts.append(hover_text)

        return go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=self.node_colorscale,
                showscale=True,
                colorbar=dict(
                    title='Coherence Score',
                    thickness=15,
                    len=0.7
                ),
                line=dict(
                    color='white',
                    width=1
                )
            ),
            text=node_labels,
            textposition="bottom center",
            hovertext=hover_texts,
            hoverinfo='text'
        )

    def _spring_layout(self, G: nx.Graph) -> Dict:
        """Enhanced spring layout with better spacing."""
        return nx.spring_layout(
            G,
            k=1/np.sqrt(len(G.nodes())),  # Optimal node spacing
            iterations=50,  # More iterations for better convergence
            seed=42  # For reproducibility
        )

    @staticmethod
    def _create_empty_figure(message: str = "No data available") -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(b=20, l=5, r=5, t=40)
        )
        
        return fig