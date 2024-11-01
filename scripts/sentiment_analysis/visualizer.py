import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from loguru import logger

class SentimentVisualizer:
    """Handles visualization of sentiment analysis results."""

    def create_basic_sentiment_plot(self, data, title=None, plot_type='histogram'):
        """Create basic sentiment distribution visualization."""
        try:
            if data is None or data.empty:
                return self._create_empty_figure("No data available")

            if plot_type == 'histogram':
                fig = go.Figure(data=[
                    go.Histogram(
                        x=data['sentiment_score'],
                        nbinsx=30,
                        name='Sentiment Distribution',
                        marker_color='rgb(55, 83, 109)',
                        hovertemplate=(
                            "Sentiment Score: %{x:.2f}<br>" +
                            "Count: %{y}<br>" +
                            "<extra></extra>"
                        )
                    )
                ])

                # Add vertical line at zero
                fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Neutral",
                    annotation_position="top right"
                )

                fig.update_layout(
                    title=title or 'Sentiment Distribution',
                    xaxis_title='Sentiment Score',
                    yaxis_title='Number of Documents',
                    showlegend=False,
                    height=500,
                    plot_bgcolor='white',
                    xaxis=dict(
                        gridcolor='lightgrey',
                        zerolinecolor='grey',
                        zerolinewidth=1
                    ),
                    yaxis=dict(
                        gridcolor='lightgrey',
                        zerolinecolor='grey',
                        zerolinewidth=1
                    ),
                    margin=dict(l=50, r=50, t=70, b=50)
                )

            elif plot_type == 'box':
                fig = go.Figure(data=[
                    go.Box(
                        y=data['sentiment_score'],
                        name='Sentiment',
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8,
                        marker_color='rgb(55, 83, 109)',
                        boxmean=True,
                        hovertemplate=(
                            "Sentiment Score: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    )
                ])

                fig.update_layout(
                    title=title or 'Sentiment Distribution',
                    yaxis_title='Sentiment Score',
                    showlegend=False,
                    height=500,
                    plot_bgcolor='white',
                    yaxis=dict(
                        gridcolor='lightgrey',
                        zerolinecolor='red',
                        zerolinewidth=1
                    ),
                    margin=dict(l=50, r=50, t=70, b=50)
                )

            elif plot_type == 'heatmap':
                # Create pivot table for heatmap
                pivot_data = pd.pivot_table(
                    data,
                    values='sentiment_score',
                    index='category',
                    columns='language',
                    aggfunc='mean'
                )

                fig = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(pivot_data.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    hovertemplate=(
                        "Category: %{y}<br>" +
                        "Language: %{x}<br>" +
                        "Average Sentiment: %{z:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))

                fig.update_layout(
                    title=title or 'Sentiment Heatmap by Category and Language',
                    height=max(400, len(pivot_data.index) * 40),
                    margin=dict(l=150, r=50, t=70, b=50)
                )

            return fig

        except Exception as e:
            logger.error(f"Error creating basic sentiment plot: {str(e)}")
            return self._create_empty_figure(f"Error: {str(e)}")

    def create_aspect_sentiment_plot(self, data, title=None):
        """Create visualization for aspect-based sentiment analysis."""
        try:
            if data is None or data.empty:
                return self._create_empty_figure("No data available")

            # Get aspect columns
            aspect_cols = [col for col in data.columns if col.endswith('_sentiment')]
            mention_cols = [col for col in data.columns if col.endswith('_mentions')]

            # Prepare data for visualization
            aspects = [col.replace('_sentiment', '') for col in aspect_cols]
            avg_sentiments = [data[col].mean() for col in aspect_cols]
            total_mentions = [data[col.replace('sentiment', 'mentions')].sum() for col in aspect_cols]

            # Create bubble chart
            fig = go.Figure(data=[
                go.Scatter(
                    x=avg_sentiments,
                    y=aspects,
                    mode='markers',
                    marker=dict(
                        size=[np.sqrt(m) * 10 for m in total_mentions],
                        color=avg_sentiments,
                        colorscale='RdBu',
                        colorbar=dict(title='Average Sentiment'),
                        showscale=True,
                        line=dict(color='black', width=1)
                    ),
                    text=[f'Mentions: {m}' for m in total_mentions],
                    hovertemplate=(
                        "<b>%{y}</b><br>" +
                        "Average Sentiment: %{x:.3f}<br>" +
                        "%{text}<br>" +
                        "<extra></extra>"
                    )
                )
            ])

            fig.update_layout(
                title=title or 'Aspect-Based Sentiment Analysis',
                xaxis=dict(
                    title='Average Sentiment',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='red',
                    gridcolor='lightgrey'
                ),
                yaxis=dict(
                    title='Aspects',
                    gridcolor='lightgrey'
                ),
                plot_bgcolor='white',
                height=max(400, len(aspects) * 40),
                margin=dict(l=150, r=50, t=70, b=50)
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating aspect sentiment plot: {str(e)}")
            return self._create_empty_figure(f"Error: {str(e)}")

    def create_temporal_sentiment_plot(self, data, title=None):
        """Create visualization for temporal sentiment analysis."""
        try:
            if data is None or data.empty:
                return self._create_empty_figure("No data available")

            fig = go.Figure()

            # Add rolling average line
            window_size = max(5, len(data) // 20)  # Adaptive window size
            rolling_avg = data['sentiment_score'].rolling(window=window_size).mean()

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['sentiment_score'],
                    mode='markers',
                    name='Individual Documents',
                    marker=dict(
                        color=data['sentiment_score'],
                        colorscale='RdBu',
                        showscale=True,
                        colorbar=dict(title='Sentiment')
                    ),
                    hovertemplate=(
                        "Document ID: %{text}<br>" +
                        "Sentiment: %{y:.3f}<br>" +
                        "<extra></extra>"
                    ),
                    text=data['doc_id']
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rolling_avg,
                    mode='lines',
                    name=f'Moving Average (window={window_size})',
                    line=dict(color='black', width=2)
                )
            )

            fig.update_layout(
                title=title or 'Temporal Sentiment Analysis',
                xaxis_title='Document Sequence',
                yaxis_title='Sentiment Score',
                plot_bgcolor='white',
                height=500,
                margin=dict(l=50, r=50, t=70, b=50),
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating temporal sentiment plot: {str(e)}")
            return self._create_empty_figure(f"Error: {str(e)}")

    def create_comparative_sentiment_plot(self, data, title=None):
        """Create visualization comparing sentiment across categories.
        
        Args:
            data (pd.DataFrame): DataFrame with columns 'category', 'avg_sentiment', 'positive_ratio', 'negative_ratio'
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            if data is None or data.empty:
                return self._create_empty_figure("No data available")

            # Create a more informative multi-trace visualization
            fig = go.Figure()

            # Add trace for average sentiment
            fig.add_trace(
                go.Bar(
                    name='Average Sentiment',
                    x=data['category'],
                    y=data['avg_sentiment'],
                    marker_color='rgb(55, 83, 109)',
                    text=data['avg_sentiment'].round(3),
                    textposition='auto',
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Average Sentiment: %{y:.3f}<br>" +
                        "<extra></extra>"
                    )
                )
            )

            # Add trace for positive ratio
            fig.add_trace(
                go.Bar(
                    name='Positive Ratio',
                    x=data['category'],
                    y=data['positive_ratio'],
                    marker_color='rgb(26, 118, 255)',
                    text=(data['positive_ratio'] * 100).round(1).astype(str) + '%',
                    textposition='auto',
                    visible='legendonly',
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Positive Ratio: %{text}<br>" +
                        "<extra></extra>"
                    )
                )
            )

            # Add trace for negative ratio
            fig.add_trace(
                go.Bar(
                    name='Negative Ratio',
                    x=data['category'],
                    y=data['negative_ratio'],
                    marker_color='rgb(255, 68, 68)',
                    text=(data['negative_ratio'] * 100).round(1).astype(str) + '%',
                    textposition='auto',
                    visible='legendonly',
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Negative Ratio: %{text}<br>" +
                        "<extra></extra>"
                    )
                )
            )

            # Add document count annotations if available
            if 'document_count' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        name='Document Count',
                        x=data['category'],
                        y=[0] * len(data),
                        text=data['document_count'].astype(str) + ' docs',
                        mode='text',
                        textposition='bottom center',
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

            # Update layout
            fig.update_layout(
                title=dict(
                    text=title or 'Comparative Sentiment Analysis',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                xaxis=dict(
                    title='Category',
                    tickangle=-45,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey'
                ),
                yaxis=dict(
                    title='Score',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='LightGrey'
                ),
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1,
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                plot_bgcolor='white',
                margin=dict(l=50, r=50, t=70, b=100)
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating comparative sentiment plot: {str(e)}")
            return self._create_empty_figure(f"Error: {str(e)}")

    @staticmethod
    def _create_empty_figure(message="No data available"):
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=70, b=50)
        )
        return fig