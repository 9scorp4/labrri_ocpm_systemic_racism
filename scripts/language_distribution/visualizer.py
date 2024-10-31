from plotly import graph_objects as go
import plotly.express as px
import pandas as pd
from loguru import logger

class LanguageDistributionVisualizer:
    """Handle visualization of language distribution analysis results."""
    
    def create_distribution_plot(self, data, title, plot_type='pie'):
        """
        Create a distribution plot for language counts.
        
        Args:
            data (pd.DataFrame): DataFrame containing language distribution data
            title (str): Title for the plot
            plot_type (str): Type of plot ('pie' or 'bar')
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            if data is None or data.empty:
                return go.Figure().add_annotation(
                    text="No data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )

            # Log the data being plotted
            logger.debug(f"Creating {plot_type} plot with data:\n{data}")

            colors = {
                'French': '#FF9999',
                'English': '#66B2FF',
                'Other': '#99FF99'
            }

            if plot_type == 'pie':
                fig = go.Figure(data=[
                    go.Pie(
                        labels=data['Language'],
                        values=data['Count'],
                        hole=0.3,
                        textinfo='label+percent',
                        marker=dict(colors=[colors.get(lang, '#CCCCCC') for lang in data['Language']]),
                        textposition='inside',
                        insidetextorientation='horizontal'
                    )
                ])
            else:  # bar
                fig = go.Figure(data=[
                    go.Bar(
                        x=data['Language'],
                        y=data['Count'],
                        text=data['Count'],
                        textposition='auto',
                        marker_color=[colors.get(lang, '#CCCCCC') for lang in data['Language']]
                    )
                ])

            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                showlegend=False,
                height=500,
                plot_bgcolor='white',
                margin=dict(t=100, l=50, r=50, b=50)
            )

            # Add custom hover template
            if plot_type == 'pie':
                fig.update_traces(
                    hovertemplate="<b>%{label}</b><br>" +
                                "Count: %{value}<br>" +
                                "Percentage: %{percent}<br>" +
                                "<extra></extra>"
                )
            else:
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>" +
                                "Count: %{y}<br>" +
                                "<extra></extra>"
                )

            return fig

        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            return go.Figure().add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )

    def create_category_comparison(self, data):
        """Create visualization comparing language distribution across categories."""
        if data is None or data.empty:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )

        try:
            # Handle MultiIndex DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                plot_data = pd.DataFrame({
                    'Category': data.index,
                    'French (%)': data[('French (%)', 'mean')],
                    'English (%)': data[('English (%)', 'mean')],
                    'Other (%)': data[('Other (%)', 'mean')]
                })
            else:
                plot_data = data.copy()

            plot_data = plot_data[plot_data['Category'].notna() & (plot_data['Category'] != '')]
            
            fig = go.Figure()

            bar_configs = [
                ('French (%)', '#FF8042', 'French'),
                ('English (%)', '#0088FE', 'English'),
                ('Other (%)', '#00C49F', 'Other')
            ]

            for col, color, label in bar_configs:
                fig.add_trace(go.Bar(
                    name=label,
                    x=plot_data['Category'],
                    y=plot_data[col],
                    marker_color=color,
                    text=plot_data[col].round(1).astype(str) + '%',
                    textposition='inside',
                    textangle=0,
                    insidetextanchor='middle',
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        f"{label}: %{{y:.1f}}%<br>" +
                        "<extra></extra>"
                    )
                ))

            fig.update_layout(
                title=dict(
                    text='Language Distribution by Category',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                barmode='group',
                xaxis=dict(
                    title='',
                    tickangle=30,
                    tickfont=dict(size=11),
                    showgrid=False
                ),
                yaxis=dict(
                    title='Percentage',
                    titlefont=dict(size=14),
                    tickfont=dict(size=12),
                    range=[0, 100],
                    ticksuffix='%',
                    gridcolor='rgba(0,0,0,0.1)',
                    showgrid=True
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                plot_bgcolor='white',
                height=600,
                width=max(800, len(plot_data) * 150),
                margin=dict(l=60, r=30, t=100, b=120)
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating category comparison: {str(e)}")
            return go.Figure().add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
        
    def debug_data(self, data):
        """Helper method to debug data structure"""
        logger.info("Debugging data structure:")
        logger.info(f"Type: {type(data)}")
        logger.info(f"Shape: {data.shape if isinstance(data, pd.DataFrame) else 'Not a DataFrame'}")
        if isinstance(data, pd.DataFrame):
            logger.info(f"Columns: {data.columns}")
            logger.info(f"Sample data:\n{data.head()}")

    def create_detailed_analysis_plots(self, data):
        """Create visualizations for detailed language analysis."""
        if data is None or data.empty:
            empty_fig = go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return empty_fig, empty_fig

        try:
            # Language percentage distribution
            fig1 = go.Figure()
            colors = {'English (%)': '#1f77b4', 'French (%)': '#ff7f0e', 'Other (%)': '#2ca02c'}
            
            for col in ['French (%)', 'English (%)', 'Other (%)']:
                fig1.add_trace(go.Box(
                    y=data[col],
                    name=col.replace(' (%)', ''),
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=colors[col],
                    boxmean=True,
                    hovertemplate=(
                        "<b>%{y:.1f}%</b><br>" +
                        "Q1: %{quartilemin:.1f}%<br>" +
                        "Median: %{median:.1f}%<br>" +
                        "Q3: %{quartilemax:.1f}%<br>" +
                        "<extra></extra>"
                    )
                ))
            
            fig1.update_layout(
                title=dict(
                    text='Distribution of Language Percentages',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                yaxis=dict(
                    title='Percentage',
                    ticksuffix='%',
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                height=500,
                plot_bgcolor='white'
            )

            # Code-switching analysis
            fig2 = go.Figure()
            
            size_ref = 2.*max(data['Code Switches'])/(40.**2)
            
            fig2.add_trace(go.Scatter(
                x=data['English (%)'],
                y=data['French (%)'],
                mode='markers',
                marker=dict(
                    size=data['Code Switches'],
                    sizeref=size_ref,
                    sizemin=4,
                    color=data['Code Switches'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title='Code Switches',
                        ticksuffix='',
                        outlinecolor='black',
                        outlinewidth=1
                    )
                ),
                text=[
                    f"Document ID: {id}<br>" +
                    f"Category: {cat}<br>" +
                    f"Code Switches: {switches}<br>" +
                    f"English: {en:.1f}%<br>" +
                    f"French: {fr:.1f}%"
                    for id, cat, switches, en, fr in zip(
                        data['Document ID'],
                        data['Category'],
                        data['Code Switches'],
                        data['English (%)'],
                        data['French (%)']
                    )
                ],
                hoverinfo='text'
            ))

            fig2.update_layout(
                title=dict(
                    text='Code-Switching Analysis',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                xaxis=dict(
                    title='English Content (%)',
                    ticksuffix='%',
                    gridcolor='rgba(0,0,0,0.1)',
                    range=[0, 100]
                ),
                yaxis=dict(
                    title='French Content (%)',
                    ticksuffix='%',
                    gridcolor='rgba(0,0,0,0.1)',
                    range=[0, 100]
                ),
                height=600,
                plot_bgcolor='white',
                showlegend=False
            )

            return fig1, fig2

        except Exception as e:
            logger.error(f"Error creating detailed analysis plots: {str(e)}")
            empty_fig = go.Figure().add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return empty_fig, empty_fig

    def create_code_switching_heatmap(self, data):
        """Create a heatmap of code-switching patterns."""
        if data is None or not isinstance(data, dict):
            return go.Figure().add_annotation(
                text="No data available for analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )

        try:
            correlations = data['correlation']
            labels = ['English', 'French', 'Code Switches']
            
            def format_corr(val):
                if pd.isna(val) or val == 0:
                    return 0.0
                return val

            corr_matrix = [
                [1.0, format_corr(correlations['en_fr']), format_corr(correlations['switches_en'])],
                [format_corr(correlations['en_fr']), 1.0, format_corr(correlations['switches_fr'])],
                [format_corr(correlations['switches_en']), format_corr(correlations['switches_fr']), 1.0]
            ]

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=labels,
                y=labels,
                text=[[f'{val:.2f}' for val in row] for row in corr_matrix],
                texttemplate='%{text}',
                textfont={"size": 12},
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))

            fig.update_layout(
                title=dict(
                    text='Correlation Between Languages and Code Switching',
                    x=0.5,
                    xanchor='center'
                ),
                width=600,
                height=500,
                xaxis_title='',
                yaxis_title='',
                xaxis={'side': 'bottom'},
                plot_bgcolor='white'
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating code switching heatmap: {str(e)}")
            return go.Figure().add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )