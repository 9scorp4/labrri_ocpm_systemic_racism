from matplotlib_venn import venn3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import matplotlib.pyplot as plt
import io
import base64
from loguru import logger

class KnowledgeTypeVisualizer:
    def create_distribution_plot(self, data, title, plot_type='pie'):
        """Create basic distribution visualization."""
        if data is None or data.empty:
            return self._create_empty_figure("No data available")
            
        try:
            if plot_type == 'pie':
                fig = go.Figure(data=[
                    go.Pie(
                        labels=data['knowledge_type'],
                        values=data['count'],
                        hole=0.3,
                        textinfo='label+percent',
                        textposition='inside',
                        marker=dict(
                            colors=px.colors.qualitative.Set3
                        )
                    )
                ])
            else:  # bar
                fig = go.Figure(data=[
                    go.Bar(
                        x=data['knowledge_type'],
                        y=data['count'],
                        text=data['percentage'].round(1).astype(str) + '%',
                        textposition='auto',
                        marker_color=px.colors.qualitative.Set3
                    )
                ])
            
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor='center'),
                showlegend=(plot_type == 'pie'),
                height=500,
                margin=dict(t=100, l=50, r=50, b=50),
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def create_venn_diagram(self, data):
        """Create Venn diagram for knowledge type intersections."""
        try:
            # Calculate set sizes for each knowledge type
            citoyen = set(data[data['Citoyen']].index)
            communautaire = set(data[data['Communautaire']].index)
            municipal = set(data[data['Municipal']].index)

            # Calculate all possible intersections
            c_com = len(citoyen.intersection(communautaire))
            c_m = len(citoyen.intersection(municipal))
            com_m = len(communautaire.intersection(municipal))
            c_com_m = len(citoyen.intersection(communautaire).intersection(municipal))

            # Calculate exclusive sets
            only_c = len(citoyen) - c_com - c_m + c_com_m
            only_com = len(communautaire) - c_com - com_m + c_com_m
            only_m = len(municipal) - c_m - com_m + c_com_m

            # Create a new figure with matplotlib
            plt.figure(figsize=(10, 10))
            
            # Create the Venn diagram
            venn3(
                subsets=(only_c, only_com, only_m, c_com - c_com_m,
                        c_m - c_com_m, com_m - c_com_m, c_com_m),
                set_labels=('Citoyen', 'Communautaire', 'Municipal'),
                set_colors=('#ff9999', '#66b3ff', '#99ff99')
            )
            
            # Save the matplotlib figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            plt.close()
            buf.seek(0)
            
            # Convert the image to base64
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Create a Plotly figure with the image
            fig = go.Figure()
            
            fig.add_layout_image(
                dict(
                    source=f'data:image/png;base64,{image_base64}',
                    x=0,
                    y=0,
                    xref="paper",
                    yref="paper",
                    sizex=1,
                    sizey=1,
                    sizing="contain",
                    layer="below"
                )
            )
            
            fig.update_layout(
                title=dict(
                    text='Knowledge Type Intersections',
                    x=0.5,
                    xanchor='center'
                ),
                width=800,
                height=800,
                margin=dict(t=50, l=50, r=50, b=50),
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Venn diagram: {str(e)}")
            return self._create_empty_figure(f"Error creating Venn diagram: {str(e)}")
    
    def create_heatmap(self, counts, percentages, category_name):
        """Create heatmap for cross-analysis visualization."""
        if counts is None or percentages is None or counts.empty or percentages.empty:
            return self._create_empty_figure("No data available")
            
        try:
            # Extract values without x/y prefixes
            z_values = percentages.values
            col_values = list(percentages.columns)
            row_values = list(percentages.index)
            text_values = counts.values

            # Create heatmap figure
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=col_values,
                y=row_values,
                colorscale='RdBu',
                text=text_values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate=(
                    "Knowledge Type: %{y}<br>" + 
                    category_name + ": %{x}<br>" +  # Concatenate the dynamic category name
                    "Count: %{text}<br>" +
                    "Percentage: %{z:.1f}%<br>" +
                    "<extra></extra>"
                )
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'Knowledge Type Distribution by {category_name.replace("_", " ").title()}',
                    x=0.5,
                    xanchor='center'
                ),
                height=max(400, len(row_values) * 40),
                width=max(800, len(col_values) * 60),
                margin=dict(t=100, l=200, r=50, b=100),
                xaxis_title=category_name.replace('_', " ").title(),
                yaxis_title='Knowledge Type',
                yaxis_tickangle=0,
                xaxis_tickangle=45,
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return self._create_empty_figure(f"Error: {str(e)}")