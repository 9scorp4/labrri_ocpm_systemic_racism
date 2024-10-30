import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, List, Tuple
from loguru import logger

class WordFrequencyVisualizer:
    """
    A class to handle word frequency visualizations with proper data structure validation.
    """
    
    @staticmethod
    def create_frequency_plot(
        data: Union[pd.DataFrame, List[Tuple[str, int]]],
        title: str,
        plot_type: str = 'bar',
    ) -> go.Figure:
        """
        Create a word frequency visualization with proper data structure handling.
        
        Args:
            data: DataFrame or list of (word, frequency) tuples
            title: Plot title
            plot_type: Type of plot ('bar' or 'pie')
            
        Returns:
            Plotly figure object
        """
        # Ensure we have a proper DataFrame
        if isinstance(data, list):
            if not data:
                logger.warning("Empty data provided for visualization")
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                return fig
                
            if isinstance(data[0], tuple):
                df = pd.DataFrame(data, columns=['Word', 'Frequency'])
            else:
                logger.error("Invalid data format provided")
                raise ValueError("Data must be a DataFrame or list of tuples")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            if 'Word' not in df.columns or 'Frequency' not in df.columns:
                # Try to handle common column name variations
                for word_col in ['word', 'Word', 'term', 'Term', 'n-gram', 'N-gram']:
                    for freq_col in ['frequency', 'Frequency', 'count', 'Count']:
                        if word_col in df.columns and freq_col in df.columns:
                            df = df.rename(columns={word_col: 'Word', freq_col: 'Frequency'})
                            break
                    if 'Word' in df.columns and 'Frequency' in df.columns:
                        break
                else:
                    logger.error(f"Invalid DataFrame columns: {df.columns}")
                    raise ValueError("DataFrame must have 'Word' and 'Frequency' columns")
        else:
            logger.error(f"Invalid data type: {type(data)}")
            raise ValueError("Data must be a DataFrame or list of tuples")

        # Create the visualization
        if plot_type == 'bar':
            fig = go.Figure(data=[
                go.Bar(
                    x=df['Frequency'],
                    y=df['Word'],
                    orientation='h',
                    marker_color='rgb(55, 83, 109)'
                )
            ])
            
            fig.update_layout(
                title=dict(
                    text=title,
                    y=0.95,
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis=dict(
                    title='Frequency',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey'
                ),
                yaxis=dict(
                    title='Words/N-grams',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey',
                    categoryorder='total ascending'
                ),
                height=max(400, len(df) * 25),
                margin=dict(l=200, r=20, t=70, b=70),
                showlegend=False,
                plot_bgcolor='white'
            )
            
        elif plot_type == 'pie':
            fig = go.Figure(data=[
                go.Pie(
                    labels=df['Word'],
                    values=df['Frequency'],
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title=dict(
                    text=title,
                    y=0.95,
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                ),
                margin=dict(l=20, r=20, t=70, b=20),
                height=600
            )
            
        return fig

    @staticmethod
    def create_comparison_plot(data, title, comparison_type='category'):
        """
        Create a comparison visualization for word frequencies across categories.
        
        Args:
            data (dict): Dictionary containing frequency data for each category
            title (str): Plot title
            comparison_type (str): Type of comparison ('language' or 'category')
        
        Returns:
            go.Figure: Plotly figure object
        """
        if not data:
            logger.warning("Empty data provided for comparison visualization")
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig

        # Create a unified set of words across all categories
        all_words = set()
        max_freq = 0
        for cat_data in data.values():
            all_words.update(cat_data['words'])
            max_freq = max(max_freq, max(cat_data['frequencies']))

        # Create traces for each category
        fig = go.Figure()
        colors = px.colors.qualitative.Set3[:len(data)]

        for (category, values), color in zip(data.items(), colors):
            # Create a dictionary for easy lookup of frequencies
            freq_dict = dict(zip(values['words'], values['frequencies']))
            
            # Get frequencies for all words (0 if not present in this category)
            frequencies = [freq_dict.get(word, 0) for word in all_words]
            
            fig.add_trace(
                go.Bar(
                    name=category,
                    x=list(all_words),
                    y=frequencies,
                    marker_color=color
                )
            )

        fig.update_layout(
            title=dict(
                text=title,
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(
                title='Words/N-grams',
                tickangle=45,
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey'
            ),
            yaxis=dict(
                title='Frequency',
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey',
                range=[0, max_freq * 1.1]  # Add 10% padding to the top
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            height=600,
            showlegend=True,
            legend=dict(
                title=comparison_type.capitalize(),
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=70, b=100)  # Increased bottom margin for rotated labels
        )

        return fig