import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from loguru import logger
from urllib.parse import quote_plus
import psycopg2

from scripts.database import Database
from scripts.word_frequency import WordFrequencyChart

class WordFrequencyDashboardApp:
    def __init__(self):
        try:
            # Get database credentials from environment variables
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = "localhost"
            db_port = "5432"
            db_name = "labrri_ocpm_systemic_racism"
            
            # Create the database connection string
            self.db_path = f"postgresql://{db_user}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}"
            
            # Initialize the database and word frequency chart
            self.db = Database(self.db_path)
            self.wfc = WordFrequencyChart(self.db_path)
            self.categories = self.db.get_unique_categories()
            self.languages = ['fr', 'en']
            self.app = self._create_app()
        except ValueError as e:
            logger.error(f"Database configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}")
            raise

    def _create_app(self):
        """Create and configure the Dash app"""
        app = Dash(__name__)
        
        app.layout = html.Div([
            html.H1('Word Frequency Analysis Dashboard', className='h1'),
            
            # Left panel - Controls
            html.Div([
                # Analysis Type Selection
                html.Div([
                    html.H3('Analysis Options'),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Category Analysis', 'value': 'category'},
                            {'label': 'Language Analysis', 'value': 'language'},
                            {'label': 'Compare Languages', 'value': 'compare-languages'},
                            {'label': 'Compare Categories', 'value': 'compare-categories'},
                            {'label': 'TF-IDF Category', 'value': 'tfidf-category'},
                            {'label': 'TF-IDF Language', 'value': 'tfidf-language'}
                        ],
                        value='category'
                    ),
                ], className='control-group'),

                # Category Selection
                html.Div([
                    dcc.Dropdown(
                        id='category-select',
                        options=[{'label': cat, 'value': cat} for cat in self.categories],
                        value=self.categories[0] if self.categories else None,
                        placeholder='Select Category',
                        style={'display': 'none'}
                    ),
                ], className='control-group'),

                # Language Selection
                html.Div([
                    dcc.Dropdown(
                        id='language-select',
                        options=[{'label': lang.upper(), 'value': lang} for lang in self.languages],
                        value='fr',
                        placeholder='Select Language',
                        style={'display': 'none'}
                    ),
                ], className='control-group'),

                # N and N-gram settings
                html.Div([
                    html.Label('Number of Words:'),
                    dcc.Slider(
                        id='n-words',
                        min=5,
                        max=50,
                        step=5,
                        value=20,
                        marks={i: str(i) for i in range(5, 51, 5)}
                    ),
                    html.Label('N-gram Size:'),
                    dcc.Slider(
                        id='n-gram',
                        min=1,
                        max=3,
                        step=1,
                        value=1,
                        marks={i: str(i) for i in range(1, 4)}
                    ),
                ], className='control-group'),

                # Categories for comparison
                html.Div([
                    dcc.Dropdown(
                        id='categories-compare',
                        options=[{'label': cat, 'value': cat} for cat in self.categories],
                        value=[self.categories[0], self.categories[1]] if len(self.categories) > 1 else [],
                        multi=True,
                        style={'display': 'none'}
                    ),
                ], className='control-group'),

                # Update and Save buttons
                html.Div([
                    html.Button('Update', id='update-button', n_clicks=0),
                    dcc.Input(id='filename-input', type='text', placeholder='Filename (optional)'),
                    html.Button('Save', id='save-button', n_clicks=0),
                ], className='button-group'),

            ], className='control-panel'),

            # Right panel - Visualization
            html.Div([
                dcc.Loading(
                    id="loading",
                    type="cube",
                    children=html.Div(id='visualization-container')
                )
            ], className='visualization-panel'),

            # Notification area
            html.Div(id='notification', className='notification')
            
        ], className='dashboard-container')

        self._setup_callbacks(app)
        return app

    def _setup_callbacks(self, app):
        @app.callback(
            [Output('category-select', 'style'),
             Output('language-select', 'style'),
             Output('categories-compare', 'style')],
            Input('analysis-type', 'value')
        )
        def toggle_inputs(analysis_type):
            category_style = {'display': 'block'} if analysis_type in ['category', 'tfidf-category'] else {'display': 'none'}
            language_style = {'display': 'block'} if analysis_type in ['language', 'tfidf-language'] else {'display': 'none'}
            categories_compare_style = {'display': 'block'} if analysis_type == 'compare-categories' else {'display': 'none'}
            return category_style, language_style, categories_compare_style

        @app.callback(
            [Output('visualization-container', 'children'),
             Output('notification', 'children')],
            [Input('update-button', 'n_clicks'),
             Input('save-button', 'n_clicks')],
            [State('analysis-type', 'value'),
             State('category-select', 'value'),
             State('language-select', 'value'),
             State('n-words', 'value'),
             State('n-gram', 'value'),
             State('categories-compare', 'value'),
             State('filename-input', 'value')]
        )
        def update_visualization(update_clicks, save_clicks, analysis_type, category, 
                               language, n_words, n_gram, categories_compare, filename):
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                if button_id == 'update-button':
                    if analysis_type == 'category':
                        result = self.wfc.top_n_words(category, n=n_words, ngram=n_gram)
                        fig = self._create_word_frequency_figure(result, f"Top {n_words} {n_gram}-grams in category: {category}")
                    elif analysis_type == 'language':
                        result = self.wfc.top_n_words(language, n=n_words, ngram=n_gram, lang=language)
                        fig = self._create_word_frequency_figure(result, f"Top {n_words} {n_gram}-grams in language: {language}")
                    elif analysis_type == 'compare-languages':
                        self.wfc.compare_languages(n=n_words, ngram=n_gram)
                        fig = self._create_language_comparison_figure(n_words, n_gram)
                    elif analysis_type == 'compare-categories':
                        self.wfc.compare_categories(categories_compare, n=n_words, ngram=n_gram)
                        fig = self._create_category_comparison_figure(categories_compare, n_words, n_gram)
                    elif analysis_type == 'tfidf-category':
                        result = self.wfc.tfidf_analysis(category)
                        fig = self._create_tfidf_figure(result, f"TF-IDF Analysis for category: {category}")
                    else:  # tfidf-language
                        result = self.wfc.tfidf_analysis(language, lang=language)
                        fig = self._create_tfidf_figure(result, f"TF-IDF Analysis for language: {language}")

                    return dcc.Graph(figure=fig), html.Div("Visualization updated successfully!", 
                                                         style={'color': 'green'})

                elif button_id == 'save-button':
                    if not filename:
                        filename = f"word_frequency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    save_path = Path('results/word_frequency') / f"{filename}.html"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save the current figure
                    if 'figure' in locals():
                        fig.write_html(str(save_path))
                        return no_update, html.Div(f"Analysis saved to {save_path}", 
                                                 style={'color': 'green'})
                    
                    return no_update, html.Div("No visualization to save", 
                                             style={'color': 'red'})

            except Exception as e:
                logger.error(f"Error in visualization update: {str(e)}")
                return no_update, html.Div(f"Error: {str(e)}", style={'color': 'red'})

    def _create_word_frequency_figure(self, result, title):
        """Create word frequency visualization using Plotly Graph Objects"""
        if result is None or result.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for the selected parameters",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title_text="No data available")
            return fig

        # Determine the correct column names based on result DataFrame
        if 'n-gram' in result.columns:
            x_col = 'Frequency'
            y_col = 'n-gram'
        elif 'Word' in result.columns:
            x_col = 'TF-IDF Score' if 'TF-IDF Score' in result.columns else 'Frequency'
            y_col = 'Word'
        else:
            gram_col = next((col for col in result.columns if 'gram' in col.lower()), None)
            if gram_col:
                x_col = 'Frequency'
                y_col = gram_col
            else:
                logger.error(f"Unexpected DataFrame columns: {result.columns}")
                fig = go.Figure()
                fig.add_annotation(
                    text="Could not process the data format",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(title_text="Error: Unexpected data format")
                return fig

        # Create the figure using Graph Objects
        fig = go.Figure()

        # Add horizontal bar trace
        fig.add_trace(
            go.Bar(
                x=result[x_col],
                y=result[y_col],
                orientation='h',
                marker_color='rgb(55, 83, 109)'
            )
        )

        # Update layout with consistent styling
        fig.update_layout(
            title=dict(
                text=title,
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(
                title=x_col,
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
            height=max(400, len(result) * 25),  # Dynamic height based on number of terms
            margin=dict(l=200, r=20, t=70, b=70),  # Adjust margins for better label visibility
            showlegend=False,
            plot_bgcolor='white'
        )

        return fig

    def _create_language_comparison_figure(self, n_words, n_gram):
        """Create language comparison visualization using Plotly Graph Objects"""
        try:
            # Get comparison data for each language
            results = {}
            for lang in self.languages:
                df = self.wfc.top_n_words(lang, n=n_words, ngram=n_gram, lang=lang)
                if df is not None and not df.empty:
                    results[lang] = df

            if not results:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(title_text="No data available for language comparison")
                return fig

            # Create figure with Graph Objects
            fig = go.Figure()

            colors = {'fr': 'rgb(55, 83, 109)', 'en': 'rgb(26, 118, 255)'}

            # Add bars for each language
            for lang, df in results.items():
                gram_col = next(col for col in df.columns if 'gram' in col.lower())
                fig.add_trace(
                    go.Bar(
                        name=lang.upper(),
                        y=df[gram_col],
                        x=df['Frequency'],
                        orientation='h',
                        offsetgroup=lang,
                        marker_color=colors.get(lang)
                    )
                )

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Top {n_words} {n_gram}-grams Comparison Across Languages",
                    y=0.95,
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis=dict(
                    title="Frequency",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey'
                ),
                yaxis=dict(
                    title=f"{n_gram}-grams",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey'
                ),
                barmode='group',
                height=max(400, n_words * 25 * len(self.languages)),
                legend=dict(
                    title="Language",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                showlegend=True,
                plot_bgcolor='white',
                margin=dict(l=200, r=20, t=70, b=70)
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating language comparison figure: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=str(e),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title_text="Error creating visualization")
            return fig

    def _create_category_comparison_figure(self, categories, n_words, n_gram):
        """
        Create category comparison visualization using Plotly Graph Objects.
        
        Args:
            categories (list): List of categories to compare
            n_words (int): Number of words to show
            n_gram (int): Size of n-grams
            
        Returns:
            go.Figure: Plotly figure object for category comparison
        """
        try:
            # Get comparison data for each category
            results = {}
            for category in categories:
                df = self.wfc.top_n_words(category, n=n_words, ngram=n_gram)
                if df is not None and not df.empty:
                    results[category] = df

            if not results:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(title_text="No data available for category comparison")
                return fig

            # Create figure with Graph Objects
            fig = go.Figure()

            # Generate a color palette based on number of categories
            colors = px.colors.qualitative.Set3[:len(categories)]
            color_map = dict(zip(categories, colors))

            # Add bars for each category
            for category, df in results.items():
                gram_col = next(col for col in df.columns if 'gram' in col.lower())
                fig.add_trace(
                    go.Bar(
                        name=category,
                        y=df[gram_col],
                        x=df['Frequency'],
                        orientation='h',
                        offsetgroup=category,
                        marker_color=color_map.get(category)
                    )
                )

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Top {n_words} {n_gram}-grams Comparison Across Categories",
                    y=0.95,
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis=dict(
                    title="Frequency",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey'
                ),
                yaxis=dict(
                    title=f"{n_gram}-grams",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey'
                ),
                barmode='group',
                height=max(400, n_words * 25 * len(categories)),
                legend=dict(
                    title="Category",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                showlegend=True,
                plot_bgcolor='white',
                margin=dict(l=200, r=20, t=70, b=70)
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating category comparison figure: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=str(e),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title_text="Error creating visualization")
            return fig    

    def _create_tfidf_figure(self, result, title):
        """Create TF-IDF visualization using Plotly Graph Objects"""
        if result is None or result.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No TF-IDF data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig

        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=result['TF-IDF Score'],
                y=result['Word'],
                orientation='h',
                marker_color='rgb(55, 83, 109)'
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
                title='TF-IDF Score',
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey'
            ),
            yaxis=dict(
                title='Words',
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey',
                categoryorder='total ascending'
            ),
            height=600,
            margin=dict(l=200, r=20, t=70, b=70),
            showlegend=False,
            plot_bgcolor='white'
        )

        return fig

    def run_server(self, debug=True, port=8051):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    # Configure logging
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"word_frequency_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger.remove()  # Remove default handler
    logger.add(
        log_file,
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Start the dashboard
    dashboard = WordFrequencyDashboardApp()
    dashboard.run_server()