import os
import sys
import json
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

from scripts.database import Database
from scripts.word_frequency.analyzer import WordFrequencyAnalyzer
from scripts.word_frequency.visualizer import WordFrequencyVisualizer

class WordFrequencyDashboardApp:
    def __init__(self):
        """Initialize the dashboard application with database connection and required components."""
        try:
            # Previous initialization code remains the same...
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = "localhost"
            db_port = "5432"
            db_name = "labrri_ocpm_systemic_racism"
            
            self.db_path = f"postgresql://{db_user}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}"
            
            self.db = Database(self.db_path)
            self.analyzer = WordFrequencyAnalyzer(self.db_path)
            self.visualizer = WordFrequencyVisualizer()
            
            self.categories = self.db.get_unique_categories()
            
            self.languages = [
                {'label': 'French', 'value': 'fr'},
                {'label': 'English', 'value': 'en'}
            ]
            
            self.current_fig = None
            self.app = self._create_app()
            
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}", exc_info=True)
            raise

    def _create_app(self):
        """Create and configure the Dash app with layout and components."""
        app = Dash(__name__, suppress_callback_exceptions=True)
        
        app.layout = html.Div([
            html.H1('Word Frequency Analysis Dashboard', className='h1'),
            
            # Analysis Mode Selection
            html.Div([
                html.H3('Analysis Mode'),
                dcc.RadioItems(
                    id='analysis-mode',
                    options=[
                        {'label': 'Single Analysis', 'value': 'single'},
                        {'label': 'Comparative Analysis', 'value': 'compare'},
                        {'label': 'TF-IDF Analysis', 'value': 'tfidf'}
                    ],
                    value='single',
                    className='radio-group'
                ),
            ], className='mode-selection'),

            # Left panel - Controls
            html.Div([
                # Analysis Type Selection (only visible in single mode)
                html.Div([
                    html.H3('Analysis Options'),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Category Analysis', 'value': 'category'},
                            {'label': 'Language Analysis', 'value': 'language'}
                        ],
                        value='category',
                        clearable=False
                    ),
                ], className='control-group', id='single-analysis-controls'),

                # Single Selection Dropdown
                html.Div([
                    dcc.Dropdown(
                        id='category-select',
                        placeholder='Select Category/Language'
                    ),
                ], className='control-group', id='single-selection-container'),

                # Multiple Selection Dropdown (for comparative analysis)
                html.Div([
                    dcc.Dropdown(
                        id='compare-select',
                        multi=True,
                        placeholder='Select Categories to Compare'
                    ),
                ], className='control-group', id='compare-selection-container'),

                # TF-IDF Analysis Controls
                html.Div([
                    html.Label('Category:'),
                    dcc.Dropdown(
                        id='tfidf-category-select',
                        options=[{'label': cat, 'value': cat} for cat in self.categories],
                        placeholder='Select Category'
                    ),
                    html.Label('Language:'),
                    dcc.Dropdown(
                        id='tfidf-language-select',
                        options=self.languages,
                        placeholder='Select Language'
                    ),
                ], className='control-group', id='tfidf-controls'),

                # N-gram Selection
                html.Div([
                    html.Label('N-gram Size:'),
                    dcc.Dropdown(
                        id='ngram-size',
                        options=[
                            {'label': 'Single Words', 'value': 1},
                            {'label': 'Bigrams', 'value': 2},
                            {'label': 'Trigrams', 'value': 3}
                        ],
                        value=1,
                        clearable=False
                    ),
                ], className='control-group'),

                # Visualization Type
                html.Div([
                    html.Label('Visualization Type:'),
                    dcc.Dropdown(
                        id='viz-type',
                        options=[
                            {'label': 'Bar Chart', 'value': 'bar'},
                            {'label': 'Pie Chart', 'value': 'pie'}
                        ],
                        value='bar',
                        clearable=False
                    ),
                ], className='control-group'),

                # Number of Words Slider
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

            # Store for current analysis
            dcc.Store(id='current-analysis'),
            
            # Notification area
            html.Div(id='notification', className='notification')
            
        ], className='dashboard-container')

        self._setup_callbacks(app)
        return app

    def _setup_callbacks(self, app):
        @app.callback(
            [Output('single-analysis-controls', 'style'),
             Output('single-selection-container', 'style'),
             Output('compare-selection-container', 'style'),
             Output('tfidf-controls', 'style')],
            Input('analysis-mode', 'value')
        )
        def toggle_analysis_controls(mode):
            """Show/hide controls based on analysis mode."""
            single_style = {'display': 'block'} if mode == 'single' else {'display': 'none'}
            compare_style = {'display': 'block'} if mode == 'compare' else {'display': 'none'}
            tfidf_style = {'display': 'block'} if mode == 'tfidf' else {'display': 'none'}
            return single_style, single_style, compare_style, tfidf_style

        @app.callback(
            [Output('category-select', 'options'),
             Output('category-select', 'value')],
            [Input('analysis-type', 'value'),
             Input('analysis-mode', 'value')]
        )
        def update_selection_dropdown(analysis_type, mode):
            """Update the selection dropdown based on analysis type."""
            if mode != 'single':
                return [], None
            
            if analysis_type == 'language':
                return self.languages, self.languages[0]['value']
            else:
                return [{'label': cat, 'value': cat} for cat in self.categories], self.categories[0]

        @app.callback(
            [Output('compare-select', 'options'),
             Output('compare-select', 'value')],
            Input('analysis-mode', 'value')
        )
        def update_compare_dropdown(mode):
            """Update the comparison dropdown options."""
            if mode != 'compare':
                return [], None
            return [{'label': cat, 'value': cat} for cat in self.categories], None

        @app.callback(
            [Output('visualization-container', 'children'),
            Output('notification', 'children'),
            Output('current-analysis', 'data')],
            [Input('update-button', 'n_clicks'),
            Input('save-button', 'n_clicks')],
            [State('analysis-mode', 'value'),
            State('analysis-type', 'value'),
            State('category-select', 'value'),
            State('compare-select', 'value'),
            State('tfidf-category-select', 'value'),
            State('tfidf-language-select', 'value'),
            State('viz-type', 'value'),
            State('n-words', 'value'),
            State('ngram-size', 'value'),
            State('filename-input', 'value'),
            State('current-analysis', 'data')]
        )
        def update_dashboard(update_clicks, save_clicks, 
                           analysis_mode, analysis_type, selection_value,
                           compare_selection, tfidf_category, tfidf_language,
                           viz_type, n_words, ngram_size,
                           filename, stored_analysis):
            """Handle dashboard updates and save operations."""
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update, no_update

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                if button_id == 'update-button':
                    current_analysis = {
                        'mode': analysis_mode,
                        'type': analysis_type,
                        'viz_type': viz_type,
                        'n_words': n_words,
                        'ngram_size': ngram_size,
                        'timestamp': datetime.now().isoformat()
                    }

                    if analysis_mode == 'single':
                        if not selection_value:
                            raise ValueError("Please make a selection")
                        
                        current_analysis['selection'] = selection_value
                        
                        if analysis_type == 'language':
                            result_df = self.analyzer.get_word_frequencies(
                                category=None,
                                n=n_words,
                                ngram=ngram_size,
                                lang=selection_value
                            )
                        else:
                            result_df = self.analyzer.get_word_frequencies(
                                category=selection_value,
                                n=n_words,
                                ngram=ngram_size
                            )

                        if result_df is None:
                            return html.Div("No data available"), html.Div(
                                "No data available for the selected parameters",
                                style={'color': 'orange'}
                            ), current_analysis

                        title = f"Word Frequencies for {selection_value}"
                        fig = self.visualizer.create_frequency_plot(result_df, title, viz_type)

                    elif analysis_mode == 'compare':
                        if not compare_selection or len(compare_selection) < 2:
                            raise ValueError("Please select at least two categories to compare")
                        
                        current_analysis = {
                            'mode': 'compare',
                            'selection': compare_selection,
                            'n_words': n_words,
                            'ngram_size': ngram_size,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        results = self.analyzer.compare_categories(
                            compare_selection,
                            n=n_words,
                            ngram=ngram_size
                        )

                        if not results:
                            return html.Div("No data available"), html.Div(
                                "No data available for comparison",
                                style={'color': 'orange'}
                            ), current_analysis

                        title = f"Word Frequency Comparison: {', '.join(compare_selection)}"
                        fig = self.visualizer.create_comparison_plot(
                            results,
                            title,
                            comparison_type='category'
                        )

                        self.current_fig = fig
                        self.current_data = pd.DataFrame({
                            'Category': [cat for cat in results.keys() for _ in results[cat]['words']],
                            'Word': [word for cat in results.keys() for word in results[cat]['words']],
                            'Frequency': [freq for cat in results.keys() for freq in results[cat]['frequencies']],
                        })

                        return dcc.Graph(figure=fig), html.Div(
                            "Comparison visualization updated successfully!",
                            style={'color': 'green'}
                        ), current_analysis

                    elif analysis_mode == 'tfidf':
                        if not tfidf_category:
                            raise ValueError("Please provide a category for TF-IDF analysis")
                        
                        current_analysis = {
                            'mode': 'tfidf',
                            'category': tfidf_category,
                            'language': tfidf_language,
                            'viz_type': viz_type,
                            'n_words': n_words,
                            'ngram_size': ngram_size,
                            'timestamp': datetime.now().isoformat()
                        }

                        # Calculate TF-IDF scores
                        result_df = self.analyzer.calculate_tfidf(
                            category=tfidf_category,
                            lang=tfidf_language
                        )

                        if result_df is None or result_df.empty:
                            return html.Div("No data available"), html.Div(
                                "No data available for TF-IDF analysis",
                                style={'color': 'orange'}
                            ), current_analysis
                        
                        # Create visualization
                        title = f"TF-IDF Analysis: {tfidf_category}"
                        if tfidf_language:
                            title += f" ({tfidf_language.upper()})"

                        # Force bar chart for TF-IDF visualization
                        viz_type = 'bar'
                        fig = self.visualizer.create_frequency_plot(
                            result_df,
                            title=title,
                            plot_type=viz_type
                        )

                    self.current_fig = fig
                    self.current_data = result_df

                    return dcc.Graph(figure=fig), html.Div(
                        "TF-IDF visualization updated successfully!",
                        style={'color': 'green'}
                    ), current_analysis

                elif button_id == 'save-button' and stored_analysis:
                    saved_files = self._save_analysis(stored_analysis, filename)
                    notification = html.Div([
                        html.P("Analysis saved successfully!", 
                            style={'color': 'green', 'font-weight': 'bold'}),
                        html.P("Saved files:"),
                        html.Ul([
                            html.Li([
                                html.Span(f"{k}: "),
                                html.A(str(v), href=str(v))
                            ]) for k, v in saved_files.items()
                        ])
                    ])
                    return no_update, notification, stored_analysis

                return no_update, html.Div("No analysis to save", style={'color': 'red'}), no_update

            except ValueError as ve:
                logger.warning(f"Validation error: {str(ve)}")
                return (
                    html.Div("No data available for visualization"),
                    html.Div(str(ve), style={'color': 'orange'}),
                    no_update
                )
            except Exception as e:
                logger.error(f"Error in callback: {str(e)}", exc_info=True)
                return (
                    html.Div("Error occurred during visualization"),
                    html.Div(str(e), style={'color': 'red'}),
                    no_update
                )

    def _save_analysis(self, analysis_data, filename=None):
        """Save the current analysis results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"word_frequency_analysis_{timestamp}"
        
        save_path = Path('results/word_frequency')
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save analysis parameters
            params_path = save_path / f"{filename}_params.json"
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            saved_files['parameters'] = params_path
            
            # Save current visualization if it exists
            if self.current_fig:
                # Save interactive HTML version
                fig_path = save_path / f"{filename}_plot.html"
                self.current_fig.write_html(str(fig_path))
                saved_files['visualization'] = fig_path
                
                # Save static PNG version
                png_path = save_path / f"{filename}_plot.png"
                self.current_fig.write_image(str(png_path))
                saved_files['image'] = png_path
                
                # If we have word frequency data, save it as CSV and Excel
                if hasattr(self, 'current_data') and self.current_data is not None:
                    csv_path = save_path / f"{filename}_data.csv"
                    self.current_data.to_csv(csv_path, index=False)
                    saved_files['csv'] = csv_path
                    
                    excel_path = save_path / f"{filename}_data.xlsx"
                    self.current_data.to_excel(excel_path, index=False)
                    saved_files['excel'] = excel_path

            # Create a summary text file
            summary_path = save_path / f"{filename}_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Word Frequency Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Type: {analysis_data['type']}\n")
                f.write(f"Selection: {analysis_data['selection']}\n")
                f.write(f"Visualization Type: {analysis_data['viz_type']}\n")
                f.write(f"Number of Words: {analysis_data['n_words']}\n")
                f.write(f"N-gram Size: {analysis_data['ngram_size']}\n")
                f.write(f"Timestamp: {analysis_data['timestamp']}\n")
            saved_files['summary'] = summary_path

            logger.info(f"Analysis saved successfully to {save_path}")
            return saved_files
                
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise

    def run_server(self, debug=True, port=8051):
        """Run the dashboard server."""
        try:
            logger.info(f"Starting dashboard server on port {port}")
            self.app.run_server(debug=debug, port=port)
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            raise

if __name__ == "__main__":
    # Configure logging
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"word_frequency_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger.remove()
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
    
    try:
        dashboard = WordFrequencyDashboardApp()
        dashboard.run_server()
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}", exc_info=True)
        raise