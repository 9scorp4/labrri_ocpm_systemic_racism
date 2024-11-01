import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
from loguru import logger
from urllib.parse import quote_plus

from scripts.database import Database
from scripts.sentiment_analysis.analyzer import SentimentAnalyzer
from scripts.sentiment_analysis.visualizer import SentimentVisualizer
from scripts.sentiment_analysis.data_handler import SentimentDataHandler

class SentimentDashboardApp:
    def __init__(self):
        try:
            # Initialize database connection
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = "localhost"
            db_port = "5432"
            db_name = "labrri_ocpm_systemic_racism"
            
            self.db_path = f"postgresql://{db_user}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}"
            
            # Initialize components
            self.analyzer = SentimentAnalyzer(self.db_path)
            self.visualizer = SentimentVisualizer()
            
            # Get available categories
            self.db = Database(self.db_path)
            self.categories = self.db.get_unique_categories()
            
            # Create Dash app
            self.app = self._create_app()
            
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}", exc_info=True)
            raise

    def _create_app(self):
        """Create and configure the Dash app"""
        app = Dash(__name__)
        
        app.layout = html.Div([
            html.H1('Sentiment Analysis Dashboard', className='h1'),
            
            # Left panel - Controls
            html.Div([
                # Analysis Type Selection
                html.Div([
                    html.H3('Analysis Type'),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Basic Sentiment Distribution', 'value': 'basic'},
                            {'label': 'Aspect-Based Analysis', 'value': 'aspect'},
                            {'label': 'Category Comparison', 'value': 'comparative'}
                        ],
                        value='basic'
                    ),
                ], className='control-group'),

                # Category Selection
                html.Div([
                    html.H3('Category Filter'),
                    dcc.Dropdown(
                        id='category-select',
                        options=[{'label': cat, 'value': cat} for cat in self.categories],
                        value='All categories',
                        multi=True
                    ),
                ], className='control-group'),

                # Visualization Type (only shown for basic analysis)
                html.Div([
                    html.H3('Visualization Type'),
                    dcc.Dropdown(
                        id='viz-type',
                        options=[
                            {'label': 'Distribution Histogram', 'value': 'histogram'},
                            {'label': 'Box Plot', 'value': 'box'},
                            {'label': 'Category Heatmap', 'value': 'heatmap'}
                        ],
                        value='histogram'
                    ),
                ], className='control-group', id='viz-type-container'),

                # Update and Save buttons
                html.Div([
                    html.Button('Update', id='update-button', n_clicks=0),
                    dcc.Input(
                        id='filename-input',
                        type='text',
                        placeholder='Filename (optional)'
                    ),
                    html.Button('Save', id='save-button', n_clicks=0),
                ], className='button-group'),

            ], className='control-panel'),

            # Right panel - Visualization
            html.Div([
                dcc.Loading(
                    id="loading",
                    type="cube",
                    children=[
                        html.Div(id='visualization-container'),
                        html.Div(id='statistics-container', className='statistics')
                    ]
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
            Output('viz-type-container', 'style'),
            Input('analysis-type', 'value')
        )
        def toggle_viz_type(analysis_type):
            """Show/hide visualization type dropdown based on analysis type."""
            if analysis_type == 'basic':
                return {'display': 'block'}
            return {'display': 'none'}

        @app.callback(
            [Output('visualization-container', 'children'),
             Output('statistics-container', 'children'),
             Output('current-analysis', 'data'),
             Output('notification', 'children')],
            [Input('update-button', 'n_clicks'),
             Input('save-button', 'n_clicks')],
            [State('analysis-type', 'value'),
             State('category-select', 'value'),
             State('viz-type', 'value'),
             State('filename-input', 'value'),
             State('current-analysis', 'data')]
        )
        def update_dashboard(update_clicks, save_clicks, 
                           analysis_type, categories, viz_type,
                           filename, current_analysis):
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update, no_update, no_update

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                if button_id == 'update-button':
                    # Ensure categories is a list
                    if isinstance(categories, str):
                        categories = [categories]
                    elif not categories:
                        categories = ["All categories"]

                    # Store current analysis parameters
                    current_analysis = {
                        'type': analysis_type,
                        'categories': categories,
                        'viz_type': viz_type,
                        'timestamp': datetime.now().isoformat()
                    }

                    if analysis_type == 'basic':
                        data = self.analyzer.get_basic_sentiment(
                            categories[0] if categories and categories[0] != 'All categories' else None
                        )
                        if data is None:
                            return self._create_error_response("No data available for basic analysis")
                        
                        formatted_data = SentimentDataHandler.validate_and_format_basic_sentiment(data)
                        if formatted_data is None:
                            return self._create_error_response("Error formatting basic sentiment data")
                        
                        # Create visualization
                        try:
                            fig = self.visualizer.create_basic_sentiment_plot(
                                formatted_data,
                                title=f"Sentiment Distribution - {categories[0]}",
                                plot_type=viz_type
                            )
                            graph = dcc.Graph(figure=fig)
                        except Exception as e:
                            logger.error(f"Error creating visualization: {str(e)}")
                            return self._create_error_response(f"Error creating visualization: {str(e)}")
                        
                        # Create statistics
                        stats = self._create_basic_stats_component(
                            SentimentDataHandler.calculate_basic_summary(formatted_data)
                        )

                        return graph, stats, current_analysis, html.Div(
                            "Analysis updated successfully",
                            style={'color': 'green'}
                        )

                    elif analysis_type == 'aspect':
                        data = self.analyzer.get_aspect_based_sentiment(categories[0] if categories else None)
                        if data is None:
                            return self._create_error_response("No data available for aspect analysis")
                        
                        formatted_data = SentimentDataHandler.validate_and_format_aspect_sentiment(data)
                        if formatted_data is None:
                            return self._create_error_response("Error formatting aspect sentiment data")
                        
                        fig = self.visualizer.create_aspect_sentiment_plot(
                            formatted_data,
                            title="Sentiment Analysis by Aspect"
                        )
                        stats = self._create_aspect_stats_component(
                            SentimentDataHandler.calculate_aspect_summary(formatted_data)
                        )

                    elif analysis_type == 'comparative':
                        if not categories or len(categories) < 2:
                            return self._create_error_response("Please select at least two categories for comparison")
                        
                        data = self.analyzer.get_comparative_sentiment(categories)
                        if data is None:
                            return self._create_error_response("No data available for comparison")
                        
                        formatted_data = SentimentDataHandler.validate_and_format_comparative_sentiment(data)
                        if formatted_data is None:
                            return self._create_error_response("Error formatting comparative sentiment data")
                        
                        fig = self.visualizer.create_comparative_sentiment_plot(
                            formatted_data,
                            title="Sentiment Comparison across Categories"
                        )
                        stats = self._create_comparative_stats_component(formatted_data)

                    return dcc.Graph(figure=fig), stats, current_analysis, html.Div(
                        "Analysis updated successfully!",
                        style={'color': 'green'}
                    )

                elif button_id == 'save-button' and current_analysis:
                    saved_files = self._save_analysis(current_analysis, filename)
                    return (
                        no_update,
                        no_update,
                        current_analysis,
                        html.Div([
                            html.P("Analysis saved successfully!", 
                                  style={'color': 'green'}),
                            html.P("Saved files:"),
                            html.Ul([html.Li(str(f)) for f in saved_files])
                        ])
                    )

            except Exception as e:
                logger.error(f"Error in callback: {str(e)}", exc_info=True)
                return self._create_error_response(str(e))

            return no_update, no_update, no_update, no_update

    def _create_error_response(self, message):
        """Create a consistent error response."""
        return (
            html.Div("Error occurred during visualization"),
            None,
            no_update,
            html.Div(message, style={'color': 'red'})
        )

    def _create_basic_stats_component(self, stats):
        """Create statistics component for basic sentiment analysis."""
        if not stats:
            return html.Div("No statistics available")
            
        return html.Div([
            html.H3("Summary Statistics"),
            html.P(f"Total Documents: {stats['total_documents']}"),
            html.P(f"Average Sentiment: {stats['average_sentiment']:.3f} (±{stats['sentiment_std']:.3f})"),
            html.Div([
                html.P("Document Distribution:"),
                html.Ul([
                    html.Li(f"Positive: {stats['positive_docs']} ({stats['positive_docs']/stats['total_documents']*100:.1f}%)"),
                    html.Li(f"Negative: {stats['negative_docs']} ({stats['negative_docs']/stats['total_documents']*100:.1f}%)"),
                    html.Li(f"Neutral: {stats['neutral_docs']} ({stats['neutral_docs']/stats['total_documents']*100:.1f}%)")
                ])
            ])
        ])

    def _create_aspect_stats_component(self, stats):
        """Create statistics component for aspect-based analysis."""
        if not stats:
            return html.Div("No statistics available")
            
        return html.Div([
            html.H3("Aspect Statistics"),
            *[
                html.Div([
                    html.H4(aspect.title()),
                    html.P(f"Average Sentiment: {data['average_sentiment']:.3f}"),
                    html.P(f"Total Mentions: {data['total_mentions']}"),
                    html.P(f"Documents with Mentions: {data['documents_with_mentions']}")
                ]) for aspect, data in stats.items()
            ]
        ])

    def _create_comparative_stats_component(self, data):
        """Create statistics component for comparative analysis."""
        return html.Div([
            html.H3("Comparative Statistics"),
            html.P(f"Categories Analyzed: {len(data)}"),
            html.P(f"Most Positive Category: {data.loc[data['avg_sentiment'].idxmax(), 'category']} "
                  f"({data['avg_sentiment'].max():.3f})"),
            html.P(f"Most Negative Category: {data.loc[data['avg_sentiment'].idxmin(), 'category']} "
                  f"({data['avg_sentiment'].min():.3f})")
        ])

    def _save_analysis(self, analysis_data, filename=None):
        """Save the current analysis results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}"
            
        save_path = Path('results/sentiment_analysis')
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Save analysis parameters
        params_file = save_path / f"{filename}_params.json"
        with open(params_file, 'w') as f:
            import json
            json.dump(analysis_data, f, indent=2)
        saved_files.append(params_file)

        return saved_files

    def run_server(self, debug=True, port=8054):
        """Run the dashboard server"""
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
    log_file = log_dir / f"sentiment_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
        dashboard = SentimentDashboardApp()
        dashboard.run_server()
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}", exc_info=True)
        raise