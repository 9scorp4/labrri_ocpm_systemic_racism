import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from urllib.parse import quote_plus

from scripts.language_distribution.analyzer import LanguageDistributionAnalyzer
from scripts.language_distribution.visualizer import LanguageDistributionVisualizer
from scripts.language_distribution.data_handler import LanguageDistributionDataHandler

class LanguageDistributionDashboard:
    def __init__(self):
        try:
            # Initialize database connection
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = "localhost"
            db_port = "5432"
            db_name = "labrri_ocpm_systemic_racism"
            
            self.db_path = f"postgresql://{db_user}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}"
            
            # Initialize analyzer and visualizer
            self.analyzer = LanguageDistributionAnalyzer(self.db_path)
            self.visualizer = LanguageDistributionVisualizer()
            
            # Pre-fetch categories for initialization
            self.categories = self._get_categories()
            
            # Create Dash app
            self.app = self._create_app()
            
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}", exc_info=True)
            raise

    def _get_categories(self):
        """Fetch all unique categories from the database."""
        try:
            # Create a database connection using the analyzer's database connection
            db = self.analyzer._get_db_connection()
            
            # Use the db.df_from_query method to execute the query
            query = """
                SELECT DISTINCT category 
                FROM documents 
                WHERE category IS NOT NULL AND category != ''
                ORDER BY category;
            """
            
            df = db.df_from_query(query)
            
            if df is not None and not df.empty:
                # Extract categories and ensure they're strings
                categories = [str(cat) for cat in df['category'].unique() if pd.notna(cat)]
                # Add 'All categories' at the beginning and sort the rest
                all_categories = ['All categories'] + sorted(categories)
                logger.info(f"Successfully fetched {len(categories)} categories from database")
                return all_categories
            else:
                logger.warning("No categories found in database")
                return ['All categories']
                
        except Exception as e:
            logger.error(f"Error fetching categories: {str(e)}")
            return ['All categories']

    def _create_app(self):
        """Create and configure the Dash app"""
        app = Dash(__name__)
        
        # Fetch categories first
        categories = self._get_categories()
        
        app.layout = html.Div([
            html.H1('Language Distribution Analysis Dashboard', className='h1'),
            
            # Left panel - Controls
            html.Div([
                # Analysis Type Selection
                html.Div([
                    html.H3('Analysis Type'),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Basic Distribution', 'value': 'basic'},
                            {'label': 'Detailed Analysis', 'value': 'detailed'},
                            {'label': 'Category Comparison', 'value': 'category'},
                            {'label': 'Code-Switching Analysis', 'value': 'code-switching'}
                        ],
                        value='basic'
                    ),
                ], className='control-group'),

                # Category Selection - Updated with explicit options
                html.Div([
                    html.H3('Category Filter'),
                    dcc.Dropdown(
                        id='category-select',
                        options=[{'label': cat, 'value': cat} for cat in categories],
                        value='All categories',
                        clearable=False
                    ),
                ], className='control-group'),

                # Visualization Options
                html.Div([
                    html.H3('Visualization Type'),
                    dcc.Dropdown(
                        id='viz-type',
                        options=[
                            {'label': 'Pie Chart', 'value': 'pie'},
                            {'label': 'Bar Chart', 'value': 'bar'}
                        ],
                        value='pie',
                        clearable=False
                    ),
                ], className='control-group'),

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
                        html.Div(id='primary-visualization'),
                        html.Div(id='secondary-visualization', style={'marginTop': '20px'})
                    ]
                )
            ], className='visualization-panel'),

            # Store for current analysis data
            dcc.Store(id='current-analysis'),
            
            # Notification area
            html.Div(id='notification', className='notification')
            
        ], className='dashboard-container')

        self._setup_callbacks(app)
        return app

    def _setup_callbacks(self, app):
        @app.callback(
            [Output('category-select', 'disabled'),
             Output('category-select', 'value')],
            Input('analysis-type', 'value')
        )
        def toggle_category_select(analysis_type):
            """Enable/disable category selection based on analysis type."""
            if analysis_type in ['basic', 'detailed', 'code-switching']:
                return False, 'All categories'
            return True, 'All categories'

        @app.callback(
            [Output('primary-visualization', 'children'),
             Output('secondary-visualization', 'children'),
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
        
        def update_visualization(update_clicks, save_clicks, analysis_type, category, viz_type, filename, current_analysis):
            """Update visualization based on user input."""
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update, no_update, no_update

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                if button_id == 'update-button':
                    # Store current analysis parameters
                    current_analysis = {
                        'type': analysis_type,
                        'category': category,
                        'viz_type': viz_type,
                        'timestamp': datetime.now().isoformat()
                    }

                    # Create visualizer instance to be used for all visualizations
                    visualizer = LanguageDistributionVisualizer()

                    if analysis_type == 'basic':
                        logger.debug(f"Fetching basic distribution for category: {category}")
                        raw_data = self.analyzer.get_language_counts(category)
                        logger.debug(f"Raw data received:\n{raw_data}")
                        
                        formatted_data = LanguageDistributionDataHandler.validate_and_format_basic_distribution(raw_data)
                        logger.debug(f"Formatted data:\n{formatted_data}")

                        if formatted_data is None or formatted_data.empty:
                            return html.Div("No data available"), None, current_analysis, html.Div(
                                "No data found for the selected category",
                                style={'color': 'orange'}
                            )

                        fig = visualizer.create_distribution_plot(
                            formatted_data,
                            f"Language Distribution - {category}",
                            viz_type
                        )
                        
                        return dcc.Graph(figure=fig), None, current_analysis, html.Div(
                            "Analysis updated successfully!",
                            style={'color': 'green'}
                        )

                    elif analysis_type == 'detailed':
                        data = self.analyzer.analyze_language_content()
                        if data is None or data.empty:
                            return (
                                html.Div("No data available"), 
                                None,
                                current_analysis,
                                html.Div("No data found for detailed analysis", style={'color': 'orange'})
                            )
                        
                        fig1, fig2 = visualizer.create_detailed_analysis_plots(data)
                        return (
                            dcc.Graph(figure=fig1),
                            dcc.Graph(figure=fig2),
                            current_analysis,
                            html.Div("Detailed analysis updated!", style={'color': 'green'})
                        )

                    elif analysis_type == 'category':
                        data = self.analyzer.get_category_summary()
                        if data is None:
                            return (
                                html.Div("No data available"),
                                None,
                                current_analysis,
                                html.Div("No data found for category comparison", style={'color': 'orange'})
                            )
                        
                        fig = visualizer.create_category_comparison(data)
                        return (
                            dcc.Graph(figure=fig),
                            None,
                            current_analysis,
                            html.Div("Category comparison updated!", style={'color': 'green'})
                        )

                    elif analysis_type == 'code-switching':
                        data = self.analyzer.get_code_switching_analysis(category)
                        if not data:
                            return (
                                html.Div("No data available"),
                                None,
                                current_analysis,
                                html.Div("No data found for code-switching analysis", style={'color': 'orange'})
                            )
                        
                        fig = visualizer.create_code_switching_heatmap(data)
                        
                        # Create summary text
                        summary = html.Div([
                            html.H3("Code-Switching Summary"),
                            html.P(f"Sample size: {data.get('sample_size', 0)} documents"),
                            html.P(f"Average switches per document: {data['avg_switches']:.2f}"),
                            html.P(f"Maximum switches in a document: {data['max_switches']}"),
                            html.P(f"Minimum switches in a document: {data['min_switches']}")
                        ])
                        
                        return (
                            dcc.Graph(figure=fig),
                            summary,
                            current_analysis,
                            html.Div("Code-Switching analysis updated!", style={'color': 'green'})
                        )

                elif button_id == 'save-button' and current_analysis:
                    self._save_analysis(current_analysis, filename)
                    return no_update, no_update, current_analysis, html.Div(
                        "Analysis saved successfully!", 
                        style={'color': 'green'}
                    )

            except Exception as e:
                logger.error(f"Error updating visualization: {e}", exc_info=True)
                return (
                    html.Div("Error occurred"),
                    None,
                    no_update,
                    html.Div(f"Error: {str(e)}", style={'color': 'red'})
                )

            return no_update, no_update, no_update, no_update

    def _save_analysis(self, analysis_data, filename=None):
        """Save the current analysis results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"language_analysis_{timestamp}"
            
        save_path = Path('results/language_distribution')
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save analysis parameters
            params_file = save_path / f"{filename}_params.json"
            with open(params_file, 'w') as f:
                import json
                json.dump(analysis_data, f, indent=2)

            # Save data based on analysis type
            if analysis_data['type'] == 'basic':
                data = self.analyzer.get_language_counts(analysis_data['category'])
                data.to_csv(save_path / f"{filename}_distribution.csv", index=False)
                data.to_excel(save_path / f"{filename}_distribution.xlsx", index=False)
            
            elif analysis_data['type'] == 'detailed':
                data = self.analyzer.analyze_language_content()
                data.to_csv(save_path / f"{filename}_detailed.csv", index=False)
                data.to_excel(save_path / f"{filename}_detailed.xlsx", index=False)
            
            logger.info(f"Analysis saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            raise

    def run_server(self, debug=True, port=8052):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    # Configure logging
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"language_distribution_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    
    try:
        dashboard = LanguageDistributionDashboard()
        dashboard.run_server()
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}", exc_info=True)
        raise