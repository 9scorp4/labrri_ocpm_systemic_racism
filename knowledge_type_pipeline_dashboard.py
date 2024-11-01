import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from urllib.parse import quote_plus

from scripts.database import Database
from scripts.knowledge_type.analyzer import KnowledgeTypeAnalyzer
from scripts.knowledge_type.visualizer import KnowledgeTypeVisualizer
from scripts.knowledge_type.data_handler import KnowledgeTypeDataHandler

class KnowledgeTypeDashboard:
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
            self.analyzer = KnowledgeTypeAnalyzer(self.db_path)
            self.visualizer = KnowledgeTypeVisualizer()
            
            # Get available categories for cross-analysis
            self.categories = self._get_categories()
            
            # Create Dash app
            self.app = self._create_app()

            # Define notification styles
            self.notification_styles = {
                'success': {'color': 'green', 'margin': '10px'},
                'error': {'color': 'red', 'margin': '10px'},
                'warning': {'color': 'orange', 'margin': '10px'},
                'info': {'color': 'blue', 'margin': '10px'}
            }
            
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}", exc_info=True)
            raise

    def _get_categories(self):
        """Fetch all possible categories for cross-analysis."""
        return [
            'document_type', 
            'category', 
            'clientele', 
            'organization',
            'language'
        ]

    def _create_app(self):
        """Create and configure the Dash app"""
        app = Dash(__name__)
        
        app.layout = html.Div([
            html.H1('Knowledge Type Analysis Dashboard', className='h1'),
            
            # Left panel - Controls
            html.Div([
                # Analysis Type Selection
                html.Div([
                    html.H3('Analysis Type'),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Distribution Analysis', 'value': 'distribution'},
                            {'label': 'Intersection Analysis', 'value': 'intersection'},
                            {'label': 'Cross Analysis', 'value': 'cross'}
                        ],
                        value='distribution'
                    ),
                ], className='control-group'),

                # Cross Analysis Settings (conditionally displayed)
                html.Div([
                    html.H3('Cross Analysis Settings'),
                    dcc.Dropdown(
                        id='cross-category',
                        options=[{'label': cat.replace('_', ' ').title(), 'value': cat} 
                                for cat in self.categories],
                        placeholder='Select category for cross analysis'
                    ),
                ], className='control-group', id='cross-analysis-controls'),

                # Visualization Type
                html.Div([
                    html.H3('Visualization Type'),
                    dcc.Dropdown(
                        id='viz-type',
                        options=[
                            {'label': 'Pie Chart', 'value': 'pie'},
                            {'label': 'Bar Chart', 'value': 'bar'},
                            {'label': 'Heatmap', 'value': 'heatmap'}
                        ],
                        value='pie'
                    ),
                ], className='control-group'),

                # Knowledge Type Filter
                html.Div([
                    html.H3('Knowledge Type Filter'),
                    dcc.Dropdown(
                        id='knowledge-type-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'Citoyen', 'value': 'Citoyen'},
                            {'label': 'Communautaire', 'value': 'Communautaire'},
                            {'label': 'Municipal', 'value': 'Municipal'}
                        ],
                        value='all',
                        multi=True
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
                        html.Div(id='visualization-container'),
                        html.Div(id='statistics-container', className='statistics')
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

    def _create_notification(self, message, type='info'):
        """Create a notification component with consistent styling."""
        return html.Div(
            message,
            style=self.notification_styles.get(type, self.notification_styles['info'])
        )

    def _create_error_response(self, error_message):
        """Create a consistent error response for callbacks."""
        return (
            html.Div("Error occurred during visualization", 
                    style={'color': 'red', 'margin': '20px'}),
            None,  # stats container
            no_update,  # current analysis
            self._create_notification(f"Error: {error_message}", 'error')
        )

    def _create_success_response(self, fig, stats, current_analysis):
        """Create a consistent success response for callbacks."""
        return (
            dcc.Graph(figure=fig),
            stats,
            current_analysis,
            self._create_notification("Analysis updated successfully!", 'success')
        )

    def _setup_callbacks(self, app):
        @app.callback(
            Output('cross-analysis-controls', 'style'),
            Input('analysis-type', 'value')
        )
        def toggle_cross_analysis_controls(analysis_type):
            """Show/hide cross analysis controls based on analysis type."""
            return {'display': 'block' if analysis_type == 'cross' else 'none'}

        @app.callback(
            [Output('visualization-container', 'children'),
             Output('statistics-container', 'children'),
             Output('current-analysis', 'data'),
             Output('notification', 'children')],
            [Input('update-button', 'n_clicks'),
             Input('save-button', 'n_clicks')],
            [State('analysis-type', 'value'),
             State('cross-category', 'value'),
             State('viz-type', 'value'),
             State('knowledge-type-filter', 'value'),
             State('filename-input', 'value'),
             State('current-analysis', 'data')]
        )
        def update_visualization(update_clicks, save_clicks, analysis_type, cross_category, 
                               viz_type, knowledge_filter, filename, current_analysis):
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update, no_update, no_update

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                if button_id == 'update-button':
                    # Store current analysis parameters
                    current_analysis = {
                        'type': analysis_type,
                        'cross_category': cross_category,
                        'viz_type': viz_type,
                        'knowledge_filter': knowledge_filter,
                        'timestamp': datetime.now().isoformat()
                    }

                    if analysis_type == 'distribution':
                        data = self.analyzer.get_distribution()
                        if knowledge_filter != 'all':
                            data = data[data['knowledge_type'].isin(knowledge_filter)]
                        
                        data = KnowledgeTypeDataHandler.validate_and_format_distribution(data)
                        if data is None:
                            return self._create_error_response("No valid data available for distribution analysis")
                            
                        fig = self.visualizer.create_distribution_plot(
                            data,
                            'Knowledge Type Distribution',
                            viz_type
                        )
                        stats = self._calculate_distribution_stats(data)
                        
                        return self._create_success_response(fig, stats, current_analysis)

                    elif analysis_type == 'intersection':
                        raw_data = self.analyzer.get_intersection_data()
                        data, stats = KnowledgeTypeDataHandler.validate_and_format_intersection_data(raw_data)
                        
                        if data is None:
                            return self._create_error_response("No valid data available for intersection analysis")
                            
                        fig = self.visualizer.create_venn_diagram(data)
                        stats_component = self._create_intersection_stats_component(stats)
                        
                        return self._create_success_response(fig, stats_component, current_analysis)

                    elif analysis_type == 'cross':
                        if not cross_category:
                            return self._create_error_response("Please select a category for cross analysis")
                            
                        data = self.analyzer.get_cross_analysis(cross_category)
                        formatted_data = KnowledgeTypeDataHandler.format_cross_analysis_data(data, cross_category)
                        
                        if formatted_data is None:
                            return self._create_error_response("No valid data available for cross analysis")
                            
                        fig = self.visualizer.create_heatmap(
                            formatted_data['counts'],
                            formatted_data['percentages'],
                            cross_category
                        )
                        stats = self._create_cross_analysis_stats_component(formatted_data['stats'])
                        
                        return self._create_success_response(fig, stats, current_analysis)

                elif button_id == 'save-button' and current_analysis:
                    try:
                        saved_files = self._save_analysis(current_analysis, filename)
                        return (
                            no_update,
                            no_update,
                            current_analysis,
                            html.Div([
                                html.P("Analysis saved successfully!", 
                                      style=self.notification_styles['success']),
                                html.P("Saved files:"),
                                html.Ul([
                                    html.Li(str(path)) for path in saved_files.values()
                                ])
                            ])
                        )
                    except Exception as e:
                        return self._create_error_response(f"Error saving analysis: {str(e)}")

            except Exception as e:
                logger.error(f"Error in callback: {str(e)}")
                return self._create_error_response(str(e))

            return no_update, no_update, no_update, no_update

    def _create_intersection_stats_component(self, stats):
        """Create a component to display intersection statistics."""
        return html.Div([
            html.H3("Intersection Statistics"),
            html.P(f"Total Documents: {stats['total_documents']}"),
            html.P(f"Documents by Type:"),
            html.Ul([
                html.Li(f"{k}: {v}") for k, v in stats['by_type'].items()
            ]),
            html.P(f"Documents with Multiple Types: {stats['multiple_types']}")
        ])

    def _create_cross_analysis_stats_component(self, stats):
        """Create a component to display cross analysis statistics."""
        return html.Div([
            html.H3("Cross Analysis Statistics"),
            html.P(f"Total Documents: {stats['total_documents']}"),
            html.P(f"Unique Knowledge Types: {stats['unique_knowledge_types']}"),
            html.P(f"Unique Categories: {stats['unique_categories']}")
        ])

    def _calculate_distribution_stats(self, data):
        """Create a component to display distribution statistics."""
        return html.Div([
            html.H3("Distribution Statistics"),
            html.P(f"Total Documents: {data['count'].sum()}"),
            html.P(f"Number of Knowledge Types: {len(data)}")
        ])

    def _calculate_intersection_stats(self, data):
        """Calculate statistics for intersection analysis."""
        try:
            if data is None or data.empty:
                return html.Div("No data available for statistics")
                
            citoyen_count = data['Citoyen'].sum()
            communautaire_count = data['Communautaire'].sum()
            municipal_count = data['Municipal'].sum()
            
            # Calculate documents with multiple types
            multiple_types = data[data[['Citoyen', 'Communautaire', 'Municipal']].sum(axis=1) > 1].shape[0]

            stats = html.Div([
                html.H3("Intersection Statistics"),
                html.P(f"Citoyen Documents: {citoyen_count}"),
                html.P(f"Communautaire Documents: {communautaire_count}"),
                html.P(f"Municipal Documents: {municipal_count}"),
                html.P(f"Documents with Multiple Types: {multiple_types}")
            ])

            return stats
        except Exception as e:
            logger.error(f"Error calculating intersection statistics: {str(e)}")
            return html.Div("Error calculating statistics", style={'color': 'red'})

    def _calculate_cross_analysis_stats(self, data, category):
        """Calculate statistics for cross analysis."""
        total_docs = data['count'].sum()
        unique_types = len(data['knowledge_type'].unique())
        unique_categories = len(data[category].unique())

        stats = html.Div([
            html.H3("Cross Analysis Statistics"),
            html.P(f"Total Documents: {total_docs}"),
            html.P(f"Unique Knowledge Types: {unique_types}"),
            html.P(f"Unique {category.replace('_', ' ').title()}: {unique_categories}")
        ])

        return stats

    def _save_analysis(self, analysis_data, filename=None):
        """Save the current analysis results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knowledge_type_analysis_{timestamp}"
            
        save_path = Path('results/knowledge_type')
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save analysis parameters
            params_file = save_path / f"{filename}_params.json"
            with open(params_file, 'w') as f:
                import json
                json.dump(analysis_data, f, indent=2)
            saved_files['params'] = params_file

            # Save current visualization if available
            if hasattr(self, 'current_fig'):
                fig_file = save_path / f"{filename}_plot.html"
                self.current_fig.write_html(str(fig_file))
                saved_files['plot'] = fig_file

            logger.info(f"Analysis saved to {save_path}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            raise

    def run_server(self, debug=True, port=8053):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    # Configure logging
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"knowledge_type_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
        dashboard = KnowledgeTypeDashboard()
        dashboard.run_server()
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}", exc_info=True)
        raise