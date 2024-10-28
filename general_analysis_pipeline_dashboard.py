import os
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

class AnalysisDashboardApp:
    def __init__(self):
        self.columns = {
            'document_type': 'Document Type',
            'language': 'Language',
            'category': 'Category',
            'clientele': 'Clientele',
            'knowledge_type': 'Knowledge Type',
            'organization': 'Organization'
        }
        self.viz_types = ['Bar Chart', 'Pie Chart', 'Crosstable', 'Heatmap']
        self.df = self._load_initial_data()
        self.app = self._create_app()

    def _load_initial_data(self):
        """Load initial data from database"""
        try:
            engine = create_engine(
                f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@localhost:5432/labrri_ocpm_systemic_racism",
                pool_size=5,
                max_overflow=10
            )
            query = """
                SELECT d.id, d.organization, d.document_type, d.category, 
                       d.clientele, d.knowledge_type, d.language
                FROM documents d
            """
            return pd.read_sql(query, engine)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_app(self):
        """Create and configure the Dash app"""
        app = Dash(__name__)
        
        app.layout = html.Div([
            html.H1('Document Analysis Dashboard', className='h1'),
            
            # Left panel - Controls
            html.Div([
                # Analysis Type Selection
                html.Div([
                    html.H3('Analysis Options'),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Single Distribution', 'value': 'single'},
                            {'label': 'Cross Analysis', 'value': 'cross'}
                        ],
                        value='single'
                    ),
                ], className='control-group'),

                # Variable Selection
                html.Div([
                    dcc.Dropdown(
                        id='primary-var',
                        options=[{'label': v, 'value': k} for k, v in self.columns.items()],
                        value=list(self.columns.keys())[0],
                        placeholder='Select Primary Variable'
                    ),
                    dcc.Dropdown(
                        id='secondary-var',
                        options=[{'label': v, 'value': k} for k, v in self.columns.items()],
                        value=list(self.columns.keys())[1],
                        placeholder='Select Secondary Variable',
                        style={'display': 'none'}
                    ),
                ], className='control-group'),

                # Visualization Type
                html.Div([
                    dcc.Dropdown(
                        id='viz-type',
                        options=[{'label': vt, 'value': vt.lower().replace(' ', '-')} 
                                for vt in self.viz_types],
                        value='bar-chart'
                    ),
                ], className='control-group'),

                # Filters
                html.Div([
                    html.H3('Filters'),
                    *[
                        html.Div([
                            html.Label(label),
                            dcc.Dropdown(
                                id=f'filter-{col}',
                                options=[{'label': 'All', 'value': 'all'}] + [
                                    {'label': str(val), 'value': str(val)} 
                                    for val in sorted(self.df[col].unique())
                                ],
                                value='all',
                                multi=True
                            )
                        ]) for col, label in self.columns.items()
                    ]
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

            # Notifications
            dcc.Store(id='current-analysis'),
            html.Div(id='notification', className='notification')
            
        ], className='dashboard-container')

        self._setup_callbacks(app)
        return app

    def _setup_callbacks(self, app):
            """Set up the callbacks for the Dash app"""
            
            @app.callback(
                Output('secondary-var', 'style'),
                Input('analysis-type', 'value')
            )
            def toggle_secondary_var(analysis_type):
                return {'display': 'block'} if analysis_type == 'cross' else {'display': 'none'}

            @app.callback(
                [Output('visualization-container', 'children'),
                Output('current-analysis', 'data'),
                Output('notification', 'children')],
                [Input('update-button', 'n_clicks'),
                Input('save-button', 'n_clicks')],
                [State('analysis-type', 'value'),
                State('primary-var', 'value'),
                State('secondary-var', 'value'),
                State('viz-type', 'value'),
                State('filename-input', 'value'),
                State('current-analysis', 'data')] +
                [State(f'filter-{col}', 'value') for col in self.columns]
            )
            def update_dashboard(update_clicks, save_clicks, 
                            analysis_type, primary_var, secondary_var, viz_type,
                            filename, stored_analysis, *filter_values):
                
                # Use callback context to determine which button was clicked
                ctx = callback_context
                if not ctx.triggered:
                    button_id = None
                else:
                    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                # Initialize default returns
                visualization = no_update
                current_analysis = no_update
                notification = no_update

                if button_id == 'update-button':
                    # Apply filters
                    filtered_df = self.df.copy()
                    for col, filter_val in zip(self.columns.keys(), filter_values):
                        if filter_val and 'all' not in filter_val:
                            filtered_df = filtered_df[filtered_df[col].isin(filter_val)]

                    # Store current analysis
                    current_analysis = {
                        'type': analysis_type,
                        'primary_var': primary_var,
                        'secondary_var': secondary_var,
                        'viz_type': viz_type,
                        'filters': dict(zip(self.columns.keys(), filter_values)),
                        'timestamp': datetime.now().isoformat()
                    }

                    # Create visualization
                    if analysis_type == 'single':
                        fig = self._create_single_visualization(filtered_df, primary_var, viz_type)
                    else:
                        fig = self._create_cross_visualization(filtered_df, primary_var, secondary_var, viz_type)

                    visualization = dcc.Graph(figure=fig)
                    notification = html.Div("Visualization updated successfully!", 
                                        style={'color': 'green'})

                elif button_id == 'save-button' and stored_analysis:
                    try:
                        saved_files = self._save_analysis(stored_analysis, filename)
                        
                        # Create a formatted list of saved files
                        files_list = html.Ul([
                            html.Li([
                                html.Span(f"{k}: "),
                                html.Span(str(v), style={'word-break': 'break-all'})
                            ]) for k, v in saved_files.items()
                        ])
                        
                        notification = html.Div([
                            html.P("Analysis saved successfully!", 
                                style={'color': 'green', 'font-weight': 'bold'}),
                            html.P("Saved files:"),
                            files_list
                        ], style={'max-width': '100%', 'overflow-wrap': 'break-word'})
                    except Exception as e:
                        logger.error(f"Error saving analysis: {str(e)}")
                        notification = html.Div(
                            f"Error saving analysis: {str(e)}", 
                            style={'color': 'red'}
                        )

                return visualization, current_analysis, notification

    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

    def _create_single_visualization(self, df, column, viz_type):
        counts = df[column].value_counts()
        
        if viz_type == 'bar-chart':
            return px.bar(
                x=counts.index,
                y=counts.values,
                title=f'{self.columns[column]} Distribution',
                labels={'x': self.columns[column], 'y': 'Count'}
            )
        elif viz_type == 'pie-chart':
            return px.pie(
                values=counts.values,
                names=counts.index,
                title=f'{self.columns[column]} Distribution'
            )
        elif viz_type == 'crosstable':
            # Return a table figure
            return go.Figure(data=[go.Table(
                header=dict(values=['Category', 'Count', 'Percentage']),
                cells=dict(values=[
                    counts.index,
                    counts.values,
                    (counts.values / counts.sum() * 100).round(1)
                ])
            )])

    def _create_cross_visualization(self, df, primary_col, secondary_col, viz_type):
        crosstab = pd.crosstab(df[primary_col], df[secondary_col])
        
        if viz_type == 'bar-chart':
            return px.bar(
                crosstab,
                barmode='group',
                title=f'{self.columns[primary_col]} by {self.columns[secondary_col]}'
            )
        elif viz_type == 'heatmap':
            normalized_crosstab = pd.crosstab(
                df[primary_col], 
                df[secondary_col], 
                normalize='all'
            ) * 100
            return px.imshow(
                normalized_crosstab,
                title=f'Heatmap of {self.columns[primary_col]} vs {self.columns[secondary_col]}',
                labels=dict(
                    x=self.columns[secondary_col],
                    y=self.columns[primary_col],
                    color="Percentage"
                ),
                aspect="auto"
            )
        elif viz_type == 'crosstable':
            return go.Figure(data=[go.Table(
                header=dict(values=[''] + list(crosstab.columns)),
                cells=dict(values=[crosstab.index] + [crosstab[col] for col in crosstab.columns])
            )])

    def _save_analysis(self, current_analysis, filename=None):
        """
        Save the current analysis parameters and results.
        
        Args:
            current_analysis (dict): Current analysis parameters and state
            filename (str, optional): Base filename for saving files
            
        Returns:
            dict: Paths to all saved files
        """
        # Generate default filename if none provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}"
        
        # Create results directory if it doesn't exist
        results_dir = Path('results/general_analysis')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}

        try:
            # Save analysis parameters to JSON
            params_path = results_dir / f"{filename}_params.json"
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(current_analysis, f, indent=2, ensure_ascii=False)
            saved_files['params'] = params_path

            # Get filtered dataframe based on current filters
            filtered_df = self.df.copy()
            for col, filter_val in current_analysis['filters'].items():
                if filter_val and 'all' not in filter_val:
                    filtered_df = filtered_df[filtered_df[col].isin(filter_val)]

            # Save results based on analysis type and visualization
            if current_analysis['type'] == 'single':
                primary_col = current_analysis['primary_var']
                counts = filtered_df[primary_col].value_counts()
                percentages = (counts / counts.sum() * 100).round(2)
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Category': counts.index,
                    'Count': counts.values,
                    'Percentage': percentages.values
                })

                # Save to CSV
                csv_path = results_dir / f"{filename}_distribution.csv"
                results_df.to_csv(csv_path, index=False, encoding='utf-8')
                saved_files['distribution_csv'] = csv_path

                # Save to Excel with formatting
                excel_path = results_dir / f"{filename}_distribution.xlsx"
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    results_df.to_excel(writer, sheet_name='Distribution', index=False)
                    workbook = writer.book
                    worksheet = writer.sheets['Distribution']
                    
                    # Add formats
                    percent_fmt = workbook.add_format({'num_format': '0.00%'})
                    worksheet.set_column('C:C', None, percent_fmt)
                saved_files['distribution_excel'] = excel_path

            else:  # Cross Analysis
                primary_col = current_analysis['primary_var']
                secondary_col = current_analysis['secondary_var']
                
                # Create different types of crosstabs
                counts = pd.crosstab(filtered_df[primary_col], filtered_df[secondary_col], margins=True)
                percentages = pd.crosstab(
                    filtered_df[primary_col], 
                    filtered_df[secondary_col], 
                    normalize='all'
                ) * 100
                row_percentages = pd.crosstab(
                    filtered_df[primary_col], 
                    filtered_df[secondary_col], 
                    normalize='index'
                ) * 100
                
                # Save to Excel with multiple sheets and formatting
                excel_path = results_dir / f"{filename}_crosstab.xlsx"
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    counts.to_excel(writer, sheet_name='Counts')
                    percentages.to_excel(writer, sheet_name='Overall Percentages')
                    row_percentages.to_excel(writer, sheet_name='Row Percentages')
                    
                    # Add formatting
                    workbook = writer.book
                    percent_fmt = workbook.add_format({'num_format': '0.00%'})
                    
                    # Format percentage sheets
                    for sheet in ['Overall Percentages', 'Row Percentages']:
                        worksheet = writer.sheets[sheet]
                        worksheet.set_column('B:Z', None, percent_fmt)
                saved_files['crosstab_excel'] = excel_path

                # Save to CSV (counts only)
                csv_path = results_dir / f"{filename}_crosstab.csv"
                counts.to_csv(csv_path, encoding='utf-8')
                saved_files['crosstab_csv'] = csv_path

            # If it's a heatmap or other visualization type, save the plot
            if current_analysis['viz_type'] in ['bar-chart', 'pie-chart', 'heatmap']:
                if current_analysis['type'] == 'single':
                    fig = self._create_single_visualization(
                        filtered_df, 
                        current_analysis['primary_var'], 
                        current_analysis['viz_type']
                    )
                else:
                    fig = self._create_cross_visualization(
                        filtered_df,
                        current_analysis['primary_var'],
                        current_analysis['secondary_var'],
                        current_analysis['viz_type']
                    )
                
                # Save as HTML (interactive)
                html_path = results_dir / f"{filename}_plot.html"
                fig.write_html(html_path)
                saved_files['plot_html'] = html_path
                
                # Save as PNG (static)
                png_path = results_dir / f"{filename}_plot.png"
                fig.write_image(png_path)
                saved_files['plot_png'] = png_path

            # Create and save a summary text file
            summary_path = results_dir / f"{filename}_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Type: {current_analysis['type']}\n")
                f.write(f"Primary Variable: {self.columns[current_analysis['primary_var']]}\n")
                if current_analysis['type'] == 'cross':
                    f.write(f"Secondary Variable: {self.columns[current_analysis['secondary_var']]}\n")
                f.write(f"Visualization Type: {current_analysis['viz_type']}\n")
                f.write(f"Timestamp: {current_analysis['timestamp']}\n\n")
                
                f.write("Applied Filters:\n")
                for col, filter_val in current_analysis['filters'].items():
                    f.write(f"- {self.columns[col]}: {filter_val}\n")
                
                f.write("\nData Summary:\n")
                f.write(f"Total Records: {len(filtered_df)}\n")
                f.write(f"Missing Values: {filtered_df.isnull().sum().to_dict()}\n")
            saved_files['summary'] = summary_path

            logger.info(f"Analysis saved successfully with {len(saved_files)} files")
            return saved_files

        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise

if __name__ == "__main__":
    dashboard = AnalysisDashboardApp()
    dashboard.run_server(debug=True)