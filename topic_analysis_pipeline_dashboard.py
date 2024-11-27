"""Topic analysis dashboard with enhanced async support and proper error handling."""
import os
import sys
from pathlib import Path
from datetime import datetime
import uvicorn
import asyncio
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, LogTransform, ServersideOutputTransform
from contextlib import asynccontextmanager
import signal
from loguru import logger

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.topic_analysis.async_db_helper import AsyncDatabaseHelper
from scripts.topic_analysis.manager import TopicAnalysisManager
from scripts.topic_analysis.visualizer import TopicAnalysisVisualizer
from scripts.topic_analysis.data_handler import TopicAnalysisDataHandler
from scripts.topic_analysis.async_manager import AsyncTaskManager
from scripts.topic_analysis.error_handlers import ErrorHandler, ServerErrorHandler

class TopicAnalysisDashboard:
    """Enhanced dashboard with proper async handling and improved error management."""

    def __init__(self, db_path: str):
        """Initialize dashboard with database connection."""
        self.db_path = db_path
        self._components_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        
        # Initialize logging first
        self._setup_logging()
        logger.info("Initializing Topic Analysis Dashboard")

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"topic_analysis_dashboard_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logger.remove()  # Remove default handler
        logger.add(
            log_file,
            rotation="1 day",
            retention="7 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        logger.add(sys.stderr, level="INFO")  # Also log to console

    async def initialize(self):
        """Initialize dashboard components."""
        if self._components_initialized:
            return

        async with self._initialization_lock:
            if self._components_initialized:
                return

            try:
                logger.info("Initializing dashboard components")
                
                # Initialize core components
                self.error_handler = ErrorHandler()
                await self.error_handler.initialize()
                self.server_error_handler = ServerErrorHandler(self.error_handler)

                # Initialize database connection
                self.db = AsyncDatabaseHelper(self.db_path)
                
                # Initialize managers and handlers
                self.task_manager = AsyncTaskManager()
                self.manager = TopicAnalysisManager(self.db_path)
                await self.manager.initialize()
                
                self.visualizer = TopicAnalysisVisualizer()
                self.data_handler = TopicAnalysisDataHandler()

                # Create Dash application
                self.app = self._create_app()
                self._setup_callbacks()

                self._components_initialized = True
                logger.info("Dashboard components initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize dashboard: {e}", exc_info=True)
                raise

    def _create_app(self) -> DashProxy:
        """Create Dash application with enhanced features."""
        # Use DashProxy with transforms for better performance
        app = DashProxy(
            name=__name__,
            suppress_callback_exceptions=True,
            transforms=[
                LogTransform(),
                ServersideOutputTransform()
            ],
            serve_locally=True
        )

        app.layout = html.Div([
            # Add Store components for state management
            dcc.Store(id='analysis-results'),
            dcc.Store(id='error-state'),
            
            # Header
            html.Div([
                html.H1('Topic Analysis Dashboard', className='text-2xl font-bold mb-4'),
                html.P('Analyze document topics with advanced NLP', className='text-gray-600')
            ], className='p-4 bg-white shadow'),

            # Main content
            html.Div([
                # Control panel
                html.Div([
                    html.H3('Analysis Settings', className='text-lg font-semibold mb-3'),
                    html.Div([
                        html.Label('Analysis Method'),
                        dcc.Dropdown(
                            id='analysis-method',
                            options=[
                                {'label': 'LDA', 'value': 'lda'},
                                {'label': 'NMF', 'value': 'nmf'},
                                {'label': 'LSA', 'value': 'lsa'}
                            ],
                            value='lda'
                        )
                    ], className='mb-4'),

                    html.Div([
                        html.Label('Number of Topics'),
                        dcc.Slider(
                            id='num-topics',
                            min=5, max=50, step=5, value=20,
                            marks={i: str(i) for i in range(5, 51, 5)}
                        )
                    ], className='mb-4'),

                    html.Button(
                        'Run Analysis',
                        id='run-analysis',
                        n_clicks=0,
                        className='w-full py-2 px-4 bg-blue-600 text-white rounded'
                    )
                ], className='bg-white p-4 rounded shadow'),

                # Results area
                html.Div([
                    dcc.Loading(
                        id="loading",
                        children=[
                            html.Div(id='analysis-display'),
                            dcc.Graph(id='topic-network'),
                            html.Div(id='error-display', className='text-red-600')
                        ],
                        type="circle"
                    )
                ], className='bg-white p-4 rounded shadow mt-4')
            ], className='p-4')
        ], className='min-h-screen bg-gray-100')

        return app

    def _setup_callbacks(self):
        """Set up dashboard callbacks with correct decorator usage."""
        if not hasattr(self, 'app'):
            raise RuntimeError("App not initialized")

        from scripts.topic_analysis.callbacks import run_global_analysis_callback

        @self.app.callback(
            [
                Output('analysis-display', 'children'),
                Output('topic-network', 'figure'),
                Output('error-display', 'children')
            ],
            [Input('run-analysis', 'n_clicks')],
            [
                State('analysis-method', 'value'),
                State('num-topics', 'value')
            ],
            prevent_initial_call=True
        )
        async def run_analysis(n_clicks, method, num_topics):
            """Run topic analysis."""
            if not n_clicks:
                raise PreventUpdate

            try:
                return await run_global_analysis_callback(
                    self.task_manager,
                    self.manager,
                    self.visualizer,
                    self.data_handler,
                    self.error_handler,
                    method,
                    num_topics,
                    0.3  # Default coherence threshold
                )
            except Exception as e:
                logger.error(f"Error in analysis callback: {e}", exc_info=True)
                return (
                    html.Div("Analysis failed"),
                    {},
                    f"Error: {str(e)}"
                )

    async def start(self, host: str = "127.0.0.1", port: int = 8055):
        """Start the dashboard server with proper WSGI/ASGI handling."""
        if not self._components_initialized:
            await self.initialize()

        try:
            config = HyperConfig()
            config.bind = [f"{host}:{port}"]
            config.use_reloader = False
            config.workers = 1
            
            logger.info(f"Starting dashboard server on http://{host}:{port}")
            await serve(self.app.server, config)
            
        except Exception as e:
            logger.error(f"Error starting server: {e}", exc_info=True)
            raise

    def run_server(self, debug: bool = True, port: int = 8055):
        """Run server synchronously for development."""
        if not self._components_initialized:
            asyncio.run(self.initialize())
        
        try:
            logger.info(f"Starting dashboard in development mode on port {port}")
            self.app.run_server(
                debug=debug,
                port=port,
                host="127.0.0.1"
            )
        except Exception as e:
            logger.error(f"Error running development server: {e}", exc_info=True)
            raise

    async def cleanup(self):
        """Clean up dashboard resources."""
        logger.info("Cleaning up dashboard resources")
        try:
            if hasattr(self, 'manager'):
                await self.manager.cleanup()
            if hasattr(self, 'task_manager'):
                await self.task_manager.cleanup()
            if hasattr(self, 'error_handler'):
                await self.error_handler.cleanup()
            
            self._components_initialized = False
            logger.info("Dashboard cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

if __name__ == "__main__":
    # Get database credentials from environment
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    if not db_user or not db_password:
        logger.error("Database credentials not found in environment variables")
        sys.exit(1)

    db_path = f"postgresql://{db_user}:{db_password}@localhost:5432/labrri_ocpm_systemic_racism"

    async def main():
        dashboard = TopicAnalysisDashboard(db_path)
        try:
            logger.info(f"Starting dashboard server on http://127.0.0.1:8055")
            await dashboard.start()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Dashboard error: {e}", exc_info=True)
        finally:
            await dashboard.cleanup()

    # Run the dashboard
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)