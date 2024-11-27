
"""Callbacks for topic analysis dashboard."""
from typing import Dict, Any, Tuple
import plotly.graph_objects as go
from dash import html, no_update
from loguru import logger
import asyncio

from .error_handlers import ErrorHandler, AnalysisErrorType

async def run_global_analysis_callback(
    task_manager,
    manager,
    visualizer,
    data_handler,
    error_handler,
    method: str,
    num_topics: int,
    threshold: float
) -> Tuple[Any, Any, str, Dict]:
    """Run global topic analysis.
    
    Args:
        task_manager: AsyncTaskManager instance
        manager: TopicAnalysisManager instance
        visualizer: TopicAnalysisVisualizer instance
        data_handler: TopicAnalysisDataHandler instance
        error_handler: ErrorHandler instance
        method: Analysis method
        num_topics: Number of topics
        threshold: Coherence threshold
        
    Returns:
        Tuple of (visualization, network figure, status message, error data)
    """
    try:
        task_id = f"global_analysis_{manager.db.get_last_update_time().timestamp()}"
        
        async def analysis_task(progress_callback):
            # Fetch documents
            progress_callback(0.1, "Fetching documents...")
            docs = await manager.db.fetch_all_async()
            if not docs:
                raise ValueError("No documents found")
            
            # Analyze topics
            progress_callback(0.3, "Analyzing topics...")
            topics_df = await manager.analyze_topics(
                docs,
                method=method,
                num_topics=num_topics,
                coherence_threshold=threshold
            )
            
            if topics_df.empty:
                raise ValueError("No topics found")
            
            # Calculate similarity matrix
            progress_callback(0.7, "Calculating topic similarities...")
            similarity_matrix = data_handler.calculate_topic_similarity_matrix(topics_df)
            
            # Create visualizations
            progress_callback(0.9, "Creating visualizations...")
            network_fig = visualizer.create_topic_network(
                topics_df.to_dict('records'),
                similarity_matrix
            )
            
            progress_callback(1.0, "Analysis complete")
            return topics_df, network_fig

        task_status = await task_manager.start_task(task_id, analysis_task)
        
        if task_status.error:
            raise Exception(task_status.error['message'])
            
        topics_df, network_fig = task_status.result
        
        # Create visualization components
        topic_list = html.Div([
            html.H3("Topics"),
            html.Ul([
                html.Li([
                    f"Topic {i+1}: {row['label']} ",
                    html.Small(f"(Coherence: {row['coherence_score']:.3f})")
                ]) for i, row in topics_df.iterrows()
            ])
        ])
        
        return (
            topic_list,
            network_fig,
            "Global analysis completed successfully",
            None
        )
        
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        error = error_handler.capture_error(e, AnalysisErrorType.ANALYSIS)
        error_handler.log_error(error)
        return (
            html.Div("Analysis failed"),
            go.Figure(),
            "Error during analysis",
            error_handler.format_user_message(error)
        )