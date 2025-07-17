#!/usr/bin/env python3
"""
Data Tools - LangChain Tool Wrappers for DataAgent Integration
Enables physics agents to interact with DataAgent through standard tool calling.
"""

import asyncio
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Global reference to DataAgent instance
_data_agent_instance: Optional[Any] = None

def set_data_agent_instance(data_agent):
    """Set the global DataAgent instance for tool access."""
    global _data_agent_instance
    _data_agent_instance = data_agent

def get_data_agent():
    """Get the global DataAgent instance."""
    if _data_agent_instance is None:
        raise RuntimeError("DataAgent instance not set. Call set_data_agent_instance() first.")
    return _data_agent_instance


class DataStatusRequest(BaseModel):
    """Schema for data status requests."""
    job_id: str = Field(description="UUID job identifier for the data processing job")


class DataPreviewRequest(BaseModel):
    """Schema for data preview requests."""
    job_id: str = Field(description="UUID job identifier for the data processing job")
    n: int = Field(default=5, description="Number of rows to preview (default: 5)")


class DataInsightsRequest(BaseModel):
    """Schema for physics insights requests."""
    job_id: str = Field(description="UUID job identifier for the data processing job")
    analysis_type: str = Field(default="physics", description="Type of analysis to perform")


class DataPublishRequest(BaseModel):
    """Schema for data publishing requests."""
    job_id: str = Field(description="UUID job identifier for the data processing job")


@tool("get_data_status", args_schema=DataStatusRequest)
def get_data_status(job_id: str) -> str:
    """
    Get the processing status and basic statistics for a data processing job.
    
    Use this to check if data processing is complete and get file statistics.
    Returns information about file size, number of rows/columns, and processing status.
    
    Args:
        job_id: UUID job identifier for the data processing job
        
    Returns:
        JSON string with status information including processing state, file info, and basic statistics
    """
    try:
        data_agent = get_data_agent()
        status = asyncio.run(data_agent.status(job_id))
        
        if "error" in status:
            return f"Error: {status['error']}"
        
        # Format status for readable output
        if status.get("status") == "completed":
            result = f"""Data Processing Status: COMPLETED

File Information:
- File: {status.get('file_path', 'Unknown')}
- Size: {status.get('file_size', 0)} bytes
- Type: {status.get('mime_type', 'Unknown')}

Data Summary:
- Rows: {status.get('rows', 0):,}
- Columns: {status.get('columns', 0)}
- Processing Time: {(status.get('completed_at', '') != status.get('started_at', '')) and 'Available' or 'Unknown'}

Status: Ready for analysis"""
            
        elif status.get("status") == "processing":
            result = f"""Data Processing Status: IN PROGRESS

File: {status.get('file_path', 'Unknown')}
Started: {status.get('started_at', 'Unknown')}

Please wait for processing to complete before requesting analysis."""
            
        elif status.get("status") == "failed":
            result = f"""Data Processing Status: FAILED

Error: {status.get('error', 'Unknown error')}
File: {status.get('file_path', 'Unknown')}

The data processing encountered an error and cannot be used for analysis."""
            
        else:
            result = f"Unknown status: {status.get('status', 'Unknown')}"
        
        return result
        
    except Exception as e:
        return f"Error accessing data status: {str(e)}"


@tool("preview_data", args_schema=DataPreviewRequest)
def preview_data(job_id: str, n: int = 5) -> str:
    """
    Get a preview of processed data showing the first few rows.
    
    Use this to understand the structure and content of uploaded data.
    Shows column names, data types, and sample values.
    
    Args:
        job_id: UUID job identifier for the data processing job
        n: Number of rows to preview (default: 5, max recommended: 10)
        
    Returns:
        Formatted preview of the data with columns and sample rows
    """
    try:
        data_agent = get_data_agent()
        preview = asyncio.run(data_agent.preview(job_id, n))
        
        if "error" in preview:
            return f"Error: {preview['error']}"
        
        # Format preview for readable output
        result = f"""Data Preview (showing {preview['rows_shown']} of {preview['total_rows']} rows)

Columns: {', '.join(preview['columns'])}

Sample Data:
"""
        
        # Format the data rows
        for i, row in enumerate(preview['data']):
            result += f"\nRow {i+1}:\n"
            for col, val in row.items():
                result += f"  {col}: {val}\n"
        
        result += f"\nTotal dataset contains {preview['total_rows']:,} rows with {len(preview['columns'])} columns."
        
        return result
        
    except Exception as e:
        return f"Error accessing data preview: {str(e)}"


@tool("get_physics_insights", args_schema=DataInsightsRequest)
def get_physics_insights(job_id: str, analysis_type: str = "physics") -> str:
    """
    Extract physics-relevant insights from processed data.
    
    Analyzes uploaded data for physics patterns, units, correlations, and experimental characteristics.
    Use this to understand how the data relates to physics concepts and theories.
    
    Args:
        job_id: UUID job identifier for the data processing job
        analysis_type: Type of analysis ("physics", "correlations", "units", "temporal")
        
    Returns:
        Detailed physics insights including detected units, patterns, and recommendations
    """
    try:
        data_agent = get_data_agent()
        insights = data_agent.get_physics_insights(job_id)
        
        if "error" in insights:
            return f"Error: {insights['error']}"
        
        # Format insights for physics analysis
        result = f"""Physics Data Analysis Results

Data Type: {insights.get('data_type', 'experimental').replace('_', ' ').title()}

"""
        
        # Unit detection
        unit_detection = insights.get('unit_detection', {})
        if unit_detection.get('detected_units'):
            result += "Detected Physics Units:\n"
            for column, unit_type in unit_detection['detected_units'].items():
                result += f"  • {column}: {unit_type.replace('_', ' ').title()}\n"
            result += "\n"
        
        # Temporal analysis
        temporal = insights.get('temporal_analysis', {})
        if temporal.get('is_time_series'):
            result += "Temporal Analysis:\n"
            result += f"  • Time Series Data: Yes\n"
            result += f"  • Time Columns: {', '.join(temporal.get('time_columns', []))}\n"
            result += "  • Suitable for dynamics and time-evolution analysis\n\n"
        
        # Physics patterns
        patterns = insights.get('physics_patterns', [])
        if patterns:
            result += "Physics Patterns Detected:\n"
            for pattern in patterns:
                result += f"  • {pattern}\n"
            result += "\n"
        
        # Recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            result += "Analysis Recommendations:\n"
            for rec in recommendations:
                result += f"  • {rec}\n"
            result += "\n"
        
        if not any([unit_detection.get('detected_units'), temporal.get('is_time_series'), patterns, recommendations]):
            result += "No specific physics patterns detected. Data appears to be general experimental data suitable for statistical analysis."
        
        return result
        
    except Exception as e:
        return f"Error analyzing physics insights: {str(e)}"


@tool("publish_data_to_lab", args_schema=DataPublishRequest) 
def publish_data_to_lab(job_id: str) -> str:
    """
    Publish processed data to the physics lab memory system.
    
    Streams all data rows to the lab memory where other agents can access it.
    Use this after analyzing data to make it available for the entire physics team.
    
    Args:
        job_id: UUID job identifier for the data processing job
        
    Returns:
        Confirmation message with number of events published
    """
    try:
        data_agent = get_data_agent()
        result = asyncio.run(data_agent.publish(job_id))
        
        return f"✅ Data Publication Complete: {result}"
        
    except Exception as e:
        return f"Error publishing data: {str(e)}"


@tool("list_available_data")
def list_available_data() -> str:
    """
    List all available data processing jobs and their status.
    
    Use this to see what data has been uploaded and is available for analysis.
    Shows job IDs, file names, processing status, and basic information.
    
    Returns:
        List of all data jobs with their current status
    """
    try:
        data_agent = get_data_agent()
        
        # Get active and completed jobs
        active_jobs = data_agent.active_jobs
        completed_jobs = data_agent.completed_jobs
        
        if not active_jobs and not completed_jobs:
            return "No data files have been uploaded yet. Upload data files through the Data Upload tab to begin analysis."
        
        result = "Available Data Jobs:\n\n"
        
        # Active jobs
        if active_jobs:
            result += "Currently Processing:\n"
            for job_id, job_info in active_jobs.items():
                result += f"  • {job_id}: {job_info.get('file_path', 'Unknown')} - {job_info.get('status', 'Unknown')}\n"
            result += "\n"
        
        # Completed jobs
        if completed_jobs:
            result += "Completed Jobs:\n"
            for job_id, job_info in completed_jobs.items():
                status = job_info.get('status', 'Unknown')
                file_path = job_info.get('file_path', 'Unknown')
                rows = job_info.get('rows', 0)
                cols = job_info.get('columns', 0)
                
                result += f"  • {job_id}: {file_path}\n"
                result += f"    Status: {status}, Rows: {rows:,}, Columns: {cols}\n"
            result += "\n"
        
        result += f"Total jobs: {len(active_jobs) + len(completed_jobs)} (Active: {len(active_jobs)}, Completed: {len(completed_jobs)})"
        
        return result
        
    except Exception as e:
        return f"Error listing data jobs: {str(e)}"


# Tool collection for easy registration
DATA_AGENT_TOOLS = [
    get_data_status,
    preview_data, 
    get_physics_insights,
    publish_data_to_lab,
    list_available_data
]


def register_data_tools_with_agent(agent):
    """
    Register data tools with a physics agent.
    
    Args:
        agent: CrewAI agent to register tools with
    """
    if hasattr(agent, 'tools'):
        if agent.tools is None:
            agent.tools = []
        agent.tools.extend(DATA_AGENT_TOOLS)
    else:
        agent.tools = DATA_AGENT_TOOLS.copy()


def create_data_context_for_task(job_ids: List[str]) -> str:
    """
    Create a comprehensive data context string for physics analysis tasks.
    
    Args:
        job_ids: List of data job IDs to include in context
        
    Returns:
        Formatted context string with data information
    """
    if not job_ids:
        return "No experimental data available for this analysis."
    
    try:
        data_agent = get_data_agent()
        context_parts = []
        
        for job_id in job_ids:
            status = asyncio.run(data_agent.status(job_id))
            if status.get("status") == "completed":
                insights = data_agent.get_physics_insights(job_id)
                
                context_part = f"""
Data File: {status.get('file_path', 'Unknown')}
Dimensions: {status.get('rows', 0):,} rows × {status.get('columns', 0)} columns
Physics Insights: {', '.join(insights.get('physics_patterns', ['Standard experimental data']))}
Units Detected: {list(insights.get('unit_detection', {}).get('detected_units', {}).keys())}
Data Type: {insights.get('data_type', 'experimental')}
Job ID: {job_id}
"""
                context_parts.append(context_part)
        
        if context_parts:
            return "AVAILABLE EXPERIMENTAL DATA:\n" + "\n".join(context_parts) + "\nUse the data tools to access detailed information and insights."
        else:
            return "Data files uploaded but not yet ready for analysis."
            
    except Exception as e:
        return f"Error creating data context: {str(e)}" 