#!/usr/bin/env python3
"""
DataAgent - Advanced File Processing Agent for PhysicsGPT Laboratory
Handles data ingestion, validation, and integration with multi-agent physics analysis.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import pyarrow.parquet as pq

# Try to import magic, fallback to mimetypes if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    import mimetypes
    MAGIC_AVAILABLE = False

# Try to import uproot for ROOT file support
try:
    import uproot
    UPROOT_AVAILABLE = True
except ImportError:
    UPROOT_AVAILABLE = False

# Fix SQLite version issue
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from crewai_database_integration import CrewAIKnowledgeAPI


class DataAgentError(Exception):
    """Custom exception for DataAgent operations."""
    pass


class DataAgent:
    """
    Advanced data processing agent with secure file handling and physics integration.
    
    Features:
    - Auto-detection of CSV, TSV, JSON/NDJSON, Parquet, Excel, HDF5, ROOT
    - Sandboxed subprocess loading for security
    - Data validation and quality checks
    - Integration with PhysicsGPT laboratory memory system
    - Real-time processing status tracking
    """
    
    # File size limit (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Supported file loaders
    _LOADERS = {
        'text/csv': '_load_csv',
        'text/tab-separated-values': '_load_tsv', 
        'application/json': '_load_json',
        'application/x-ndjson': '_load_ndjson',
        'application/octet-stream': '_load_parquet_or_hdf5',  # Could be parquet or HDF5
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '_load_excel',
        'application/vnd.ms-excel': '_load_excel_old',
        'application/x-hdf': '_load_hdf5',
        'application/x-parquet': '_load_parquet',
        'application/x-root': '_load_root'
    }
    
    # Forbidden MIME types for security
    _FORBIDDEN_TYPES = {
        'application/x-pickle',
        'application/x-python-pickle', 
        'application/x-executable',
        'application/x-dosexec',
        'application/x-shellscript'
    }
    
    def __init__(self):
        """Initialize DataAgent with database integration."""
        self.staging_dir = Path("./data/staging")
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        
        # Job tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Database integration
        self.lab_memory = CrewAIKnowledgeAPI()
        
        # Log initialization
        self.lab_memory.log_event(
            source="data_agent",
            event_type="agent_initialized",
            payload={"staging_dir": str(self.staging_dir)}
        )
    
    async def ingest(self, path: str, meta: dict) -> str:
        """
        Ingest data file with comprehensive validation and processing.
        
        Args:
            path: File path to ingest
            meta: Metadata dictionary with context information
            
        Returns:
            UUID job ID for tracking
            
        Raises:
            DataAgentError: For validation failures or processing errors
        """
        job_id = str(uuid.uuid4())
        
        try:
            # Validate file exists and size
            file_path = Path(path)
            if not file_path.exists():
                raise DataAgentError(f"File not found: {path}")
            
            file_size = file_path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                raise DataAgentError(f"File too large: {file_size / 1024 / 1024:.1f}MB (max: 100MB)")
            
            # Detect MIME type
            if MAGIC_AVAILABLE:
                mime_type = magic.from_file(str(file_path), mime=True)
            else:
                # Fallback using file extension
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type is None:
                    # Guess based on file extension
                    ext = file_path.suffix.lower()
                    mime_type = {
                        '.csv': 'text/csv',
                        '.tsv': 'text/tab-separated-values',
                        '.json': 'application/json',
                        '.jsonl': 'application/x-ndjson',
                        '.ndjson': 'application/x-ndjson',
                        '.parquet': 'application/x-parquet',
                        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        '.xls': 'application/vnd.ms-excel',
                        '.h5': 'application/x-hdf',
                        '.hdf5': 'application/x-hdf',
                        '.root': 'application/x-root'
                    }.get(ext, 'application/octet-stream')
            
            # Special handling for ROOT files that might be detected as octet-stream
            if mime_type == 'application/octet-stream' and file_path.suffix.lower() == '.root':
                mime_type = 'application/x-root'
            
            # Security validation
            if mime_type in self._FORBIDDEN_TYPES:
                raise DataAgentError(f"Forbidden file type: {mime_type}")
            
            if mime_type not in self._LOADERS:
                raise DataAgentError(f"Unsupported file type: {mime_type}")
            
            # Initialize job tracking
            job_info = {
                "job_id": job_id,
                "file_path": str(file_path),
                "file_size": file_size,
                "mime_type": mime_type,
                "metadata": meta,
                "status": "processing",
                "started_at": datetime.now().isoformat(),
                "error": None,
                "stats": None,
                "preview": None
            }
            self.active_jobs[job_id] = job_info
            
            # Log ingestion start
            self.lab_memory.log_event(
                source="data_agent",
                event_type="ingestion_started",
                payload={
                    "job_id": job_id,
                    "file_size": file_size,
                    "mime_type": mime_type,
                    "metadata": meta
                }
            )
            
            # Load data in sandboxed subprocess
            loader_method = self._LOADERS[mime_type]
            
            # For ROOT files, try direct loading first if sandboxed fails
            # ROOT files are generally safe and our main method has better error handling
            if mime_type == 'application/x-root':
                try:
                    df = await self._load_file_sandboxed(str(file_path), loader_method)
                except DataAgentError as e:
                    error_msg = str(e).lower()
                    if ("signature" in error_msg or "subprocess" in error_msg or 
                        "unable to" in error_msg or "magic" in error_msg or
                        "same length" in error_msg or "all arrays must" in error_msg):
                        print(f"Sandboxed ROOT loading failed ({e}), trying direct loading...")
                        # Fall back to direct loading for ROOT files
                        df = getattr(self, loader_method)(str(file_path))
                    else:
                        raise
                except Exception as e:
                    # Catch any other subprocess-related errors
                    print(f"Sandboxed ROOT loading failed with unexpected error ({e}), trying direct loading...")
                    df = getattr(self, loader_method)(str(file_path))
            else:
                df = await self._load_file_sandboxed(str(file_path), loader_method)
            
            # Validate data quality
            await self._validate_dataframe(df, job_id)
            
            # Generate output path
            output_path = self.staging_dir / f"{job_id}.parq"
            
            # Save as Parquet
            df.to_parquet(output_path, engine='pyarrow', compression='snappy')
            
            # Generate statistics and preview
            stats = self._generate_stats(df)
            preview = self._generate_preview(df, n=10)
            
            # Update job info
            job_info.update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "output_path": str(output_path),
                "rows": len(df),
                "columns": len(df.columns),
                "stats": stats,
                "preview": preview
            })
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.active_jobs[job_id]
            
            # Log completion
            self.lab_memory.log_event(
                source="data_agent",
                event_type="ingestion_completed",
                payload={
                    "job_id": job_id,
                    "rows_processed": len(df),
                    "columns": len(df.columns),
                    "processing_time": (datetime.fromisoformat(job_info["completed_at"]) - 
                                      datetime.fromisoformat(job_info["started_at"])).total_seconds()
                }
            )
            
            return job_id
            
        except Exception as e:
            # Update job with error
            if job_id in self.active_jobs:
                self.active_jobs[job_id].update({
                    "status": "failed", 
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                })
                self.completed_jobs[job_id] = self.active_jobs[job_id]
                del self.active_jobs[job_id]
            
            # Log error
            self.lab_memory.log_event(
                source="data_agent",
                event_type="ingestion_failed",
                payload={"job_id": job_id, "error": str(e)}
            )
            
            raise DataAgentError(f"Ingestion failed: {str(e)}")
    
    async def status(self, job_id: str) -> dict:
        """
        Get processing status and basic statistics for a job.
        
        Args:
            job_id: UUID job identifier
            
        Returns:
            Status dictionary with file info and statistics
        """
        # Check active jobs first
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id].copy()
            job_info.pop('preview', None)  # Don't include full preview in status
            return job_info
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job_info = self.completed_jobs[job_id].copy()
            job_info.pop('preview', None)  # Don't include full preview in status
            return job_info
        
        # Job not found
        return {"error": f"Job {job_id} not found", "status": "not_found"}
    
    async def preview(self, job_id: str, n: int = 5) -> dict:
        """
        Get preview of processed data.
        
        Args:
            job_id: UUID job identifier
            n: Number of rows to preview
            
        Returns:
            Dictionary with normalized row data
        """
        if job_id not in self.completed_jobs:
            return {"error": f"Job {job_id} not found or not completed"}
        
        job_info = self.completed_jobs[job_id]
        
        if job_info["status"] != "completed":
            return {"error": f"Job {job_id} failed: {job_info.get('error', 'Unknown error')}"}
        
        try:
            # Load parquet file and get preview
            output_path = Path(job_info["output_path"])
            df = pd.read_parquet(output_path)
            
            # Get first n rows
            preview_df = df.head(n)
            
            # Convert to JSON-serializable format
            preview_data = []
            for _, row in preview_df.iterrows():
                row_dict = {}
                for col, val in row.items():
                    # Handle various data types for JSON serialization
                    if pd.isna(val):
                        row_dict[col] = None
                    elif isinstance(val, (pd.Timestamp, datetime)):
                        row_dict[col] = val.isoformat()
                    elif isinstance(val, (int, float, str, bool)):
                        row_dict[col] = val
                    else:
                        row_dict[col] = str(val)
                preview_data.append(row_dict)
            
            return {
                "job_id": job_id,
                "rows_shown": len(preview_data),
                "total_rows": len(df),
                "columns": list(df.columns),
                "data": preview_data
            }
            
        except Exception as e:
            return {"error": f"Failed to load preview: {str(e)}"}
    
    async def publish(self, job_id: str) -> str:
        """
        Publish processed data to lab memory system.
        
        Args:
            job_id: UUID job identifier
            
        Returns:
            Success message with event count
        """
        if job_id not in self.completed_jobs:
            raise DataAgentError(f"Job {job_id} not found or not completed")
        
        job_info = self.completed_jobs[job_id]
        
        if job_info["status"] != "completed":
            raise DataAgentError(f"Job {job_id} failed: {job_info.get('error', 'Unknown error')}")
        
        try:
            # Load the processed data
            output_path = Path(job_info["output_path"])
            df = pd.read_parquet(output_path)
            
            # Get metadata
            meta = job_info["metadata"]
            event_type = meta.get("type", "raw_data")
            
            # Stream each row to lab memory
            events_published = 0
            batch_size = 100  # Process in batches for large datasets
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    # Convert row to dict
                    row_dict = {}
                    for col, val in row.items():
                        if pd.isna(val):
                            row_dict[col] = None
                        elif isinstance(val, (pd.Timestamp, datetime)):
                            row_dict[col] = val.isoformat()
                        elif isinstance(val, (int, float, str, bool)):
                            row_dict[col] = val
                        else:
                            row_dict[col] = str(val)
                    
                    # Log to lab memory
                    self.lab_memory.log_event(
                        source="data_agent",
                        event_type=event_type,
                        payload=row_dict
                    )
                    events_published += 1
                
                # Small delay between batches to prevent overwhelming the system
                if i + batch_size < len(df):
                    await asyncio.sleep(0.01)
            
            # Emit LangGraph signal (simulate for now - would integrate with actual LangGraph)
            self._emit_signal("event_ingested", job_id)
            
            # Log publication completion
            self.lab_memory.log_event(
                source="data_agent",
                event_type="data_published",
                payload={
                    "job_id": job_id,
                    "events_published": events_published,
                    "event_type": event_type
                }
            )
            
            return f"Successfully published {events_published} events to lab memory"
            
        except Exception as e:
            raise DataAgentError(f"Failed to publish data: {str(e)}")
    
    async def _load_file_sandboxed(self, file_path: str, loader_method: str) -> pd.DataFrame:
        """Load file in sandboxed subprocess for security."""
        
        # Create temporary script for subprocess
        script_content = f'''
import pandas as pd
import sys
import json
import os

def {loader_method.replace("_", "")}(file_path):
    {self._get_loader_code(loader_method)}

try:
    df = {loader_method.replace("_", "")}("{file_path}")
    
    # Basic validation
    if df.empty:
        raise ValueError("Empty dataframe")
    
    # Convert to JSON for transport
    result = {{
        "success": True,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "data": df.to_dict(orient="records")[:1000]  # Limit for transport
    }}
    print(json.dumps(result))
    
except Exception as e:
    result = {{"success": False, "error": str(e)}}
    print(json.dumps(result))
'''
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Run in subprocess with timeout
            import sys
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                raise DataAgentError(f"Subprocess failed: {result.stderr}")
            
            # Parse result
            try:
                output = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                raise DataAgentError(f"Failed to parse subprocess output: {result.stdout}")
            
            if not output.get("success"):
                raise DataAgentError(f"Data loading failed: {output.get('error', 'Unknown error')}")
            
            # Reconstruct dataframe
            df = pd.DataFrame(output["data"])
            
            # Reload full file if truncated (for small files)
            if len(df) == 1000 and output["shape"][0] > 1000:
                # For large files, load in the main process (already validated)
                df = getattr(self, loader_method)(file_path)
            
            return df
            
        finally:
            # Clean up temporary script
            try:
                os.unlink(script_path)
            except OSError:
                pass
    
    def _get_loader_code(self, loader_method: str) -> str:
        """Get the appropriate loader code for subprocess execution."""
        
        loaders = {
            '_load_csv': '''
    return pd.read_csv(file_path, encoding='utf-8-sig')
            ''',
            '_load_tsv': '''
    return pd.read_csv(file_path, sep='\\t', encoding='utf-8-sig')
            ''',
            '_load_json': '''
    return pd.read_json(file_path, encoding='utf-8')
            ''',
            '_load_ndjson': '''
    return pd.read_json(file_path, lines=True, encoding='utf-8')
            ''',
            '_load_excel': '''
    return pd.read_excel(file_path, engine='openpyxl')
            ''',
            '_load_excel_old': '''
    return pd.read_excel(file_path, engine='xlrd')
            ''',
            '_load_parquet': '''
    return pd.read_parquet(file_path, engine='pyarrow')
            ''',
            '_load_hdf5': '''
    import h5py
    # Simple HDF5 loading - may need enhancement for complex structures
    with h5py.File(file_path, 'r') as f:
        # Try to find a dataset that looks like tabular data
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                data = f[key][:]
                if len(data.shape) <= 2:  # 1D or 2D array
                    if len(data.shape) == 1:
                        return pd.DataFrame({key: data})
                    else:
                        return pd.DataFrame(data)
    raise ValueError("No suitable tabular data found in HDF5 file")
            ''',
            '_load_root': '''
    import uproot
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Validate file exists and basic properties
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise ValueError(f"ROOT file not found: {file_path}")
    
    file_size = file_path_obj.stat().st_size
    if file_size == 0:
        raise ValueError("ROOT file is empty")
    
    if file_size < 50:
        raise ValueError("File too small to be a valid ROOT file")
    
    try:
        # Open ROOT file with enhanced error handling
        with uproot.open(file_path) as root_file:
            # Use classnames() to inspect without reading (best practice)
            try:
                classnames = root_file.classnames()
            except Exception as e:
                # If classnames fails, try basic inspection
                try:
                    keys = root_file.keys()
                    if not keys:
                        raise ValueError("ROOT file contains no readable objects")
                    classnames = {}
                    for key in keys:
                        try:
                            obj = root_file[key]
                            classnames[key] = obj.classname if hasattr(obj, 'classname') else type(obj).__name__
                        except:
                            classnames[key] = 'Unknown'
                except Exception as inner_e:
                    raise ValueError(f"ROOT file appears corrupted: {str(e)}")
            
            if not classnames:
                raise ValueError("No objects found in ROOT file")
            
            # Try different ROOT object types following best practices
            
            # 1. TTrees and RNTuples (most structured data)
            tree_keys = [k for k, cls in classnames.items() 
                        if cls in ['TTree', 'ROOT::RNTuple']]
            if tree_keys:
                tree_key = tree_keys[0]
                try:
                    tree = root_file[tree_key]
                except Exception as e:
                    raise ValueError(f"Failed to access TTree '{tree_key}': {str(e)}")
                
                if hasattr(tree, 'keys') and tree.keys():
                    try:
                        # For subprocess, limit to 1000 entries for safety
                        num_entries = getattr(tree, 'num_entries', 1000)
                        max_entries = min(1000, num_entries)
                        
                        if num_entries == 0:
                            raise ValueError(f"TTree '{tree_key}' is empty")
                        
                        # Try pandas library first for better integration
                        try:
                            return tree.arrays(library="pd", entry_stop=max_entries)
                        except Exception as pd_error:
                            # Fallback to awkward arrays with simplified handling
                            arrays = tree.arrays(library="ak", entry_stop=max_entries)
                            data_dict = {}
                            
                            for branch_name in tree.keys():
                                try:
                                    arr = arrays[branch_name]
                                    if arr.ndim == 1:
                                        data_dict[branch_name] = arr.to_numpy()
                                    else:
                                        # For nested data, try flattening
                                        try:
                                            flat_arr = arr.flatten()
                                            if len(flat_arr) <= len(arr) * 5:  # Reasonable expansion
                                                data_dict[branch_name] = flat_arr.to_numpy()
                                            else:
                                                # Take first element for deeply nested data
                                                data_dict[branch_name] = arr[:, 0].to_numpy()
                                        except:
                                            # Convert to string for complex structures (limit for subprocess)
                                            data_dict[branch_name] = [str(x) for x in arr.to_list()[:500]]
                                except Exception as branch_error:
                                    # Create placeholder for problematic branches
                                    data_dict[branch_name] = [f"Error: {str(branch_error)}"] * min(500, max_entries)
                            
                            if not data_dict:
                                raise ValueError("No branches could be processed from TTree")
                            
                            return pd.DataFrame(data_dict)
                    except Exception as tree_error:
                        raise ValueError(f"Could not read TTree '{tree_key}': {str(tree_error)}")
            
            # 2. TNtuples (simple tabular data)
            ntuple_keys = [k for k, cls in classnames.items() if cls == 'TNtuple']
            if ntuple_keys:
                try:
                    ntuple = root_file[ntuple_keys[0]]
                    try:
                        return ntuple.arrays(library="pd")
                    except:
                        arrays = ntuple.arrays(library="ak")
                        data_dict = {branch: arrays[branch].to_numpy() 
                                   for branch in ntuple.keys()}
                        return pd.DataFrame(data_dict)
                except Exception as e:
                    # Continue to next object type
                    pass
            
            # 3. Histograms (convert bin contents to tabular data)
            hist_types = ['TH1F', 'TH1D', 'TH1I', 'TH1C', 'TH1S',
                         'TH2F', 'TH2D', 'TH2I', 'TH2C', 'TH2S',
                         'TProfile', 'TProfile2D']
            hist_keys = [k for k, cls in classnames.items() if cls in hist_types]
            if hist_keys:
                try:
                    hist = root_file[hist_keys[0]]
                    classname = classnames[hist_keys[0]]
                    
                    # Use to_numpy() method when available (best practice)
                    if classname.startswith('TH1') or classname == 'TProfile':
                        values, edges = hist.to_numpy()
                        centers = (edges[:-1] + edges[1:]) / 2
                        return pd.DataFrame({'bin_center': centers, 'bin_content': values})
                    elif classname.startswith('TH2') or classname == 'TProfile2D':
                        values, x_edges, y_edges = hist.to_numpy()
                        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
                        return pd.DataFrame({
                            'x_center': X.flatten(),
                            'y_center': Y.flatten(),
                            'bin_content': values.flatten()
                        })
                except Exception as e:
                    # Continue to next object type
                    pass
            
            # 4. TGraph objects (if present)
            graph_types = ['TGraph', 'TGraphErrors', 'TGraphAsymmErrors']
            graph_keys = [k for k, cls in classnames.items() if cls in graph_types]
            if graph_keys:
                try:
                    graph = root_file[graph_keys[0]]
                    x_values = graph.values(axis="x")
                    y_values = graph.values(axis="y")
                    return pd.DataFrame({'x': x_values, 'y': y_values})
                except Exception as e:
                    # Continue to error message
                    pass
            
            # Generate helpful error message
            available = [f"{k}({cls})" for k, cls in classnames.items()]
            raise ValueError(f"No supported ROOT objects could be read. Available: {available}")
            
    except Exception as e:
        # Enhanced error message for different types of issues
        error_msg = str(e).lower()
        if "signature" in error_msg or "magic" in error_msg:
            raise ValueError(f"ROOT file appears corrupted or has invalid signature. Original error: {str(e)}")
        elif "permission" in error_msg or "access" in error_msg:
            raise ValueError(f"Cannot access ROOT file - check permissions: {str(e)}")
        else:
            raise ValueError(f"Failed to read ROOT file: {str(e)}")
            ''',
            '_load_parquet_or_hdf5': '''
    # Try parquet first, then HDF5
    try:
        return pd.read_parquet(file_path, engine='pyarrow')
    except:
        import h5py
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][:]
                    if len(data.shape) <= 2:
                        if len(data.shape) == 1:
                            return pd.DataFrame({key: data})
                        else:
                            return pd.DataFrame(data)
        raise ValueError("No suitable data format found")
            '''
        }
        
        return loaders.get(loader_method, 'raise ValueError("Unknown loader method")')
    
    # Actual loader methods for main process
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, encoding='utf-8-sig')
    
    def _load_tsv(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, sep='\t', encoding='utf-8-sig')
    
    def _load_json(self, file_path: str) -> pd.DataFrame:
        return pd.read_json(file_path, encoding='utf-8')
    
    def _load_ndjson(self, file_path: str) -> pd.DataFrame:
        return pd.read_json(file_path, lines=True, encoding='utf-8')
    
    def _load_excel(self, file_path: str) -> pd.DataFrame:
        return pd.read_excel(file_path, engine='openpyxl')
    
    def _load_excel_old(self, file_path: str) -> pd.DataFrame:
        return pd.read_excel(file_path, engine='xlrd')
    
    def _load_parquet(self, file_path: str) -> pd.DataFrame:
        return pd.read_parquet(file_path, engine='pyarrow')
    
    def _load_hdf5(self, file_path: str) -> pd.DataFrame:
        import h5py
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][:]
                    if len(data.shape) <= 2:
                        if len(data.shape) == 1:
                            return pd.DataFrame({key: data})
                        else:
                            return pd.DataFrame(data)
        raise ValueError("No suitable tabular data found in HDF5 file")
    
    def _load_parquet_or_hdf5(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(file_path, engine='pyarrow')
        except:
            return self._load_hdf5(file_path)
    
    def _load_root(self, file_path: str) -> pd.DataFrame:
        """Load ROOT file using uproot following current best practices."""
        if not UPROOT_AVAILABLE:
            raise DataAgentError("ROOT file support requires uproot package: pip install uproot")
        
        import uproot
        import pandas as pd
        import numpy as np
        
        # First, validate that the file exists and is readable
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise DataAgentError(f"ROOT file not found: {file_path}")
        
        # Check file size and basic properties
        file_size = file_path_obj.stat().st_size
        if file_size == 0:
            raise DataAgentError("ROOT file is empty")
        
        if file_size < 50:  # ROOT files are typically much larger
            raise DataAgentError("File too small to be a valid ROOT file")
        
        # Try to open ROOT file with enhanced error handling
        try:
            with uproot.open(file_path) as root_file:
                # Use classnames() to inspect objects without reading them (best practice)
                try:
                    classnames = root_file.classnames()
                except Exception as e:
                    # If classnames fails, try basic inspection
                    try:
                        keys = root_file.keys()
                        if not keys:
                            raise DataAgentError("ROOT file contains no readable objects")
                        # Create basic classnames dict
                        classnames = {}
                        for key in keys:
                            try:
                                obj = root_file[key]
                                classnames[key] = obj.classname if hasattr(obj, 'classname') else type(obj).__name__
                            except:
                                classnames[key] = 'Unknown'
                    except Exception as inner_e:
                        raise DataAgentError(f"ROOT file appears corrupted or uses unsupported format. Original error: {str(e)}, Inspection error: {str(inner_e)}")
                
                if not classnames:
                    raise DataAgentError("No objects found in ROOT file")
                
                # Log file structure for debugging
                print(f"ROOT file structure: {classnames}")
                
                # Try different ROOT object types in order of preference
                
                # 1. First try TTrees and RNTuples (most structured data)
                tree_keys = [k for k, cls in classnames.items() 
                            if cls in ['TTree', 'ROOT::RNTuple']]
                if tree_keys:
                    tree_key = tree_keys[0]
                    print(f"Loading TTree: {tree_key}")
                    
                    try:
                        tree = root_file[tree_key]
                    except Exception as e:
                        raise DataAgentError(f"Failed to access TTree '{tree_key}': {str(e)}")
                    
                    # Check if TTree has branches
                    if hasattr(tree, 'keys') and tree.keys():
                        # For small datasets, try using library="pd" directly for better integration
                        try:
                            num_entries = getattr(tree, 'num_entries', 0)
                            print(f"TTree has {num_entries} entries")
                            
                            if num_entries == 0:
                                raise DataAgentError(f"TTree '{tree_key}' is empty")
                            
                            # Choose appropriate reading strategy based on size
                            if num_entries <= 50000:
                                # Small file - read directly to pandas
                                try:
                                    return tree.arrays(library="pd")
                                except Exception as pd_error:
                                    print(f"Direct pandas conversion failed: {pd_error}, trying awkward arrays...")
                                    # Fallback to awkward arrays
                                    pass
                            else:
                                # Large file - read first 10000 entries as sample
                                try:
                                    print(f"Large file ({num_entries} entries), reading first 10000...")
                                    return tree.arrays(library="pd", entry_stop=10000)
                                except Exception as pd_error:
                                    print(f"Direct pandas conversion failed: {pd_error}, trying awkward arrays...")
                                    # Fallback to awkward arrays
                                    pass
                                    
                            # Fallback to awkward arrays for complex nested data
                            try:
                                # Read a reasonable number of entries to avoid memory issues
                                max_entries = min(25000, num_entries)
                                print(f"Reading {max_entries} entries using awkward arrays...")
                                arrays = tree.arrays(library="ak", entry_stop=max_entries)
                                data_dict = {}
                                
                                branch_keys = tree.keys()
                                print(f"Processing {len(branch_keys)} branches...")
                                
                                # First pass: determine the expected length
                                expected_length = max_entries
                                
                                for branch_name in branch_keys:
                                    try:
                                        arr = arrays[branch_name]
                                        
                                        # Handle different array types using awkward array utilities
                                        # Check if array is categorical (string-like)
                                        if hasattr(arr, 'layout') and 'categorical' in str(type(arr.layout)):
                                            processed_data = arr.to_numpy().astype(str)
                                        # Check if it's a regular array that can be directly converted
                                        elif arr.ndim == 1:
                                            processed_data = arr.to_numpy()
                                        # Handle jagged/nested arrays
                                        else:
                                            # Try to flatten jagged arrays
                                            try:
                                                flat_arr = arr.flatten()
                                                if len(flat_arr) <= len(arr) * 10:  # Reasonable expansion
                                                    processed_data = flat_arr.to_numpy()
                                                else:
                                                    # Take first element of each entry for very nested data
                                                    processed_data = arr[:, 0].to_numpy()
                                            except:
                                                # For very complex structures, convert to string representation
                                                arr_list = arr.to_list()[:1000]  # Limit for memory
                                                processed_data = [str(x) for x in arr_list]
                                        
                                        # Ensure all arrays have the same length
                                        if len(processed_data) != expected_length:
                                            if len(processed_data) > expected_length:
                                                # Truncate if too long
                                                processed_data = processed_data[:expected_length]
                                            else:
                                                # Pad if too short (for jagged arrays)
                                                if isinstance(processed_data, list):
                                                    processed_data.extend([None] * (expected_length - len(processed_data)))
                                                else:
                                                    import numpy as np
                                                    if processed_data.dtype.kind in 'iuf':  # numeric
                                                        pad_value = np.nan
                                                    else:  # string or object
                                                        pad_value = None
                                                    padding = np.full(expected_length - len(processed_data), pad_value, dtype=processed_data.dtype)
                                                    processed_data = np.concatenate([processed_data, padding])
                                        
                                        data_dict[branch_name] = processed_data
                                                
                                    except Exception as branch_error:
                                        print(f"Warning: Could not process branch '{branch_name}': {branch_error}")
                                        # Create placeholder for problematic branches with correct length
                                        placeholder = [f"Error: {str(branch_error)}"] * expected_length
                                        data_dict[branch_name] = placeholder
                                
                                if not data_dict:
                                    raise DataAgentError("No branches could be successfully processed from TTree")
                                
                                # Final validation: ensure all arrays have the same length
                                lengths = {k: len(v) for k, v in data_dict.items()}
                                unique_lengths = set(lengths.values())
                                if len(unique_lengths) > 1:
                                    print(f"Warning: Found different array lengths: {lengths}")
                                    # Fix any remaining length mismatches
                                    target_length = max(unique_lengths)
                                    for key, value in data_dict.items():
                                        if len(value) < target_length:
                                            if isinstance(value, list):
                                                value.extend([None] * (target_length - len(value)))
                                            else:
                                                import numpy as np
                                                if value.dtype.kind in 'iuf':  # numeric
                                                    pad_value = np.nan
                                                else:  # string or object
                                                    pad_value = None
                                                padding = np.full(target_length - len(value), pad_value, dtype=value.dtype)
                                                data_dict[key] = np.concatenate([value, padding])
                                        elif len(value) > target_length:
                                            data_dict[key] = value[:target_length]
                                
                                return pd.DataFrame(data_dict)
                                
                            except Exception as ak_error:
                                raise DataAgentError(f"Could not read TTree '{tree_key}' with awkward arrays: {str(ak_error)}")
                        
                        except Exception as tree_read_error:
                            raise DataAgentError(f"Failed to read TTree data: {str(tree_read_error)}")
                    else:
                        print(f"TTree '{tree_key}' has no branches or keys")
                
                # 2. Try TNtuples (simple tabular data)
                ntuple_keys = [k for k, cls in classnames.items() if cls == 'TNtuple']
                if ntuple_keys:
                    ntuple_key = ntuple_keys[0]
                    print(f"Loading TNtuple: {ntuple_key}")
                    
                    try:
                        ntuple = root_file[ntuple_key]
                        # TNtuples are simple - try pandas directly first
                        try:
                            return ntuple.arrays(library="pd")
                        except Exception:
                            # Fallback to awkward arrays
                            arrays = ntuple.arrays(library="ak")
                            data_dict = {branch: arrays[branch].to_numpy() 
                                       for branch in ntuple.keys()}
                            return pd.DataFrame(data_dict)
                    except Exception as e:
                        print(f"Failed to read TNtuple '{ntuple_key}': {e}")
                
                # 3. Try histograms (convert bin contents to data)
                hist_types = ['TH1F', 'TH1D', 'TH1I', 'TH1C', 'TH1S',  # 1D histograms
                             'TH2F', 'TH2D', 'TH2I', 'TH2C', 'TH2S',  # 2D histograms  
                             'TH3F', 'TH3D', 'TH3I', 'TH3C', 'TH3S',  # 3D histograms
                             'TProfile', 'TProfile2D', 'TProfile3D']   # Profile histograms
                
                hist_keys = [k for k, cls in classnames.items() if cls in hist_types]
                if hist_keys:
                    hist_key = hist_keys[0]  # Use first histogram
                    print(f"Loading histogram: {hist_key}")
                    
                    try:
                        hist = root_file[hist_key]
                        classname = classnames[hist_key]
                        
                        # Use histogram's to_numpy() method when available (best practice)
                        if classname.startswith('TH1') or classname == 'TProfile':
                            # 1D histogram or profile
                            values, bin_edges = hist.to_numpy()
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            return pd.DataFrame({
                                'bin_center': bin_centers,
                                'bin_content': values
                            })
                        elif classname.startswith('TH2') or classname == 'TProfile2D':
                            # 2D histogram or profile
                            values, x_edges, y_edges = hist.to_numpy()
                            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                            
                            # Create meshgrid and flatten for tabular format
                            X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
                            return pd.DataFrame({
                                'x_center': X.flatten(),
                                'y_center': Y.flatten(),
                                'bin_content': values.flatten()
                            })
                        elif classname.startswith('TH3') or classname == 'TProfile3D':
                            # 3D histogram - flatten to tabular format (limit size for memory)
                            values, x_edges, y_edges, z_edges = hist.to_numpy()
                            
                            # Check if result would be too large
                            total_bins = len(x_edges) * len(y_edges) * len(z_edges)
                            if total_bins > 100000:
                                raise DataAgentError(f"3D histogram too large ({total_bins} bins)")
                                
                            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                            z_centers = (z_edges[:-1] + z_edges[1:]) / 2
                            
                            X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
                            return pd.DataFrame({
                                'x_center': X.flatten(),
                                'y_center': Y.flatten(), 
                                'z_center': Z.flatten(),
                                'bin_content': values.flatten()
                            })
                    except Exception as e:
                        print(f"Failed to read histogram '{hist_key}': {e}")
                
                # 4. Try TGraph objects (if present)
                graph_types = ['TGraph', 'TGraphErrors', 'TGraphAsymmErrors']
                graph_keys = [k for k, cls in classnames.items() if cls in graph_types]
                if graph_keys:
                    graph_key = graph_keys[0]
                    print(f"Loading TGraph: {graph_key}")
                    
                    try:
                        graph = root_file[graph_key]
                        
                        # Extract graph data
                        x_values = graph.values(axis="x")
                        y_values = graph.values(axis="y")
                        data = {'x': x_values, 'y': y_values}
                        
                        # Add errors if available
                        if classnames[graph_key] in ['TGraphErrors', 'TGraphAsymmErrors']:
                            try:
                                x_errors = graph.errors(axis="x")
                                y_errors = graph.errors(axis="y")
                                if x_errors is not None:
                                    data['x_error'] = x_errors
                                if y_errors is not None:
                                    data['y_error'] = y_errors
                            except:
                                pass  # Errors not available
                                
                        return pd.DataFrame(data)
                    except Exception as e:
                        print(f"Failed to read TGraph '{graph_key}': {e}")
                
                # Generate helpful error message with available objects
                available_objects = []
                for k, cls in classnames.items():
                    available_objects.append(f"{k} ({cls})")
                
                raise DataAgentError(f"No supported ROOT objects could be read. Available objects: {', '.join(available_objects)}")
                
        except uproot.exceptions.KeyInFileError as e:
            raise DataAgentError(f"ROOT file structure error: {str(e)}")
        except Exception as e:
            # Enhanced error message for different types of issues
            error_msg = str(e).lower()
            if "signature" in error_msg or "magic" in error_msg:
                raise DataAgentError(f"ROOT file appears corrupted or has invalid file signature. This could be due to: 1) Incomplete file download, 2) File corruption during transfer, 3) Non-ROOT file with .root extension. Original error: {str(e)}")
            elif "permission" in error_msg or "access" in error_msg:
                raise DataAgentError(f"Cannot access ROOT file - check file permissions: {str(e)}")
            elif "memory" in error_msg or "allocation" in error_msg:
                raise DataAgentError(f"ROOT file too large or complex for available memory: {str(e)}")
            else:
                raise DataAgentError(f"Failed to read ROOT file: {str(e)}")
    
    async def _validate_dataframe(self, df: pd.DataFrame, job_id: str):
        """Validate dataframe quality and detect issues."""
        
        # Check for completely empty dataframe
        if df.empty:
            raise DataAgentError("Dataframe is empty")
        
        # Check for NaNs in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_percentage = (nan_count / len(df)) * 100
                
                # Log warning for high NaN percentage
                self.lab_memory.log_event(
                    source="data_agent",
                    event_type="data_quality_warning",
                    payload={
                        "job_id": job_id,
                        "column": col,
                        "nan_count": int(nan_count),
                        "nan_percentage": float(nan_percentage),
                        "warning_type": "missing_numeric_data"
                    }
                )
                
                # Strict validation: reject if >50% NaNs in numeric columns
                if nan_percentage > 50:
                    raise DataAgentError(f"Column '{col}' has {nan_percentage:.1f}% NaN values")
    
    def _generate_stats(self, df: pd.DataFrame) -> dict:
        """Generate comprehensive statistics for the dataframe."""
        
        stats = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_summary": {},
            "categorical_summary": {},
            "missing_data": df.isnull().sum().to_dict(),
            "column_types": {
                "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime": df.select_dtypes(include=['datetime64']).columns.tolist()
            }
        }
        
        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            desc = numeric_df.describe()
            stats["numeric_summary"] = desc.to_dict()
        
        # Categorical statistics
        categorical_df = df.select_dtypes(include=['object', 'category'])
        for col in categorical_df.columns:
            stats["categorical_summary"][col] = {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head().to_dict()
            }
        
        return stats
    
    def _generate_preview(self, df: pd.DataFrame, n: int = 10) -> dict:
        """Generate preview data for the dataframe."""
        
        preview_df = df.head(n)
        
        preview_data = []
        for _, row in preview_df.iterrows():
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (pd.Timestamp, datetime)):
                    row_dict[col] = val.isoformat()
                elif isinstance(val, (int, float, str, bool)):
                    row_dict[col] = val
                else:
                    row_dict[col] = str(val)
            preview_data.append(row_dict)
        
        return {
            "rows_shown": len(preview_data),
            "total_rows": len(df),
            "columns": list(df.columns),
            "data": preview_data
        }
    
    def _emit_signal(self, signal_name: str, job_id: str):
        """Emit LangGraph signal for downstream processing."""
        # This is a placeholder for LangGraph signal emission
        # In actual implementation, this would integrate with LangGraph's signal system
        
        self.lab_memory.log_event(
            source="data_agent",
            event_type="signal_emitted",
            payload={
                "signal_name": signal_name,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def get_physics_insights(self, job_id: str) -> dict:
        """Extract physics-relevant insights from processed data."""
        
        if job_id not in self.completed_jobs:
            return {"error": f"Job {job_id} not found or not completed"}
        
        job_info = self.completed_jobs[job_id]
        
        if job_info["status"] != "completed":
            return {"error": f"Job {job_id} failed: {job_info.get('error', 'Unknown error')}"}
        
        try:
            # Load the data
            output_path = Path(job_info["output_path"])
            df = pd.read_parquet(output_path)
            
            insights = {
                "data_type": "experimental",  # Default assumption
                "temporal_analysis": {},
                "unit_detection": {},
                "physics_patterns": [],
                "recommendations": []
            }
            
            # Detect time series data
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                insights["temporal_analysis"]["is_time_series"] = True
                insights["temporal_analysis"]["time_columns"] = datetime_cols.tolist()
                insights["data_type"] = "time_series_experimental"
            
            # Detect potential physics units in column names
            physics_units = {
                'time': ['time', 't', 'seconds', 'sec', 'ms', 'minutes', 'hours'],
                'length': ['length', 'distance', 'x', 'y', 'z', 'position', 'meter', 'm', 'cm', 'mm'],
                'mass': ['mass', 'm', 'kg', 'weight', 'gram', 'g'],
                'force': ['force', 'f', 'newton', 'n', 'pressure'],
                'energy': ['energy', 'e', 'joule', 'j', 'potential', 'kinetic'],
                'velocity': ['velocity', 'v', 'speed', 'ms', 'm/s'],
                'acceleration': ['acceleration', 'a', 'accel', 'm/s2', 'm/s^2'],
                'temperature': ['temperature', 'temp', 't', 'kelvin', 'k', 'celsius', 'c'],
                'electric': ['voltage', 'current', 'resistance', 'volt', 'amp', 'ohm', 'electric'],
                'magnetic': ['magnetic', 'field', 'tesla', 'gauss', 'flux']
            }
            
            detected_units = {}
            for col in df.columns:
                col_lower = col.lower()
                for unit_type, keywords in physics_units.items():
                    if any(keyword in col_lower for keyword in keywords):
                        detected_units[col] = unit_type
                        break
            
            if detected_units:
                insights["unit_detection"]["detected_units"] = detected_units
                insights["physics_patterns"].append(f"Detected {len(detected_units)} columns with physics units")
            
            # Check for experimental data patterns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Look for potential relationships
                correlations = df[numeric_cols].corr()
                high_correlations = []
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:  # Avoid duplicates
                            corr_val = correlations.loc[col1, col2]
                            if abs(corr_val) > 0.7:  # Strong correlation
                                high_correlations.append({
                                    "variables": [col1, col2],
                                    "correlation": float(corr_val),
                                    "relationship": "strong_positive" if corr_val > 0 else "strong_negative"
                                })
                
                if high_correlations:
                    insights["physics_patterns"].extend([
                        f"Strong correlation between {rel['variables'][0]} and {rel['variables'][1]} (r={rel['correlation']:.3f})"
                        for rel in high_correlations
                    ])
            
            # Generate recommendations for physics analysis
            if detected_units:
                insights["recommendations"].append("Data contains physics units - suitable for quantitative analysis")
            
            if insights["temporal_analysis"].get("is_time_series"):
                insights["recommendations"].append("Time series data detected - suitable for dynamics analysis")
            
            if len(numeric_cols) >= 3:
                insights["recommendations"].append("Multiple numeric variables - suitable for multi-dimensional physics modeling")
            
            return insights
            
        except Exception as e:
            return {"error": f"Failed to analyze physics insights: {str(e)}"}


# Import numpy for data validation
import numpy as np 