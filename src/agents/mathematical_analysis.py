"""
Mathematical Analysis Agent for statistical analysis and pattern recognition.
Handles curve fitting, correlation analysis, and mathematical relationship discovery.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import logging
from scipy import stats, optimize
from scipy.optimize import curve_fit
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import sympy as sp

from .base import BaseAgent
from ..database.knowledge_api import KnowledgeAPI
from ..memory.stores import get_memory_store


class MathematicalAnalysisAgent(BaseAgent):
    """
    Agent specialized in mathematical analysis of experimental data.
    
    Capabilities:
    - Statistical analysis (mean, std, correlation, regression)
    - Curve fitting (linear, polynomial, exponential, power law)
    - Pattern recognition and relationship discovery
    - Mathematical model validation
    - Physics law discovery from data
    - Uncertainty quantification
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_api = KnowledgeAPI()
        self.logger = logging.getLogger(__name__)
        
        # Analysis thresholds
        self.correlation_threshold = 0.7  # Minimum correlation for significance
        self.r2_threshold = 0.8  # Minimum R² for good fit
        self.p_value_threshold = 0.05  # Statistical significance threshold
        
        # Common physics relationships to detect
        self.physics_models = {
            'linear': lambda x, a, b: a * x + b,
            'quadratic': lambda x, a, b, c: a * x**2 + b * x + c,
            'power_law': lambda x, a, b: a * x**b,
            'exponential': lambda x, a, b: a * np.exp(b * x),
            'inverse': lambda x, a, b: a / x + b,
            'logarithmic': lambda x, a, b: a * np.log(x) + b
        }
        
        # Physics law templates
        self.physics_laws = {
            'newtons_second_law': {
                'equation': 'F = m * a',
                'variables': ['force', 'mass', 'acceleration'],
                'relationship': 'F = m * a',
                'expected_pattern': 'linear_product'
            },
            'kinematic_equation': {
                'equation': 's = v₀*t + 0.5*a*t²',
                'variables': ['displacement', 'initial_velocity', 'time', 'acceleration'],
                'relationship': 's = v₀*t + 0.5*a*t²',
                'expected_pattern': 'quadratic_time'
            },
            'conservation_energy': {
                'equation': 'KE + PE = constant',
                'variables': ['kinetic_energy', 'potential_energy'],
                'relationship': 'KE + PE = constant',
                'expected_pattern': 'conservation'
            }
        }
        
    def _get_default_system_message(self) -> str:
        return """You are a Mathematical Analysis Agent specialized in discovering patterns and relationships in experimental physics data.
        
        Your responsibilities:
        - Perform statistical analysis on experimental datasets
        - Fit mathematical models to data (linear, polynomial, exponential, etc.)
        - Discover correlations and relationships between variables
        - Identify physics laws from data patterns
        - Validate mathematical models and quantify uncertainty
        - Generate mathematical expressions for discovered relationships
        
        Always maintain statistical rigor and provide confidence intervals for your analyses."""
    
    async def analyze_experimental_data(self, data_id: int, 
                                      analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive mathematical analysis on experimental data.
        
        Args:
            data_id: ID of experimental data to analyze
            analysis_type: Type of analysis (comprehensive, correlation, curve_fitting, physics_discovery)
            
        Returns:
            Dict containing analysis results and discovered patterns
        """
        try:
            # Retrieve experimental data
            data_records = await self.knowledge_api.get_experimental_data()
            data_record = next((d for d in data_records if d['id'] == data_id), None)
            
            if not data_record:
                return {
                    "success": False,
                    "error": f"Data record {data_id} not found"
                }
            
            # Parse the data
            data = json.loads(data_record['data_json'])
            experiment_type = data_record['experiment_type']
            
            # Perform analysis based on type
            analysis_results = {
                "data_id": data_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "experiment_type": experiment_type,
                "analysis_type": analysis_type,
                "results": {}
            }
            
            if analysis_type in ["comprehensive", "statistics"]:
                stats_results = await self._perform_statistical_analysis(data)
                analysis_results["results"]["statistical_analysis"] = stats_results
            
            if analysis_type in ["comprehensive", "correlation"]:
                corr_results = await self._perform_correlation_analysis(data)
                analysis_results["results"]["correlation_analysis"] = corr_results
            
            if analysis_type in ["comprehensive", "curve_fitting"]:
                curve_results = await self._perform_curve_fitting(data)
                analysis_results["results"]["curve_fitting"] = curve_results
            
            if analysis_type in ["comprehensive", "physics_discovery"]:
                physics_results = await self._discover_physics_laws(data, experiment_type)
                analysis_results["results"]["physics_discovery"] = physics_results
            
            # Store analysis results
            await self._store_analysis_results(data_id, analysis_results)
            
            # Log analysis event
            await self.knowledge_api.log_event(
                source="mathematical_analysis_agent",
                event_type="data_analyzed",
                payload={
                    "data_id": data_id,
                    "analysis_type": analysis_type,
                    "patterns_found": len(analysis_results["results"]),
                    "significant_correlations": len([r for r in analysis_results["results"].get("correlation_analysis", {}).get("significant_correlations", [])])
                }
            )
            
            return {
                "success": True,
                "analysis_results": analysis_results,
                "message": f"Mathematical analysis completed for data {data_id}"
            }
            
        except Exception as e:
            self.logger.error(f"Mathematical analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _perform_statistical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        data_points = data.get('data_points', [])
        columns = data.get('columns', [])
        
        if not data_points or not columns:
            return {"error": "No data points or columns found"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data_points)
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats_results = {
            "summary_statistics": {},
            "distributions": {},
            "normality_tests": {},
            "outlier_analysis": {}
        }
        
        # Summary statistics for each numeric column
        for col in numeric_columns:
            values = df[col].dropna()
            if len(values) > 0:
                stats_results["summary_statistics"][col] = {
                    "count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "skewness": float(stats.skew(values)),
                    "kurtosis": float(stats.kurtosis(values))
                }
                
                # Normality test
                if len(values) >= 3:
                    try:
                        stat, p_value = stats.shapiro(values)
                        stats_results["normality_tests"][col] = {
                            "statistic": float(stat),
                            "p_value": float(p_value),
                            "is_normal": bool(p_value > self.p_value_threshold)
                        }
                    except:
                        stats_results["normality_tests"][col] = {"test_failed": bool(True)}
                
                # Outlier detection using IQR method
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                stats_results["outlier_analysis"][col] = {
                    "outlier_count": int(len(outliers)),
                    "outlier_percentage": float(len(outliers) / len(values) * 100),
                    "outlier_values": outliers.tolist() if len(outliers) < 10 else outliers[:10].tolist()
                }
        
        return stats_results
    
    async def _perform_correlation_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform correlation analysis between variables."""
        data_points = data.get('data_points', [])
        
        if not data_points:
            return {"error": "No data points found"}
        
        # Convert to DataFrame
        df = pd.DataFrame(data_points)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
        
        # Calculate correlation matrix
        correlation_matrix = df[numeric_columns].corr()
        
        # Find significant correlations
        significant_correlations = []
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) >= self.correlation_threshold:
                        # Calculate p-value
                        try:
                            _, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                            significant_correlations.append({
                                "variable1": col1,
                                "variable2": col2,
                                "correlation": float(corr_value),
                                "p_value": float(p_value),
                                "significance": "significant" if p_value < self.p_value_threshold else "not_significant",
                                "strength": self._interpret_correlation_strength(abs(corr_value))
                            })
                        except:
                            pass
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "significant_correlations": significant_correlations,
            "correlation_threshold": self.correlation_threshold
        }
    
    async def _perform_curve_fitting(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform curve fitting with multiple models."""
        data_points = data.get('data_points', [])
        
        if not data_points:
            return {"error": "No data points found"}
        
        df = pd.DataFrame(data_points)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            return {"error": "Need at least 2 numeric columns for curve fitting"}
        
        fitting_results = {}
        
        # Try all pairs of variables
        for i, x_col in enumerate(numeric_columns):
            for j, y_col in enumerate(numeric_columns):
                if i != j:  # Don't fit variable to itself
                    pair_key = f"{y_col}_vs_{x_col}"
                    
                    # Clean data (remove NaN values)
                    clean_data = df[[x_col, y_col]].dropna()
                    if len(clean_data) < 3:
                        continue
                    
                    x_data = clean_data[x_col].values
                    y_data = clean_data[y_col].values
                    
                    # Try different models
                    model_results = {}
                    
                    for model_name, model_func in self.physics_models.items():
                        try:
                            fit_result = self._fit_model(x_data, y_data, model_func, model_name)
                            if fit_result:
                                model_results[model_name] = fit_result
                        except Exception as e:
                            self.logger.debug(f"Failed to fit {model_name} for {pair_key}: {e}")
                    
                    if model_results:
                        # Find best fitting model
                        best_model = max(model_results.items(), key=lambda x: x[1]['r2_score'])
                        fitting_results[pair_key] = {
                            "x_variable": x_col,
                            "y_variable": y_col,
                            "best_model": best_model[0],
                            "best_fit": best_model[1],
                            "all_models": model_results
                        }
        
        return fitting_results
    
    def _fit_model(self, x_data: np.ndarray, y_data: np.ndarray, 
                   model_func: Callable, model_name: str) -> Optional[Dict[str, Any]]:
        """Fit a specific model to data."""
        try:
            # Handle special cases for certain models
            if model_name == 'power_law' and np.any(x_data <= 0):
                return None
            if model_name == 'logarithmic' and np.any(x_data <= 0):
                return None
            
            # Initial parameter guess
            if model_name == 'linear':
                p0 = [1, 0]
            elif model_name == 'quadratic':
                p0 = [1, 1, 0]
            elif model_name == 'exponential':
                p0 = [1, 0.1]
            elif model_name == 'power_law':
                p0 = [1, 1]
            elif model_name == 'inverse':
                p0 = [1, 0]
            elif model_name == 'logarithmic':
                p0 = [1, 0]
            else:
                p0 = None
            
            # Fit the model
            if p0:
                popt, pcov = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=1000)
            else:
                popt, pcov = curve_fit(model_func, x_data, y_data, maxfev=1000)
            
            # Calculate goodness of fit
            y_pred = model_func(x_data, *popt)
            r2 = r2_score(y_data, y_pred)
            mse = mean_squared_error(y_data, y_pred)
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            return {
                "parameters": popt.tolist(),
                "parameter_errors": param_errors.tolist(),
                "r2_score": float(r2),
                "mse": float(mse),
                "model_name": model_name,
                "equation": self._get_equation_string(model_name, popt),
                "quality": "excellent" if r2 > 0.95 else "good" if r2 > 0.8 else "fair" if r2 > 0.6 else "poor"
            }
            
        except Exception as e:
            self.logger.debug(f"Model fitting failed for {model_name}: {e}")
            return None
    
    def _get_equation_string(self, model_name: str, params: np.ndarray) -> str:
        """Generate equation string for the fitted model."""
        if model_name == 'linear':
            return f"y = {params[0]:.3f}*x + {params[1]:.3f}"
        elif model_name == 'quadratic':
            return f"y = {params[0]:.3f}*x² + {params[1]:.3f}*x + {params[2]:.3f}"
        elif model_name == 'power_law':
            return f"y = {params[0]:.3f}*x^{params[1]:.3f}"
        elif model_name == 'exponential':
            return f"y = {params[0]:.3f}*exp({params[1]:.3f}*x)"
        elif model_name == 'inverse':
            return f"y = {params[0]:.3f}/x + {params[1]:.3f}"
        elif model_name == 'logarithmic':
            return f"y = {params[0]:.3f}*ln(x) + {params[1]:.3f}"
        else:
            return f"y = f(x) with parameters {params}"
    
    async def _discover_physics_laws(self, data: Dict[str, Any], experiment_type: str) -> Dict[str, Any]:
        """Discover physics laws from data patterns."""
        data_points = data.get('data_points', [])
        columns = data.get('columns', [])
        
        if not data_points:
            return {"error": "No data points found"}
        
        df = pd.DataFrame(data_points)
        discovery_results = {
            "discovered_laws": [],
            "potential_relationships": [],
            "experiment_type": experiment_type
        }
        
        # Check for specific physics laws based on experiment type
        if experiment_type == "motion":
            # Look for Newton's second law: F = ma
            newton_result = await self._check_newtons_second_law(df, columns)
            if newton_result:
                discovery_results["discovered_laws"].append(newton_result)
            
            # Look for kinematic relationships
            kinematic_result = await self._check_kinematic_relationships(df, columns)
            if kinematic_result:
                discovery_results["potential_relationships"].extend(kinematic_result)
        
        # General relationship discovery
        general_relationships = await self._discover_general_relationships(df)
        discovery_results["potential_relationships"].extend(general_relationships)
        
        return discovery_results
    
    async def _check_newtons_second_law(self, df: pd.DataFrame, columns: List[str]) -> Optional[Dict[str, Any]]:
        """Check if data follows Newton's second law F = ma."""
        # Look for force, mass, and acceleration columns
        force_cols = [col for col in columns if 'force' in col.lower()]
        mass_cols = [col for col in columns if 'mass' in col.lower()]
        accel_cols = [col for col in columns if 'accel' in col.lower()]
        
        if not (force_cols and mass_cols and accel_cols):
            return None
        
        # Use the first matching column of each type
        force_col = force_cols[0]
        mass_col = mass_cols[0]
        accel_col = accel_cols[0]
        
        # Clean data
        clean_data = df[[force_col, mass_col, accel_col]].dropna()
        if len(clean_data) < 3:
            return None
        
        # Calculate predicted force: F_predicted = m * a
        predicted_force = clean_data[mass_col] * clean_data[accel_col]
        actual_force = clean_data[force_col]
        
        # Calculate correlation and R²
        correlation = np.corrcoef(predicted_force, actual_force)[0, 1]
        r2 = r2_score(actual_force, predicted_force)
        
        # Check if relationship holds
        if correlation > 0.9 and r2 > 0.8:
            return {
                "law_name": "Newton's Second Law",
                "equation": "F = m × a",
                "variables": {
                    "force": force_col,
                    "mass": mass_col,
                    "acceleration": accel_col
                },
                "correlation": float(correlation),
                "r2_score": float(r2),
                "confidence": "high" if r2 > 0.95 else "medium",
                "evidence": f"Strong correlation ({correlation:.3f}) between F and m×a"
            }
        
        return None
    
    async def _check_kinematic_relationships(self, df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
        """Check for kinematic relationships in motion data."""
        relationships = []
        
        # Look for time, velocity, acceleration, displacement columns
        time_cols = [col for col in columns if 'time' in col.lower() or 't' == col.lower()]
        velocity_cols = [col for col in columns if 'velocity' in col.lower() or 'speed' in col.lower()]
        accel_cols = [col for col in columns if 'accel' in col.lower()]
        disp_cols = [col for col in columns if 'displacement' in col.lower() or 'distance' in col.lower()]
        
        # Check v = v₀ + at (if we have initial velocity)
        if time_cols and velocity_cols and accel_cols:
            # This would require more complex analysis with initial conditions
            pass
        
        # Check s = ½at² (for motion from rest)
        if time_cols and disp_cols and accel_cols:
            time_col = time_cols[0]
            disp_col = disp_cols[0]
            accel_col = accel_cols[0]
            
            clean_data = df[[time_col, disp_col, accel_col]].dropna()
            if len(clean_data) > 3:
                # Fit s = ½at² model
                try:
                    def kinematic_model(t, a):
                        return 0.5 * a * t**2
                    
                    popt, _ = curve_fit(kinematic_model, clean_data[time_col], clean_data[disp_col])
                    predicted_disp = kinematic_model(clean_data[time_col], popt[0])
                    r2 = r2_score(clean_data[disp_col], predicted_disp)
                    
                    if r2 > 0.8:
                        relationships.append({
                            "relationship_name": "Kinematic Equation",
                            "equation": "s = ½at²",
                            "variables": {
                                "displacement": disp_col,
                                "time": time_col,
                                "acceleration": accel_col
                            },
                            "fitted_acceleration": float(popt[0]),
                            "r2_score": float(r2),
                            "confidence": "high" if r2 > 0.95 else "medium"
                        })
                except:
                    pass
        
        return relationships
    
    async def _discover_general_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Discover general mathematical relationships in the data."""
        relationships = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Look for proportional relationships
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i < j:
                    clean_data = df[[col1, col2]].dropna()
                    if len(clean_data) > 3:
                        # Check for linear relationship through origin (proportionality)
                        x_data = clean_data[col1].values
                        y_data = clean_data[col2].values
                        
                        # Fit y = kx (proportional relationship)
                        try:
                            k = np.sum(x_data * y_data) / np.sum(x_data**2)
                            y_pred = k * x_data
                            r2 = r2_score(y_data, y_pred)
                            
                            if r2 > 0.8:
                                relationships.append({
                                    "relationship_type": "proportional",
                                    "equation": f"{col2} = {k:.3f} × {col1}",
                                    "variables": [col1, col2],
                                    "proportionality_constant": float(k),
                                    "r2_score": float(r2),
                                    "confidence": "high" if r2 > 0.95 else "medium"
                                })
                        except:
                            pass
        
        return relationships
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength."""
        if correlation >= 0.9:
            return "very_strong"
        elif correlation >= 0.7:
            return "strong"
        elif correlation >= 0.5:
            return "moderate"
        elif correlation >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    async def _store_analysis_results(self, data_id: int, analysis_results: Dict[str, Any]):
        """Store mathematical analysis results."""
        # Store as a knowledge entry
        await self.knowledge_api.store_knowledge(
            title=f"Mathematical Analysis - Data {data_id}",
            content=json.dumps(analysis_results),
            domain="mathematical_analysis",
            confidence_score=0.9,
            source_type="analysis_agent",
            metadata={
                "data_id": data_id,
                "analysis_type": analysis_results.get("analysis_type", "comprehensive"),
                "timestamp": analysis_results.get("analysis_timestamp")
            }
        )
    
    def _build_graph(self):
        """Build the agent's processing graph."""
        from langgraph.graph import StateGraph, MessagesState
        
        workflow = StateGraph(MessagesState)
        workflow.add_node("process", self._process_message)
        workflow.set_entry_point("process")
        workflow.set_finish_point("process")
        
        return workflow.compile()
    
    async def _process_message(self, state):
        """Process incoming messages."""
        return {"messages": [{"role": "assistant", "content": "Mathematical analysis agent ready"}]}
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this mathematical analysis agent."""
        return {
            "name": "MathematicalAnalysisAgent",
            "type": "mathematical_analysis",
            "description": "A specialized agent for mathematical analysis and pattern recognition in experimental physics data",
            "capabilities": [
                "Statistical analysis (descriptive and inferential)",
                "Correlation analysis and significance testing",
                "Curve fitting with multiple models",
                "Pattern recognition and relationship discovery",
                "Physics law discovery from data",
                "Mathematical model validation"
            ],
            "supported_models": list(self.physics_models.keys()),
            "physics_laws": list(self.physics_laws.keys()),
            "thresholds": {
                "correlation_threshold": self.correlation_threshold,
                "r2_threshold": self.r2_threshold,
                "p_value_threshold": self.p_value_threshold
            },
            "tools": [
                "scipy statistical functions",
                "scikit-learn regression models",
                "numpy mathematical operations",
                "curve fitting algorithms",
                "sympy symbolic mathematics"
            ],
            "version": "1.0.0"
        } 