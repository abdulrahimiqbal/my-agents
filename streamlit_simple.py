"""
PhysicsGPT - Simplified Streamlit Interface
A streamlined physics expert chat interface that works reliably in cloud environments.
"""

import streamlit as st
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

# Core dependencies
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError as e:
    st.error(f"Missing required dependencies: {e}")
    st.error("Please install: pip install langchain-openai pydantic-settings")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PhysicsGPT - AI Physics Expert",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Settings class
class Settings(BaseSettings):
    """Simplified settings for PhysicsGPT."""
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    default_model: str = Field(default="gpt-4o-mini", env="PHYSICS_AGENT_MODEL")
    default_temperature: float = Field(default=0.1, env="PHYSICS_AGENT_TEMPERATURE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Physics constants database
PHYSICS_CONSTANTS = {
    "speed_of_light": {"value": 299792458, "unit": "m/s", "description": "Speed of light in vacuum"},
    "planck_constant": {"value": 6.62607015e-34, "unit": "J⋅s", "description": "Planck constant"},
    "elementary_charge": {"value": 1.602176634e-19, "unit": "C", "description": "Elementary charge"},
    "electron_mass": {"value": 9.1093837015e-31, "unit": "kg", "description": "Electron rest mass"},
    "proton_mass": {"value": 1.67262192369e-27, "unit": "kg", "description": "Proton rest mass"},
    "avogadro_number": {"value": 6.02214076e23, "unit": "mol⁻¹", "description": "Avogadro constant"},
    "boltzmann_constant": {"value": 1.380649e-23, "unit": "J/K", "description": "Boltzmann constant"},
    "gas_constant": {"value": 8.314462618, "unit": "J/(mol⋅K)", "description": "Universal gas constant"},
    "gravitational_constant": {"value": 6.67430e-11, "unit": "m³/(kg⋅s²)", "description": "Gravitational constant"},
    "vacuum_permeability": {"value": 1.25663706212e-6, "unit": "H/m", "description": "Vacuum permeability"},
    "vacuum_permittivity": {"value": 8.8541878128e-12, "unit": "F/m", "description": "Vacuum permittivity"},
    "fine_structure_constant": {"value": 7.2973525693e-3, "unit": "dimensionless", "description": "Fine-structure constant"},
}

# Unit conversion factors (to SI base units)
UNIT_CONVERSIONS = {
    "length": {
        "m": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "μm": 1e-6, "nm": 1e-9,
        "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.34,
        "au": 1.496e11, "ly": 9.461e15, "pc": 3.086e16
    },
    "mass": {
        "kg": 1, "g": 0.001, "mg": 1e-6, "μg": 1e-9,
        "lb": 0.453592, "oz": 0.0283495, "ton": 1000,
        "u": 1.66054e-27, "me": 9.109e-31, "mp": 1.673e-27
    },
    "time": {
        "s": 1, "ms": 0.001, "μs": 1e-6, "ns": 1e-9,
        "min": 60, "h": 3600, "day": 86400, "year": 3.156e7
    },
    "energy": {
        "J": 1, "kJ": 1000, "MJ": 1e6, "GJ": 1e9,
        "eV": 1.602e-19, "keV": 1.602e-16, "MeV": 1.602e-13, "GeV": 1.602e-10,
        "cal": 4.184, "kcal": 4184, "kWh": 3.6e6
    },
    "temperature": {
        "K": lambda x: x, "C": lambda x: x + 273.15, "F": lambda x: (x + 459.67) * 5/9
    }
}

# CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .physics-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def create_physics_system_message(difficulty_level: str, specialty: Optional[str] = None) -> str:
    """Create physics expert system message."""
    base_message = """You are PhysicsGPT, a world-class physics expert and educator. You help users understand physics concepts, solve problems, and explore physics.

## Your Expertise:
- Classical Mechanics, Electromagnetism, Thermodynamics
- Quantum Mechanics, Relativity, Optics
- Particle Physics, Cosmology, Condensed Matter

## Your Approach:
1. Break down complex problems step-by-step
2. Explain physics intuition behind equations
3. Use proper units and dimensional analysis
4. Connect physics to real-world phenomena
5. Provide multiple solution approaches when possible

## Tools Available:
- Physics constants database
- Unit conversions
- Mathematical calculations
- Step-by-step problem solving"""

    level_customization = {
        "high_school": "\n\nFocus: High school level - emphasize concepts and simple math",
        "undergraduate": "\n\nFocus: Undergraduate level - balance concepts with mathematical rigor", 
        "graduate": "\n\nFocus: Graduate level - use advanced formalism and theory",
        "research": "\n\nFocus: Research level - discuss cutting-edge developments"
    }
    
    base_message += level_customization.get(difficulty_level, "")
    
    if specialty:
        base_message += f"\n\nSpecialty Focus: {specialty.title()}"
    
    return base_message

def get_physics_constant(name: str) -> Optional[Dict]:
    """Get a physics constant by name."""
    return PHYSICS_CONSTANTS.get(name.lower().replace(" ", "_"))

def convert_units(value: float, from_unit: str, to_unit: str, quantity: str) -> Optional[float]:
    """Convert between units of the same physical quantity."""
    if quantity not in UNIT_CONVERSIONS:
        return None
    
    conversions = UNIT_CONVERSIONS[quantity]
    
    if from_unit not in conversions or to_unit not in conversions:
        return None
    
    # Special handling for temperature
    if quantity == "temperature":
        # Convert to Kelvin first, then to target
        kelvin_value = conversions[from_unit](value)
        if to_unit == "K":
            return kelvin_value
        elif to_unit == "C":
            return kelvin_value - 273.15
        elif to_unit == "F":
            return kelvin_value * 9/5 - 459.67
    else:
        # Standard conversion through SI base unit
        si_value = value * conversions[from_unit]
        return si_value / conversions[to_unit]

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'settings' not in st.session_state:
        st.session_state.settings = Settings()

def create_header():
    """Create the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>⚛️ PhysicsGPT</h1>
        <p style="font-size: 1.2em; margin: 0;">Your AI Physics Expert & Problem Solver</p>
        <p style="opacity: 0.9; margin: 0.5rem 0 0 0;">Simplified Cloud-Ready Version</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the sidebar with settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Physics Agent Settings")
        
        difficulty_level = st.selectbox(
            "🎓 Difficulty Level",
            ["high_school", "undergraduate", "graduate", "research"],
            index=1
        )
        
        specialty = st.selectbox(
            "🔬 Physics Specialty",
            [None, "mechanics", "electromagnetism", "quantum", "thermodynamics", 
             "relativity", "optics", "particle_physics", "cosmology"],
        )
        
        st.session_state.difficulty_level = difficulty_level
        st.session_state.specialty = specialty
        
        # Physics Tools
        st.markdown("## 🧮 Physics Tools")
        
        # Constants lookup
        with st.expander("📚 Physics Constants"):
            const_name = st.selectbox("Select constant:", list(PHYSICS_CONSTANTS.keys()))
            if const_name:
                const = PHYSICS_CONSTANTS[const_name]
                st.write(f"**{const_name.replace('_', ' ').title()}**")
                st.write(f"Value: {const['value']}")
                st.write(f"Unit: {const['unit']}")
                st.write(f"Description: {const['description']}")
        
        # Unit converter
        with st.expander("🔄 Unit Converter"):
            quantity = st.selectbox("Quantity:", list(UNIT_CONVERSIONS.keys()))
            if quantity and quantity != "temperature":
                units = list(UNIT_CONVERSIONS[quantity].keys())
                col1, col2 = st.columns(2)
                with col1:
                    from_unit = st.selectbox("From:", units)
                    value = st.number_input("Value:", value=1.0)
                with col2:
                    to_unit = st.selectbox("To:", units)
                
                if st.button("Convert"):
                    result = convert_units(value, from_unit, to_unit, quantity)
                    if result:
                        st.success(f"{value} {from_unit} = {result:.6g} {to_unit}")

def main():
    """Main application function."""
    initialize_session_state()
    create_header()
    create_sidebar()
    
    # Check for API key
    if not st.session_state.settings.openai_api_key:
        st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment or secrets.")
        st.stop()
    
    # Initialize LLM
    try:
        llm = ChatOpenAI(
            model=st.session_state.settings.default_model,
            temperature=st.session_state.settings.default_temperature,
            api_key=st.session_state.settings.openai_api_key
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.stop()
    
    # Main chat interface
    st.markdown("## 💬 Physics Chat")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about physics..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking about physics..."):
                try:
                    # Create system message
                    system_msg = SystemMessage(content=create_physics_system_message(
                        st.session_state.difficulty_level,
                        st.session_state.specialty
                    ))
                    
                    # Get recent conversation context
                    recent_messages = st.session_state.chat_history[-6:]  # Last 6 messages
                    messages = [system_msg]
                    
                    for msg in recent_messages:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        # Note: We'd add AI messages here if we had them in the right format
                    
                    # Generate response
                    response = llm.invoke(messages)
                    response_content = response.content
                    
                    st.write(response_content)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response_content
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    
    # Example problems
    st.markdown("## 🎯 Example Physics Problems")
    
    examples = [
        "What is the speed of light in vacuum?",
        "Convert 100 km/h to m/s",
        "Explain Newton's second law",
        "Calculate the energy of a photon with wavelength 500 nm",
        "What is the uncertainty principle?",
        "Derive the kinetic energy formula"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": example})
                st.rerun()

if __name__ == "__main__":
    main() 