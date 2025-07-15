"""Physics research tools for scientific literature and information."""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import requests
import json
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlencode
import re


@tool
def search_arxiv(query: str, max_results: int = 5, category: str = "physics") -> str:
    """Search ArXiv for physics papers and preprints.
    
    Args:
        query: Search query (keywords, author, title)
        max_results: Maximum number of results to return (1-20)
        category: ArXiv category filter (physics, math, cs, etc.)
        
    Returns:
        Formatted search results with abstracts
    """
    try:
        # Construct ArXiv API query
        base_url = "http://export.arxiv.org/api/query?"
        
        # Add category filter if specified
        if category and category.lower() != "all":
            if category.lower() == "physics":
                # Include all physics categories
                cat_filter = "cat:physics.* OR cat:quant-ph OR cat:gr-qc OR cat:hep-* OR cat:nucl-* OR cat:astro-ph OR cat:cond-mat.*"
            else:
                cat_filter = f"cat:{category}*"
            
            search_query = f"({query}) AND ({cat_filter})"
        else:
            search_query = query
        
        # Limit results
        max_results = min(max(1, max_results), 20)
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        url = base_url + urlencode(params)
        
        # Make request to ArXiv API
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Extract namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}
        
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            return f"No papers found for query: {query}"
        
        results = f"**ArXiv Search Results for: {query}**\n\n"
        
        for i, entry in enumerate(entries, 1):
            # Extract paper information
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text[:10]  # Just the date
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            
            # Get categories
            categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]
            primary_category = entry.find('arxiv:primary_category', ns)
            if primary_category is not None:
                primary_cat = primary_category.get('term')
            else:
                primary_cat = categories[0] if categories else "Unknown"
            
            # Format result
            results += f"**{i}. {title}**\n"
            results += f"Authors: {', '.join(authors[:3])}"
            if len(authors) > 3:
                results += f" (+{len(authors)-3} more)"
            results += f"\nArXiv ID: {arxiv_id}\n"
            results += f"Category: {primary_cat}\n"
            results += f"Published: {published}\n"
            results += f"Abstract: {summary[:300]}{'...' if len(summary) > 300 else ''}\n"
            results += f"URL: https://arxiv.org/abs/{arxiv_id}\n\n"
        
        return results
    
    except requests.RequestException as e:
        return f"Error accessing ArXiv: {str(e)}"
    except ET.ParseError as e:
        return f"Error parsing ArXiv response: {str(e)}"
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"


@tool
def search_physics_wikipedia(topic: str, sections: str = "summary") -> str:
    """Search Wikipedia for physics-related information.
    
    Args:
        topic: Physics topic to search for
        sections: Which sections to return (summary, all, or specific section name)
        
    Returns:
        Wikipedia content formatted for physics context
    """
    try:
        # Wikipedia API endpoint
        base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        
        # Clean and encode the topic
        topic_encoded = quote_plus(topic.strip())
        url = base_url + topic_encoded
        
        # Make request
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            # Try a search if direct page not found
            return _search_wikipedia_pages(topic)
        
        response.raise_for_status()
        data = response.json()
        
        # Extract information
        title = data.get('title', topic)
        extract = data.get('extract', 'No summary available')
        page_url = data.get('content_urls', {}).get('desktop', {}).get('page', '')
        
        result = f"**Wikipedia: {title}**\n\n"
        result += f"{extract}\n\n"
        
        if page_url:
            result += f"**Full Article:** {page_url}\n\n"
        
        # Add physics context if available
        physics_context = _get_wikipedia_physics_context(title, extract)
        if physics_context:
            result += f"**Physics Context:**\n{physics_context}\n\n"
        
        return result
    
    except requests.RequestException as e:
        return f"Error accessing Wikipedia: {str(e)}"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


def _search_wikipedia_pages(query: str) -> str:
    """Search for Wikipedia pages when direct lookup fails."""
    try:
        search_url = "https://en.wikipedia.org/api/rest_v1/page/search/"
        params = {'q': query, 'limit': 3}
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get('pages', [])
        if not pages:
            return f"No Wikipedia pages found for: {query}"
        
        result = f"**Wikipedia Search Results for: {query}**\n\n"
        
        for i, page in enumerate(pages, 1):
            title = page.get('title', 'Unknown')
            description = page.get('description', 'No description')
            
            result += f"**{i}. {title}**\n"
            result += f"Description: {description}\n"
            result += f"URL: https://en.wikipedia.org/wiki/{quote_plus(title)}\n\n"
        
        return result
    
    except Exception as e:
        return f"Error in Wikipedia search: {str(e)}"


def _get_wikipedia_physics_context(title: str, content: str) -> str:
    """Add physics-specific context to Wikipedia content."""
    physics_keywords = {
        'quantum': 'Quantum mechanics - fundamental theory describing nature at atomic scales',
        'relativity': 'Theory of relativity - Einstein\'s theories of space, time, and gravity',
        'thermodynamics': 'Study of heat, work, temperature, and energy transfer',
        'electromagnetism': 'Study of electric and magnetic phenomena and their interactions',
        'mechanics': 'Study of motion, forces, and energy in physical systems',
        'optics': 'Study of light and its properties, behavior, and interactions',
        'particle': 'Particle physics - study of fundamental constituents of matter',
        'nuclear': 'Nuclear physics - study of atomic nuclei and nuclear reactions',
        'cosmology': 'Study of the origin, evolution, and structure of the universe',
        'condensed matter': 'Study of physical properties of condensed phases of matter'
    }
    
    found_contexts = []
    title_lower = title.lower()
    content_lower = content.lower()
    
    for keyword, description in physics_keywords.items():
        if keyword in title_lower or keyword in content_lower:
            found_contexts.append(description)
    
    return '\n'.join(f"• {context}" for context in found_contexts[:3])


@tool
def lookup_physics_equation(equation_name: str) -> str:
    """Look up famous physics equations and their meanings.
    
    Args:
        equation_name: Name or description of the physics equation
        
    Returns:
        Equation details, meaning, and applications
    """
    try:
        # Database of famous physics equations
        equations_db = {
            # Classical Mechanics
            "newton's second law": {
                "equation": "F = ma",
                "description": "Newton's Second Law of Motion",
                "meaning": "Force equals mass times acceleration",
                "variables": "F = force (N), m = mass (kg), a = acceleration (m/s²)",
                "applications": "Fundamental law for analyzing motion and forces in classical mechanics",
                "field": "Classical Mechanics"
            },
            "kinetic energy": {
                "equation": "KE = ½mv²",
                "description": "Kinetic Energy",
                "meaning": "Energy of motion",
                "variables": "KE = kinetic energy (J), m = mass (kg), v = velocity (m/s)",
                "applications": "Energy calculations, collision analysis, mechanical systems",
                "field": "Classical Mechanics"
            },
            "potential energy": {
                "equation": "PE = mgh",
                "description": "Gravitational Potential Energy",
                "meaning": "Energy stored due to position in gravitational field",
                "variables": "PE = potential energy (J), m = mass (kg), g = gravity (m/s²), h = height (m)",
                "applications": "Energy conservation, pendulums, projectile motion",
                "field": "Classical Mechanics"
            },
            
            # Electromagnetism
            "coulomb's law": {
                "equation": "F = k(q₁q₂)/r²",
                "description": "Coulomb's Law",
                "meaning": "Force between electric charges",
                "variables": "F = force (N), k = Coulomb constant, q₁,q₂ = charges (C), r = distance (m)",
                "applications": "Electrostatics, atomic structure, electrical interactions",
                "field": "Electromagnetism"
            },
            "ohm's law": {
                "equation": "V = IR",
                "description": "Ohm's Law",
                "meaning": "Relationship between voltage, current, and resistance",
                "variables": "V = voltage (V), I = current (A), R = resistance (Ω)",
                "applications": "Circuit analysis, electrical engineering, power calculations",
                "field": "Electromagnetism"
            },
            
            # Quantum Mechanics
            "schrodinger equation": {
                "equation": "iℏ ∂ψ/∂t = Ĥψ",
                "description": "Time-dependent Schrödinger Equation",
                "meaning": "Fundamental equation of quantum mechanics",
                "variables": "ψ = wave function, ℏ = reduced Planck constant, Ĥ = Hamiltonian operator",
                "applications": "Quantum systems, atomic physics, molecular physics",
                "field": "Quantum Mechanics"
            },
            "planck's equation": {
                "equation": "E = hf",
                "description": "Planck's Energy-Frequency Relation",
                "meaning": "Energy of a photon",
                "variables": "E = energy (J), h = Planck constant (J⋅s), f = frequency (Hz)",
                "applications": "Quantum mechanics, photoelectric effect, blackbody radiation",
                "field": "Quantum Mechanics"
            },
            "de broglie wavelength": {
                "equation": "λ = h/p",
                "description": "de Broglie Wavelength",
                "meaning": "Wave-like properties of matter",
                "variables": "λ = wavelength (m), h = Planck constant, p = momentum (kg⋅m/s)",
                "applications": "Wave-particle duality, electron microscopy, quantum mechanics",
                "field": "Quantum Mechanics"
            },
            
            # Relativity
            "mass energy equivalence": {
                "equation": "E = mc²",
                "description": "Mass-Energy Equivalence",
                "meaning": "Mass and energy are interchangeable",
                "variables": "E = energy (J), m = mass (kg), c = speed of light (m/s)",
                "applications": "Nuclear reactions, particle physics, cosmology",
                "field": "Relativity"
            },
            "lorentz factor": {
                "equation": "γ = 1/√(1 - v²/c²)",
                "description": "Lorentz Factor",
                "meaning": "Time dilation and length contraction factor",
                "variables": "γ = Lorentz factor, v = velocity (m/s), c = speed of light (m/s)",
                "applications": "Special relativity, particle accelerators, GPS corrections",
                "field": "Relativity"
            },
            
            # Thermodynamics
            "ideal gas law": {
                "equation": "PV = nRT",
                "description": "Ideal Gas Law",
                "meaning": "Relationship between pressure, volume, and temperature for ideal gases",
                "variables": "P = pressure (Pa), V = volume (m³), n = amount (mol), R = gas constant, T = temperature (K)",
                "applications": "Gas behavior, thermodynamic cycles, atmospheric physics",
                "field": "Thermodynamics"
            },
            "stefan boltzmann law": {
                "equation": "j = σT⁴",
                "description": "Stefan-Boltzmann Law",
                "meaning": "Power radiated by a blackbody",
                "variables": "j = energy flux (W/m²), σ = Stefan-Boltzmann constant, T = temperature (K)",
                "applications": "Thermal radiation, stellar physics, climate science",
                "field": "Thermodynamics"
            },
            
            # Wave Physics
            "wave equation": {
                "equation": "v = fλ",
                "description": "Wave Speed Equation",
                "meaning": "Relationship between wave speed, frequency, and wavelength",
                "variables": "v = wave speed (m/s), f = frequency (Hz), λ = wavelength (m)",
                "applications": "All wave phenomena, acoustics, optics, electromagnetic waves",
                "field": "Wave Physics"
            }
        }
        
        # Normalize the input
        equation_name_lower = equation_name.lower().strip()
        
        # Try exact match first
        if equation_name_lower in equations_db:
            eq_data = equations_db[equation_name_lower]
            
            result = f"**{eq_data['description']}**\n\n"
            result += f"**Equation:** {eq_data['equation']}\n\n"
            result += f"**Meaning:** {eq_data['meaning']}\n\n"
            result += f"**Variables:** {eq_data['variables']}\n\n"
            result += f"**Applications:** {eq_data['applications']}\n\n"
            result += f"**Field:** {eq_data['field']}\n\n"
            
            return result
        
        # Try partial matches
        matches = []
        for key, data in equations_db.items():
            if (equation_name_lower in key or 
                equation_name_lower in data['description'].lower() or
                any(word in key for word in equation_name_lower.split())):
                matches.append((key, data))
        
        if matches:
            if len(matches) == 1:
                key, eq_data = matches[0]
                result = f"**{eq_data['description']}**\n\n"
                result += f"**Equation:** {eq_data['equation']}\n\n"
                result += f"**Meaning:** {eq_data['meaning']}\n\n"
                result += f"**Variables:** {eq_data['variables']}\n\n"
                result += f"**Applications:** {eq_data['applications']}\n\n"
                result += f"**Field:** {eq_data['field']}\n\n"
                return result
            else:
                result = f"**Multiple equations found for '{equation_name}':**\n\n"
                for i, (key, data) in enumerate(matches[:5], 1):
                    result += f"**{i}. {data['description']}** - {data['equation']}\n"
                result += "\nPlease be more specific in your search."
                return result
        
        # If no matches found, suggest available equations
        available_equations = list(equations_db.keys())
        result = f"Equation '{equation_name}' not found.\n\n"
        result += "**Available equations:**\n"
        for eq in sorted(available_equations):
            result += f"• {equations_db[eq]['description']} - {equations_db[eq]['equation']}\n"
        
        return result
    
    except Exception as e:
        return f"Error looking up equation: {str(e)}"


@tool
def physics_concept_search(concept: str, depth: str = "overview") -> str:
    """Search for physics concepts with explanations at different levels.
    
    Args:
        concept: Physics concept to search for
        depth: Level of detail (overview, detailed, advanced)
        
    Returns:
        Concept explanation with appropriate depth
    """
    try:
        # Comprehensive physics concept database
        concepts_db = {
            "time travel": {
                "overview": """Time travel refers to the hypothetical ability to move between different points in time, analogous to moving between different points in space. In physics, this concept is primarily explored through Einstein's theories of relativity.

**Key Physics Principles:**
• **Special Relativity**: Time dilation occurs when traveling at high speeds relative to the speed of light
• **General Relativity**: Massive objects can curve spacetime, potentially creating closed timelike curves
• **Causality**: The principle that cause must precede effect, which time travel could violate

**Current Scientific Status:**
Time travel to the future is theoretically possible and has been experimentally verified through time dilation effects. However, travel to the past remains highly speculative and faces significant theoretical obstacles including paradoxes and energy requirements.

**Real-World Applications:**
• GPS satellites must account for time dilation effects
• Particle accelerators demonstrate time dilation at high speeds
• Gravitational time dilation affects clocks at different altitudes""",
                
                "detailed": """Time travel in physics is governed by Einstein's theories of relativity, which fundamentally changed our understanding of space and time as a unified spacetime continuum.

**Mathematical Framework:**
The spacetime interval: ds² = -c²dt² + dx² + dy² + dz²
For time travel to be possible, we need closed timelike curves (CTCs) where ds² < 0.

**Theoretical Mechanisms:**
1. **Wormholes (Einstein-Rosen Bridges)**: Solutions to Einstein's field equations that could connect distant regions of spacetime
2. **Rotating Black Holes (Kerr Metric)**: The ergosphere of a rotating black hole might allow closed timelike curves
3. **Cosmic Strings**: Hypothetical one-dimensional defects in spacetime that could create time loops
4. **Alcubierre Drive**: Theoretical faster-than-light travel by contracting space in front and expanding behind

**Physical Constraints:**
• **Energy Requirements**: Most time travel scenarios require exotic matter with negative energy density
• **Quantum Effects**: Quantum fluctuations may prevent the formation of closed timelike curves
• **Chronology Protection**: Hawking's conjecture that the laws of physics prevent time travel paradoxes

**Experimental Evidence:**
• Muon decay experiments confirm time dilation at high speeds
• Hafele-Keating experiment measured time dilation in aircraft
• GPS satellites demonstrate gravitational time dilation""",
                
                "advanced": """Time travel represents one of the most profound challenges to our understanding of causality and the structure of spacetime. Current research focuses on quantum gravity approaches and the resolution of temporal paradoxes.

**Advanced Theoretical Frameworks:**
1. **Closed Timelike Curves in General Relativity**: Solutions like the Gödel universe, van Stockum dust, and Tipler cylinders
2. **Quantum Mechanics of Time Travel**: Novikov self-consistency principle and quantum computation with closed timelike curves
3. **String Theory Implications**: Extra dimensions and their potential role in time travel scenarios
4. **Loop Quantum Gravity**: Discrete spacetime structure and its implications for temporal loops

**Current Research Frontiers:**
• **Quantum Information Theory**: How information behaves in the presence of closed timelike curves
• **Holographic Principle**: AdS/CFT correspondence and time travel in higher dimensions
• **Causal Set Theory**: Discrete approaches to spacetime and causality
• **Emergent Gravity**: Time travel in emergent spacetime theories

**Paradox Resolution Mechanisms:**
• **Novikov Self-Consistency**: Events arrange themselves to prevent paradoxes
• **Many-Worlds Interpretation**: Time travel creates parallel timelines
• **Quantum Decoherence**: Quantum effects prevent macroscopic time travel

**Open Questions:**
• Can quantum effects provide a chronology protection mechanism?
• What is the fundamental nature of time in quantum gravity?
• Are closed timelike curves physically realizable or merely mathematical curiosities?
• How do information-theoretic constraints limit time travel scenarios?"""
            },
            
            "quantum mechanics": {
                "overview": """Quantum mechanics is the fundamental theory describing the behavior of matter and energy at the atomic and subatomic scale. It reveals that particles exhibit both wave-like and particle-like properties.

**Key Principles:**
• **Wave-Particle Duality**: Particles can exhibit both wave and particle characteristics
• **Uncertainty Principle**: Cannot simultaneously know exact position and momentum
• **Superposition**: Particles can exist in multiple states simultaneously
• **Entanglement**: Particles can be correlated in ways that seem to defy classical physics

**Applications:**
• Lasers and LED technology
• Computer processors and memory
• Medical imaging (MRI, PET scans)
• Quantum computing development""",
                
                "detailed": """Quantum mechanics describes the probabilistic nature of physical systems at microscopic scales through wave functions and operators.

**Mathematical Framework:**
• Schrödinger equation: iℏ ∂ψ/∂t = Ĥψ
• Wave function ψ contains all information about a quantum system
• Observables are represented by Hermitian operators
• Measurement collapses the wave function probabilistically

**Key Phenomena:**
• **Quantum Tunneling**: Particles can pass through energy barriers
• **Spin**: Intrinsic angular momentum of particles
• **Pauli Exclusion Principle**: No two fermions can occupy the same quantum state
• **Quantum Decoherence**: Loss of quantum coherence due to environmental interaction

**Interpretations:**
• Copenhagen interpretation (measurement causes collapse)
• Many-worlds interpretation (all possibilities occur)
• Hidden variable theories (Bell's theorem violations)""",
                
                "advanced": """Quantum mechanics at the research level involves advanced mathematical formalism and exploration of foundational questions about the nature of reality.

**Advanced Topics:**
• **Quantum Field Theory**: Relativistic quantum mechanics
• **Quantum Information**: Entanglement, quantum computing, quantum cryptography
• **Quantum Gravity**: Attempts to quantize general relativity
• **Quantum Foundations**: Measurement problem, quantum-to-classical transition

**Current Research:**
• Quantum error correction and fault-tolerant quantum computing
• Quantum simulation of complex many-body systems
• Tests of quantum nonlocality and contextuality
• Quantum thermodynamics and quantum biology

**Open Questions:**
• What is the correct interpretation of quantum mechanics?
• How does quantum mechanics emerge from more fundamental theories?
• Can quantum mechanics be unified with general relativity?"""
            }
        }
        
        # Normalize the concept
        concept_lower = concept.lower().strip()
        
        # Check if we have specific information about this concept
        if concept_lower in concepts_db:
            concept_data = concepts_db[concept_lower]
            
            if depth.lower() in concept_data:
                return concept_data[depth.lower()]
            else:
                # Default to overview if depth not found
                return concept_data.get("overview", concept_data[list(concept_data.keys())[0]])
        
        # For concepts not in our database, provide a structured response
        result = f"**Physics Concept: {concept.title()}**\n\n"
        
        if depth.lower() == "overview":
            result += "**Overview Level Analysis**\n"
            result += f"The concept of {concept} in physics involves fundamental principles that govern how matter, energy, space, and time interact. "
            result += "This concept connects to broader physical theories and has both theoretical implications and practical applications.\n\n"
            result += "**Key Considerations:**\n"
            result += "• Theoretical foundations and governing equations\n"
            result += "• Experimental evidence and observations\n"
            result += "• Practical applications and technologies\n"
            result += "• Connections to other physics concepts\n\n"
        elif depth.lower() == "detailed":
            result += "**Detailed Level Analysis**\n"
            result += f"A comprehensive analysis of {concept} requires examining its mathematical formulation, experimental verification, and theoretical implications.\n\n"
            result += "**Mathematical Framework:**\n"
            result += "• Fundamental equations and their derivations\n"
            result += "• Boundary conditions and constraints\n"
            result += "• Symmetries and conservation laws\n\n"
            result += "**Experimental Aspects:**\n"
            result += "• Key experiments and observations\n"
            result += "• Measurement techniques and precision\n"
            result += "• Technological applications\n\n"
        elif depth.lower() == "advanced":
            result += "**Advanced Level Analysis**\n"
            result += f"Advanced study of {concept} involves cutting-edge research, theoretical developments, and open questions in modern physics.\n\n"
            result += "**Current Research:**\n"
            result += "• Recent theoretical developments\n"
            result += "• Experimental frontiers and new technologies\n"
            result += "• Interdisciplinary connections\n\n"
            result += "**Open Questions:**\n"
            result += "• Unresolved theoretical issues\n"
            result += "• Future research directions\n"
            result += "• Potential paradigm shifts\n\n"
        
        # Add suggestion for more specific information
        result += "**For More Specific Information:**\n"
        result += f"Consider searching for specific aspects of {concept} such as mathematical formulations, experimental techniques, or particular applications.\n"
        
        return result
    
    except Exception as e:
        return f"Error searching physics concept: {str(e)}"


def get_physics_research_tools() -> List:
    """Get all physics research tools.
    
    Returns:
        List of physics research tools
    """
    return [
        search_arxiv,
        search_physics_wikipedia,
        lookup_physics_equation,
        physics_concept_search
    ] 