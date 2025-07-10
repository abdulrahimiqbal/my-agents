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
        # This would integrate with multiple sources
        # For now, provide a structured approach to concept explanation
        
        result = f"**Physics Concept: {concept.title()}**\n\n"
        
        if depth.lower() == "overview":
            result += "**Overview Level Explanation**\n"
            result += "• Basic definition and key principles\n"
            result += "• Simple examples and analogies\n"
            result += "• Real-world applications\n\n"
        elif depth.lower() == "detailed":
            result += "**Detailed Level Explanation**\n"
            result += "• Mathematical formulation\n"
            result += "• Derivations and proofs\n"
            result += "• Multiple examples and problem-solving\n"
            result += "• Connections to other concepts\n\n"
        elif depth.lower() == "advanced":
            result += "**Advanced Level Explanation**\n"
            result += "• Rigorous mathematical treatment\n"
            result += "• Current research and developments\n"
            result += "• Theoretical implications\n"
            result += "• Open questions and frontiers\n\n"
        
        # Suggest using other tools for comprehensive information
        result += "**Recommended Actions:**\n"
        result += f"1. Search ArXiv for recent papers: search_arxiv('{concept}')\n"
        result += f"2. Get Wikipedia overview: search_physics_wikipedia('{concept}')\n"
        result += f"3. Look up related equations: lookup_physics_equation('[equation name]')\n"
        
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