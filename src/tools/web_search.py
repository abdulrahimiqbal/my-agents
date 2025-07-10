"""
Web search tools for internet research.

Provides web search capabilities using Tavily API, following LangChain Academy
tool creation patterns from Module 1.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, tool

from ..config import get_settings


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for information using Tavily.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    settings = get_settings()
    
    if not settings.tavily_api_key:
        return "Error: Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
    
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=settings.tavily_api_key)
        
        # Perform search
        results = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )
        
        if not results.get("results"):
            return f"No results found for query: {query}"
        
        # Format results
        formatted_results = f"Search results for '{query}':\n\n"
        
        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content available")
            
            formatted_results += f"{i}. **{title}**\n"
            formatted_results += f"   URL: {url}\n"
            formatted_results += f"   Summary: {content[:200]}{'...' if len(content) > 200 else ''}\n\n"
        
        return formatted_results
        
    except ImportError:
        return "Error: Tavily package not installed. Run 'pip install tavily-python' to use web search."
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def search_news(query: str, max_results: int = 3) -> str:
    """
    Search for recent news articles.
    
    Args:
        query: News search query
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        Formatted news results with titles, URLs, and snippets
    """
    settings = get_settings()
    
    if not settings.tavily_api_key:
        return "Error: Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
    
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=settings.tavily_api_key)
        
        # Perform news search
        results = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_domains=["news.google.com", "reuters.com", "bbc.com", "cnn.com", "npr.org"]
        )
        
        if not results.get("results"):
            return f"No news results found for query: {query}"
        
        # Format results
        formatted_results = f"Recent news for '{query}':\n\n"
        
        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content available")
            
            formatted_results += f"{i}. **{title}**\n"
            formatted_results += f"   Source: {url}\n"
            formatted_results += f"   Summary: {content[:150]}{'...' if len(content) > 150 else ''}\n\n"
        
        return formatted_results
        
    except ImportError:
        return "Error: Tavily package not installed. Run 'pip install tavily-python' to use news search."
    except Exception as e:
        return f"Error performing news search: {str(e)}"


@tool
def get_website_content(url: str) -> str:
    """
    Get the main content from a website URL.
    
    Args:
        url: Website URL to extract content from
        
    Returns:
        Main text content from the website
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit length
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return f"Content from {url}:\n\n{text}"
        
    except ImportError:
        return "Error: Required packages not installed. Run 'pip install requests beautifulsoup4' to use website content extraction."
    except requests.exceptions.RequestException as e:
        return f"Error fetching website content: {str(e)}"
    except Exception as e:
        return f"Error extracting website content: {str(e)}"


def get_web_search_tools() -> List[BaseTool]:
    """
    Get all web search tools.
    
    Returns:
        List of web search tools
    """
    return [
        search_web,
        search_news,
        get_website_content,
    ] 