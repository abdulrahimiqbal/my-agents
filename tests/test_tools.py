"""
Tests for tool implementations.
"""

import pytest
from unittest.mock import Mock, patch

from src.tools.calculator import (
    add, subtract, multiply, divide, power, square_root, 
    calculate_expression, get_calculator_tools
)
from src.tools.web_search import (
    search_web, search_news, get_website_content, get_web_search_tools
)


class TestCalculatorTools:
    """Tests for calculator tools."""
    
    def test_add(self):
        """Test addition tool."""
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0.5, 0.5) == 1.0
    
    def test_subtract(self):
        """Test subtraction tool."""
        assert subtract(5, 3) == 2
        assert subtract(0, 5) == -5
        assert subtract(10.5, 0.5) == 10.0
    
    def test_multiply(self):
        """Test multiplication tool."""
        assert multiply(3, 4) == 12
        assert multiply(-2, 5) == -10
        assert multiply(0.5, 8) == 4.0
    
    def test_divide(self):
        """Test division tool."""
        assert divide(10, 2) == 5
        assert divide(7, 2) == 3.5
        assert divide(-10, 5) == -2
    
    def test_divide_by_zero(self):
        """Test division by zero raises error."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)
    
    def test_power(self):
        """Test power tool."""
        assert power(2, 3) == 8
        assert power(5, 0) == 1
        assert power(4, 0.5) == 2.0
    
    def test_square_root(self):
        """Test square root tool."""
        assert square_root(4) == 2.0
        assert square_root(9) == 3.0
        assert square_root(0) == 0.0
    
    def test_square_root_negative(self):
        """Test square root of negative number raises error."""
        with pytest.raises(ValueError, match="Cannot calculate square root of negative number"):
            square_root(-1)
    
    def test_calculate_expression_simple(self):
        """Test simple expression calculation."""
        assert calculate_expression("2 + 3") == 5.0
        assert calculate_expression("10 - 4") == 6.0
        assert calculate_expression("3 * 4") == 12.0
        assert calculate_expression("8 / 2") == 4.0
    
    def test_calculate_expression_complex(self):
        """Test complex expression calculation."""
        assert calculate_expression("2 + 3 * 4") == 14.0
        assert calculate_expression("(2 + 3) * 4") == 20.0
        assert calculate_expression("sqrt(16) + 2") == 6.0
    
    def test_calculate_expression_with_constants(self):
        """Test expression with mathematical constants."""
        result = calculate_expression("pi * 2")
        assert abs(result - 6.283185307179586) < 1e-10
    
    def test_calculate_expression_division_by_zero(self):
        """Test expression with division by zero."""
        result = calculate_expression("5 / 0")
        assert "Error: Division by zero" in str(result)
    
    def test_calculate_expression_dangerous_operations(self):
        """Test that dangerous operations are blocked."""
        dangerous_expressions = [
            "import os",
            "__import__('os')",
            "exec('print(1)')",
            "eval('1+1')"
        ]
        
        for expr in dangerous_expressions:
            result = calculate_expression(expr)
            assert "Error: Expression contains potentially dangerous operations" in str(result)
    
    def test_calculate_expression_invalid_syntax(self):
        """Test invalid expression syntax."""
        result = calculate_expression("2 + + 3")
        assert "Error: Invalid expression" in str(result)
    
    def test_get_calculator_tools(self):
        """Test that all calculator tools are returned."""
        tools = get_calculator_tools()
        
        assert len(tools) == 7
        tool_names = [tool.name for tool in tools]
        expected_names = ["add", "subtract", "multiply", "divide", "power", "square_root", "calculate_expression"]
        
        for name in expected_names:
            assert name in tool_names


class TestWebSearchTools:
    """Tests for web search tools."""
    
    @patch('src.tools.web_search.get_settings')
    def test_search_web_no_api_key(self, mock_get_settings):
        """Test web search without API key."""
        mock_settings = Mock()
        mock_settings.tavily_api_key = None
        mock_get_settings.return_value = mock_settings
        
        result = search_web("test query")
        assert "Error: Tavily API key not configured" in result
    
    @patch('src.tools.web_search.get_settings')
    @patch('src.tools.web_search.TavilyClient')
    def test_search_web_success(self, mock_tavily_client, mock_get_settings):
        """Test successful web search."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.tavily_api_key = "test-key"
        mock_get_settings.return_value = mock_settings
        
        # Mock Tavily client
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "content": "Test content for the search result"
                }
            ]
        }
        mock_tavily_client.return_value = mock_client_instance
        
        result = search_web("test query")
        
        assert "Search results for 'test query'" in result
        assert "Test Result" in result
        assert "https://example.com" in result
        assert "Test content" in result
    
    @patch('src.tools.web_search.get_settings')
    @patch('src.tools.web_search.TavilyClient')
    def test_search_web_no_results(self, mock_tavily_client, mock_get_settings):
        """Test web search with no results."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.tavily_api_key = "test-key"
        mock_get_settings.return_value = mock_settings
        
        # Mock Tavily client with empty results
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {"results": []}
        mock_tavily_client.return_value = mock_client_instance
        
        result = search_web("test query")
        assert "No results found for query: test query" in result
    
    @patch('src.tools.web_search.get_settings')
    def test_search_web_import_error(self, mock_get_settings):
        """Test web search with missing Tavily package."""
        mock_settings = Mock()
        mock_settings.tavily_api_key = "test-key"
        mock_get_settings.return_value = mock_settings
        
        with patch('src.tools.web_search.TavilyClient', side_effect=ImportError):
            result = search_web("test query")
            assert "Error: Tavily package not installed" in result
    
    @patch('src.tools.web_search.get_settings')
    def test_search_news_no_api_key(self, mock_get_settings):
        """Test news search without API key."""
        mock_settings = Mock()
        mock_settings.tavily_api_key = None
        mock_get_settings.return_value = mock_settings
        
        result = search_news("test query")
        assert "Error: Tavily API key not configured" in result
    
    @patch('src.tools.web_search.requests.get')
    def test_get_website_content_success(self, mock_get):
        """Test successful website content extraction."""
        # Mock response
        mock_response = Mock()
        mock_response.content = b"<html><body><h1>Test Title</h1><p>Test content</p></body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = get_website_content("https://example.com")
        
        assert "Content from https://example.com" in result
        assert "Test Title" in result
        assert "Test content" in result
    
    @patch('src.tools.web_search.requests.get')
    def test_get_website_content_request_error(self, mock_get):
        """Test website content extraction with request error."""
        mock_get.side_effect = Exception("Network error")
        
        result = get_website_content("https://example.com")
        assert "Error extracting website content: Network error" in result
    
    def test_get_website_content_import_error(self):
        """Test website content extraction with missing packages."""
        with patch('src.tools.web_search.requests', side_effect=ImportError):
            result = get_website_content("https://example.com")
            assert "Error: Required packages not installed" in result
    
    def test_get_web_search_tools(self):
        """Test that all web search tools are returned."""
        tools = get_web_search_tools()
        
        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        expected_names = ["search_web", "search_news", "get_website_content"]
        
        for name in expected_names:
            assert name in tool_names


class TestToolIntegration:
    """Integration tests for tools."""
    
    def test_calculator_tools_are_callable(self):
        """Test that calculator tools can be called."""
        tools = get_calculator_tools()
        
        for tool in tools:
            assert callable(tool.func)
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
    
    def test_web_search_tools_are_callable(self):
        """Test that web search tools can be callable."""
        tools = get_web_search_tools()
        
        for tool in tools:
            assert callable(tool.func)
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')


if __name__ == "__main__":
    pytest.main([__file__]) 