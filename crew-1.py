#!/usr/bin/env python3
"""
Product Research and Sales Guide CrewAI Project

This script implements a CrewAI project with multiple agents working together
to research products and create sales guides.

Agents:
- Searcher: Finds relevant product information online
- Scraper: Extracts detailed information from websites
- Analyst: Analyzes data for inconsistencies and errors
- Writer: Creates comprehensive sales guides

Usage:
    python crew-1.py

Requirements:
    - crewai
    - langchain
    - beautifulsoup4
    - requests
    - duckduckgo-search
    - ollama (optional, for local models)
    - openai (optional, for OpenAI models)
"""

import os
import sys
import json
import time
import logging
import asyncio
import functools
from typing import List, Dict, Any, Optional, Union, Callable, ClassVar
from pydantic import Field
from enum import Enum

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# LangChain imports
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

# Web scraping imports
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define LLM provider options
class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

# Error handling decorator
def with_error_handling(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Network error: {e}. Retrying in {wait_time}s ({retry_count}/{max_retries})")
                time.sleep(wait_time)
            except TimeoutError:
                logger.error(f"Operation timed out. Proceeding with partial results.")
                return {"error": "timeout", "partial_result": True, "data": {}}
            except Exception as e:
                logger.error(f"Critical error: {e}")
                return {"error": str(e), "success": False}
                
        # If we've exhausted retries
        logger.error(f"Failed after {max_retries} attempts")
        return {"error": "max_retries_exceeded", "success": False}
    
    return wrapper

# Custom tools for the agents
class WebSearchTool(BaseTool):
    """Tool for searching the web for information about a product."""
    
    name: str = "Web Search"
    description: str = "Search the web for information about a product"
    
    def __init__(self):
        super().__init__()
        self._search_engine = DuckDuckGoSearchRun()
    
    @with_error_handling
    def _run(self, query: str) -> str:
        """Run the web search for the given query."""
        logger.info(f"Searching for: {query}")
        results = self._search_engine.run(query)
        return results
    
    async def _arun(self, query: str) -> str:
        """Run the web search asynchronously."""
        return self._run(query)

class WebScraperTool(BaseTool):
    """Tool for scraping information from websites."""
    
    name: str = "Web Scraper"
    description: str = "Scrape information from a website URL"
    
    @with_error_handling
    def _run(self, url: str) -> Dict[str, Any]:
        """Scrape the given URL for product information."""
        logger.info(f"Scraping URL: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=45)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic information
        title = soup.title.text.strip() if soup.title else "No title found"
        
        # Extract all paragraphs
        paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
        
        # Extract product information (generic approach)
        product_info = {}
        
        # Look for price information
        price_elements = soup.find_all(text=lambda text: "$" in text if text else False)
        prices = [element.strip() for element in price_elements if element.strip()]
        if prices:
            product_info["possible_prices"] = prices[:5]  # Limit to first 5 to avoid noise
        
        # Look for product specifications
        spec_tables = soup.find_all('table')
        if spec_tables:
            product_info["specifications"] = []
            for table in spec_tables[:3]:  # Limit to first 3 tables
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        key = cells[0].text.strip()
                        value = cells[1].text.strip()
                        if key and value:
                            product_info["specifications"].append({key: value})
        
        # Look for product features (often in lists)
        feature_lists = soup.find_all(['ul', 'ol'])
        if feature_lists:
            product_info["features"] = []
            for feature_list in feature_lists[:5]:  # Limit to first 5 lists
                items = feature_list.find_all('li')
                features = [item.text.strip() for item in items if item.text.strip()]
                if features:
                    product_info["features"].extend(features)
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else "No description found"
        
        return {
            "url": url,
            "title": title,
            "description": description,
            "content_summary": "\n".join(paragraphs[:10]),  # First 10 paragraphs
            "product_info": product_info,
            "success": True
        }
    
    async def _arun(self, url: str) -> Dict[str, Any]:
        """Run the web scraper asynchronously."""
        return self._run(url)

class DataAnalysisTool(BaseTool):
    """Tool for analyzing and validating product data."""
    
    name: str = "Data Analyzer"
    description: str = "Analyze product data for inconsistencies and errors"
    
    @with_error_handling
    def _run(self, data: str) -> Dict[str, Any]:
        """Analyze the given product data for inconsistencies and errors."""
        logger.info("Analyzing product data")
        
        # Parse the input data
        try:
            product_data = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError:
            # If not valid JSON, treat as text
            return {
                "analysis": "The provided data is not in a structured format. Please provide structured data.",
                "confidence": 0.0,
                "success": False
            }
        
        analysis_results = {
            "inconsistencies": [],
            "missing_information": [],
            "confidence_scores": {},
            "overall_confidence": 0.0,
            "success": True
        }
        
        # Check for missing critical information
        critical_fields = ["title", "description", "features", "specifications"]
        for field in critical_fields:
            if field not in product_data or not product_data[field]:
                analysis_results["missing_information"].append(field)
        
        # Check for inconsistencies in pricing
        if "possible_prices" in product_data and len(product_data["possible_prices"]) > 1:
            price_values = []
            for price_str in product_data["possible_prices"]:
                try:
                    # Extract numeric value from price string (e.g., "$99.99" -> 99.99)
                    price_value = float(''.join(c for c in price_str if c.isdigit() or c == '.'))
                    price_values.append(price_value)
                except ValueError:
                    continue
            
            if price_values:
                min_price = min(price_values)
                max_price = max(price_values)
                
                # If price difference is significant, flag as inconsistency
                if max_price > min_price * 1.5:  # 50% difference threshold
                    analysis_results["inconsistencies"].append(
                        f"Price inconsistency detected: values range from {min_price} to {max_price}"
                    )
        
        # Calculate confidence scores
        confidence_scores = {}
        
        # Title confidence
        if "title" in product_data and product_data["title"] != "No title found":
            confidence_scores["title"] = 1.0
        else:
            confidence_scores["title"] = 0.0
        
        # Description confidence
        if "description" in product_data and product_data["description"] != "No description found":
            confidence_scores["description"] = 1.0
        else:
            confidence_scores["description"] = 0.0
        
        # Features confidence
        if "features" in product_data and len(product_data.get("features", [])) > 0:
            confidence_scores["features"] = min(1.0, len(product_data["features"]) / 5)
        else:
            confidence_scores["features"] = 0.0
        
        # Specifications confidence
        if "specifications" in product_data and len(product_data.get("specifications", [])) > 0:
            confidence_scores["specifications"] = min(1.0, len(product_data["specifications"]) / 5)
        else:
            confidence_scores["specifications"] = 0.0
        
        # Calculate overall confidence
        if confidence_scores:
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            overall_confidence = 0.0
        
        analysis_results["confidence_scores"] = confidence_scores
        analysis_results["overall_confidence"] = overall_confidence
        
        return analysis_results
    
    async def _arun(self, data: str) -> Dict[str, Any]:
        """Run the data analyzer asynchronously."""
        return self._run(data)

# Define the agents
def create_agents(llm_provider: LLMProvider, verbose: bool = True):
    """Create and return the agents for the crew."""
    
    # Configure the LLM based on the provider
    if llm_provider == LLMProvider.OPENAI:
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            sys.exit(1)
    
    # Create the searcher agent
    searcher_agent = Agent(
        role="Product Search Specialist",
        goal="Find comprehensive and accurate information about products online",
        backstory="""You are an expert at finding information about products online. 
        You know how to craft effective search queries and identify the most relevant 
        and reliable sources of information.""",
        verbose=verbose,
        allow_delegation=True,
        tools=[WebSearchTool()],
        llm_provider=llm_provider
    )
    
    # Create the scraper agent
    scraper_agent = Agent(
        role="Web Scraping Expert",
        goal="Extract detailed and structured product information from websites",
        backstory="""You are a skilled web scraper who can extract valuable information 
        from various websites. You know how to navigate different website structures and 
        identify the most important product details.""",
        verbose=verbose,
        allow_delegation=True,
        tools=[WebScraperTool()],
        llm_provider=llm_provider
    )
    
    # Create the analyst agent
    analyst_agent = Agent(
        role="Data Analysis Specialist",
        goal="Analyze product data for inconsistencies, errors, and gaps",
        backstory="""You are a meticulous data analyst with an eye for detail. 
        You can spot inconsistencies and errors in product information and provide 
        corrections and improvements.""",
        verbose=verbose,
        allow_delegation=True,
        tools=[DataAnalysisTool()],
        llm_provider=llm_provider
    )
    
    # Create the writer agent
    writer_agent = Agent(
        role="Sales Guide Writer",
        goal="Create comprehensive and persuasive sales guides for products",
        backstory="""You are a talented writer specializing in creating sales guides. 
        You know how to highlight product features and benefits in a way that resonates 
        with potential customers.""",
        verbose=verbose,
        allow_delegation=True,
        llm_provider=llm_provider
    )
    
    return {
        "searcher": searcher_agent,
        "scraper": scraper_agent,
        "analyst": analyst_agent,
        "writer": writer_agent
    }

# Define the tasks
def create_tasks(agents, product_name: str):
    """Create and return the tasks for the crew."""
    
    # Task 1: Search for product information
    search_task = Task(
        description=f"""Search for comprehensive information about {product_name}.
        Find the most relevant and reliable sources of information.
        Focus on official product pages, reviews, and technical specifications.
        Return a list of the most relevant URLs along with a summary of what each source contains.""",
        agent=agents["searcher"],
        expected_output="""A list of relevant URLs with brief descriptions of what information 
        each source contains. The list should be formatted as JSON."""
    )
    
    # Task 2: Scrape product information
    scrape_task = Task(
        description=f"""Using the URLs provided by the searcher, scrape detailed information about {product_name}.
        Extract product features, specifications, pricing, and any other relevant details.
        Organize the information in a structured format.
        Focus on accuracy and completeness.""",
        agent=agents["scraper"],
        expected_output="""Structured product information including features, specifications, 
        pricing, and other relevant details. The information should be formatted as JSON.""",
        context=[search_task]
    )
    
    # Task 3: Analyze product data
    analyze_task = Task(
        description=f"""Analyze the scraped information about {product_name} for inconsistencies, errors, and gaps.
        Identify any contradictions or questionable information.
        Assess the confidence level for different pieces of information.
        Provide corrections and improvements where necessary.""",
        agent=agents["analyst"],
        expected_output="""Analysis results including identified inconsistencies, 
        corrections, and confidence scores. The results should be formatted as JSON.""",
        context=[scrape_task]
    )
    
    # Task 4: Create sales guide
    write_task = Task(
        description=f"""Create a comprehensive sales guide for {product_name} based on the analyzed information.
        Highlight the key features and benefits of the product.
        Address potential customer questions and concerns.
        Use persuasive language and formatting to make the guide engaging and effective.
        The guide should be well-structured with clear sections and bullet points where appropriate.""",
        agent=agents["writer"],
        expected_output="""A complete sales guide for the product, formatted with 
        clear sections, bullet points, and persuasive language.""",
        context=[analyze_task]
    )
    
    return [search_task, scrape_task, analyze_task, write_task]

# Main function
def main():
    """Main function to run the CrewAI project."""
    
    print("\n" + "="*50)
    print("Product Research and Sales Guide CrewAI Project")
    print("="*50 + "\n")
    
    # Get product name from user
    product_name = input("Enter the product name to research: ")
    if not product_name:
        print("Error: Product name cannot be empty.")
        sys.exit(1)
    
    # Get LLM provider from user
    print("\nSelect LLM provider:")
    print("1. OpenAI")
    print("2. Ollama")
    provider_choice = input("Enter your choice (1 or 2): ")
    
    if provider_choice == "1":
        llm_provider = LLMProvider.OPENAI
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            api_key = input("Enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
    elif provider_choice == "2":
        llm_provider = LLMProvider.OLLAMA
    else:
        print("Invalid choice. Defaulting to OpenAI.")
        llm_provider = LLMProvider.OPENAI
    
    # Get verbosity preference
    verbose_choice = input("Enable verbose mode for all agents? (y/n): ")
    verbose = verbose_choice.lower() == "y"
    
    print("\nInitializing agents...")
    agents = create_agents(llm_provider, verbose)
    
    print("Creating tasks...")
    tasks = create_tasks(agents, product_name)
    
    print("Assembling crew...")
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        verbose=verbose,
        process=Process.sequential
    )
    
    print("\n" + "="*50)
    print(f"Starting research for: {product_name}")
    print("="*50 + "\n")
    
    try:
        # Run the crew
        result = crew.kickoff()
        
        print("\n" + "="*50)
        print("Research Complete!")
        print("="*50 + "\n")
        
        print(result)
        
        # Save the result to a file
        filename = f"{product_name.replace(' ', '_').lower()}_sales_guide.txt"
        with open(filename, "w") as f:
            f.write(result)
        
        print(f"\nSales guide saved to: {filename}")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
