from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from math_tools import get_math_tool

from model.language_models import GLM4

calculate = get_math_tool(GLM4())
search = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="the search query") - a search engine.',
)

tools = [search, calculate]

calculate.invoke(
    {
        "problem": "What's the temp of sf + 5?",
        "context": ["Thet empreature of sf is 32 degrees"],
    }
)