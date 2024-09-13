import os
import json
import asyncio
import logging
import ssl
import certifi
import requests
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import google.generativeai as genai
from github import Github
from github.Repository import Repository
from github.GithubException import UnknownObjectException, GithubException
from diskcache import Cache
import aiohttp
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import re
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class Config:
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_PROJECT: str = os.getenv("GEMINI_PROJECT", "shiny260")
    GEMINI_LOCATION: str = os.getenv("GEMINI_LOCATION", "us-central1")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    CACHE_DIR: str = "./github_cache"
    CACHE_EXPIRY: int = int(1.5 * 3600)  # 1.5 hours in seconds
    MAX_RETRIES: int = 5
    INITIAL_DELAY: int = 1

config = Config()

# Validate configuration
if not config.GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is not set in the environment variables.")
if not config.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
if not config.HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY is not set in the environment variables.")

# Configure the Gemini API
genai.configure(api_key=config.GOOGLE_API_KEY)

# Initialize embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=config.HUGGINGFACE_API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize GitHub client, cache, and SSL context
g = Github(config.GITHUB_TOKEN)
cache = Cache(config.CACHE_DIR)
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Pre-defined questions
PRE_DEFINED_QUESTIONS = [
    "What is the overall activity level of the repository?",
    "How engaged is the community with this project?",
    "What are the main strengths and weaknesses of the project based on the data?",
    "Are there any noticeable trends in the repository's development over time?"
]

# Constants for data limits
COMMIT_LIMIT = 100
CONTRIBUTOR_LIMIT = 50
ISSUE_LIMIT = 100
PR_LIMIT = 100

def extract_repo_name(url: str) -> Optional[str]:
    """Extract the repository name from a GitHub URL."""
    match = re.match(r"https://github\.com/([^/]+/[^/]+)/?.*", url)
    if match:
        return match.group(1)
    return None

def get_user_input(repo_url: str) -> str:
    """Get the GitHub repository full name from the user input."""
    if re.match(r"https://github\.com/[\w-]+/[\w.-]+/?$", repo_url):
        return "/".join(repo_url.split("/")[-2:])
    else:
        # Log the error and return a special value
        print("Invalid GitHub URL:", repo_url)
        return None  # or return an empty string, or some other placeholder

async def fetch_repository_data(repo_full_name: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch raw data from a GitHub repository."""
    if not force_refresh:
        cached_data = cache.get(repo_full_name)
        if cached_data is not None:
            logger.info(f"Using cached data for {repo_full_name}")
            return cached_data

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        try:
            github_repo = g.get_repo(repo_full_name)
        except UnknownObjectException:
            logger.error(f"Repository not found: {repo_full_name}")
            return {"error": f"Repository not found: {repo_full_name}"}
        except GithubException as e:
            logger.error(f"GitHub API error for {repo_full_name}: {str(e)}")
            return {"error": f"GitHub API error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error accessing repository {repo_full_name}: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

        data = {
            "name": github_repo.name,
            "full_name": github_repo.full_name,
            "description": github_repo.description,
            "created_at": github_repo.created_at.isoformat(),
            "updated_at": github_repo.updated_at.isoformat(),
            "pushed_at": github_repo.pushed_at.isoformat(),
            "size": github_repo.size,
            "stargazers_count": github_repo.stargazers_count,
            "watchers_count": github_repo.watchers_count,
            "forks_count": github_repo.forks_count,
            "open_issues_count": github_repo.open_issues_count,
            "language": github_repo.language,
            "has_issues": github_repo.has_issues,
            "has_projects": github_repo.has_projects,
            "has_wiki": github_repo.has_wiki,
            "default_branch": github_repo.default_branch,
            "license": github_repo.license.name if github_repo.license else None,
            "topics": github_repo.get_topics(),
            "commits": [],
            "branches": [],
            "contributors": [],
            "issues": [],
            "pull_requests": [],
            "readme": "",
        }

        # Fetch commits
        try:
            data["commits"] = [{"sha": commit.sha, 
                                "author": commit.author.login if commit.author else None,
                                "date": commit.commit.author.date.isoformat(),
                                "message": commit.commit.message}
                               for commit in github_repo.get_commits()[:COMMIT_LIMIT]]
        except Exception as e:
            logger.error(f"Error fetching commits: {str(e)}")
            data["commits"] = []

        # Fetch branches
        try:
            data["branches"] = [branch.name for branch in github_repo.get_branches()]
        except Exception as e:
            logger.error(f"Error fetching branches: {str(e)}")
            data["branches"] = []

        # Fetch contributors
        try:
            data["contributors"] = [{"login": contributor.login, "contributions": contributor.contributions} 
                                    for contributor in github_repo.get_contributors()[:CONTRIBUTOR_LIMIT]]
        except Exception as e:
            logger.error(f"Error fetching contributors: {str(e)}")
            data["contributors"] = []

        # Fetch issues
        try:
            data["issues"] = [{"number": issue.number, 
                               "state": issue.state, 
                               "created_at": issue.created_at.isoformat(), 
                               "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
                               "title": issue.title}
                              for issue in github_repo.get_issues(state='all')[:ISSUE_LIMIT]]
        except Exception as e:
            logger.error(f"Error fetching issues: {str(e)}")
            data["issues"] = []

        # Fetch pull requests
        try:
            data["pull_requests"] = [{"number": pr.number, 
                                      "state": pr.state, 
                                      "created_at": pr.created_at.isoformat(), 
                                      "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
                                      "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                                      "title": pr.title}
                                     for pr in github_repo.get_pulls(state='all')[:PR_LIMIT]]
        except Exception as e:
            logger.error(f"Error fetching pull requests: {str(e)}")
            data["pull_requests"] = []

        # Fetch README content
        try:
            readme = github_repo.get_readme()
            data["readme"] = readme.decoded_content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error fetching README: {str(e)}")
            data["readme"] = ""

        # Fetch code content
        try:
            readme = github_repo.get_readme()
            data["readme"] = readme.decoded_content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error fetching README: {str(e)}")
            data["readme"] = ""

        cache.set(repo_full_name, data, expire=config.CACHE_EXPIRY)
        return data

async def analyze_with_gemini(prompt: str) -> str:
    """Analyze repository data using Gemini 1.5 Flash API with formatted output."""
    formatting_instructions = """
    Please format your response using the following structure:

    # Summary
    [Provide a brief summary of the analysis]

    ## Key Points
    - [First key point]
    - [Second key point]
    - [Third key point]

    # Detailed Analysis
    ## [Subtopic 1]
    [Detailed analysis for subtopic 1]

    ## [Subtopic 2]
    [Detailed analysis for subtopic 2]

    # Conclusion
    [Provide a concise conclusion]

    Please ensure that your response adheres to this structure for all questions, including those about trends or patterns. For questions about trends, focus on the main trends observed, if any, or explain why trends cannot be determined within the existing structure.
    """

    full_prompt = f"{formatting_instructions}\n\n{prompt}"

    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        
        if response.parts:
            logger.info(f"Gemini API response received. Length: {len(response.text)} characters")
            logger.debug(f"Response preview: {response.text[:100]}...")  # Log the first 100 characters
            return response.text
        else:
            logger.error("The Gemini API returned an empty response.")
            return "The Gemini API returned an empty response."
    
    except genai.types.GoogleGenerativeAIError as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)}"
    except ValueError as e:
        logger.error(f"Invalid argument to Gemini API: {str(e)}")
        return f"Gemini API error: Invalid argument - {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error using Gemini API: {str(e)}")
        return f"Unexpected error: {str(e)}"
    
def generate_base_prompt(data: Dict[str, Any], initial_analysis: Dict[str, Any]) -> str:
    """Generate a base prompt for LLM analysis."""
    return f"""
    Repository: {data['full_name']}
    Description: {data['description']}
    Created: {data['created_at']}
    Last Updated: {data['updated_at']}
    Stars: {data['stargazers_count']}
    Forks: {data['forks_count']}
    Open Issues: {data['open_issues_count']}
    Main Language: {data['language']}
    
    Initial Analysis:
    {json.dumps(initial_analysis, indent=2)}
    
    Based on this information about the GitHub repository, please provide a detailed analysis.
    """

async def analyze_pre_defined_questions(data: Dict[str, Any], initial_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    """Analyze pre-defined questions about the repository."""
    base_prompt = generate_base_prompt(data, initial_analysis)
    answers = []

    for question in PRE_DEFINED_QUESTIONS:
        full_prompt = f"""
        {base_prompt}

        Please answer the following question about the repository:

        {question}

        Your response:
        """
        response = await analyze_with_gemini(full_prompt)
        answers.append({"question": question, "answer": response})

    return answers

async def analyze_user_question(data: Dict[str, Any], initial_analysis: Dict[str, Any], user_question: str, vector_store: FAISS) -> str:
    """Analyze a user-defined question about the repository."""
    try:
        # Use the vector store to find relevant content
        relevant_docs = vector_store.similarity_search(user_question, k=3)
        relevant_content = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Generate a focused prompt for the user question
        base_prompt = generate_base_prompt(data, initial_analysis)
        full_prompt = f"""
        {base_prompt}

        Based on the following relevant content from the GitHub repository, please answer this question:
        
        {user_question}

        Relevant Repository Content:
        {relevant_content}

        Please provide a concise and data-driven answer to the question, using the provided relevant content.
        """

        # Perform LLM analysis
        answer = await analyze_with_gemini(full_prompt)

        return answer

    except Exception as e:
        logger.error(f"An error occurred while analyzing the user question: {str(e)}")
        return f"An error occurred: {str(e)}"

async def process_repository_content(data: Dict[str, Any]) -> FAISS:
    """Process repository content: split text and create embeddings."""
    # Combine all text content
    all_text = data["readme"] + "\n\n"
    for commit in data["commits"]:
        all_text += f"Commit: {commit['message']}\n"
    for issue in data["issues"]:
        all_text += f"Issue: {issue['title']}\n"
    for pr in data["pull_requests"]:
        all_text += f"Pull Request: {pr['title']}\n"

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(all_text)

    # Create embeddings
    try:
        vector_store = FAISS.from_texts(texts, embeddings)
        logger.info("Successfully created vector store")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        raise

def perform_initial_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform initial analysis on the repository data."""
    return {
        "total_commits": len(data["commits"]),
        "total_contributors": len(data["contributors"]),
        "total_issues": len(data["issues"]),
        "total_pull_requests": len(data["pull_requests"]),
        "average_commits_per_contributor": len(data["commits"]) / len(data["contributors"]) if data["contributors"] else 0,
        "open_issues_ratio": sum(1 for issue in data["issues"] if issue["state"] == "open") / len(data["issues"]) if data["issues"] else 0,
        "merged_pr_ratio": sum(1 for pr in data["pull_requests"] if pr["merged_at"]) / len(data["pull_requests"]) if data["pull_requests"] else 0,
    }

async def fetch_file_content(session: aiohttp.ClientSession, url: str, file_name: str) -> Optional[str]:
    """Fetch file content from GitHub."""
    try:
        async with session.get(url, ssl=ssl_context) as response:
            content = await response.read()
            return content.decode('utf-8')
    except UnicodeDecodeError:
        logger.warning(f"Unable to decode {file_name} as UTF-8. Skipping this file.")
    except Exception as e:
        logger.error(f"Error fetching {file_name}: {str(e)}")
    return None

def generate_visualizations_as_objects(data: Dict[str, Any]) -> Dict[str, Figure]:
    """Generate visualizations based on the repository data and return as plot objects."""
    visualizations = {}

    # 1. Commit Activity Over Time
    if data.get("commits"):
        commit_dates = [datetime.fromisoformat(commit["date"]) for commit in data["commits"] if commit.get("date")]
        if commit_dates:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(commit_dates, bins=20, color='blue', edgecolor='black')
            ax.set_title("Commit Activity Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Commits")
            plt.xticks(rotation=45)
            plt.tight_layout()
            visualizations["commit_activity"] = fig

    # 2. Top Contributors
    if data.get("commits"):
        author_commits = {}
        for commit in data["commits"]:
            author = commit.get("author")
            if author:
                author_commits[author] = author_commits.get(author, 0) + 1
        top_contributors = sorted(author_commits.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_contributors:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar([c[0] for c in top_contributors], [c[1] for c in top_contributors], color='green')
            ax.set_title("Top 5 Contributors")
            ax.set_xlabel("Contributor")
            ax.set_ylabel("Number of Commits")
            plt.xticks(rotation=45)
            plt.tight_layout()
            visualizations["top_contributors"] = fig

    # 3. File Type Distribution
    if data.get("code"):
        file_types = {}
        for file in data["code"]:
            file_type = file.split(".")[-1] if "." in file else "unknown"
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        if file_types:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(file_types.values(), labels=file_types.keys(), autopct='%1.1f%%', colors=plt.cm.Paired(range(len(file_types))))
            ax.set_title("File Type Distribution")
            ax.axis('equal')
            plt.tight_layout()
            visualizations["file_types"] = fig

    return visualizations

async def analyze_repository(repo_url: str, user_question: Optional[str] = None, fetch_data: bool = True) -> Dict[str, Any]:
    """Perform a comprehensive analysis of the repository."""
    try:
        # NEW: Extract repository name from URL
        repo_full_name = extract_repo_name(repo_url)
        if not repo_full_name:
            return {"error": "Invalid GitHub repository URL"}

        # MODIFIED: Added caching logic
        if fetch_data:
            data = await fetch_repository_data(repo_full_name)
            if "error" in data:
                return {"error": data["error"]}
            
            initial_analysis = perform_initial_analysis(data)
            vector_store = await process_repository_content(data)
            
            # NEW: Store data for future use
            cache.set(f"{repo_full_name}_data", data)
            cache.set(f"{repo_full_name}_initial_analysis", initial_analysis)
            cache.set(f"{repo_full_name}_vector_store", vector_store)
        else:
            # NEW: Use cached data
            data = cache.get(f"{repo_full_name}_data")
            initial_analysis = cache.get(f"{repo_full_name}_initial_analysis")
            vector_store = cache.get(f"{repo_full_name}_vector_store")
            
            if not all([data, initial_analysis, vector_store]):
                return {"error": "Cached data not found. Please perform an initial analysis first."}

        # Analyze pre-defined questions (only if fetch_data is True)
        pre_defined_analysis = None
        if fetch_data:
            pre_defined_analysis = await analyze_pre_defined_questions(data, initial_analysis)

        # Analyze user question if provided
        user_question_analysis = None
        if user_question:
            user_question_analysis = await analyze_user_question(data, initial_analysis, user_question, vector_store)

        # NEW: Generate visualizations
        visualizations = generate_visualizations_as_objects(data)

        # Combine results
        final_report = {
            "repository": repo_full_name,
            "initial_analysis": initial_analysis,
            "visualizations": visualizations  # NEW: Added visualizations to the report
        }
        
        if pre_defined_analysis:
            final_report["pre_defined_analysis"] = pre_defined_analysis
        
        if user_question_analysis:
            final_report["user_question_analysis"] = user_question_analysis

        return final_report

    except Exception as e:
        logger.error(f"An error occurred during repository analysis: {str(e)}")
        return {"error": str(e)}
# NEW: Added function to generate visualizations
def generate_visualizations_as_objects(data: Dict[str, Any]) -> Dict[str, Figure]:
    """Generate visualizations based on the repository data and return as plot objects."""
    visualizations = {}

    # 1. Commit Activity Over Time
    if data.get("commits"):
        commit_dates = [datetime.fromisoformat(commit["date"]) for commit in data["commits"] if commit.get("date")]
        if commit_dates:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(commit_dates, bins=20, color='blue', edgecolor='black')
            ax.set_title("Commit Activity Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Commits")
            plt.xticks(rotation=45)
            plt.tight_layout()
            visualizations["commit_activity"] = fig

    # 2. Top Contributors
    if data.get("commits"):
        author_commits = {}
        for commit in data["commits"]:
            author = commit.get("author")
            if author:
                author_commits[author] = author_commits.get(author, 0) + 1
        top_contributors = sorted(author_commits.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_contributors:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar([c[0] for c in top_contributors], [c[1] for c in top_contributors], color='green')
            ax.set_title("Top 5 Contributors")
            ax.set_xlabel("Contributor")
            ax.set_ylabel("Number of Commits")
            plt.xticks(rotation=45)
            plt.tight_layout()
            visualizations["top_contributors"] = fig

    return visualizations

async def main(repo_url: str, user_question: Optional[str] = None, force_refresh: bool = False):
    try:
        result = await analyze_repository(repo_url, user_question, fetch_data=force_refresh)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    repo_url = input("Enter the GitHub repository URL: ")
    user_question = input("Enter your question about the repository (optional): ")
    force_refresh = input("Force refresh data? (y/n): ").lower() == 'y'
    
    if not user_question:
        user_question = None
    
    asyncio.run(main(repo_url, user_question, force_refresh))
