# Shiny AI GitHub Repository Analyzer

## Description

This project is a comprehensive GitHub repository finder and analysis tool that integrates with the GitHub API and leverages AI capabilities such as the Gemini 1.5 Flash LLM. It provides a detailed analysis of repositories, including visualizations and insights. The application uses a Python-based frontend developed using the Shiny library and a robust backend that handles API requests, data fetching, and AI-powered analysis.

## Features

- **GitHub Integration**: Fetch repository data using the GitHub API.
- **AI-Enhanced Analysis**: Analyze repositories with AI models like Gemini 1.5 and LangChain.
- **Data Visualization**: Generate visualizations of repository metrics such as commit activity and top contributors.
- **Custom Queries**: Users can input custom questions for AI-powered responses.
- **User-Friendly Interface**: Developed with Shiny, providing an interactive and intuitive UI.

## Requirements

Ensure you have the following installed:

- Python 3.8 or higher
- Pip (Python package manager)
- GitHub Account and Personal Access Token
- Hugging Face Account and API Key
- Google AI Studio Account and Gemini API Key

Python Libraries:

- `shiny`
- `aiohttp`
- `requests`
- `matplotlib`
- `google-generativeai`
- `langchain`
- `github`
- `diskcache`
- `faiss`

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. **Set Up a Virtual Environment** (Optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```

3. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Environment Variables**:
    - Create a `.env` file in the root directory.
    - Add your GitHub token, Hugging Face API key, and Gemini API key to the `.env` file as follows:
      ```
      GITHUB_TOKEN=your_github_token_here
      HUGGINGFACE_API_KEY=your_huggingface_api_key_here
      GOOGLE_API_KEY=your_gemini_api_key_here
      ```

5. **Prepare the Backend**:
    - The backend is implemented in `repo_analysis.py`. This handles data fetching, AI analysis, and more.

6. **Prepare the Frontend**:
    - The frontend interface is implemented in `app.py` using the Shiny library. This script defines the layout and interactive components of the app.

## Usage

1. **Run the Application**:
    ```bash
    python app.py
    ```

2. **Access the Interface**:
    - Once the server starts, the Shiny app will be accessible in your web browser. You can input a GitHub repository URL, select or enter a custom question, and analyze the repository.

3. **Explore Results**:
    - Navigate through the different panels (`GitHub Repository Analysis`, `Visualization`, `Data Flowchart`) to explore the results of your analysis.

## Deployment

To deploy this application using ShinyApps.io or any other Shiny server, follow the deployment instructions [here](https://shiny.posit.co/py/docs/deploy.html)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure to update the documentation as necessary.
