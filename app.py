from pathlib import Path
from shiny import App, ui, render, reactive
from shiny.types import ImgData
import logging
import re
from repo_analysis import get_user_input, aiohttp, generate_visualizations_as_objects, analyze_repository, fetch_repository_data, PRE_DEFINED_QUESTIONS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the path to the image
DataFlowChart = Path(__file__).parent

# Define Shiny app UI
app_ui = ui.page_fluid(
    ui.page_fillable(
        ui.div(
            ui.div(
                ui.h1("GitHub Repository [AI] Analysis Dashboard", style="font-family: Georgia, serif; color: #000000; font-size: 4.0em; text-align: center; padding: 10px;"),
                ui.h2(
                    ui.HTML(
                        '<span style="font-family: \'Helvetica Neue\', Arial, sans-serif; font-size: 2.0em; text-align: center; font-weight: 400;">'
                        '<span style="color: #6A0DAD;">lonely</span>'
                        '<span style="color: #444444;">octopus</span> - '
                        '<span style="font-family: \'Courier New\', Courier, monospace; color: #333333; font-weight: bold; text-shadow: 2px 2px 4px #aaa;">Posit</span>'
                        '<span style="font-family: \'Arial\', sans-serif; color: #000000; font-weight: normal;"> collaboration</span>'
                        '</span>'
                    ),
                    style="padding: 10px; text-align: center;"
                ),
                ui.markdown("This project is a GitHub repository finder and analytical tool that uses the GitHub API and the Gemini 1.5 Flash LLM to recommend and analyze repositories. The backend code is designed to fetch repository data, perform initial analysis, and generate visualizations to provide a comprehensive overview of the repository's health and activity."),
                class_="header-wrapper"
            ),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h3("**Input GitHub Repository Info**", class_="info-title"),
                    ui.input_text("repo_url", "Enter GitHub Repository URL (E.g. https://github.com/username/repo):", value=""),
                    ui.div(
                        ui.h4(ui.tags.u("Pre-defined and Custom Questions"), style="font-size: 1.2em;"),
                        ui.input_select(
                            "pre_defined_question",
                            "Select a pre-defined question (optional):",
                            choices=["None"] + [f"{i+1}: {q}" for i, q in enumerate(PRE_DEFINED_QUESTIONS)],
                            width='63%',
                        ),
                        style="margin-bottom: 10px;",  # Adjust the value as needed
                        class_="card-input",
                    ),
                    ui.input_text_area("user_question", "Enter your custom question :", placeholder="Type your question here..."),
                    ui.input_action_button("submit", "Analyze Repository", style="background-color: #ea8148; color: white;"),
                    class_="card-input",
                    width=300
                ),
                ui.navset_pill(
                    ui.nav_panel("GitHub Repository Analysis",
                        ui.div(
                            ui.h2("GitHub Repository Analysis", class_="section-header"),
                            ui.div(
                                ui.output_ui("loading_indicator"),
                                ui.output_ui("analysis_status"),
                                ui.output_ui("pre_defined_question_result"),
                                ui.output_ui("user_question_result"),
                                class_="plot-card1 area-content"
                            ),
                            class_="area-content"
                        )
                    ),
                    ui.nav_panel("Visualization",
                        ui.div(
                            ui.h2("Visualization", class_="section-header"),
                            ui.div(
                                ui.div(
                                    ui.h5("Commit Activity"),
                                    ui.output_plot("commit_activity_plot"),
                                    class_="plot-card2"
                                ),
                                ui.div(
                                    ui.h5("Top Contributors"),
                                    ui.output_plot("top_contributors_plot"),
                                    class_="plot-card2"
                                ),
                                class_="plots-wrapper"
                            ),
                            class_="area-content"
                        )
                    ),
                    ui.nav_panel("Data Flowchart",
                        ui.div(
                            ui.h2("Data FlowChart", class_="section-header"),
                            ui.output_image("image", width= '100%' , height= '50%'),
                            class_="image-container"
                        )
                    ),
                ),
                class_="layout-sidebar-no-padding"
            ),
            class_="background-wrapper"
        )
    ),
    ui.tags.style(
        """
        /* Styles for the overall UI, areas, and cards */
        body {
            background-color: #011936; /* Outer background color */
            color: #24292e; /* Text color */
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            margin: 0;
        }
        .background-wrapper {
            background-color: #ede0d4 !important; /* Matching background color */
            padding: 15px;
            border-radius: 15px; /* Curved square brackets */
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* Add subtle shadow */
        }
        .header-wrapper {
            background-color: #ffd6a5; /* Header background color */
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 0px; /* Adds space below the header section */
        }
        .card-input {
            background-color: #faedcd !important; /* Sidebar background color with !important */
            color: #000000; /* Black text */
            height: 100%; /* Makes the card fill the available height */
            width: 250px; /* Increase sidebar width */
            overflow-wrap: break-word; /* Ensures long words break to fit */
            word-break: break-word; /* Breaks long words or URLs */
            white-space: normal; /* Allows text to wrap */
        }
        .area-background {
            background-color: #faedcd; /* Same color as sidebar */
            padding: 20px; /* Adjust padding as needed */
            border-radius: 15px; /* Curved edges for consistency */
            margin-top: 0px; /* Minimize gap between top border and columns */
        }
        .layout-sidebar-no-padding {
            padding: 0 !important; /* Remove any padding around the sidebar */
            margin: 0 !important; /* Remove any margin around the sidebar */
            border-radius: 15px; /* Curved edges for consistency */
        }
        .area-content {
            min-height: 500px; /* Ensure the area content has a minimum height to accommodate larger images */
            background-color: #faedcd; /* Same color as sidebar */
            border: 2px solid #ea8148; /* Border color and thickness */
            color: #000000; /* Black text */
            padding: 15px;
            border-radius: 15px; /* Curved edges for consistency */
            margin-bottom: 20px; /* Adds some spacing between the areas */
        }
        .plot-card1 {
            background-color: #f8edeb; /* Internal background color */
            border: 2px solid #fcd5ce; /* Border color and thickness */
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Add subtle shadow */
            margin-bottom: 5px; /* Adds spacing between plot cards */
        }
        .plot-card2 {
            background-color: #f8edeb; /* Internal background color */
            border: 2px solid #fcd5ce; /* Border color and thickness */
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Add subtle shadow */
            margin-bottom: 5px; /* Adds spacing between plot cards */
            text-decoration: underline #000000; /* Underlines the text */
        }
        .btn-primary {
            background-color: #2d72d9; /* GitHub Blue */
            border-color: #2d72d9; /* GitHub Blue */
            color: #ffffff; /* White text */
        }
        .btn-primary:hover {
            background-color: #1d4ed8; /* Darker Blue */
            border-color: #1d4ed8; /* Darker Blue */
        }
        .output-text {
            color: #24292e; /* Dark Gray */
        }
        .border {
            border-color: #d0d7de; /* Light Border Gray */
        }
        .image-container  {
            width: 100%;  /* Full width of its parent container */
            max-width: 1000px; /* Removes any maximum width restriction */
            height: 100%;  /* Adjust height automatically based on the aspect ratio */
            overflow: visible;  /* Ensures that the content is not clipped */
            text-align: center;  /* Center-align the content within the container */
            margin: auto; /* Centers the container if smaller than the parent */
            padding: 10px; /* Optional: Adds some space around the image */
        }
        .plots-wrapper {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); /* Flexible grid layout */
            gap: 20px; /* Spacing between grid items */
        }
        .section-header {
            background-color: #fbe5da; /* Header background color */
            color: #000000; /* Text color */
            padding: 15px 20px; /* Padding for space around text */
            border-radius: 15px; /* Rounded corners for header */
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for better separation */
            font-size: 1.4em; /* Adjust font size as needed */
            font-weight: bold; /* Bold text */
            margin-bottom: 20px; /* Space below each header */
        }
        .info-title {
            font-weight: bold; /* Make the title bold */
            font-size: 1.5em; /* Increase the font size */
            color: #ea8148; /* Use a distinct color for emphasis */
            padding: 10px 0; /* Add padding for spacing */
            text-align: center; /* Center the title */
            border-bottom: 2px solid #ea8148; /* Add an underline for emphasis */
        }
        .info-card {
            background-color: #faedcd !important; /* Matching background color */
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* Add subtle shadow */
        }
        .navset_pill {
            min-height: 600px; /* Provide enough vertical space for the image */
        }
        """
    )
)


def clean_text(html_text):
    # Replace <br> tags with line breaks
    text = html_text.replace("<br>", "\n")
    
    # Add a newline after each full stop, except if there's already a newline
    text = re.sub(r'(?<!\n)\.\s+', '.\n\n', text)
    
    # Fix spacing and line breaks around bullet points and bold formatting
    text = re.sub(r'\*\s*\*\*\s*', '* **', text)  # Proper formatting of list items with bold text
    text = re.sub(r'\*\*\s*\*\s*', '** ', text)   # Fix cases where bold and list markers are misaligned
    text = re.sub(r'\*\*\s+', '** ', text)        # Remove extra spaces after bold markers
    
    # Remove any other unwanted HTML tags (if any)
    text = re.sub(r'<[^>]+>', '', text).strip()
    
    # Normalize spaces around punctuation
    text = re.sub(r'\s+([.,!?;])', r'\1', text)
    
    # Remove extra newlines to avoid excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    
    return text


def server(input, output, session):
    analysis_data = reactive.Value({})
    is_loading = reactive.Value(False)

    def extract_repo_name(url):
        match = re.match(r"https://github\.com/([^/]+/[^/]+)/?.*", url)
        if match:
            return match.group(1)
        return None

    @render.image
    def image():
        # Always return the image source
        return {"src": DataFlowChart / "DataFlowChart.png", "width": "auto", "height": "700px"}
    pass
    
    @reactive.Effect
    @reactive.event(input.submit)
    async def update_analysis():
        is_loading.set(True)
        repo_url = input.repo_url().strip()
        user_question = input.user_question().strip()
        repo_full_name = get_user_input(repo_url)

        # Fetch data from the repository using the backend function
        async with aiohttp.ClientSession() as session:
            data = await fetch_repository_data(repo_full_name, session)

        if not data:
            return "Failed to retrieve repository data. Please check the repository name."

        # Generate visualizations
        visualizations = generate_visualizations_as_objects(data)


        if not repo_url:
            ui.notification_show("Repository URL cannot be empty", type="error")
            is_loading.set(False)
            return

        repo_name = extract_repo_name(repo_url)
        if not repo_name:
            ui.notification_show("Invalid GitHub URL format", type="error")
            is_loading.set(False)
            return

        try:
            logger.info(f"Starting analysis for repository: {repo_name}")
            result = await analyze_repository(repo_name, user_question if user_question else None)
            logger.info("Analysis completed. Processing results...")
            if "error" in result:
                logger.error(f"Analysis error: {result['error']}")
                ui.notification_show(f"Analysis error: {result['error']}", type="error")
            else:
                logger.info("Setting analysis data...")
                analysis_data.set(result)
                logger.info("Analysis data set successfully")
                ui.notification_show("Analysis complete!", type="message")
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {str(e)}")
            ui.notification_show(f"An unexpected error occurred: {str(e)}", type="error")
        finally:
            logger.info("Analysis process completed")
            is_loading.set(False)

        return visualizations      
    

    @output
    @render.ui
    def analysis_status():
        data = analysis_data.get()
        if not data:
            return ui.p("Click 'Analyze Repository' to start")

        logger.info("Rendering analysis status")
        return ui.div(
            ui.h5("Analysis Complete"),
            ui.p("View results for pre-defined or custom questions below.")
        )
        
    @output
    @render.ui
    def pre_defined_question_result():
        data = analysis_data.get()
        selected_question = input.pre_defined_question()
        
        if not data or selected_question == "None":
            return ui.div()
        
        logger.info(f"Rendering result for pre-defined question: {selected_question}")
        pre_defined_analysis = data.get("pre_defined_analysis", [])
        question_index = int(selected_question.split(":")[0]) - 1
        
        if 0 <= question_index < len(pre_defined_analysis):
            qa = pre_defined_analysis[question_index]
            cleaned_answer1 = clean_text(qa['answer'])
            return ui.div(
                ui.h3("Pre-defined Question Analysis"),
                ui.p(ui.strong("Question: "), qa['question']),
                ui.p(ui.strong("Answer: "), cleaned_answer1)
            )
        else:
            logger.warning(f"Selected question not found: {selected_question}")
            return ui.p("Selected question not found in the analysis results.")

    @output
    @render.ui
    def user_question_result():
        data = analysis_data.get()
        user_question = input.user_question().strip()
        
        if not data or not user_question or "user_question_analysis" not in data:
            return ui.div()
        cleaned_answer2 = clean_text(data["user_question_analysis"])
        logger.info("Rendering result for user-defined question")
        return ui.div(
            ui.h3("Custom Question Analysis"),
            ui.p(ui.strong("Question: "), user_question),
            ui.div(ui.strong("Answer: "), cleaned_answer2)
        )
    @output
    @render.ui
    def loading_indicator():
        if is_loading.get():
            logger.info("Displaying loading indicator")
            return ui.div(
                ui.p("Analysis in progress..."),
                ui.progress(value=0.5, striped=True, animated=True)
            )
        return ui.div()
    
    @output
    @render.plot
    def commit_activity_plot():
        data = analysis_data.get()
        if data and "visualizations" in data:
            fig = data["visualizations"].get("commit_activity")
            if fig:
                return fig
        return None

    @output
    @render.plot
    def top_contributors_plot():
        data = analysis_data.get()
        if data and "visualizations" in data:
            fig = data["visualizations"].get("top_contributors")
            if fig:
                return fig
        return None
    


# Create the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
