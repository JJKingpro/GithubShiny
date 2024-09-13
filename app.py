from pathlib import Path
from shiny import App, ui, render, reactive
from shiny.types import ImgData
import logging
import re
from repo_analysis import (
    aiohttp, 
    analyze_repository, 
    PRE_DEFINED_QUESTIONS,
    extract_repo_name
)

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
                ui.h1("Shiny AI GitHub Repository Analyzer", style="font-family: Georgia, serif; color: #000000; font-size: 4.0em; text-align: center; padding: 10px;"),
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
                    ui.h4("Input GitHub Repository Info", class_="info-title"),
                    ui.input_text("repo_url", "Enter GitHub Repository URL (E.g. https://github.com/username/repo) and click on the ☀️Analyze Repository☀️ button below", value=""),
                    ui.input_action_button("submit", "Analyze Repository", class_="btn-primary"),
                    ui.hr(style="margin-top: 5px; margin-bottom: 5px;"),  # Adjust the margins and style as needed
                    ui.h4("Select or Enter your questions here", class_="info-title"),
                    ui.div(
                        ui.h4(ui.tags.u("Pre-defined Questions"), style="font-size: 1.2em;"),
                        ui.input_select(
                            "pre_defined_question",
                            "Select a pre-defined question (optional):",
                            choices=["None"] + [f"{i+1}: {q}" for i, q in enumerate(PRE_DEFINED_QUESTIONS)],
                            width='100%',
                        ),
                        style="margin-bottom: 10px;",
                        class_="card-input",
                    ),
                    ui.div(
                        ui.h4(ui.tags.u("Custom Questions"), style="font-size: 1.2em;"),
                        ui.input_text_area(
                            "user_question",
                            "Enter your custom question:", 
                            placeholder="Type your question here and press the Enter key to submit..."
                        ),
                        ui.input_action_button("submit_question", "Submit Question", style="display: none;"),
                        style="margin-bottom: 10px; padding: 10px; background-color: white; border: 1px solid lightgrey; border-radius: 5px;",
                        class_="card-input",
                        width=350
                    ),
                    width=400
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
                                    ui.output_plot("top_contributors_plot"),
                                    class_="plot-card2"
                                ),
                                ui.div(
                                    ui.h5("Top Contributors"),
                                    ui.output_plot("commit_activity_plot"),
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
                            ui.output_image("image", width='100%', height='50%'),
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
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f8edeb;
        color: #24292e;
        line-height: 1.6;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        margin: 0;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        margin-bottom: 0.5em;
    }

    h1 { 
        font-size: 2.0em; 
        font-family: Georgia, serif; 
        color: #000000; 
        text-align: center; 
        padding: 10px;
    }
    h2 { font-size: 1.5em; }
    h3 { font-size: 1.5em; }
    
    /* Layout */
    .background-wrapper {
        background-color: #faedcd;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .header-wrapper {
        background-color: #ffd6a5;
        padding: 40px;
        border-radius: 15px;
        margin-bottom: 30px;
    }

    .layout-sidebar-no-padding {
        padding: 0 !important;
        margin: 0 !important;
    }

    .area-content {
        min-height: 500px;
        background-color: #faedcd;
        border: 2px solid #ea8148;
        color: #000000;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    }

    /* Cards */
    .card-input, .plot-card1, .plot-card2, .info-card {
        background-color: #ffffff;
        border: 1px solid #fcd5ce;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 25px;
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }

    .card-input:hover, .plot-card1:hover, .plot-card2:hover, .info-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }

    /* Buttons */
    .btn-primary {
        background-color: #ea8148;
        border-color: #ea8148;
        color: #ffffff;
        padding: 12px 24px;
        border-radius: 5px;
        transition: background-color 0.3s ease, transform 0.3s ease;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .btn-primary:hover {
        background-color: #d67339;
        border-color: #d67339;
        transform: translateY(-2px);
    }

    /* Section Headers */
    .section-header {
        background-color: #fbe5da;
        color: #000000;
        padding: 20px 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 25px;
    }

    .info-title {
        font-weight: bold;
        font-size: 1.6em;
        color: #ea8148;
        padding: 15px 0;
        text-align: center;
        border-bottom: 2px solid #ea8148;
    }

    /* Plots and Images */
    .plots-wrapper {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
    }

    .image-container {
        width: 100%;
        max-width: 1000px;
        height: auto;
        overflow: visible;
        text-align: center;
        margin: auto;
        padding: 10px;
    }

    .navset_pill {
        min-height: 600px;
    }

    /* Loading Indicator */
    .loading-indicator {
        text-align: center;
        padding: 30px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .loading-indicator .progress {
        height: 15px;
        border-radius: 7px;
    }

    /* Additional Styles */
    .output-text {
        color: #24292e;
    }

    .border {
        border-color: #d0d7de;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .background-wrapper {
            padding: 15px;
        }

        .header-wrapper {
            padding: 20px;
        }

        .card-input, .plot-card1, .plot-card2, .info-card {
            padding: 15px;
        }

        .card-input {
            width: 100%;
        }
        
        .plots-wrapper {
            grid-template-columns: 1fr;
        }
    }
    """
    ),
    ui.tags.script("""
    document.getElementById('user_question').addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            document.getElementById('submit_question').click();
        }
    });
    """)
)

def server(input, output, session):
    analysis_data = reactive.Value({})
    is_loading = reactive.Value(False)
    repo_analyzed = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.submit, input.submit_question)
    async def update_analysis():
        is_loading.set(True)
        repo_url = input.repo_url().strip()
        user_question = input.user_question().strip()

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
            
            if not repo_analyzed.get():
                # Initial analysis
                result = await analyze_repository(repo_url, user_question if user_question else None)
                repo_analyzed.set(True)
            else:
                # Subsequent questions
                result = await analyze_repository(repo_url, user_question, fetch_data=False)
            
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
            logger.error(f"An unexpected error occurred: {str(e)}")
            ui.notification_show(f"An unexpected error occurred: {str(e)}", type="error")
        finally:
            logger.info("Analysis process completed")
            is_loading.set(False)

    @output
    @render.ui
    def analysis_status():
        data = analysis_data.get()
        if not data:
            return ui.p(
                ui.span("Click on the ☀️Analyze Repository☀️ button to begin analyzing the contents of the Github repository"), 
                ui.br(),
                ui.span("(The button is located on the sidebar to the left of this message)")
            )
        return ui.div()  # Return an empty div instead of the "Analysis Complete" message


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
            return ui.div(
                ui.p(ui.strong("Pre-defined Question:"), style="font-size: 1.2em;"),
                ui.p(qa['question'], style="font-size: 1.2em; margin-bottom: 20px;"),
                ui.p(ui.strong("Answer:"), style="font-size: 1.1em;"),
                ui.markdown(qa['answer'])
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
        
        logger.info("Rendering result for user-defined question")
        return ui.div(
            ui.p(ui.strong("Custom Question:"), style="font-size: 1.2em;"),
            ui.p(user_question, style="font-size: 1.2em; margin-bottom: 20px;"),
            ui.p(ui.strong("Answer:"), style="font-size: 1.1em;"),
            ui.markdown(data["user_question_analysis"])
        )
    
    @output
    @render.ui
    def user_question_result():
        data = analysis_data.get()
        user_question = input.user_question().strip()
        
        if not data or not user_question or "user_question_analysis" not in data:
            return ui.div()
        
        logger.info("Rendering result for user-defined question")
        return ui.div(
            ui.p(ui.strong("Custom Question:"), style="font-size: 1.2em;"),
            ui.p(user_question, style="font-size: 1.2em; margin-bottom: 20px;"),
            ui.p(ui.strong("Answer:"), style="font-size: 1.1em;"),
            ui.markdown(data["user_question_analysis"])
        )
    
    @output
    @render.ui
    def loading_indicator():
        if is_loading.get():
            logger.info("Displaying loading indicator")
            return ui.div(
                ui.h4("Analysis in progress..."),
                ui.progress(value=0.5, striped=True, animated=True),
                class_="loading-indicator"
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

    @output
    @render.image
    def image():
        return {"src": DataFlowChart / "DataFlowChart.png", "width": "auto", "height": "700px"}

# Create the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()






