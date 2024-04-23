import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]


st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr NL2MongoDB🧑🏼‍💻")
st.markdown("### Welcome to the Lyzr NL2MongoDB!")
st.markdown("Convert natural language queries into MongoDB commands effortlessly with Lyzr NL2MongoDB !!!")

code = st.text_input("Enter your text: ",placeholder=f""""Type Natural Language Query here!""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def code_translation(code):
    translation_agent = Agent(
        role="MongoDB QUERY TRANSLATOR expert",
        prompt_persona=f"Your task is to convert the natural language user input to MongoDB query effectively."
    )

    prompt = f"""
    You are an Expert MONGODB QUERY TRANSLATOR. Your task is to INTERPRET natural language user input and CONVERT it into precise MongoDB queries. You MUST ensure that the resulting code accurately reflects the user's intent, is SYNTACTICALLY CORRECT.
    
    Here is how you should approach this TASK:

    1. LISTEN carefully to the USER'S INPUT and identify KEY TERMS such as collection names, field names, and desired operations (e.g., find, insert, update).

    2. BREAK DOWN the request into logical components that correspond to MongoDB query structure: filter criteria, projection, sorting order, etc.

    3. TRANSLATE these components into a MongoDB query using the appropriate syntax for commands and operators.

    4. VERIFY that each part of the query aligns with MongoDB's best practices for efficiency and performance.

    5. PRESENT the query back to the user with a step-by-step EXPLANATION of each segment to ensure they grasp how it fulfills their request.


    """

    translation_task = Task(
        name="MongoDB Translation",
        model=open_ai_text_completion_model,
        agent=translation_agent,
        instructions=prompt,
        default_input=code,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return translation_task 
   
if st.button("Convert"):
    solution = code_translation(code)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent Optimize your code. For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)