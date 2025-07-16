#pip install langchain_openai
#pip install pydantic
#pip install langchain-core
#pip install gradio
#pip install mcp
#pip install langchain_mcp_adapters
#pip install langgraph
import asyncio
import gradio as gr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from enum import Enum
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

#Enum (Enumeration)
#sabit değerlerin anlamlı adlarla gruplandırılmasını sağlar.
class Category(str, Enum):
    get_weather_temperature = 'The weather temperature of a city'
    capitalize_each_word = 'Capitalize each word'
    count_each_word = 'Count each word'
    other = 'Other'

#BaseModel
#veri doğrulama (data validation) ve modelleme için kullanılır.

class ResultModel(BaseModel):
    result: Category

#result alanının mutlaka Category türünde olması gerektiğini söyler.

candidate_labels = [e.value for e in Category]


#OpenAI API Key ortam değişkenine atanmalı
os.environ['OPENAI_API_KEY'] = ''


system_prompt = (
    "You are a function-calling AI agent.\n"
    "The function you need to call will be selected according to the user's input for you and it will be informed to you\n"
    "Only provide the outcome of the function, do not make any additonal comments or statements\n"
)

#chat_model = ChatOllama(model="llama3.2").with_config({"system_prompt": system_prompt})
chat_model = ChatOpenAI(model="gpt-3.5-turbo",
 temperature=0).with_config({
    "system_prompt": system_prompt
})


### === AGENT WORKFLOW === ###
async def agent_pipeline(user_input):
    parser = PydanticOutputParser(pydantic_object=ResultModel)
    intent_model = ChatOpenAI(model="gpt-3.5-turbo",
    temperature=0)
    prompt = (
    "You are an AI assistant that classifies user queries into a category.\n"
    "Choose one of the following categories:\n"
    f"{', '.join(candidate_labels)}\n\n"
    f"User query: \"{user_input}\"\n\n"
    f"Respond **only** in this exact JSON format:\n\n"
    "{\"result\": \"<one of: The weather temperature of a city, Capitalize each word, Count each word, Other>\"}"
    )

    try:
        response = intent_model.invoke(prompt)
        parsed = parser.parse(response.content)

        category = parsed.result
        print(category)

        if category == Category.other:
            return "Out of scope as function!"

        if category == Category.count_each_word:
            top_label = "count_words"
        elif category == Category.capitalize_each_word:
            top_label = "capitalize_words"
        elif category == Category.get_weather_temperature:
            top_label = "get_weather_temperature"
        else:
            return "Out of scope as function!"

        server_params = StdioServerParameters(command="python", args=["test_tools_mcp_server.py"])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                selected_tools = [tool for tool in tools if tool.name == top_label]
                agent = create_react_agent(chat_model, selected_tools)
                msg = {"messages": user_input}
                response = await agent.ainvoke(msg)

                for m in response["messages"]:
                    if m.type == "tool":
                        return m.content
                return "No function output returned."

    except Exception as e:
        return f"Agent error: {e}"


### === GRADIO ARAYÜZÜ === ###
with gr.Blocks(title="LDAP + AI Agent Chat") as demo:

    with gr.Column(visible=True) as chat_area:
        gr.Markdown("##MCP AI Agent")
        user_query = gr.Textbox(label="Your Query: (for example: weather query, capitalize words or count words)")
        agent_response = gr.Textbox(label="Agent Output")
        query_btn = gr.Button("Run MCP Agent")

    async def handle_query_wrapper(query):
        return await agent_pipeline(query)

    query_btn.click(
        fn=handle_query_wrapper,
        inputs=[user_query],
        outputs=agent_response
    )

demo.launch()
