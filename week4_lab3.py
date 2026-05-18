### Using an existing function and converting it into a tool

from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper
# GoogleSerperAPIWrapper: Useful function in LangChain community

from langchain_core.tools import Tool  
# Tool: LangChain wrapper class for converting functions into tools

load_dotenv(override=True)

serper = GoogleSerperAPIWrapper()

# print(serper.run("What is the capital of Paris?"))

tool_search = Tool(
    name="search",
    func=serper.run,
    description="Useful for when you need more information from an online search"
)

print(tool_search.invoke("What is the capital of Paris?"))  # Trying out the tool LangChain way
