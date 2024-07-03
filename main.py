from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun

llm =ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.5,
    verbose=True,
    google_api_key="AIzaSyCPD-4v0Hklwzn_Xk4SnMQWLWtkx5DNvf4"
)

search_tool=DuckDuckGoSearchRun()


query=input("Hi! How can I help you today? ")


identifying_agent = Agent(
role='Identify the product, what does the customer want to know about the product and if the customer is facing any '
     'issue with the product',
goal=f""" Analyze the query: {query} from the customer and try and find out 
about what product is the customer asking about and what exactly does he wants to know about the product.
Also figure out the exact problem the customer is having with the product.
If you can't figure out the product skip it and continue forward.""",
backstory="""You are a master at understanding what a customer wants and understanding their problems. When they input
 you a query you are able to categorize it in a useful way 
and summarize the customer needs accordingly. You are experienced and can easily figure out.
You fully read the statement again and again until you find out the meaning of the query.""",
  verbose=True,
    llm=llm
)

research_agent = Agent(
role='Info Researcher Agent',
goal=f"""take in the query: {query} from the customer and with the help of the identifying agent
research on the internet about the product which has been asked by the customer in the query and also find
a viable solution to the problem which the customer is facing with the product.
Deeply research about what the customer wants to know about the product.
Also research on more general topics on which the product and the problem cannot be identified. 
You must come up with a solution always to the problem that the customer is facing or with the information
about the product which the customer has inquired about.
Let the customer know if not enough information has been provided.""",
backstory="""You are a master at understanding what needs to be researched on the internet and our an expert 
in coming up with great solutions to each of the problems""",
  verbose=True,
    llm=llm
)

writer_agent = Agent(
role='Response writer agent',
goal="""Understand the results of the research performed by the research agent and professionally draft a reply of the 
to the customer query in a polite and friendly manner.Also let the customer know if the information provided by him is
not enough.""",
backstory="""You are a master at responding to queries of customers and have quite a lot of experience
drafting and writing to customer issues""",
  verbose=True,
    llm=llm
)


task1 = Task(
  description='Help to identify the product and the what issue is the customer asking.'
              'Also find out what the the customers wants to inquire about.'
              'If you are unable to identify the product, no problem, move on',
  expected_output='Give a proper output about what is the product being talked about in the query, '
                  'what does the user want to know about that product and'
    'what exactly is the issue the customer is facing with the product.'
                  'Also there is no issue if you are unable to identify the product let it continue with the information it has.',
  agent=identifying_agent,
  tools=[search_tool],

)

task2 = Task(
  description='Thoroughly research the product and query being inquired about and also come up with a viable '
              'solution to any problem that the customer is facing. Also let the customer know if not enough'
              'is provided',
  expected_output='A well defined note about the product and a valid answer in bullets to the query'
                  'of the customer. Also a well defined list of solutions to the customer problems.'
                  'if you cant figure out what the problem or query is let the customer know not enough information'
                  'provided.',
  agent=research_agent,
tools =[search_tool],
)

task3 = Task(
  description="Write a friendly knowledgeable and respectable response to the customer query. Also let the customer "
              "know if the information provided by him is not enough ",
  expected_output='A well drafted reply with proper use of vocabulary and looks presentable',
  agent=writer_agent,
  context=[task1, task2]
)

crew=Crew(
    agents=[identifying_agent, research_agent, writer_agent],
    tasks=[task1, task2, task3],
    verbose=True,
    process=Process.sequential
)


results= crew.kickoff()
print("Crew work results:")
print(results)
