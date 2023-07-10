# June 2023
# Nate Lalor

# imports
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain import OpenAI
import openai
import os

  # do you have external data that you want use?
      # a chance for user to give dataset ?
  
  # take user input (dataset)
  # embed
  # check length
  # if long, summarize it (langchain?)
  # 
  # with user_focus and user_context and summary, make a table of contents
    #prompt: "If there is context to reference, reference the context.
    #             Context: [not available] OR IF THERE IS SUMMARY, [summary]"
  # based on the user_focus and user_context, AND the dataset:
      # fill out the sections of the table of contents
      # embed section header
      # query that embedding in database and pull top 10?? similar-est results
      # then:
        # prompt: "using the following context from our database, and using the focus of user_focus:
        #           #1; wguohqwgouqwhrgwoqhu
        #           #2; qoughqwoguhwegowuehg
        #           #3; oqigfhqwopifqjwqpif
        # 
        #           write out the [i] section of the table of contents"
  



def main():

    # initializes API specifics
    llm, openai_ = llm_initialization()

    # the starting "prompt" for the ChatCompletion
    messages=[
            {"role": "system", "content": "You are a Top-tier Management Consultant with an MBA and Outstanding Expertise in the Field, Renowned for Major Contributions to International Business Strategy and Consultancy. You want to make a detailed report for a client, but you need to know the purpose first. You will ask each of the following questions one at a time, and wait for the client to respond before proceeding. 1. First you will ask what is the purpose of this deliverable? 2. Then you ask clarifying questions to get any additional context to help you do a better job. Then, using those first 2 questions, you will make a table of contents for a report on the client's subject. Make sure the table of contents has exactly 6 sections with subsections. Start:"},
            {"role": "assistant", "content": "Let us start at step 1, about the purpose of the deliverable. Ready for my first question?"},

        ]
    
    # for iteration of while loop
    counter = 1

    # to capture the conversation contents
    user_input_array = []
    ai_response_array = []

    # the conversation
    print("Ready to begin?")
    while counter != 4:
      user_input = input()
      messages = update_chat(messages, "user", str(user_input))
      user_input_array.append(user_input)
      model_response = get_chatgpt_response(messages)
      print(model_response)
      ai_response_array.append(model_response)
      messages = update_chat(messages, "assistant", model_response)
      counter += 1
    
    # now that conversation is over, harness the user_inputs for their focus of deliverable
    # as well as their further context
    user_focus = user_input_array[1]
    user_context = user_input_array[2]

    # also the table of contents
    table_of_contents_outline = ai_response_array[2]

    final_report = []
    counter = 1
    print("Please wait while we process your report...")
    while counter != 12:
      # create the prompt to send to new completion

      completion_prompt = (
          """I want to write a report on {0}. Specifically, I want to focus on: 
                          {1}. 
                          
                          Based on the following table of contents:
                          {2}
                          I want you to write out subsection {3} of the report, following the table of contents and the topic and focus of the report.
                          Start: 
                          """.format(user_focus, user_context, table_of_contents_outline, counter)
      )

      # minimize any other prose 

      # TRIMMING STRING ATTEMPTS - DID NOT WORK!

      # temp = completion_prompt.strip()
      # first = temp.find("\n")
      # last = temp.rfind("\n")
      # trimmed = ""
      # print("FIRST:", first)
      # print("LAST:", last)
      # for x in range(first+1, first+50):
      #    trimmed = trimmed + completion_prompt[x]
      #    print(trimmed)
      # print("TRIMMED!!!!!!!!!!!!!!!!!!")
      # print(trimmed)

      response = openai.Completion.create(
        model="text-davinci-003",
        prompt=completion_prompt,
        temperature=1,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )

      # print(response["choices"][0]["text"])

      final_report.append(response["choices"][0]["text"])
      # final_report = final_report + "\n"
      counter += 1
    
    # final report
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FINAL REPORT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for x in final_report:
       print(x)
#       text_to_txt(final_report)


# make chat completions gpt4
# or experiment with gpt3 with larger context window

# mess around with completion prompt








# def text_to_txt(transcription_text):
#   with open('focused_report.txt', 'w') as f:
#     f.write(str(transcription_text))
#   f.close()
#   # user info
#   print("Successfully created 'focused_report.txt' in current directory.")

# finish that ^^^^ conversation with user, (maybe shorten it? stop after step 2 and dont show user table of contents? maybe try this first)
# then after, make a Completion (not chat) (????? maybe???) where you paste the table of contents and ask it to write out part one.
# (then we need more info for part 1 --- how do we do that?)


# a helper function to append to messages as well as return
# the AI's response
def get_chatgpt_response(messages):
  response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=messages
)
  return  response['choices'][0]['message']['content']

# a helper function to append to messages
def update_chat(messages, role, content):
  messages.append({"role": role, "content": content})
  return messages
    
    








# # ----------------------------------------------------------- #


# # initializes the llm and OPENAI_API_KEY variables,
# # basically preparing to use OpenAI's API
def llm_initialization():
    # LLM setup
    OPENAI_API_KEY = "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"
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    return llm, OPENAI_API_KEY


# # langchain_execution takes all the important information in: lg_docs,
# # the set of split up text, llm, the language learning model (OpenAI),
# # and the user's purpose/focus for this deliverable. It sets up prompts
# # then makes a call to map_reduce chain through Langchain which produces
# # our nice result
# def langchain_execution(llm, lg_docs, user_input):
#     # map prompt : given to produce each chunk
#     map_prompt = (
#         """
#                  Write a concise summary focusing on %s:
#                  "{text}"
#                  CONCISE SUMMARY:
#                  """
#         % user_input
#     )

#     # make a PromptTemplate object using the s-string above
#     map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

#     # combine prompt : the prompt it gives to "summarize", or how to sum up content into a final product.
#     combine_prompt = (
#         """Given the extracted content, create a detailed and thorough 3 paragraph report. 
#                         The report should use the following extracted content and focus the content towards %s.
                        

#                                 EXTRACTED CONTENT:
#                                 {text}
#                                 YOUR REPORT:
#                                 """
#         % user_input
#     )

#     # make a PromptTemplate object using the s-string above
#     combine_prompt_template = PromptTemplate(
#         template=combine_prompt, input_variables=["text"]
#     )

#     # line up all the data to our chain variable before the run execution below
#     chain = load_summarize_chain(
#         llm=llm,
#         chain_type="map_reduce",
#         map_prompt=map_prompt_template,
#         combine_prompt=combine_prompt_template,
#         verbose=False,
#     )

#     # execute the chain on the new split up doc
#     summarized_log_doc = chain.run(lg_docs)

#     # return the result
#     return summarized_log_doc


if __name__ == "__main__":
    main()
