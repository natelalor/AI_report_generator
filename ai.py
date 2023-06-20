# June 19th, 2023
# Nate Lalor

import os
import openai

def main():
    print("Welcome to main")
    
    embed_data()


    openai.api_key = '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'

    response = openai.Completion.create(
        engine = 'davinci',  # Specify the language model to use, e.g., 'davinci' or 'curie'
        prompt = 'Once upon a time',
        max_tokens = 100  # Define the desired length of the generated text
    )

    generated_text = response.choices[0].text.strip()
    print(generated_text)









    # set up vector database and add all the embedded data




    # use langchain (or openAI's "function calling") to create a link maybe between vector database
    # and 
    
    # creation of database?




    # Your main code logic goes here
    # userin = input("Purpose of report: ")
    userin = "end of main"
    print(userin)


def embed_data():
    #embed data (through openai embed model)
    print("in embed_data")
    #first, connect to openai
    connect_openai()

# helper function to embed_data() to establish a connection
# to openai API

def connect_openai():
    #connect data here
    print("in connect_openai")

if __name__ == '__main__':
    main()
