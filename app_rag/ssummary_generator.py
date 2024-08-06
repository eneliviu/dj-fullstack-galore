import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv,  find_dotenv
from langchain_postgres import PGVector


def generate_story(words):
    '''
        Call the OpenAI API to generate the story
    '''
    response = get_short_story(words)
    # Format and return the response
    return format_response(response)


def get_short_story(words):
    '''
        Construct the system prompt
    '''
    system_prompt = f"""You are a short story generator.
    Write a short story using the following words: {words}.
    Do not go beyond one paragraph."""
    # Make the API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": system_prompt
        }],
        temperature=0.8,
        max_tokens=1000
    )

    # Return the API response
    return response


def format_response(response):
    '''
        Extract the generated story from the response
    '''
    story = response.choices[0].message.content
    # Remove any unwanted text or formatting
    story = story.strip()
    # Return the formatted story
    return story
