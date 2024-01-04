# OpenAI Assistant Manager 

This repo contains a class to help manage the interaction with the OpenAI assistant API.

Many of these examples were inspired by taking the Udemy course:

```text
https://www.udemy.com/course/openai-assistants-with-openai-python-api/
```
I would recommend this course.

The examples used have been reworked to utilize the `OpenAIAssistant` class from this repo.

## Manager Class

see: `openai_assistant.py`

## Retrieval Examples

### Better with Bacon band

File: `examples/bwb_assistant.py`

Create a retrieval assistant using a text document about the band `Better with Bacon` and ask questions about the band.

Before running the assistant, go to chatGPT and verify that it does not know anything about the band.  Then
run the assistant and see that it can pull information from the uploaded document.

### Single File Assistant

File: `examples/single_file_assistant.py`

Create a retrieval assistant that uses a single file.  Much like the BwB example

### Multiple File Assistant

File: `examples/multiple_file_assistant.py`

Create a retrieval assistant that used mulitple uploaded files.

## Functions

### Databot

File: `examples/databot_function_example.py`

This example uses the databot sensor device along with the OpenAI assistant to read realtime values from the databot device.

This example assumes you are running the databot webserver from the `databot-py` repo.

### President Quiz

File: `examples/function_excerise`

Example is from the Udemy course.  This example is a little prone to some 'odd' behavior.

## Code Retrieval

### Mortgage Bot

File: `examples/code_interpreter_assistant.py`

### Stock Visualizer

File: `examples/file_code_interpreter_assistant.py`
