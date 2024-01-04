from dotenv import load_dotenv
import time
from openai_assistant import OpenAIAssistant

def ask_question(question):
    print("*"*30)
    print("Next question")
    # create user prompt
    assistant.submit_user_prompt(user_prompt=question, wait_for_completion=True)

    messages = assistant.message_history
    for message in messages:
        print(message)

if __name__ == '__main__':

    load_dotenv()
    assistant = OpenAIAssistant()

    # load a file
    assistant.add_file_to_assistant(file_path="./files/better_with_bacon.txt")

    # create assistant
    print("create assistant")
    assistant_instructions = """
    You answer questions about the band called "better with bacon" based on the documents available in your knowledge base attached to this assistant.
    Be brief in your answers and do not mention the file in your answers.
    """
    assistant.create_assistant(name="Better with Bacon Assistant",
                                tools=['retrieval'],
                               include_files=True,
                               instructions=assistant_instructions)

    # create user prompt
    ask_question("What kind of music does BwB play?")

    ask_question("Where is the band from?")

    ask_question("Who are the members in the band?")

    ask_question("Who was the original bass player?")

    ask_question("Where did he go?")

    ask_question("How long has the band been together?")

    ask_question("Who plays the keyboards in the band?")

    ask_question("What does BwB stand for?")

    ask_question("Is anything better than bacon?")

    print("delete files")
    assistant.delete_files()

    print("delete assistant")
    resp = assistant.delete_assistant()
    print(resp)
