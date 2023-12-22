from dotenv import load_dotenv
import time
from openai_assistant import OpenAIAssistant


if __name__ == '__main__':

    load_dotenv()
    assistant = OpenAIAssistant()

    print("create assistant")
    assistant.create_assistant(name="Mortgage Bot",
                                tools=['code_interpreter'],
                               instructions="You use Python code to help answer questions about mortgage and interest payments.")


    time.sleep(3)

    assistant.submit_user_prompt(user_prompt="I want to buy a house that costs $2.1 Milion on a 30-yr fixed loan at 7.8% interest. What will my monthly payments be?")

    time.sleep(2)
    response = assistant.get_run_status()
    print(response)
    while response != "completed":
        time.sleep(1)
        response = assistant.get_run_status()
        print(response)

    conversation = assistant.get_assistant_conversation()
    for message in conversation:
        print(message)

    print("*"*20)

    assistant.submit_user_prompt(user_prompt="What if I put a down payment of $200k on the house, how would that change the monthly payments?")

    time.sleep(2)
    response = assistant.get_run_status()
    print(response)
    while response != "completed":
        time.sleep(1)
        response = assistant.get_run_status()
        print(response)

    conversation = assistant.get_assistant_conversation()
    for message in conversation:
        print(message)



    print("delete assistant")
    resp = assistant.delete_assistant()
    print(resp)

