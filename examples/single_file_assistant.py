from dotenv import load_dotenv
import time
from openai_assistant import OpenAIAssistant


if __name__ == '__main__':

    load_dotenv()
    assistant = OpenAIAssistant()

    # load a file
    assistant.add_file_to_assistant(file_path="./files/Wonka Chocolate Facility Rules.pdf")

    # incase there are other files in openai, go ahead and get them.  only new ones will be added to this
    # instance of the assistant
    files = assistant.get_assistant_files(refresh_from_openai=True)
    for file in files:
        print(file)
        assistant.add_file_id_to_assistant(file_id=file.file_id)

    print("Uploading file to OpenAI.  Give it some time to process the uploaded file.")
    for i in range(10,0,-1):
        print(i)
        time.sleep(1) # give it time to process the file

    # create assistant
    print("create assistant")
    assistant.create_assistant(name="Rules Explainer",
                                tools=['retrieval'],
                               instructions="You answer rules based on your knowledge base of files")

    # create user prompt
    assistant.submit_user_prompt(user_prompt="Can I bring my cat to the Wonka Chocolate Facility?", include_files=True)

    messages = assistant.poll_for_assistant_conversation()
    for message in messages:
        print(message)



    print("delete files")
    assistant.delete_files()

    print("delete assistant")
    resp = assistant.delete_assistant()
    print(resp)
