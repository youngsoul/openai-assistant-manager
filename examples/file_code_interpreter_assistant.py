from dotenv import load_dotenv
import time
from openai_assistant import OpenAIAssistant
import random, string

if __name__ == '__main__':

    load_dotenv()
    assistant = OpenAIAssistant()

    # load a file
    assistant.add_file_to_assistant(file_path="./files/SP500_Prices_5Year.csv")

    #
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
    assistant.create_assistant(name="Stock Visualizer",
                                tools=['retrieval', 'code_interpreter'],
                               instructions="You use code and files to help visualize stock data")

    print("Submit user prompt")
    assistant.submit_user_prompt(user_prompt="Can you create a plot of the historical adjusted closing price of the SP500?", include_files=True)

    print("Poll for response")
    messages = assistant.poll_for_assistant_conversation()
    for message in messages:
        if message.get_type() == "image_file":
            x = ''.join(
                random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(6))

            with open(f"./generated/test_image_{x}.png", "wb") as f:
                f.write(assistant.get_file_content(message.get_file_id()))

        print(message)

    print("delete files")
    assistant.delete_files()

    print("delete assistant")
    resp = assistant.delete_assistant()
    print(resp)


