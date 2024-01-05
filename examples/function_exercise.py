import json

from dotenv import load_dotenv
import time
from openai_assistant import OpenAIAssistant, FunctionDefinition, FunctionParameter


def president_home_state(president, state_options):
    '''
    INPUTS:
        president str = The string name of a US President
        state_options list[str] = A list of potential states that the President was born in, with one of them being correct!

    OUTPUTS:
        response str = The response the user chose as the correct birthplace state.

    '''
    print("Hello! Let's test your knowledge of the home states of US Presidents!")
    print(f"In what state was this president born: {president}\n")

    for num, option in enumerate(state_options):
        print('\n')
        print(f"Definition #{num} is: {option}")

    print('\n')
    num_choice = input("What is your choice? (Return the single number option)")

    return state_options[int(num_choice)]

class QuizOpenAiAssistant(OpenAIAssistant):
    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)

    def handle_requires_action(self, tool_call, function_name, function_args) -> dict:
        print(tool_call)
        print(function_name)
        print(function_args)
        args = json.loads(function_args)

        result = president_home_state(args['president'], args['state_options'])
        print(result)

        return result

if __name__ == '__main__':
    load_dotenv()
    assistant = QuizOpenAiAssistant()

    try:
        function_definition = FunctionDefinition(
            name="president_home_state",
            description="Function that takes as parameters a US Presidents name, and a list of possible states that the US President was born in.  The function will return the selected home state",
            parameters=[
                FunctionParameter(
                    name="president",
                    description="The name of a randomly selected US President as a string",
                    type="string",
                    required=True
                ),
                FunctionParameter(
                    name="state_options",
                    description="A list of potential states that the President was born in, with one of them being the correct birthplace state",
                    type="array",
                    required=True,
                    array_items_type="string"
                )
            ]
        )

        assistant.add_function(function_definition)

        # funcs_json = assistant.create_function_definition_json()
        # for func_json in funcs_json:
        #     print(json.dumps(func_json, indent=2))

        # create assistant
        print("create assistant")
        assistant.create_assistant(name="US President Quiz Bot", model='gpt-4-1106-preview',
                                    tools=['function'],
                                   instructions="You help create a quiz where you give a US President and a list of birthplace states, where only one is the correct birthplace state of the president. Later you check if answers returned are correct."
                )

        # create user prompt
        user_prompt = "Create a new quiz question with a US President and a list of options for the home state of birth. Then I will reply with a single state and let me know if I got it right."
        assistant.submit_user_prompt(user_prompt=user_prompt, include_files=False)

        messages = assistant.poll_for_assistant_conversation()
        for message in messages:
            print(message)
    finally:
        print("delete assistant")
        resp = assistant.delete_assistant()
        print(resp)
