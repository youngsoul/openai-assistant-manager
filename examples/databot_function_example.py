import json
from typing import List
from dotenv import load_dotenv
from databot.PyDatabot import databot_sensors
from openai_assistant import OpenAIAssistant, FunctionDefinition, FunctionParameter
import pandas as pd
import requests


def get_databot_values(sensor_names: List) -> str:
    try:
        print(f"Get values for: {sensor_names}")
        url = "http://localhost:8321/"
        response = requests.get(url)
        return json.dumps(response.json())
    except:
        return "There was an error trying to access the databot device.  Make sure it is turned on and running the webserver."

def get_databot_friendly_names() -> List:
    df = pd.DataFrame(data=databot_sensors.values()).sort_values(by="friendly_name")
    f_names = df['friendly_name'].to_list()
    return f_names



class DatabotOpenAiAssistant(OpenAIAssistant):
    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)

    def handle_requires_action(self, tool_call, function_name: str, function_args:str) -> str:
        output = None
        try:
            print(tool_call)
            print(function_name)
            print(function_args)
            args = json.loads(function_args)

            sensor_value = get_databot_values(args['sensor_names'])
            print(sensor_value)
            output = f"{sensor_value}"

        except:
            output = "unknown"

        return output

if __name__ == '__main__':
    load_dotenv()
    fnames = get_databot_friendly_names()
    print(fnames)
    assistant = DatabotOpenAiAssistant()

    try:
        function_definition = FunctionDefinition(
            name="get_databot_values",
            description="""Get sensor values from the databot.  If there are multiple sensor values, a list of sensor names can be provided.
                            This function can only provide information on the current values from the databot.  
                            This function CANNOT describe what the sensor is measuring.
                            """,
            parameters=[
                FunctionParameter(
                    name="sensor_names",
                    description="""List of the friendly human readable sensor value names.""",
                    type="string",
                    required=True,
                    enum_values=get_databot_friendly_names()
                )
            ]
        )

        assistant.add_function(function_definition)

        funcs_json = assistant.create_function_definition_json()
        for func_json in funcs_json:
            print(json.dumps(func_json, indent=2))

        # create assistant
        print("create assistant")
        assistant.create_assistant(name="Databot Assistant", model="gpt-3.5-turbo-1106",
                                    tools=['function'],
                                   instructions="You help answer questions about the databot sensor device and can call function to retrieve values from the databot."
                )

        # create user prompt
        user_prompt = "What is the current external temperature sensor value 2?"
        assistant.submit_user_prompt(user_prompt=user_prompt)

        messages = assistant.poll_for_assistant_conversation()
        for message in messages:
            print(message)
    finally:
        print("delete assistant")
        resp = assistant.delete_assistant()
        print(resp)

