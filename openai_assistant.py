import os
import threading
import json
import time
from dataclasses import dataclass, field
from typing import Literal, List

from openai import OpenAI
from openai.types.beta import AssistantDeleted
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.thread_message import ThreadMessage
from openai.types.file_object import FileObject


@dataclass
class FunctionParameter:
    name: str
    type: str
    description: str
    required: bool
    enum_values: List[str] = field(default_factory=list)
    # array_items_types is only needed if the type is 'array'
    array_items_type: str = field(default=None)

@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: List[FunctionParameter] = field(default_factory=list)


@dataclass
class AssistantFile:
    file_id: str
    file_path: str = field(default=None)
    file_object: FileObject = field(default=None)


class AssistantThreadMessage:

    def __init__(self, thread_message: ThreadMessage):
        self.thread_message = thread_message

    def get_thread_message(self) -> ThreadMessage:
        return self.thread_message

    def get_id(self) -> str:
        return self.thread_message.id

    def get_file_id(self) -> str:
        if self.get_type() == "image_file":
            return self.thread_message.content[0].image_file.file_id
        else:
            return ""

    def get_type(self) -> str:
        return self.thread_message.content[0].type

    def get_message(self) -> List[str]:
        if self.get_type() == "text":
            return [self.thread_message.content[0].text.value]
        elif self.get_type() == "image_file":
            return [self.thread_message.content[0].image_file.file_id, self.thread_message.content[1].text.value]

    def get_message_annotations(self) -> List[str]:
        if self.get_type() == "text":
            return [self.thread_message.content[0].text.annotations]
        return []


    def __str__(self):
        if self.get_type() == "text":
            return self.thread_message.content[0].text.value
        elif self.get_type() == "image_file":
            return f"FileID: {self.thread_message.content[0].image_file.file_id}\n{self.thread_message.content[1].text.value}"
        else:
            return f"Unknown type: {self.get_type()}, MessageID: {self.thread_message.content[0]}"

class OpenAIAssistant:
    def __init__(self, api_key: str=None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.openai_client = OpenAI(api_key=api_key)

        self.assistant = None
        self.thread = None
        self.run = None
        self.files: List[AssistantFile] = []
        self.functions: List[FunctionDefinition] = []

    def create_function_definition_json(self) -> List[dict]:
        result = []
        for function_definition in self.functions:
            function_template = {
                "type": "function",
                "function": {
                    "name": function_definition.name,
                    "description": function_definition.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }

            for parameter in function_definition.parameters:
                function_template["function"]["parameters"]["properties"][parameter.name] = {
                    "type": parameter.type,
                    "description": parameter.description
                }
                if parameter.enum_values:
                    function_template["function"]["parameters"]["properties"][parameter.name][
                        'enum'] = parameter.enum_values

                if parameter.type == 'array':
                    function_template["function"]["parameters"]["properties"][parameter.name]['items'] = {
                        'type': parameter.array_items_type}
                if parameter.required:
                    function_template["function"]["parameters"]["required"].append(parameter.name)

            # result.append(json.dumps(function_template))
            result.append(function_template)

        return result

    def add_function(self, function: FunctionDefinition):
        self.functions.append(function)

    def delete_file(self, file_id: str):
        self.openai_client.files.delete(file_id=file_id)

    def delete_files(self):
        for file in self.files:
            self.delete_file(file.file_id)

    def get_file_content(self, file_id: str):
        myfile = self.openai_client.files.content(file_id=file_id)
        return myfile.content

    def add_file_id_to_assistant(self, file_id: str):
        """
        Sometimes all you have is the file_id so inject that in to the assistant.

        :param file_id:
        :return:
        """
        for file in self.files:
            if file.file_id == file_id:
                break
        else:
            self.files.append(AssistantFile(
                file_id=file_id
            ))

    def add_file_to_assistant(self, file_path: str) -> str:

        file = self.openai_client.files.create(file=open(file_path, "rb"), purpose="assistants")
        self.files.append(AssistantFile(
            file_id=file.id,
            file_path=file_path,
            file_object=file
        ))
        return file.id

    def get_assistant_files(self, refresh_from_openai: bool = False) -> List[AssistantFile]:
        """
        :param refresh_from_openai: A boolean value indicating whether to retrieve all files from OpenAI or only the locally stored ones. Default is False.
        :return: A list of AssistantFile objects.

        This method retrieves the assistant files either from OpenAI or from the locally stored files. If `all_from_openai` is set to True, it will fetch all files from OpenAI and return them
        * as a list of AssistantFile objects. If `all_from_openai` is set to False, it will return the locally stored files.
        """
        if refresh_from_openai:
            resp: List[AssistantFile] = []
            files = self.openai_client.files.list()
            for file in files:
                resp.append(AssistantFile(
                    file_id=file.id,
                    file_path=file.filename,
                    file_object=file
                ))
            return resp
        else:
            return self.files

    def create_assistant(self, name: str, instructions: str,
                         tools: List[Literal["retrieval", "code_interpreter", "function"]] = ["retrieval"],
                         model: Literal["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"] = "gpt-3.5-turbo-1106",
                         include_files: bool = False):
        tool_list = []
        for tool in tools:
            if tool == "function":
                function_json_obj_list = self.create_function_definition_json()
                for function in function_json_obj_list:
                    tool_list.append(function)
            else:
                tool_list.append({
                    "type": tool
                })
        file_ids = []
        if include_files:
            for file in self.files:
                file_ids.append(file.file_id)

        self.assistant = self.openai_client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tool_list,
            model=model,
            file_ids=file_ids
        )

    def delete_assistant(self) -> AssistantDeleted:
        response = self.openai_client.beta.assistants.delete(self.assistant.id)
        return response

    def _create_conversation(self):
        if self.thread is None:
            self.thread = self.openai_client.beta.threads.create()

    def _add_user_prompt(self, user_prompt: str, include_files: bool = False) -> ThreadMessage:
        self._create_conversation()
        file_ids = []
        if include_files:
            for file in self.files:
                file_ids.append(file.file_id)

        message = self.openai_client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=user_prompt,
            file_ids=file_ids
        )
        return message

    def submit_user_prompt(self, user_prompt: str, instructions: str = "", include_files: bool = False) -> Run:
        self._add_user_prompt(user_prompt, include_files)

        if include_files:
            instructions = instructions + f"\n Use files with ids: {','.join([f.file_id for f in self.files])}"

        self.run = self.openai_client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=instructions
        )
        return self.run

    def get_run(self) -> Run:
        the_run = self.openai_client.beta.threads.runs.retrieve(
            thread_id=self.thread.id,
            run_id=self.run.id
        )
        return the_run


    def get_assistant_conversation(self) -> List[AssistantThreadMessage]:
        messages = self.openai_client.beta.threads.messages.list(
            thread_id=self.thread.id
        )
        thread_messages = []
        for thread_message in messages.data[::-1]:
            thread_messages.append(AssistantThreadMessage(thread_message))

        return thread_messages

    def poll_for_assistant_conversation(self, max_wait_time: int = 60) -> List[AssistantThreadMessage]:
        the_run = self.get_run()
        response = the_run.status
        max_timeout = max_wait_time

        # we cannot just look for != 'complete' because it might be 'requires_action'
        # when we need to call a function
        while response == "queued" or response == "in_progress" or response == "requires_action":
            the_run = self.get_run()
            response = the_run.status

            self.run_response_callback(the_run=the_run)

            if response == "requires_action":
                tool_outputs = []
                tool_calls = the_run.required_action.submit_tool_outputs.tool_calls
                for tool_call in tool_calls:
                    tool_output = self.handle_requires_action(tool_call)
                    tool_outputs.append(tool_output)
                self.openai_client.beta.threads.runs.submit_tool_outputs(thread_id=self.thread.id,
                                                                         run_id=the_run.id, tool_outputs=tool_outputs)
                time.sleep(1)
                the_run = self.get_run()
                response = the_run.status
                continue

            time.sleep(1)
            max_timeout -= 1
            if max_timeout == 0:
                break
        print(response)

        if max_timeout == 0:
            return ["Timeout occurred. Please try again"]
        else:
            conversation = self.get_assistant_conversation()
            messages = []
            for message in conversation:
                messages.append(message)
            return messages

    def handle_requires_action(self, tool_call) -> dict:
        raise NotImplementedError("handle_requires_action is not implemented.  Expected to be implemented in base classes")

    def run_response_callback(self, the_run: Run):
        print(f"The Run Status is: {the_run.status}")
