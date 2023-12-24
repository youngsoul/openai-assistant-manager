import json
from typing import List
from openai_assistant import FunctionDefinition, FunctionParameter


def _create_function_definition_json(function_definitions: List[FunctionDefinition]) -> List[str]:
    result = []
    for function_definition in function_definitions:
        function_template = {
            "type": "function",
            "function": {
                "name": function_definition.name,
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
                function_template["function"]["parameters"]["properties"][parameter.name]['enum'] = parameter.enum_values

            if parameter.type == 'array':
                function_template["function"]["parameters"]["properties"][parameter.name]['items'] = {
                    'type': parameter.array_items_type}
            if parameter.required:
                function_template["function"]["parameters"]["required"].append(parameter.name)

        result.append(json.dumps(function_template))

    return result


def test_function_to_json():
    function_definition = FunctionDefinition(
        name="president_home_state",
        description="Function that takes as parameters a US Presidents name, and a list of possible states that the US President was born in.  The function will return the selected home state",
        parameters=[
            FunctionParameter(
                name="president",
                description="The name of the US President as a string",
                type="string",
                required=True
            ),
            FunctionParameter(
                name="state_options",
                description="A list of potential states that the President was born in, with one of them being the correct birthplace state",
                type="array",
                required=True,
                array_items_type="string"
            ),
            FunctionParameter(
                name="temperature_format",
                description="the temperature format to use for the temperature value.",
                type="string",
                required=True,
                enum_values=["celsius", "fahrenheit"]
            ),

        ]
    )

    result = _create_function_definition_json([function_definition])
    assert result is not None
    expected_result = ['{"type": "function", "function": {"name": "president_home_state", "parameters": {"type": "object", "properties": {"president": {"type": "string", "description": "The name of the US President as a string"}, "state_options": {"type": "array", "description": "A list of potential states that the President was born in, with one of them being the correct birthplace state", "items": {"type": "string"}}, "temperature_format": {"type": "string", "description": "the temperature format to use for the temperature value.", "enum": ["celsius", "fahrenheit"]}}, "required": ["president", "state_options", "temperature_format"]}}}']
    assert result == expected_result
