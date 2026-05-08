# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from __future__ import annotations

from typing import Dict, Tuple


def get_world_descr(
    world_graph,
    agent_uid=0,
    include_room_name=False,
    add_state_info=False,
    centralized=False,
):
    """
    Builds a string description of the environment from the world graph for single step planners.

    :param world_graph: The world graph representing the environment.
    :return: A string description of the environment, including rooms and their furniture, objects held by the agent, and locations of objects in the house.
    """
    ## house description -- rooms and their furniture list
    furn_room = world_graph.group_furniture_by_room()
    house_info = ""
    for k, v in furn_room.items():
        furn_names = [furn.name for furn in v]
        all_furn = ", ".join(furn_names)
        house_info += k + ": " + all_furn + "\n"

    all_furniture = world_graph.get_all_furnitures()
    furn_with_faucets = [
        fur for fur in all_furniture if "faucet" in fur.properties.get("components", [])
    ]
    faucet_info = "The following furnitures have a faucet: " + ", ".join(
        [fur.name for fur in furn_with_faucets]
    )

    objs_info = get_objects_descr(
        world_graph, agent_uid, include_room_name, add_state_info, centralized
    )
    return f"Furniture:\n{house_info}\n{faucet_info}\nObjects:\n{objs_info}"


def state_dict_to_string(state_dict):
    """
    Transforms a state dictionary into a human-readable string.

    :param state_dict: A dictionary of states, e.g., {'is_clean': False, 'is_powered_on': True}
    :return: A string describing the states, e.g., "is not clean and is powered on"
    """
    state_strings = []

    for state, value in state_dict.items():
        if state.startswith("is_"):
            state = state[3:]  # Remove 'is_' prefix
        state = state.replace("_", " ")
        state_strings.append(f"{state}: {value}")
    return ", ".join(state_strings)


def get_objects_descr(
    world_graph,
    agent_uid=0,
    include_room_name=False,
    add_state_info=False,
    centralized=False,
):
    """
    Builds a string description of objects in the environment.

    :param world_graph: The world graph representing the environment.
    :return: A string description of objects, including their locations.
    """

    # lazy import to avoid importing habitat module if not necessary
    from habitat_llm.world_model.entity import Room

    all_objs = world_graph.get_all_objects()
    if not all_objs:
        return "No objects found yet"
    else:
        obj_strings = []
        for obj in all_objs:
            obj_info = ""
            rooms_path = world_graph.find_path(root_node=obj, end_node_types=[Room])
            if rooms_path is None:
                room_name = "an unknown room"
            else:
                rooms = [x for x in rooms_path if isinstance(x, Room)]
                if len(rooms) == 0:
                    room_name = "an unknown room"
                else:
                    if len(rooms) > 1:
                        raise ValueError(
                            f"Multiple rooms detected for object {obj.name}"
                        )
                    room_name = rooms[0].name
            holding_agent = world_graph.get_agent_holding_object(obj)
            agent_name = f"agent_{agent_uid}"
            if (
                not centralized
                and holding_agent is not None
                and holding_agent.name == agent_name
            ):
                obj_info += obj.name + ": held by the agent"
            elif not centralized and holding_agent is not None:
                obj_info += obj.name + f": held by {holding_agent.name}"
            elif centralized and holding_agent is not None:
                obj_info += obj.name + f": held by {holding_agent.name}"
            else:
                furn_node = world_graph.find_furniture_for_object(obj)
                furn_name = "unknown" if furn_node is None else furn_node.name
                if include_room_name:
                    obj_info += obj.name + ": " + furn_name + " in " + room_name
                else:
                    obj_info += obj.name + ": " + furn_name
            if (add_state_info) and ("states" in obj.properties):
                state_string = state_dict_to_string(obj.properties["states"])
                if len(state_string) > 0:
                    obj_info += ". States: " + state_string
            obj_strings.append(obj_info)
        return "\n".join(obj_strings)


def get_rearranged_objects_descr(
    obj_descr_t_1,
    obj_descr_t,
):
    """
    Builds a string description of objects that were updated by latest agent actions execution

    :param object description string at t-1 and at t
    :return: A string description of objects that are updated by agent actions, including their updated locations.
    """

    objs_list_t_1 = obj_descr_t_1.split("\n")
    objs_list_t = obj_descr_t.split("\n")
    updated_objs = []

    objs_t_1 = any(":" in s for s in objs_list_t_1)
    objs_t = any(":" in s for s in objs_list_t)

    if not objs_t_1 and objs_t:
        return "\n".join(objs_list_t)

    if len(objs_list_t) == len(objs_list_t_1):
        for obj_id, obj in enumerate(objs_list_t):
            if obj != objs_list_t_1[obj_id]:
                updated_objs.append(obj)
    else:
        updated_objs = [obj for obj in objs_list_t if obj not in objs_list_t_1]

    return "\n".join(updated_objs)


def has_valid_square_brackets(input_string):
    return "[" in input_string and "]" in input_string


def remove_non_alpha_left(input_string):
    """
    This method strips non alphabetical characters from the left part of the string until first alphabetical character is found.
    Useful to handle cases such as, '- Agent', '** Agent_' ' Agent_' etc. which are
    the result of LLM not following the correct syntax.
    """
    for i, char in enumerate(input_string):
        if char.isalpha():
            return input_string[i:]
    return ""


def actions_parser(
    agents, input_string, params=None
) -> Dict[int, Tuple[str, str, str]]:
    """
    Actions parser used by planners to convert LLM generation
    into a structured representation.
    """

    # Container to store parser output
    actions_dict: Dict[int, Tuple[str, str, str]] = {}

    # Split input string
    lines = input_string.strip().split("\n")

    for line in lines:
        line = line.strip()
        line = remove_non_alpha_left(line)
        if line.startswith("Agent") and ("_Action" in line):
            # Extract agent info and actions info
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue

            agent_id, action_info = parts[0].strip(), parts[1].strip()

            # Extracting the numerical part of the agent ID
            if "_" in agent_id:
                agent_id = int(agent_id.split("_")[1])
            else:
                agent_id_list = [int(i) for i in parts[0].split() if i.isdigit()]
                if len(agent_id_list) < 1:
                    continue
                agent_id = agent_id_list[0]

            # Make sure that agent uid is valid
            true_agent_ids = [agent.uid for agent in agents]
            if agent_id not in true_agent_ids:
                for true_agent_id in true_agent_ids:
                    actions_dict[true_agent_id] = (
                        None,
                        None,
                        f"Invalid Agent ID in Action directive. Only valid Agent IDs are {true_agent_ids}!",
                    )
                continue

            # Make syntax exception for Wait command
            if "Wait" in action_info:
                action_info = "Wait[]"

            # Add error message to indicate if the line does not have complete square brackets
            if not has_valid_square_brackets(action_info):
                actions_dict[agent_id] = (
                    None,
                    None,
                    'SyntaxError in Action directive. Opening "[" or closing "]" square bracket is missing!',
                )
                continue

            if params and not any(tool in action_info for tool in params["tool_list"]):
                actions_dict[agent_id] = (
                    None,
                    None,
                    "This tool/action is invalid for your agent. No action will be assigned to the agent.",
                )

            # Split the action info into action name and action arguments (inputs)
            action_name, action_input = action_info.split("[", 1)
            action_input = action_input.rstrip("]")

            # Set action_input to None if its empty
            # Useful in handling cases like Wait[], FindAgentAction[]
            if action_input == "":
                action_input = None

            actions_dict[agent_id] = (action_name, action_input, None)

    return actions_dict
