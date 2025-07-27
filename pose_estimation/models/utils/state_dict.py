# Author: Hiroyasu Akada

def fix_model_state_dict(state_dict, key="module."):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k

        name = name[len(key):]

        new_state_dict[name] = v
    return new_state_dict