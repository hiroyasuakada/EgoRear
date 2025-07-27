# Author: Hiroyasu Akada



def fix_model_state_dict(state_dict, rm_name="module.", key=None):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k

        if name.startswith(rm_name):
            name = name[len(rm_name):]
        else:
            name = name.replace(rm_name, "")

        if key:
            if name.startswith(key):
                new_state_dict[name] = v
        else:
            new_state_dict[name] = v

    return new_state_dict