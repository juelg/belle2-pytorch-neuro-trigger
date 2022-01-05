
def nested_dict(dic, *args, default=None):
    para = dic
    for arg in args:
        if arg in para:
            para = para[arg]
        else:
            return default
    return para
    