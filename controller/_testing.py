import os

# code used for testing but useful to have importable (e.g. can refer to remote functions)

def do_bogus_tensor_work(x, y, fail_rank=None):
    if fail_rank is not None and int(os.environ["RANK"]) != fail_rank:
        return x
    return x @ y
