# these are remote functions we want accessible in our test suite
import os

def do_bogus_tensor_work(x, y, fail_rank=None):
    if fail_rank is not None and int(os.environ["RANK"]) != fail_rank:
        return x
    return x @ y
