from supervisor import get_message_queue
from typing import NamedTuple

class Reply(NamedTuple):
    a: int
    b: int
    x: int


def reply_hello(a, b, x):
    q = get_message_queue()
    q.send(Reply(a, b, x))

class Mapper:
    def map(self, items):
        return sum(x*2 for x in items)
    def reduce(self, items):
        return sum(items)
    def finish(self, result):
        return result
