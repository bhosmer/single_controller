from . import get_message_queue
import importlib.util
import sys

if __name__ == "__main__":
    q = get_message_queue()
    _, call = q.recv()
    filename, *rest = call.target.split(':', 1)
    if not rest:
        modulename, funcname = filename.rsplit('.', 1)
        module = importlib.import_module(modulename)
    else:
        spec = importlib.util.spec_from_file_location('__entry__', filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['__entry__'] = module
        funcname = rest[0]
    func = getattr(module, funcname)
    func(*call.args, **call.kwargs)
