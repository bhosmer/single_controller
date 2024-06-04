class Ref:
    def __init__(self, id: int):
        self.id = id

    def __repr__(self):
        return f'r{self.id}'

    def __reduce__(self):
        return Ref, (self.id,)

class Referenceable:
    def delete_ref(self, ref):
        raise NotImplementedError("no delete_ref method")

    def define_ref(self):
        raise NotImplementedError("undefined ref with no define_ref method")

    def __reduce_ex__(self, protocol):
        if self.ref is None:
            self.ref = self.define_ref()
        return Ref, (self.ref,)

    def __del__(self):
        if self.ref is not None:
            self.delete_ref(self.ref)
