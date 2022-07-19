class Registry(dict):

    def register(self, o):
        self[o.__name__] = o


MODEL_REGISTRY = Registry()
