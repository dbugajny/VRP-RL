from collections import namedtuple


class State(namedtuple("State", ("capacity", "demands", "mask"))):
    pass
