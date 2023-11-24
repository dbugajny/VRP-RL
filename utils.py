from collections import namedtuple


class State(namedtuple("State", ("current_capacity", "demands", "mask"))):
    pass
