class Player:
    def __init__(self, id, resource):
        self.id = id
        self.resource = resource
        self.opponent_id = 1 - id
        self.using_resource = 0