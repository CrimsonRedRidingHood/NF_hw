class SessionStorage:
    def __init__(self):
        self.d = {}
        self.idx = 1
    
    def __getitem__(self, key):
        if key not in self.d:
            self.d[key] = self.idx
            self.idx += 1
            return self.idx - 1
        else:
            return self.d[key]
