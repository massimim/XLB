import warp as wp

class Backend:
    def __init__(self):
        wp.init()
        self.type = 'cpu'
        self.ids = [0]

    def forEachDevice(self, foo):
        def getDeviceName(id):
            if self.type == 'cpu':
                return 'cpu'
            return 'gpu' + str(id)

        for id in self.ids:
            wp.set_device(getDeviceName(id))
            foo(self.type, id)
