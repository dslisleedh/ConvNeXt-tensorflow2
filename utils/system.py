import gin


class Runner:
    def __init__(self, f):
        self.f = f

    def __call__(self):
        try:
            self.f()
            gin.clear_config()
        except Exception as e:
            print(e)
            gin.clear_config()
            raise e
