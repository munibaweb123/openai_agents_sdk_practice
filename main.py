class Hello:
    def __init__(self,name:str):
        self.name = name

    def say_hello(self):
        print(f"Hello, {self.name}!")

class HelloHuman(Hello):
    def say_hello(self):
        return super().say_hello()
    
print(Hello.mro())
print(HelloHuman.mro())