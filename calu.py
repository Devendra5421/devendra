class Calculator:
    def __init__(self,a=0,b=30) -> None:
        self.a=a
        self.b=b
        
    def addition(self) -> int:
        return self.a+self.b
    def substraction(self) -> float:
        return self.a-self.b 
    def multiplication(self) -> int:
        return self.a*self.b
    def divison(self) -> float:
        return self.a/self.b 
    
cal = Calculator(a=4,b=4)
print(f"Adding the values:{cal.addition()}")
print(f"substraction the values:{cal.substraction()}")
print(f"multiplication the values:{cal.multiplication()}")
print(f"divison the values:{cal.divison()}")