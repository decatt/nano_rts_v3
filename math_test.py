n = 300000.0
p = 1.0/n
k = 3.3

expect = 1.0-(1.0-p)**(n*k)
print(expect)