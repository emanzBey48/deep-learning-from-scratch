a= float(input("ilk input "))
b= float(input("ikinci input"))
c= float(input("üçüncü input"))
e=2.71828
aw=3.65
bw=4.2
cw=-7.1

toplam=((a*aw)+(b*bw)+(c*cw)-10)


sigmoid=1/(1+(e**(-toplam)))



print (sigmoid)