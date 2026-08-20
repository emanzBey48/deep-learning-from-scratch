a= float(input("ilk input "))
b= float(input("ikinci input"))
c= float(input("üçüncü input"))
e=2.71828
aw=3.65
bw=4.2
cw=-7.1
a2w=4.4
b2w=8.9
c2w=-6.7
a3w=0.7
b3w=5.6
c3w=1.1

toplam1=((a*aw)+(b*bw)+(c*cw)-10)
toplam2=((a*a2w)+(b*b2w)+(c*c2w)-19)
toplam3=((a*a3w)+(b*b3w)+(c*c3w)-5)

sigmoid1=1/(1+(e**(-toplam1)))
sigmoid2=1/(1+(e**(-toplam2)))
sigmoid3=1/(1+(e**(-toplam3)))

print (sigmoid1, "", sigmoid2,"",sigmoid3)