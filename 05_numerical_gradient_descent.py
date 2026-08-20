def loss(grck,thmn):
    fark=(grck - thmn)
    lloss=fark**2
    return lloss


gr=int(input("gercek"))
thinp=int(input("tahmin"))
w=4.0
th=thinp*w
adim=0.1
h=0.001
for i in range(20):
    th=thinp*w
    w1=loss(gr,th)
    wh=w+h
    th=thinp*wh
    w2=loss(gr,th)
    diff=(w2-w1)/h
    print (i," ",w1," ",diff," ",w)
    w-=adim*diff
