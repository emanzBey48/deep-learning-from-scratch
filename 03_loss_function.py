def loss(grck,thmn):
    fark=(grck - thmn)
    lloss=fark**2
    print (lloss)
    return lloss
gr=int(input("gercek"))
th=int(input("tahmin"))
w=2.3
th=th*w
loss(gr,th)
