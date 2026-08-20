import matplotlib.pyplot as plt
def loss(grck,thmn):
    fark=(grck - thmn)
    lloss=fark**2
    return lloss


input=5.2
degr=4
w=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0]
lloss=[]
for i in range (len(w)):
    thmn=input*w[i]
    lloss.append(loss(degr,thmn))

print(lloss)



plt.plot(w, lloss)
plt.show()
