#!/usr/bin/env python3
"""Perceptron & multi-layer perceptron — zero-dep."""
import math, random

def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))
def sigmoid_d(x): s=sigmoid(x); return s*(1-s)
def relu(x): return max(0,x)
def relu_d(x): return 1.0 if x>0 else 0.0

class Perceptron:
    def __init__(self,n_in,lr=0.1):
        self.w=[random.gauss(0,0.1) for _ in range(n_in)]
        self.b=0.0; self.lr=lr
    def predict(self,x): return 1 if sum(w*xi for w,xi in zip(self.w,x))+self.b>0 else 0
    def train(self,X,y,epochs=100):
        for _ in range(epochs):
            for xi,yi in zip(X,y):
                p=self.predict(xi); err=yi-p
                self.w=[w+self.lr*err*x for w,x in zip(self.w,xi)]
                self.b+=self.lr*err

class MLP:
    def __init__(self,layers,lr=0.5):
        self.lr=lr; self.layers=layers
        self.W=[]; self.B=[]
        for i in range(len(layers)-1):
            self.W.append([[random.gauss(0,0.5) for _ in range(layers[i])] for _ in range(layers[i+1])])
            self.B.append([0.0]*layers[i+1])
    def forward(self,x):
        acts=[x]; zs=[]
        for W,B in zip(self.W,self.B):
            z=[sum(w*a for w,a in zip(row,acts[-1]))+b for row,b in zip(W,B)]
            zs.append(z); acts.append([sigmoid(v) for v in z])
        return acts,zs
    def train(self,X,y,epochs=1000):
        for _ in range(epochs):
            for xi,yi in zip(X,y):
                acts,zs=self.forward(xi)
                if not isinstance(yi,list): yi=[yi]
                delta=[2*(a-t)*sigmoid_d(z) for a,t,z in zip(acts[-1],yi,zs[-1])]
                for l in range(len(self.W)-1,-1,-1):
                    new_delta=[0.0]*len(acts[l])
                    for j in range(len(self.W[l])):
                        for k in range(len(self.W[l][j])):
                            new_delta[k]+=self.W[l][j][k]*delta[j]
                            self.W[l][j][k]-=self.lr*delta[j]*acts[l][k]
                        self.B[l][j]-=self.lr*delta[j]
                    if l>0: delta=[nd*sigmoid_d(z) for nd,z in zip(new_delta,zs[l-1])]

if __name__=="__main__":
    p=Perceptron(2); p.train([[0,0],[0,1],[1,0],[1,1]],[0,0,0,1],100)
    print("AND perceptron:",[p.predict(x) for x in [[0,0],[0,1],[1,0],[1,1]]])
    mlp=MLP([2,4,1],lr=2.0); mlp.train([[0,0],[0,1],[1,0],[1,1]],[[0],[1],[1],[0]],5000)
    print("XOR MLP:",[f"{mlp.forward(x)[0][-1][0]:.3f}" for x in [[0,0],[0,1],[1,0],[1,1]]])
