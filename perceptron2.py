#!/usr/bin/env python3
"""Perceptron and Adaline (adaptive linear neuron)."""
import sys, random

class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr, self.epochs = lr, epochs
    def fit(self, X, y):
        self.w = [0.0]*len(X[0]); self.b = 0.0
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = 1 if sum(w*x for w,x in zip(self.w, xi))+self.b >= 0 else 0
                err = yi - pred
                if err != 0:
                    for j in range(len(self.w)): self.w[j] += self.lr*err*xi[j]
                    self.b += self.lr*err; errors += 1
            if errors == 0: break
    def predict(self, X):
        return [1 if sum(w*x for w,x in zip(self.w,xi))+self.b >= 0 else 0 for xi in X]

class Adaline:
    def __init__(self, lr=0.001, epochs=100):
        self.lr, self.epochs = lr, epochs
    def fit(self, X, y):
        self.w = [0.0]*len(X[0]); self.b = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                output = sum(w*x for w,x in zip(self.w, xi)) + self.b
                err = yi - output
                for j in range(len(self.w)): self.w[j] += self.lr*err*xi[j]
                self.b += self.lr*err
    def predict(self, X):
        return [1 if sum(w*x for w,x in zip(self.w,xi))+self.b >= 0.5 else 0 for xi in X]

def main():
    X = [[0,0],[0,1],[1,0],[1,1]]; y_or = [0,1,1,1]
    p = Perceptron(lr=0.1, epochs=10); p.fit(X, y_or)
    print(f"Perceptron OR: {p.predict(X)}")
    a = Adaline(lr=0.1, epochs=20); a.fit(X, y_or)
    print(f"Adaline OR:    {a.predict(X)}")

if __name__ == "__main__": main()
