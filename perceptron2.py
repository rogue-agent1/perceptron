#!/usr/bin/env python3
"""perceptron2 - Perceptron learning algorithm."""
import sys, random
class Perceptron:
    def __init__(self, n_features, lr=0.1):
        self.w = [0.0]*n_features; self.b = 0.0; self.lr = lr
    def predict(self, x):
        return 1 if sum(wi*xi for wi,xi in zip(self.w,x))+self.b > 0 else 0
    def train(self, X, Y, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for x, y in zip(X, Y):
                pred = self.predict(x); err = y - pred
                if err != 0:
                    for j in range(len(self.w)): self.w[j] += self.lr * err * x[j]
                    self.b += self.lr * err; errors += 1
            if errors == 0: print(f"Converged at epoch {epoch}"); return
        print(f"Finished {epochs} epochs")
if __name__=="__main__":
    random.seed(42)
    X = [[random.gauss(-2,1), random.gauss(-2,1)] for _ in range(50)] + [[random.gauss(2,1), random.gauss(2,1)] for _ in range(50)]
    Y = [0]*50 + [1]*50
    p = Perceptron(2)
    p.train(X, Y)
    correct = sum(1 for x,y in zip(X,Y) if p.predict(x)==y)
    print(f"Accuracy: {correct}/100 = {correct}%")
