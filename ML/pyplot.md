---
layout: page
title: ABC of Pyplot
---





# 2d plots


We need to import ```pyplot```:

```python
import matplotlib.pyplot as plt
```

The structure is: a ```figure``` as a canvas which contains ```axes```. 

The plotting can be done by using the 'state based interface' and call

```python
plt.plot(x,y)
plt.imshow(img)
```

There is no explicit creation of a ```figure``` or ```axes```. Nowadays it is common to do as follows, showing the structure explicitluy, as follows

```python
fig, ax = plt.subplots(2,2,figsize=(6,6))
```

It will create a canvas with a $$2\times 2$$ table of plots (axes). The latter have a structure of an array. To plot we assign to each ```ax``` what we want to plot. For instance, for a line, an image, or a list of points we would use correspondingly

```python
ax.plot(x,y)
```

```python
ax.imshow(img)
```
and

```python
ax.scatter()
```

To actually plot we finish with

```python
plt.show()
```

There are different parameters to handle: 

```python
ax.set_title(label)
ax.axis('off')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Plot's name")
```
etc.


Here is a complete working example

```python
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(1,100)

fig, ax = plt.subplots(2,2)
ax[0,0].plot(x,x)
ax[0,0].set_title("Plot 1")
ax[0,1].plot(x,x**2)
ax[0,1].set_title("Plot 2")
ax[1,0].plot(x,x**3)
ax[1,0].set_title("Plot 3")
ax[1,1].plot(x,x**4)
ax[1,1].set_title("Plot 4")
plt.show()
```





# 3d plots


In this case on top of that we we need to import
```python
from mpl_toolkits.mplot3d import axes3d
```

and we have to use an explicit construction

```python
fig = plt.figure(figsize=(6,6))
```

instead of

<span style="color: red;">WRONG!!</span> ```fig, ax = plt.subplots(2,2,figsize=(6,6))```

since the latter creates explicitly 2d axes. 

As the next step we add plots explicitly

```python
ax = fig.add_subplot(projection='3d')
```

We could also make it in one shot

```python
ax = plt.figure(figsize=(6,6)).add_subplot(projection='3d')
```

For multiple plots we would do
```python
ax = figure.add_subplot(ncols,nrows,pos,projection='3d')
```
with ```ncols, nrows, pos``` being the number of columns, rows, and the position of the plot (counted left to right, top to bottom)

As before, there are various self-evident methods

```python
ax.plot(x, y, z) # For lines
ax.scatter(x, y, z) # For a list of points
ax.plot_surface(X, Y, Z)
ax.plot_wireframe(X, Y, Z)
```

A complete working example

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

x=np.linspace(-1,1,100)
y=np.linspace(-1,1,100)

X, Y = np.meshgrid(x,y)

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1,projection='3d')
ax4 = fig.add_subplot(2,2,4,projection='3d')

ax1.plot_wireframe(X,Y,X**2+Y**2,alpha=0.3) # Transparency 0.3
ax4.plot_wireframe(X,Y,X**2-Y**2,alpha=0.3) # Transparency 0.3

plt.show()
```