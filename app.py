import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

st.title("Gradient and Direction of Steepest Ascent")

st.write("This app visualises the gradient of a function f(x, y).")

func_input = st.text_input("Enter function f(x, y):", "x**2 + y**2")

x0 = st.slider("x₀", -5.0, 5.0, 1.0)
y0 = st.slider("y₀", -5.0, 5.0, 1.0)

x, y = sp.symbols('x y')
f = sp.sympify(func_input)

fx = sp.diff(f, x)
fy = sp.diff(f, y)

grad_x = fx.subs({x: x0, y: y0})
grad_y = fy.subs({x: x0, y: y0})

st.latex(r"\nabla f = (" + str(grad_x) + ", " + str(grad_y) + ")")

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = sp.lambdify((x, y), f, 'numpy')(X, Y)

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.update_layout(title="3D Surface Plot")

st.plotly_chart(fig)
# ----- Contour Plot -----
st.subheader("Contour Plot")

fig2 = go.Figure(
    data=go.Contour(
        x=X[0],
        y=Y[:, 0],
        z=Z,
        contours=dict(showlabels=True),
        colorscale='Viridis'
    )
)

# Gradient arrow
fig2.add_annotation(
    x=x0, y=y0,
    ax=x0 + float(grad_x),
    ay=y0 + float(grad_y),
    arrowhead=3,
    arrowwidth=2,
    arrowcolor="red"
)

st.plotly_chart(fig2)
