import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

# =========================
# Title and description
# =========================
st.title("Gradient and Direction of Steepest Ascent")
st.write(
    "This application visualises the gradient of a function of two variables "
    "and shows the direction of steepest ascent using surface and contour plots."
)

# =========================
# User inputs
# =========================
func_input = st.text_input("Enter function f(x, y):", "x**2 + y**2")

x0 = st.slider("x₀ value", -5.0, 5.0, 1.0)
y0 = st.slider("y₀ value", -5.0, 5.0, 1.0)

# =========================
# Symbolic computation
# =========================
x, y = sp.symbols('x y')
f = sp.sympify(func_input)

fx = sp.diff(f, x)
fy = sp.diff(f, y)

grad_x = float(fx.subs({x: x0, y: y0}))
grad_y = float(fy.subs({x: x0, y: y0}))

st.subheader("Gradient at the selected point")
st.latex(r"\nabla f(x_0, y_0) = (" + str(round(grad_x, 3)) + ", " + str(round(grad_y, 3)) + ")")

# =========================
# Numerical grid
# =========================
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = sp.lambdify((x, y), f, "numpy")(X, Y)

# =========================
# 3D Surface Plot
# =========================
surface_fig = go.Figure(
    data=[go.Surface(x=X, y=Y, z=Z, showscale=False)]
)

surface_fig.update_layout(
    title="3D Surface Plot of f(x, y)",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="f(x, y)"
    )
)

st.plotly_chart(surface_fig, use_container_width=True)

# =========================
# Contour Plot with Gradient Arrow
# =========================
contour_fig = go.Figure()

# Contour
contour_fig.add_trace(
    go.Contour(
        x=X[0],
        y=Y[:, 0],
        z=Z,
        contours_coloring="lines",
        line_width=2
    )
)

# Gradient arrow
arrow_scale = 0.8
contour_fig.add_trace(
    go.Scatter(
        x=[x0, x0 + arrow_scale * grad_x],
        y=[y0, y0 + arrow_scale * grad_y],
        mode="lines+markers",
        line=dict(width=3),
        marker=dict(size=8),
        name="Gradient Direction"
    )
)

contour_fig.update_layout(
    title="Contour Plot with Gradient Direction",
    xaxis_title="x",
    yaxis_title="y",
    showlegend=True
)

st.plotly_chart(contour_fig, use_container_width=True)

# =========================
# Explanation
# =========================
st.subheader("Explanation")
st.write(
    "The gradient vector points in the direction of the steepest ascent of the function. "
    "On the contour plot, the gradient arrow is perpendicular to the contour lines and "
    "indicates the direction in which the function increases most rapidly."
)
